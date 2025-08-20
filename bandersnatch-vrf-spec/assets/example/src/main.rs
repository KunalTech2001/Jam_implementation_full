use ark_vrf::reexports::{
    ark_ec::AffineRepr,
    ark_serialize::{self, CanonicalDeserialize, CanonicalSerialize},
};
use ark_vrf::{pedersen::PedersenSuite, ring::RingSuite, suites::bandersnatch};
use bandersnatch::{
    AffinePoint, BandersnatchSha512Ell2, IetfProof, Input, Output, Public, RingProof,
    RingProofParams, Secret,
};
use axum::{extract::Json, routing::{get, post}, Router, http::StatusCode};
use serde::{Deserialize, Serialize};
use tokio::net::TcpListener;
use std::collections::HashMap as StdHashMap;
use std::sync::{Arc, Mutex};
use uuid::Uuid; // Ensure Uuid is in scope
const RING_SIZE: usize = 6;

// This is the IETF `Prove` procedure output as described in section 2.2
// of the Bandersnatch VRF specification
#[derive(CanonicalSerialize, CanonicalDeserialize)]
struct IetfVrfSignature {
    output: Output,
    proof: IetfProof,
}

// This is the IETF `Prove` procedure output as described in section 4.2
// of the Bandersnatch VRF specification
#[derive(CanonicalSerialize, CanonicalDeserialize)]
struct RingVrfSignature {
    output: Output,
    // This contains both the Pedersen proof and actual ring proof.
    proof: RingProof,
}

// "Static" ring proof parameters.
fn ring_proof_params() -> &'static RingProofParams {
    use std::sync::OnceLock;
    static PARAMS: OnceLock<RingProofParams> = OnceLock::new();
    PARAMS.get_or_init(|| {
        use bandersnatch::PcsParams;
        use std::{fs::File, io::Read};
        let manifest_dir =
            std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is not set");
        let filename = format!("{}/data/zcash-srs-2-11-uncompressed.bin", manifest_dir);
        let mut file = File::open(filename).unwrap();
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).unwrap();
        let pcs_params = PcsParams::deserialize_uncompressed_unchecked(&mut &buf[..]).unwrap();
        RingProofParams::from_pcs_params(RING_SIZE, pcs_params).unwrap()
    })
}

// Construct VRF Input Point from arbitrary data (section 1.2)
fn vrf_input_point(vrf_input_data: &[u8]) -> Input {
    Input::new(vrf_input_data).unwrap()
}

// Prover actor.
struct Prover {
    pub prover_idx: usize,
    pub secret: Secret,
    pub ring: Vec<Public>,
}

impl Prover {
    pub fn new(ring: Vec<Public>, prover_idx: usize) -> Self {
        Self {
            prover_idx,
            secret: Secret::from_seed(&prover_idx.to_le_bytes()),
            ring,
        }
    }

    /// VRF output hash.
    pub fn vrf_output(&self, vrf_input_data: &[u8]) -> Vec<u8> {
        let input = vrf_input_point(vrf_input_data);
        let output = self.secret.output(input);
        output.hash()[..32].try_into().unwrap()
    }

    /// Anonymous VRF signature.
    ///
    /// Used for tickets submission.
    pub fn ring_vrf_sign(&self, vrf_input_data: &[u8], aux_data: &[u8]) -> Vec<u8> {
        use ark_vrf::ring::Prover as _;

        let input = vrf_input_point(vrf_input_data);
        let output = self.secret.output(input);

        // Backend currently requires the wrapped type (plain affine points)
        let pts: Vec<_> = self.ring.iter().map(|pk| pk.0).collect();

        // Proof construction
        let params = ring_proof_params();
        let prover_key = params.prover_key(&pts);
        let prover = params.prover(prover_key, self.prover_idx);
        let proof = self.secret.prove(input, output, aux_data, &prover);

        // Output and Ring Proof bundled together (as per section 2.2)
        let signature = RingVrfSignature { output, proof };
        let mut buf = Vec::new();
        signature.serialize_compressed(&mut buf).unwrap();
        buf
    }

    /// Non-Anonymous VRF signature.
    ///
    /// Used for ticket claiming during block production.
    /// Not used with Safrole test vectors.
    pub fn ietf_vrf_sign(&self, vrf_input_data: &[u8], aux_data: &[u8]) -> Vec<u8> {
        use ark_vrf::ietf::Prover as _;

        let input = vrf_input_point(vrf_input_data);
        let output = self.secret.output(input);

        let proof = self.secret.prove(input, output, aux_data);

        // Output and IETF Proof bundled together (as per section 2.2)
        let signature = IetfVrfSignature { output, proof };
        let mut buf = Vec::new();
        signature.serialize_compressed(&mut buf).unwrap();
        buf
    }
}

type RingCommitment = ark_vrf::ring::RingCommitment<BandersnatchSha512Ell2>;

// Verifier actor.
struct Verifier {
    pub commitment: RingCommitment,
    pub ring: Vec<Public>,
}

impl Verifier {
    fn new(ring: Vec<Public>) -> Self {
        // Backend currently requires the wrapped type (plain affine points)
        let pts: Vec<_> = ring.iter().map(|pk| pk.0).collect();
        let verifier_key = ring_proof_params().verifier_key(&pts);
        let commitment = verifier_key.commitment();
        Self { ring, commitment }
    }

    /// Anonymous VRF signature verification.
    ///
    /// Used for tickets verification.
    ///
    /// On success returns the VRF output hash.
    pub fn ring_vrf_verify(
        &self,
        vrf_input_data: &[u8],
        aux_data: &[u8],
        signature: &[u8],
    ) -> Result<[u8; 32], ()> {
        use ark_vrf::ring::Verifier as _;

        let signature = RingVrfSignature::deserialize_compressed(signature).unwrap();

        let input = vrf_input_point(vrf_input_data);
        let output = signature.output;

        let params = ring_proof_params();

        let verifier_key = params.verifier_key_from_commitment(self.commitment.clone());
        let verifier = params.verifier(verifier_key);
        if Public::verify(input, output, aux_data, &signature.proof, &verifier).is_err() {
            println!("Ring signature verification failure");
            return Err(());
        }
        println!("Ring signature verified");

        let vrf_output_hash: [u8; 32] = output.hash()[..32].try_into().unwrap();
        println!(" vrf-output-hash: {}", hex::encode(vrf_output_hash));
        Ok(vrf_output_hash)
    }

    /// Non-Anonymous VRF signature verification.
    ///
    /// Used for ticket claim verification during block import.
    /// Not used with Safrole test vectors.
    ///
    /// On success returns the VRF output hash.
    pub fn ietf_vrf_verify(
        &self,
        vrf_input_data: &[u8],
        aux_data: &[u8],
        signature: &[u8],
        signer_key_index: usize,
    ) -> Result<[u8; 32], ()> {
        use ark_vrf::ietf::Verifier as _;

        let signature = IetfVrfSignature::deserialize_compressed(signature).unwrap();

        let input = vrf_input_point(vrf_input_data);
        let output = signature.output;

        let public = &self.ring[signer_key_index];
        if public
            .verify(input, output, aux_data, &signature.proof)
            .is_err()
        {
            println!("IETF signature verification failure");
            return Err(());
        }
        println!("IETF signature verified");

        let vrf_output_hash: [u8; 32] = output.hash()[..32].try_into().unwrap();
        println!(" vrf-output-hash: {}", hex::encode(vrf_output_hash));
        Ok(vrf_output_hash)
    }
}

fn print_point(name: &str, p: AffinePoint) {
    println!("------------------------------");
    println!("[{name}]");
    println!("X: {}", p.x);
    println!("Y: {}", p.y);
    let mut buf = Vec::new();
    p.serialize_compressed(&mut buf).unwrap();
    println!("Compressed: 0x{}", hex::encode(buf));
}

fn print_points() {
    println!("==============================");
    print_point("Group Base", AffinePoint::generator());
    print_point("Blinding Base", BandersnatchSha512Ell2::BLINDING_BASE);
    print_point("Ring Padding", BandersnatchSha512Ell2::PADDING);
    print_point("Accumulator Base", BandersnatchSha512Ell2::ACCUMULATOR_BASE);
    println!("==============================");
}

// In main.rs

fn compose_gamma_z(public_keys: &[String]) -> Vec<u8> {
    const PADDING_KEY_HEX: &str = "0000000000000000000000000000000000000000000000000000000000000000";
    let mut ring_keys = Vec::new();

    // DO NOT SORT THE KEYS. Process them in the order received from the client.
    for pk_hex in public_keys {
        if pk_hex.trim_start_matches("0x") == PADDING_KEY_HEX {
            // Handle the padding key to prevent crashes
            let padding_point = Public::from(BandersnatchSha512Ell2::PADDING);
            ring_keys.push(padding_point);
        } else {
            // Handle the normal keys
            let bytes = hex::decode(pk_hex.trim_start_matches("0x")).unwrap();
            let pk = Public::deserialize_compressed(&bytes[..]).unwrap();
            ring_keys.push(pk);
        }
    }

    // The rest of the function remains the same
    let pts: Vec<_> = ring_keys.iter().map(|pk| pk.0).collect();
    let params = ring_proof_params();
    let verifier_key = params.verifier_key(&pts);
    let commitment = verifier_key.commitment();
    let mut buf = Vec::new();
    commitment.serialize_compressed(&mut buf).unwrap();
    buf
}
// API Request/Response types
#[derive(Deserialize)]
struct GammaZRequest {
    public_keys: Vec<String>,
}

#[derive(Serialize)]
struct GammaZResponse {
    gamma_z: String,
}

#[derive(Deserialize)]
struct CreateProverRequest {
    public_keys: Vec<String>,
    prover_index: usize,
}

#[derive(Serialize)]
struct CreateProverResponse {
    prover_id: String,
    public_key: String,
}

#[derive(Deserialize)]
struct VrfOutputRequest {
    prover_id: String,
    vrf_input_data: String,
}

#[derive(Serialize)]
struct VrfOutputResponse {
    vrf_output_hash: String,
}

#[derive(Deserialize)]
struct RingVrfSignRequest {
    prover_id: String,
    vrf_input_data: String,
    aux_data: String,
}

#[derive(Serialize)]
struct RingVrfSignResponse {
    signature: String,
}

#[derive(Deserialize)]
struct IetfVrfSignRequest {
    prover_id: String,
    vrf_input_data: String,
    aux_data: String,
}

#[derive(Serialize)]
struct IetfVrfSignResponse {
    signature: String,
}

#[derive(Deserialize)]
struct CreateVerifierRequest {
    public_keys: Vec<String>,
}

#[derive(Serialize)]
struct CreateVerifierResponse {
    verifier_id: String,
    commitment: String,
}

#[derive(Deserialize)]
struct RingVrfVerifyRequest {
    verifier_id: String,
    vrf_input_data: String,
    aux_data: String,
    signature: String,
}

#[derive(Deserialize)]
struct RingVrfVerifyPayloadRequest {
    gamma_z: String,
    ring_set: Vec<String>,
    eta2_prime: String,
    extrinsic: Vec<ExtrinsicItem>,
}

#[derive(Deserialize)]
struct ExtrinsicItem {
    attempt: u8,
    signature: String,
}

#[derive(Serialize)]
struct RingVrfVerifyResponse {
    verified: bool,
    vrf_output_hash: Option<String>,
}

#[derive(Serialize)]
struct RingVrfVerifyPayloadResponse {
    results: Vec<TicketVerificationResult>,
}

#[derive(Serialize)]
struct TicketVerificationResult {
    attempt: u8,
    ok: bool,
    output_hash: Option<String>,
    message: String,
}

#[derive(Deserialize)]
struct IetfVrfVerifyRequest {
    verifier_id: String,
    vrf_input_data: String,
    aux_data: String,
    signature: String,
    signer_key_index: usize,
}

#[derive(Serialize)]
struct IetfVrfVerifyResponse {
    verified: bool,
    vrf_output_hash: Option<String>,
}

#[derive(Serialize)]
struct PointInfo {
    name: String,
    x: String,
    y: String,
    compressed: String,
}

#[derive(Serialize)]
struct ConstantPointsResponse {
    points: Vec<PointInfo>,
}

#[derive(Serialize)]
struct ApiEndpoint {
    method: String,
    path: String,
    description: String,
}

#[derive(Serialize)]
struct ApiDocResponse {
    endpoints: Vec<ApiEndpoint>,
}

// Global storage for provers and verifiers
type ProverStorage = Arc<Mutex<StdHashMap<String, Prover>>>;
type VerifierStorage = Arc<Mutex<StdHashMap<String, Verifier>>>;

// API Handlers
async fn compose_gamma_z_handler(Json(req): Json<GammaZRequest>) -> Json<GammaZResponse> {
    let gamma_z_bytes = compose_gamma_z(&req.public_keys);
    Json(GammaZResponse {
        gamma_z: format!("0x{}", hex::encode(gamma_z_bytes)),
    })
}

async fn create_prover_handler(
    axum::extract::State((prover_storage, _)): axum::extract::State<(ProverStorage, VerifierStorage)>,
    Json(req): Json<CreateProverRequest>,
) -> Result<Json<CreateProverResponse>, StatusCode> {
    let mut ring_keys = Vec::new();
    for pk_hex in &req.public_keys {
        let bytes = hex::decode(pk_hex.trim_start_matches("0x"))
            .map_err(|_| StatusCode::BAD_REQUEST)?;
        let pk = Public::deserialize_compressed(&bytes[..])
            .map_err(|_| StatusCode::BAD_REQUEST)?;
        ring_keys.push(pk);
    }

    if req.prover_index >= ring_keys.len() {
        return Err(StatusCode::BAD_REQUEST);
    }

    let prover = Prover::new(ring_keys.clone(), req.prover_index);
    // This line requires the `v4` feature in `Cargo.toml` for the `uuid` crate.
    let prover_id = Uuid::new_v4().to_string();

    let public_key = &ring_keys[req.prover_index];
    let mut pk_buf = Vec::new();
    public_key.serialize_compressed(&mut pk_buf).unwrap();

    prover_storage.lock().unwrap().insert(prover_id.clone(), prover);

    Ok(Json(CreateProverResponse {
        prover_id,
        public_key: format!("0x{}", hex::encode(pk_buf)),
    }))
}

async fn vrf_output_handler(
    axum::extract::State((prover_storage, _)): axum::extract::State<(ProverStorage, VerifierStorage)>,
    Json(req): Json<VrfOutputRequest>,
) -> Result<Json<VrfOutputResponse>, StatusCode> {
    let vrf_input_data = hex::decode(req.vrf_input_data.trim_start_matches("0x"))
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    let storage = prover_storage.lock().unwrap();
    let prover = storage.get(&req.prover_id).ok_or(StatusCode::NOT_FOUND)?;

    let output_hash = prover.vrf_output(&vrf_input_data);

    Ok(Json(VrfOutputResponse {
        vrf_output_hash: format!("0x{}", hex::encode(output_hash)),
    }))
}

async fn ring_vrf_sign_handler(
    axum::extract::State((prover_storage, _)): axum::extract::State<(ProverStorage, VerifierStorage)>,
    Json(req): Json<RingVrfSignRequest>,
) -> Result<Json<RingVrfSignResponse>, StatusCode> {
    let vrf_input_data = hex::decode(req.vrf_input_data.trim_start_matches("0x"))
        .map_err(|_| StatusCode::BAD_REQUEST)?;
    let aux_data = hex::decode(req.aux_data.trim_start_matches("0x"))
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    let storage = prover_storage.lock().unwrap();
    let prover = storage.get(&req.prover_id).ok_or(StatusCode::NOT_FOUND)?;

    let signature = prover.ring_vrf_sign(&vrf_input_data, &aux_data);

    Ok(Json(RingVrfSignResponse {
        signature: format!("0x{}", hex::encode(signature)),
    }))
}

async fn ietf_vrf_sign_handler(
    axum::extract::State((prover_storage, _)): axum::extract::State<(ProverStorage, VerifierStorage)>,
    Json(req): Json<IetfVrfSignRequest>,
) -> Result<Json<IetfVrfSignResponse>, StatusCode> {
    let vrf_input_data = hex::decode(req.vrf_input_data.trim_start_matches("0x"))
        .map_err(|_| StatusCode::BAD_REQUEST)?;
    let aux_data = hex::decode(req.aux_data.trim_start_matches("0x"))
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    let storage = prover_storage.lock().unwrap();
    let prover = storage.get(&req.prover_id).ok_or(StatusCode::NOT_FOUND)?;

    let signature = prover.ietf_vrf_sign(&vrf_input_data, &aux_data);

    Ok(Json(IetfVrfSignResponse {
        signature: format!("0x{}", hex::encode(signature)),
    }))
}

async fn create_verifier_handler(
    axum::extract::State((_, verifier_storage)): axum::extract::State<(ProverStorage, VerifierStorage)>,
    Json(req): Json<CreateVerifierRequest>,
) -> Result<Json<CreateVerifierResponse>, StatusCode> {
    let mut ring_keys = Vec::new();
    for pk_hex in &req.public_keys {
        let bytes = hex::decode(pk_hex.trim_start_matches("0x"))
            .map_err(|_| StatusCode::BAD_REQUEST)?;
        let pk = Public::deserialize_compressed(&bytes[..])
            .map_err(|_| StatusCode::BAD_REQUEST)?;
        ring_keys.push(pk);
    }

    let verifier = Verifier::new(ring_keys);
    // This line requires the `v4` feature in `Cargo.toml` for the `uuid` crate.
    let verifier_id = Uuid::new_v4().to_string();

    let mut commitment_buf = Vec::new();
    verifier.commitment.serialize_compressed(&mut commitment_buf).unwrap();

    verifier_storage.lock().unwrap().insert(verifier_id.clone(), verifier);

    Ok(Json(CreateVerifierResponse {
        verifier_id,
        commitment: format!("0x{}", hex::encode(commitment_buf)),
    }))
}

async fn ring_vrf_verify_handler(
    axum::extract::State((_, verifier_storage)): axum::extract::State<(ProverStorage, VerifierStorage)>,
    Json(req): Json<RingVrfVerifyRequest>,
) -> Result<Json<RingVrfVerifyResponse>, StatusCode> {
    let vrf_input_data = hex::decode(req.vrf_input_data.trim_start_matches("0x"))
        .map_err(|_| StatusCode::BAD_REQUEST)?;
    let aux_data = hex::decode(req.aux_data.trim_start_matches("0x"))
        .map_err(|_| StatusCode::BAD_REQUEST)?;
    let signature = hex::decode(req.signature.trim_start_matches("0x"))
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    let storage = verifier_storage.lock().unwrap();
    let verifier = storage.get(&req.verifier_id).ok_or(StatusCode::NOT_FOUND)?;

    match verifier.ring_vrf_verify(&vrf_input_data, &aux_data, &signature) {
        Ok(output_hash) => Ok(Json(RingVrfVerifyResponse {
            verified: true,
            vrf_output_hash: Some(format!("0x{}", hex::encode(output_hash))),
        })),
        Err(_) => Ok(Json(RingVrfVerifyResponse {
            verified: false,
            vrf_output_hash: None,
        })),
    }
}

async fn ring_vrf_verify_payload_handler(
    Json(req): Json<RingVrfVerifyPayloadRequest>,
) -> Result<Json<RingVrfVerifyPayloadResponse>, StatusCode> {
    //==============================================================================
    // 1. PARSE THE INPUT DATA
    //==============================================================================
    println!("Verifying payload with gamma_z: {}", req.gamma_z.trim_start_matches("0x"));
    // Parse gamma_z (ring commitment)
    let gamma_z_bytes = hex::decode(req.gamma_z.trim_start_matches("0x"))
        .map_err(|_| StatusCode::BAD_REQUEST)?;
    
    // Parse ring_set (public keys)
    let mut ring: Vec<Public> = Vec::new();
    for hex_str in &req.ring_set {
        let bytes = hex::decode(hex_str.trim_start_matches("0x"))
            .map_err(|_| StatusCode::BAD_REQUEST)?;
        let public_key = Public::deserialize_compressed(&bytes[..])
            .map_err(|_| StatusCode::BAD_REQUEST)?;
        ring.push(public_key);
    }
    
    // Parse eta2_prime
    let eta2_prime_bytes = hex::decode(req.eta2_prime.trim_start_matches("0x"))
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    //==============================================================================
    // 2. SETUP THE VERIFIER BY CALCULATING THE COMMITMENT FROM THE RING
    //==============================================================================
    println!("Calculating commitment from the ring of {} public keys...", ring.len());
    let verifier = Verifier::new(ring);

    // Verify that our calculated commitment matches the provided gamma_z
    let mut calculated_commitment_bytes = Vec::new();
    verifier.commitment.serialize_compressed(&mut calculated_commitment_bytes).unwrap();
    let calculated_commitment_hex = hex::encode(&calculated_commitment_bytes);
    let expected_gamma_z_hex = hex::encode(&gamma_z_bytes);
    
    if calculated_commitment_hex != expected_gamma_z_hex {
        return Ok(Json(RingVrfVerifyPayloadResponse {
            results: vec![TicketVerificationResult {
                attempt: 0,
                ok: false,
                output_hash: None,
                message: "Calculated commitment (gamma_z) does not match the provided gamma_z".to_string(),
            }],
        }));
    }
    println!("✓ Commitment matches provided gamma_z.");

    //==============================================================================
    // 3. VERIFY EACH TICKET IN THE EXTRINSIC
    //==============================================================================
    let mut results = Vec::new();
    
    for extrinsic_item in req.extrinsic.iter() {
        println!("\n--- Verifying Ticket for Attempt {} ---", extrinsic_item.attempt);
        
        // Try to decode the signature, handle errors gracefully
        println!("Signature: {}", extrinsic_item.signature);
        let signature_result = hex::decode(extrinsic_item.signature.trim_start_matches("0x"));
        if let Err(e) = signature_result {
            println!("✗ Failed to decode signature for attempt {}: {}", extrinsic_item.attempt, e);
            results.push(TicketVerificationResult {
                attempt: extrinsic_item.attempt,
                ok: false,
                output_hash: None,
                message: format!("Failed to decode signature: {}", e),
            });
            continue;
        }
        
        let signature = signature_result.unwrap();
        
        // As per the JAM specification notation `○V[] γ′Z ⟨XT ⌢ η′2 ⌢ e⟩`:
        // The message `m` is empty, which maps to `aux_data`.
        let aux_data = b"";

        // The context `x` is the full concatenation, which maps to `vrf_input_data`.
        let domain_separator = b"jam_ticket_seal";
        let mut vrf_input_data = Vec::new();
        vrf_input_data.extend_from_slice(domain_separator);
        vrf_input_data.extend_from_slice(&eta2_prime_bytes);
        vrf_input_data.push(extrinsic_item.attempt); // Use big-endian

        // Verifier checks the signature anonymously.
        match verifier.ring_vrf_verify(&vrf_input_data, aux_data, &signature) {
            Ok(output_hash) => {
                println!("✓ Ticket {} verified successfully", extrinsic_item.attempt);
                results.push(TicketVerificationResult {
                    attempt: extrinsic_item.attempt,
                    ok: true,
                    output_hash: Some(format!("0x{}", hex::encode(output_hash))),
                    message: format!("Ticket {} verified successfully", extrinsic_item.attempt),
                });
            },
            Err(_) => {
                println!("✗ Ticket {} verification failed", extrinsic_item.attempt);
                results.push(TicketVerificationResult {
                    attempt: extrinsic_item.attempt,
                    ok: false,
                    output_hash: None,
                    message: format!("Ticket {} verification failed", extrinsic_item.attempt),
                });
            }
        }
    }

    Ok(Json(RingVrfVerifyPayloadResponse { results }))
}

async fn ietf_vrf_verify_handler(
    axum::extract::State((_, verifier_storage)): axum::extract::State<(ProverStorage, VerifierStorage)>,
    Json(req): Json<IetfVrfVerifyRequest>,
) -> Result<Json<IetfVrfVerifyResponse>, StatusCode> {
    let vrf_input_data = hex::decode(req.vrf_input_data.trim_start_matches("0x"))
        .map_err(|_| StatusCode::BAD_REQUEST)?;
    let aux_data = hex::decode(req.aux_data.trim_start_matches("0x"))
        .map_err(|_| StatusCode::BAD_REQUEST)?;
    let signature = hex::decode(req.signature.trim_start_matches("0x"))
        .map_err(|_| StatusCode::BAD_REQUEST)?;

    let storage = verifier_storage.lock().unwrap();
    let verifier = storage.get(&req.verifier_id).ok_or(StatusCode::NOT_FOUND)?;

    match verifier.ietf_vrf_verify(&vrf_input_data, &aux_data, &signature, req.signer_key_index) {
        Ok(output_hash) => Ok(Json(IetfVrfVerifyResponse {
            verified: true,
            vrf_output_hash: Some(format!("0x{}", hex::encode(output_hash))),
        })),
        Err(_) => Ok(Json(IetfVrfVerifyResponse {
            verified: false,
            vrf_output_hash: None,
        })),
    }
}

async fn constant_points_handler() -> Json<ConstantPointsResponse> {
    let points = vec![
        ("Group Base", AffinePoint::generator()),
        ("Blinding Base", BandersnatchSha512Ell2::BLINDING_BASE),
        ("Ring Padding", BandersnatchSha512Ell2::PADDING),
        ("Accumulator Base", BandersnatchSha512Ell2::ACCUMULATOR_BASE),
    ];

    let point_infos: Vec<PointInfo> = points.into_iter().map(|(name, point)| {
        let mut buf = Vec::new();
        point.serialize_compressed(&mut buf).unwrap();
        PointInfo {
            name: name.to_string(),
            x: point.x.to_string(),
            y: point.y.to_string(),
            compressed: format!("0x{}", hex::encode(buf)),
        }
    }).collect();

    Json(ConstantPointsResponse { points: point_infos })
}

async fn api_docs_handler() -> Json<ApiDocResponse> {
    let endpoints = vec![
        ApiEndpoint {
            method: "GET".to_string(),
            path: "/".to_string(),
            description: "API documentation".to_string(),
        },
        ApiEndpoint {
            method: "GET".to_string(),
            path: "/constant_points".to_string(),
            description: "Get constant elliptic curve points used in Bandersnatch VRF".to_string(),
        },
        ApiEndpoint {
            method: "POST".to_string(),
            path: "/compose_gamma_z".to_string(),
            description: "Compose gamma_z (ring commitment) from public keys".to_string(),
        },
        ApiEndpoint {
            method: "POST".to_string(),
            path: "/prover/create".to_string(),
            description: "Create a new prover instance with a ring of public keys".to_string(),
        },
        ApiEndpoint {
            method: "POST".to_string(),
            path: "/prover/vrf_output".to_string(),
            description: "Generate VRF output hash for given input data".to_string(),
        },
        ApiEndpoint {
            method: "POST".to_string(),
            path: "/prover/ring_vrf_sign".to_string(),
            description: "Create anonymous VRF signature (ring signature)".to_string(),
        },
        ApiEndpoint {
            method: "POST".to_string(),
            path: "/prover/ietf_vrf_sign".to_string(),
            description: "Create non-anonymous VRF signature (IETF standard)".to_string(),
        },
        ApiEndpoint {
            method: "POST".to_string(),
            path: "/verifier/create".to_string(),
            description: "Create a new verifier instance with a ring of public keys".to_string(),
        },
        ApiEndpoint {
            method: "POST".to_string(),
            path: "/verifier/ring_vrf_verify".to_string(),
            description: "Verify anonymous VRF signature (ring signature)".to_string(),
        },
        ApiEndpoint {
            method: "POST".to_string(),
            path: "/verifier/ring_vrf_verify_payload".to_string(),
            description: "Verify ring VRF signature with payload (gamma_z, ring_set, eta2_prime, extrinsic)".to_string(),
        },
        ApiEndpoint {
            method: "POST".to_string(),
            path: "/verifier/ietf_vrf_verify".to_string(),
            description: "Verify non-anonymous VRF signature (IETF standard)".to_string(),
        },
    ];

    Json(ApiDocResponse { endpoints })
}

#[tokio::main]
async fn main() {
    print_points();

    let prover_storage: ProverStorage = Arc::new(Mutex::new(StdHashMap::new()));
    let verifier_storage: VerifierStorage = Arc::new(Mutex::new(StdHashMap::new()));
    let state = (prover_storage, verifier_storage);

    let app = Router::new()
        .route("/", get(api_docs_handler))
        .route("/constant_points", get(constant_points_handler))
        .route("/compose_gamma_z", post(compose_gamma_z_handler))
        .route("/prover/create", post(create_prover_handler))
        .route("/prover/vrf_output", post(vrf_output_handler))
        .route("/prover/ring_vrf_sign", post(ring_vrf_sign_handler))
        .route("/prover/ietf_vrf_sign", post(ietf_vrf_sign_handler))
        .route("/verifier/create", post(create_verifier_handler))
        .route("/verifier/ring_vrf_verify", post(ring_vrf_verify_handler))
        .route("/verifier/ring_vrf_verify_payload", post(ring_vrf_verify_payload_handler))
        .route("/verifier/ietf_vrf_verify", post(ietf_vrf_verify_handler))
        .with_state(state);

    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], 3000));
    println!("🚀 Bandersnatch VRF API Server listening on http://{}", addr);
    println!("📖 Visit http://{} for API documentation", addr);

    let listener = TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// ADD THIS AT THE END OF YOUR main.rs FILE
#[cfg(test)]
mod tests {
    use super::*; // Import everything from the parent module (main.rs)
    use ark_vrf::ring::{Prover, Verifier};

    #[test]
    fn sign_and_verify_loop_should_pass() {
        println!("--- Running final Rust sign-and-verify test ---");

        // 1. Define the ring and prover index from the test vector
        let public_keys_hex = vec![
            "0xff71c6c03ff88adb5ed52c9681de1629a54e702fc14729f6b50d2f0a76f185b3",
            "0xdee6d555b82024f1ccf8a1e37e60fa60fd40b1958c4bb3006af78647950e1b91",
            "0x9326edb21e5541717fde24ec085000b28709847b8aab1ac51f84e94b37ca1b66",
            "0x0746846d17469fb2f95ef365efcab9f4e22fa1feb53111c995376be8019981cc",
            "0x151e5c8fe2b9d8a606966a79edd2f9e5db47e83947ce368ccba53bf6ba20a40b",
            "0x2105650944fcd101621fd5bb3124c9fd191d114b7ad936c1d79d734f9f21392e"
        ];
        let ring: Vec<Public> = public_keys_hex.iter().map(|hex| {
            let bytes = hex::decode(hex.trim_start_matches("0x")).unwrap();
            Public::deserialize_compressed(&bytes[..]).unwrap()
        }).collect();
        let prover_index = 3;

        // 2. Construct the VRF inputs
        // CORRECTED: All context goes into `vrf_input_data`. `aux_data` is empty.
        let jam_ticket_seal_bytes = b"$jam_ticket_seal";
        let historical_entropy_eta2_bytes = hex::decode("bb30a42c1e62f0afda5f0a4e8a562f7a13a24cea00ee81917b86b89e801314aa").unwrap();
        let attempt_index_bytes = vec![0b00000100]; // Compact encoding for `1`
        
        let mut vrf_input_data = Vec::new();
        vrf_input_data.extend_from_slice(jam_ticket_seal_bytes);
        vrf_input_data.extend_from_slice(&historical_entropy_eta2_bytes);
        vrf_input_data.extend_from_slice(&attempt_index_bytes);
        
        let aux_data = Vec::new(); // `aux_data` must be empty

        // 3. Create Prover and generate a signature
        let secret = Secret::from_seed(&prover_index.to_le_bytes());
        let prover = Prover::new(ring.clone(), prover_index);
        let signature_bytes = prover.ring_vrf_sign(&vrf_input_data, &aux_data);
        println!("Signature generated successfully.");

        // 4. Create Verifier
        let verifier = Verifier::new(ring.clone());
        println!("Verifier created successfully.");

        // 5. Verify the generated signature
        let verification_result = verifier.ring_vrf_verify(&vrf_input_data, &aux_data, &signature_bytes);

        // 6. Assert that the verification must succeed
        assert!(verification_result.is_ok(), "Verification failed! The prover and verifier are inconsistent.");
        
        println!("\n✅✅✅ SUCCESS: The self-generated signature was verified correctly in Rust. ✅✅✅");
        println!("This confirms the correct input construction.");
    }
}