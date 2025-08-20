//! WASM exports for Bandersnatch RingVRF (Safrole/JAM)

use wasm_bindgen::prelude::*;
use ark_vrf::reexports::{
    ark_serialize::{CanonicalDeserialize, CanonicalSerialize},
};
use bandersnatch::{Public, Secret, RingProofParams, Input, Output, RingProof};
use ark_vrf::suites::bandersnatch;
use std::sync::OnceLock;
use ark_vrf::ring::{Prover, Verifier};


#[wasm_bindgen]
pub struct WasmRingVRFKey {
    pub_bytes: Vec<u8>,
}

#[wasm_bindgen]
impl WasmRingVRFKey {
    #[wasm_bindgen(constructor)]
    pub fn new(pub_bytes: Vec<u8>) -> WasmRingVRFKey {
        WasmRingVRFKey { pub_bytes }
    }
}

const SRS_BYTES: &[u8] = include_bytes!("../data/zcash-srs-2-11-uncompressed.bin");

fn ring_proof_params() -> &'static RingProofParams {
    static PARAMS: OnceLock<RingProofParams> = OnceLock::new();
    PARAMS.get_or_init(|| {
        use bandersnatch::PcsParams;
        let pcs_params = PcsParams::deserialize_uncompressed_unchecked(&mut &SRS_BYTES[..]).unwrap();
        RingProofParams::from_pcs_params(1023, pcs_params).unwrap()
    })
}

fn vrf_input_point(vrf_input_data: &[u8]) -> Input {
    Input::new(vrf_input_data).unwrap()
}

#[wasm_bindgen]
pub fn ringvrf_prove(message: &[u8], ring: &[u8], priv_key: &[u8]) -> Vec<u8> {
    // ring: concatenated public keys (32 bytes each)
    let ring_keys: Vec<Public> = ring.chunks(32).map(|b| Public::deserialize_compressed(b).unwrap()).collect();
    let secret = Secret::deserialize_compressed(priv_key).unwrap();
    let input = vrf_input_point(message);
    let output = secret.output(input);
    let pts: Vec<_> = ring_keys.iter().map(|pk| pk.0).collect();
    let params = ring_proof_params();
    let prover_key = params.prover_key(&pts);
    let prover = params.prover(prover_key, 0); // index unknown, not needed for proof
    let proof = secret.prove(input, output, b"", &prover);
    // Output and Ring Proof bundled together
    let signature = crate::RingVrfSignature { output, proof };
    let mut buf = Vec::new();
    signature.serialize_compressed(&mut buf).unwrap();
    buf
}

#[wasm_bindgen]
pub fn ringvrf_verify(message: &[u8], ring: &[u8], proof: &[u8]) -> JsValue {
    let ring_keys: Vec<Public> = ring.chunks(32).map(|b| Public::deserialize_compressed(b).unwrap()).collect();
    let input = vrf_input_point(message);
    let params = ring_proof_params();
    let pts: Vec<_> = ring_keys.iter().map(|pk| pk.0).collect();
    let verifier_key = params.verifier_key(&pts);
    let commitment = verifier_key.commitment();
    let verifier = params.verifier(verifier_key);
    let signature = crate::RingVrfSignature::deserialize_compressed(proof).unwrap();
    let output = signature.output;
    let ok = bandersnatch::Public::verify(input, output, b"", &signature.proof, &verifier).is_ok();
    let output_bytes = output.hash()[..32].to_vec();
    // Return JS object: { ok: bool, output: Uint8Array }
    let result = js_sys::Object::new();
    js_sys::Reflect::set(&result, &"ok".into(), &JsValue::from_bool(ok)).unwrap();
    js_sys::Reflect::set(&result, &"output".into(), &js_sys::Uint8Array::from(&output_bytes[..])).unwrap();
    result.into()
}

#[wasm_bindgen]
pub fn compose_ring_root(ring: &[u8]) -> Vec<u8> {
    let ring_keys: Vec<Public> = ring.chunks(32)
        .map(|b| Public::deserialize_compressed(b).unwrap())
        .collect();
    let pts: Vec<_> = ring_keys.iter().map(|pk| pk.0).collect();
    let params = ring_proof_params();
    let verifier_key = params.verifier_key(&pts);
    let commitment = verifier_key.commitment();
    let mut buf = Vec::new();
    commitment.serialize_compressed(&mut buf).unwrap();
    buf
}

// Helper struct for serialization
#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct RingVrfSignature {
    output: Output,
    proof: RingProof,
} 