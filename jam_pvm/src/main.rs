#![cfg(feature = "std")]

use axum::{routing::post, Json, Router};
use hex::FromHex;
use parity_scale_codec::{Decode, Encode};
use jam_types::{
    AccumulateItem, AuthOutput, AuthParam, CodeHash, CoreIndex, Hash, RefineContext, ServiceId,
    Slot, TransferRecord, WorkOutput, WorkPackage, WorkPackageHash, WorkPayload,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use tracing_subscriber::{fmt, EnvFilter};
use std::sync::Mutex; 
use jam_pvm::authorizer::MyJamAuthorizer;
use jam_pvm::service::MyJamService;
use jam_pvm_common::Authorizer as _; // trait for is_authorized
use jam_pvm_common::Service as _; // trait for service fns
use once_cell::sync::Lazy; 

static SERVICE_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

#[tokio::main]
async fn main() {
    let _ = fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .try_init();

    let app = Router::new()
        .route("/authorizer/is_authorized", post(authorizer_is_authorized))
        .route("/service/refine", post(service_refine))
        .route("/service/accumulate", post(service_accumulate))
        .route("/service/accumulate_json", post(service_accumulate_json))
        .route("/service/on_transfer", post(service_on_transfer));

    let addr = SocketAddr::from(([127, 0, 0, 1], 8080));
    tracing::info!("listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr).await.expect("bind failed");
    axum::serve(listener, app).await.expect("server failed");
}

// ---- JSON-friendly accumulate endpoint ----

#[derive(Deserialize)]
struct AccumulateItemJson {
    auth_output_hex: String,   // raw bytes hex for AuthOutput
    payload_hash_hex: String,  // 32-byte hex
    result_ok: bool,
    work_output_hex: Option<String>, // hex bytes if ok
    package_hash_hex: String,  // 32-byte hex
    exports_root_hex: String,  // 32-byte hex
    authorizer_hash_hex: String, // 32-byte hex
}

#[derive(Deserialize)]
struct AccumulateJsonInput {
    slot: u32,
    service_id: u32,
    items: Vec<AccumulateItemJson>,
}

fn hex32_to_array(hex_str: &str) -> Result<[u8; 32], String> {
    let v = hex_to_vec(hex_str)?;
    if v.len() != 32 {
        return Err("expected 32 bytes".to_string());
    }
    let mut arr = [0u8; 32];
    arr.copy_from_slice(&v);
    Ok(arr)
}

async fn service_accumulate_json(
    Json(input): Json<AccumulateJsonInput>,
) -> Result<Json<serde_json::Value>, String> {
    // Convert JSON types into jam_types for reuse of service logic
    let slot: Slot = input.slot;
    let id: ServiceId = input.service_id;

    let mut items_conv: Vec<AccumulateItem> = Vec::with_capacity(input.items.len());
    for it in input.items.iter() {
        // auth_output
        let auth_bytes = hex_to_vec(&it.auth_output_hex)?;
        let auth_output = jam_types::AuthOutput(auth_bytes);

        // payload
        let payload_hash = hex32_to_array(&it.payload_hash_hex)?;
        let payload = jam_types::PayloadHash(payload_hash.into());

        // result: support only successful result for now
        if !it.result_ok {
            return Err("result_ok=false not supported in accumulate_json; use SCALE endpoint for error variants".to_string());
        }
        let wo_bytes = it
            .work_output_hex
            .as_ref()
            .ok_or_else(|| "work_output_hex missing while result_ok=true".to_string())
            .and_then(|s| hex_to_vec(s))?;
        let wo = jam_types::WorkOutput(wo_bytes);
        let result = Ok(wo);

        // hashes
        let package_hash_arr = hex32_to_array(&it.package_hash_hex)?;
        let exports_root_arr = hex32_to_array(&it.exports_root_hex)?;
        let authorizer_hash_arr = hex32_to_array(&it.authorizer_hash_hex)?;

        let acc_item = jam_types::AccumulateItem {
            auth_output,
            payload,
            result,
            package: jam_types::WorkPackageHash(package_hash_arr),
            exports_root: exports_root_arr.into(),
            authorizer_hash: authorizer_hash_arr.into(),
        };
        items_conv.push(acc_item);
    }

    // --- Acquire lock before accessing state ---
    let _guard = SERVICE_LOCK.lock().unwrap();
    let _out: Option<Hash> = <MyJamService as jam_pvm_common::Service>::accumulate(slot, id, items_conv);

    Ok(Json(serde_json::json!({"status":"ok"})))
}

// ---- Models ----

#[derive(Deserialize)]
struct HexInput {
    // SCALE-encoded hex string (no 0x prefix) for AuthParam
    param_hex: String,
    // SCALE-encoded hex for WorkPackage
    package_hex: String,
    // SCALE-encoded CoreIndex
    core_index_hex: String,
}

#[derive(Serialize)]
struct HexOutput {
    // SCALE-encoded hex for AuthOutput
    output_hex: String,
}

#[derive(Deserialize)]
struct RefineInput {
    service_id_hex: String, // ServiceId (SCALE hex)
    payload_hex: String,    // WorkPayload (SCALE hex)
    package_hash_hex: String, // WorkPackageHash (SCALE hex)
    context_hex: String,      // RefineContext (SCALE hex)
    auth_code_hash_hex: String, // CodeHash (SCALE hex)
}

#[derive(Serialize)]
struct RefineOutput {
    work_output_hex: String, // WorkOutput (SCALE hex)
}

#[derive(Deserialize)]
struct AccumulateInput {
    slot_hex: String,          // Slot (SCALE hex)
    service_id_hex: String,    // ServiceId
    items_hex: String,         // Vec<AccumulateItem>
}

#[derive(Serialize)]
struct AccumulateOutput {
    hash_hex: Option<String>, // Option<Hash>
}

#[derive(Deserialize)]
struct OnTransferInput {
    slot_hex: String,          // Slot
    service_id_hex: String,    // ServiceId
    transfers_hex: String,     // Vec<TransferRecord>
}

// ---- Helpers ----

fn hex_to_vec(s: &str) -> Result<Vec<u8>, String> {
    Vec::from_hex(s).map_err(|e| format!("invalid hex: {}", e))
}

fn decode_scale<T: Decode>(hex_str: &str) -> Result<T, String> {
    let bytes = hex_to_vec(hex_str)?;
    T::decode(&mut bytes.as_slice()).map_err(|_| "failed to decode".to_string())
}

fn encode_scale<T: Encode>(value: &T) -> String {
    hex::encode(value.encode())
}

// ---- Handlers ----

async fn authorizer_is_authorized(Json(input): Json<HexInput>) -> Result<Json<HexOutput>, String> {
    let param_bytes = hex_to_vec(&input.param_hex)?;
    let param = AuthParam(param_bytes);
    let package: WorkPackage = decode_scale(&input.package_hex)?;
    let core_index: CoreIndex = decode_scale(&input.core_index_hex)?;

    let out: AuthOutput = <MyJamAuthorizer as jam_pvm_common::Authorizer>::is_authorized(
        param,
        package,
        core_index,
    );

    // --- THIS IS THE FIX ---
    // The `out` variable already contains the raw bytes we need (`out.0`).
    // We should hex-encode these bytes directly, NOT SCALE-encode the `AuthOutput` struct again.
    Ok(Json(HexOutput {
        output_hex: hex::encode(&out.0),
    }))
    // --- END OF FIX ---
}

async fn service_refine(Json(input): Json<RefineInput>) -> Result<Json<RefineOutput>, String> {
    let id: ServiceId = decode_scale(&input.service_id_hex)?;
    let payload: WorkPayload = decode_scale(&input.payload_hex)?;
    let package_hash: WorkPackageHash = decode_scale(&input.package_hash_hex)?;
    let context: RefineContext = decode_scale(&input.context_hex)?;
    let auth_code_hash: CodeHash = decode_scale(&input.auth_code_hash_hex)?;

    let out: WorkOutput = <MyJamService as jam_pvm_common::Service>::refine(
        id,
        payload,
        package_hash,
        context,
        auth_code_hash,
    );
    // Ok(Json(RefineOutput {
    //     work_output_hex: encode_scale(&out),
    // }))
    Ok(Json(RefineOutput {
        work_output_hex: hex::encode(&out.0),
    }))
}

async fn service_accumulate(
    Json(input): Json<AccumulateInput>,
) -> Result<Json<AccumulateOutput>, String> {
    let slot: Slot = decode_scale(&input.slot_hex)?;
    let id: ServiceId = decode_scale(&input.service_id_hex)?;
    let items: Vec<AccumulateItem> = decode_scale(&input.items_hex)?;

    // --- FIX: Acquire lock before accessing state ---
    let _guard = SERVICE_LOCK.lock().unwrap();
    let out: Option<Hash> = <MyJamService as jam_pvm_common::Service>::accumulate(slot, id, items);
    
    Ok(Json(AccumulateOutput {
        hash_hex: out.map(|h| encode_scale(&h)),
    }))
}

async fn service_on_transfer(Json(input): Json<OnTransferInput>) -> Result<Json<serde_json::Value>, String> {
    let slot: Slot = decode_scale(&input.slot_hex)?;
    let id: ServiceId = decode_scale(&input.service_id_hex)?;
    let transfers: Vec<TransferRecord> = decode_scale(&input.transfers_hex)?;

    // --- FIX: Acquire lock before accessing state ---
    let _guard = SERVICE_LOCK.lock().unwrap();
    <MyJamService as jam_pvm_common::Service>::on_transfer(slot, id, transfers);

    Ok(Json(serde_json::json!({"status":"ok"})))
}
