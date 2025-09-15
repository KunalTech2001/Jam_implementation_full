extern crate alloc;

use crate::types::AuthCredentials;
use ed25519_dalek::{ Signature, VerifyingKey };
use jam_codec::Decode;
use jam_pvm_common::{ declare_authorizer, info, Authorizer };
use jam_types::{ AuthOutput, AuthParam, CoreIndex, WorkPackage };
use serde::{Serialize, Deserialize};
use sha2::{ Digest, Sha256 };
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Mutex;

lazy_static::lazy_static! {
    static ref AUTH_STATE: Mutex<AuthState> = Mutex::new(load_auth_state());
}

#[derive(Debug, Serialize, Deserialize, Default)]
pub struct AuthState {
    pub nonces: HashMap<[u8; 32], u64>,
    pub authorizations: HashMap<String, AuthRecord>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct AuthRecord {
    pub public_key: String,
    pub nonce: u64,
    pub last_updated: String,
    pub payload: serde_json::Value,
}

const STATE_FILE: &str = "../server/updated_state.json";

fn load_auth_state() -> AuthState {
    let path = Path::new(STATE_FILE);
    if !path.exists() {
        return AuthState::default();
    }

    match fs::read_to_string(path) {
        Ok(contents) => {
            serde_json::from_str(&contents).unwrap_or_else(|_| {
                eprintln!("Failed to parse auth state, using default");
                AuthState::default()
            })
        }
        Err(e) => {
            eprintln!("Failed to read auth state: {}", e);
            AuthState::default()
        }
    }
}

fn save_auth_state(state: &AuthState) -> std::io::Result<()> {
    let serialized = serde_json::to_string_pretty(state)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    fs::write(STATE_FILE, serialized)
}

pub struct MyJamAuthorizer;

declare_authorizer!(MyJamAuthorizer);

impl Authorizer for MyJamAuthorizer {
    fn is_authorized(param: AuthParam, package: WorkPackage, _core_index: CoreIndex) -> AuthOutput {
        info!(target = "authorizer", "Executing is_authorized logic.");

        // --- Decode incoming credentials ---
        let creds: AuthCredentials = match AuthCredentials::decode(&mut param.0.as_slice()) {
            Ok(creds) => creds,
            Err(_) => {
                return AuthOutput(Sha256::digest(b"DECODE_ERROR").to_vec());
            }
        };

        // --- NONCE VERIFICATION ---
        let mut state = AUTH_STATE.lock().unwrap();
        let public_key_hex = hex::encode(creds.public_key);
        let expected_nonce = state.nonces.get(&creds.public_key).cloned().unwrap_or(0);

        if creds.nonce != expected_nonce {
            info!(
                target= "authorizer",
                "Auth failed: Invalid nonce for {}. Expected {}, got {}.",
                public_key_hex,
                expected_nonce,
                creds.nonce
            );
            return AuthOutput(Sha256::digest(b"INVALID_NONCE").to_vec());
        }

        // Update nonce for next time
        state.nonces.insert(creds.public_key, creds.nonce + 1);

        // Save authorization record
        let record = AuthRecord {
            public_key: public_key_hex,
            nonce: creds.nonce,
            last_updated: chrono::Utc::now().to_rfc3339(),
            payload: serde_json::from_slice(&package.items[0].payload)
                .unwrap_or_else(|_| serde_json::json!({ "error": "invalid_payload" })),
        };

        // Save the updated state
        if let Err(e) = save_auth_state(&state) {
            eprintln!("Failed to save auth state: {}", e);
            return AuthOutput(Sha256::digest(b"STATE_SAVE_ERROR").to_vec());
        }

        // --- PAYLOAD & SIGNATURE CHECK ---
        let Some(first_item) = package.items.get(0) else {
            return AuthOutput(Sha256::digest(b"NO_PAYLOAD").to_vec());
        };
        let payload_hash = Sha256::digest(first_item.payload.as_slice());

        let public_key = match VerifyingKey::from_bytes(&creds.public_key) {
            Ok(pk) => pk,
            Err(_) => {
                return AuthOutput(Sha256::digest(b"INVALID_PUBKEY").to_vec());
            }
        };

        let signature = Signature::from_bytes(&creds.signature);

        if public_key.verify_strict(&payload_hash, &signature).is_ok() {
            info!(target = "authorizer", "Authorization successful.");
            AuthOutput(param.0) // success
        } else {
            AuthOutput(Sha256::digest(b"SIGNATURE_INVALID").to_vec())
        }
    }
}
