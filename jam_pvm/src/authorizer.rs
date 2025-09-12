extern crate alloc;

use crate::types::AuthCredentials;
use crate::service::GLOBAL_STATE;
use ed25519_dalek::{ Signature, VerifyingKey };
use jam_codec::Decode;
use jam_pvm_common::{ declare_authorizer, info, Authorizer };
use jam_types::{ AuthOutput, AuthParam, CoreIndex, WorkPackage };
use sha2::{ Digest, Sha256 };

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
        let state = GLOBAL_STATE.lock().unwrap();
        let expected_nonce = state.nonces.get(&creds.public_key).cloned().unwrap_or(0);

        if creds.nonce != expected_nonce {
            info!(
                target = "authorizer",
                "Auth failed: Invalid nonce. Expected {}, got {}.",
                expected_nonce,
                creds.nonce
            );
            return AuthOutput(Sha256::digest(b"INVALID_NONCE").to_vec());
        }
        drop(state); // release lock before further logic

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
