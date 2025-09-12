// src/types.rs
#![cfg_attr(not(feature = "std"), no_std)]
extern crate alloc;

use alloc::collections::BTreeMap;
use jam_codec::{Decode, Encode};

/// Credentials for authorizing a WorkPackage, including a nonce.
#[derive(Encode, Decode, Clone, Debug, PartialEq)]
pub struct AuthCredentials {
    pub public_key: [u8; 32],
    pub signature: [u8; 64],
    pub nonce: u64,
}

/// The persistent state of MyJamService, now including an admin and nonces.
#[derive(Encode, Decode, Clone, Debug, PartialEq)]
pub struct ServiceState {
    pub counter: u64,
    pub last_payload_hash: [u8; 32],
    pub admin: u64, // <-- ADD THIS LINE
    pub nonces: BTreeMap<[u8; 32], u64>,
}

// Add this manual implementation of `Default`
impl Default for ServiceState {
    fn default() -> Self {
        Self {
            counter: 0,
            last_payload_hash: [0; 32],
            admin: 0, // Default admin to 0 (no admin)
            nonces: BTreeMap::new(),
        }
    }
}

/// A command that can be sent to the service via the `on_transfer` memo field.
#[derive(Encode, Decode, Clone, Debug, PartialEq)]
pub enum ServiceCommand {
    IncrementCounter { by: u64 },
    ResetState,
}
