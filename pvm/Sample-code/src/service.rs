// src/service.rs
extern crate alloc;

use crate::types::{ AuthCredentials, ServiceCommand, ServiceState };
use alloc::vec::Vec;
use jam_codec::{ Decode, Encode };
use jam_pvm_common::{ declare_service, info, Service };
use jam_types::{
    AccumulateItem,
    CodeHash,
    Hash,
    RefineContext,
    ServiceId,
    Slot,
    TransferRecord,
    WorkOutput,
    WorkPackageHash,
    WorkPayload,
};
use lazy_static::lazy_static;
use std::sync::Mutex;

// Global in-memory state instead of set_storage/get_storage
lazy_static! {
    pub static ref GLOBAL_STATE: Mutex<ServiceState> = Mutex::new(ServiceState::default());
}

pub struct MyJamService;

declare_service!(MyJamService);

impl Service for MyJamService {
    fn refine(
        _id: ServiceId,
        payload: WorkPayload,
        _package_hash: WorkPackageHash,
        _context: RefineContext,
        _auth_code_hash: CodeHash
    ) -> WorkOutput {
        info!(target = "service::refine", "Executing refine logic.");
        let payload_slice = payload.take();
        let output_data = [b"Refined: ", payload_slice.as_slice()].concat();
        info!(target = "service::refine", "Produced output of length {}.", output_data.len());
        output_data.into()
    }

    fn accumulate(_slot: Slot, _id: ServiceId, items: Vec<AccumulateItem>) -> Option<Hash> {
        info!(
            target = "service::accumulate",
            "Executing accumulate logic with {} item(s).",
            items.len()
        );

        let mut state = GLOBAL_STATE.lock().unwrap();

        if let Some(item) = items.first() {
            if item.result.is_ok() {
                state.counter += 1;
                state.last_payload_hash = item.payload.0.clone();

                // Try decode
                match AuthCredentials::decode(&mut item.auth_output.0.as_slice()) {
                    Ok(creds) => {
                        let nonce = state.nonces.entry(creds.public_key.clone()).or_insert(0);
                        *nonce += 1;
                       println!("✅ Nonce for pk {:?} incremented to {}.", creds.public_key, *nonce);
                    }
                    Err(e) => {
                        info!(
                            target = "service::accumulate",
                            "⚠️ Failed to decode AuthCredentials from auth_output: {:?}. Raw auth_output = {:?}",
                            e,
                            item.auth_output.0
                        );
                    }
                }
            } else {
                info!(
                    target = "service::accumulate",
                    "⚠️ Item.result was Err, skipping state update."
                );
            }
        } else {
            info!(target = "service::accumulate", "⚠️ No items passed to accumulate.");
        }

        println!(
            "DEBUG: State before saving: counter = {}, nonces = {:?}",
            state.counter,
            state.nonces
        );

        info!(
            target = "service::accumulate",
            "Successfully updated state: counter = {}.",
            state.counter
        );

        None
    }

    fn on_transfer(_slot: Slot, _id: ServiceId, transfers: Vec<TransferRecord>) {
        info!(
            target = "service::on_transfer",
            "Executing on_transfer logic with {} record(s).",
            transfers.len()
        );

        if transfers.is_empty() {
            return;
        }

        let mut state = GLOBAL_STATE.lock().unwrap();

        info!(target = "service::on_transfer", "Read initial state: counter = {}.", state.counter);

        for transfer in transfers {
            if let Ok(command) = ServiceCommand::decode(&mut &transfer.memo.0[..]) {
                info!(target = "service::on_transfer", "Decoded command: {:?}.", command);
                match command {
                    ServiceCommand::IncrementCounter { by } => {
                        state.counter += by;
                    }
                    ServiceCommand::ResetState => {
                        if u64::from(transfer.source) == state.admin {
                            *state = ServiceState::default(); // ✅ reset cleanly
                            state.admin = u64::from(transfer.source);
                        } else {
                            info!(
                                target = "service::on_transfer",
                                "ACCESS DENIED: ResetState is admin-only."
                            );
                        }
                    }
                }
            } else {
                info!(
                    target = "service::on_transfer",
                    "Could not decode command from transfer memo."
                );
            }
        }

        info!(target = "service::on_transfer", "Updated state: counter = {}.", state.counter);
    }
}
