"""
Safrole Manager - Core JAM Protocol Implementation

This module contains the main SafroleManager class that handles
the JAM protocol state transitions and block processing.
"""

import json
import copy
import requests
import sys

from ..utils.helpers import (
    hex_to_bytes,
    bytes_to_hex,
    deep_clone,
    get_epoch_and_slot_phase,
    process_validator_keys_for_offenders,
    z,
    get_gamma_z_from_rust_server,
)
from ..utils.crypto_bridge import CryptoBridge
from ..protocols.fallback_condition import calculate_fallback_gamma_s


class SafroleManager:
    """Main manager class for JAM protocol state transitions."""
    
    def __init__(self, initial_state, test_vector_file_path=None):
        """Initialize the SafroleManager with initial state.
        
        Args:
            initial_state: The initial state dictionary
            test_vector_file_path: Optional path to test vector file for test mode
        """
        self.state = deep_clone(initial_state)
        is_tiny = test_vector_file_path and "/tiny/" in str(test_vector_file_path)

        # Set default values only if they don't exist in initial_state
        if "E" not in self.state:
            self.state["E"] = 12 if is_tiny else 12  # Epoch length
        if "Y" not in self.state:
            self.state["Y"] = 11 if is_tiny else 11  # Submission period
        if "N" not in self.state:
            self.state["N"] = 3 if is_tiny else 3    # Number of validators

        # Initialize gamma_a with proper type conversion if it exists
        initial_gamma_a = self.state.get("gamma_a", [])
        if initial_gamma_a and isinstance(initial_gamma_a, list):
            self.state["gamma_a"] = [
                {
                    "index": t.get("attempt", i),  # Use existing attempt or index as fallback
                    "randomness": hex_to_bytes(t["id"]) if isinstance(t.get("id"), str) else t.get("id", b""),
                    "proof": t.get("proof", b"")
                }
                for i, t in enumerate(initial_gamma_a)
            ]
        else:
            self.state["gamma_a"] = []

    def batch_ring_vrf_verify(self, gamma_z, ring_set, eta2_prime, extrinsic):
        """Perform batch ring VRF verification."""
        try:
            payload = {
                "gamma_z": gamma_z,
                "ring_set": ring_set,
                "eta2_prime": eta2_prime,
                "extrinsic": extrinsic,
            }
            response = requests.post(
                "http://127.0.0.1:3000/verifier/ring_vrf_verify_payload", json=payload
            )
            response.raise_for_status()
            return response.json()["results"]
        except requests.exceptions.RequestException as e:
            print(f"Rust server batch verification error: {e}", file=sys.stderr)
            raise Exception("rust_server_batch_verify_failed")

    def process_block(self, block_input):
        """Process a block and update the state."""
        pre_state = self.state
        print(f"Prestate state****************************: {pre_state}");
        if block_input["slot"] <= pre_state["tau"]:
            raise ValueError("bad_slot")

        is_gap_block = block_input["slot"] > pre_state["tau"] + 1
        if is_gap_block and block_input.get("extrinsic"):
            raise ValueError("unexpected_ticket")

        current_state = deep_clone(pre_state)

        prev_epoch, prev_m = get_epoch_and_slot_phase(
            pre_state["tau"], current_state["E"]
        )
        tau_prime = block_input["slot"]
        next_epoch, m_prime = get_epoch_and_slot_phase(tau_prime, current_state["E"])

        eta0_bytes = hex_to_bytes(pre_state["eta"][0])
        vrf_output = hex_to_bytes(block_input["entropy"])
        new_eta0_bytes = CryptoBridge.calculate_eta_update(eta0_bytes, vrf_output)
        new_eta0 = bytes_to_hex(new_eta0_bytes)

        if next_epoch > prev_epoch:
            current_state["eta"] = [
                new_eta0,
                pre_state["eta"][0],
                pre_state["eta"][1],
                pre_state["eta"][2],
            ]
        else:
            current_state["eta"] = [
                new_eta0,
                pre_state["eta"][1],
                pre_state["eta"][2],
                pre_state["eta"][3],
            ]

        extrinsic = block_input.get("extrinsic", [])
        if m_prime < current_state["Y"] and extrinsic:
            for ticket in extrinsic:
                if ticket["attempt"] >= current_state["N"]:
                    raise ValueError("bad_ticket_attempt")

            verification_results = self.batch_ring_vrf_verify(
                pre_state["gamma_z"],
                [k["bandersnatch"] for k in pre_state["gamma_k"]],
                current_state["eta"][2],
                extrinsic,
            )

            if any(not r["ok"] or not r["output_hash"] for r in verification_results):
                raise ValueError("bad_ticket_proof")

            submitted_randomness = [r["output_hash"] for r in verification_results]

            for i in range(len(submitted_randomness) - 1):
                if submitted_randomness[i] > submitted_randomness[i + 1]:
                    raise ValueError("bad_ticket_order")

            if len(set(submitted_randomness)) != len(submitted_randomness):
                raise ValueError("duplicate_ticket")

            # Pre-compute existing ticket IDs for O(1) lookup
            existing_ticket_ids = {
                bytes_to_hex(t["randomness"]) for t in current_state["gamma_a"]
            }
            
            # Create a lookup dictionary for extrinsic data to avoid O(n) search
            extrinsic_lookup = {ex["attempt"]: ex for ex in extrinsic}
            
            # Process verification results with optimized data structures
            new_tickets = []
            for result in verification_results:
                randomness_hex = result["output_hash"]
                if randomness_hex in existing_ticket_ids:
                    raise ValueError("duplicate_ticket")

                attempt = result["attempt"]
                if attempt not in extrinsic_lookup:
                    raise ValueError(f"missing_extrinsic_for_attempt_{attempt}")
                
                original_extrinsic = extrinsic_lookup[attempt]
                new_tickets.append({
                    "index": attempt,
                    "proof": hex_to_bytes(original_extrinsic["signature"]),
                    "randomness": hex_to_bytes(randomness_hex),
                })

            current_state["gamma_a"].extend(new_tickets)
            current_state["gamma_a"].sort(key=lambda t: t["randomness"])
            current_state["gamma_a"] = current_state["gamma_a"][: current_state["E"]]

        if next_epoch > prev_epoch:
            current_state["lambda_"] = deep_clone(pre_state["kappa"])
            current_state["kappa"] = deep_clone(pre_state["gamma_k"])
            current_state["gamma_k"] = process_validator_keys_for_offenders(
                pre_state["iota"], pre_state["post_offenders"]
            )
            current_state["gamma_z"] = get_gamma_z_from_rust_server(
                current_state["gamma_k"], pre_state["post_offenders"]
            )

            is_prev_accumulator_saturated = len(pre_state["gamma_a"]) == pre_state["E"]
            is_immediate_epoch_transition = next_epoch == prev_epoch + 1

            if (
                is_immediate_epoch_transition
                and prev_m >= pre_state["Y"]
                and is_prev_accumulator_saturated
            ):
                winning_tickets = [
                    {"id": bytes_to_hex(t["randomness"]), "attempt": t["index"]}
                    for t in z(pre_state["gamma_a"])
                ]
                current_state["gamma_s"] = {"tickets": winning_tickets}
            else:
                current_state["gamma_s"] = {
                    "keys": calculate_fallback_gamma_s(current_state)
                }

            current_state["gamma_a"] = []

        current_state["tau"] = tau_prime

        epoch_mark, tickets_mark = None, None
        if next_epoch > prev_epoch:
            epoch_mark = {
                "entropy": pre_state["eta"][0],
                "tickets_entropy": pre_state["eta"][1],
                "validators": [
                    {"bandersnatch": k["bandersnatch"], "ed25519": k["ed25519"]}
                    for k in current_state["gamma_k"]
                ],
            }

        end_of_submission = (
            (next_epoch == prev_epoch)
            and (prev_m < current_state["Y"])
            and (m_prime >= current_state["Y"])
        )
        if end_of_submission and len(current_state["gamma_a"]) == current_state["E"]:
            tickets_mark = [
                {"id": bytes_to_hex(t["randomness"]), "attempt": t["index"]}
                for t in z(current_state["gamma_a"])
            ]

        header = {
            "slot": tau_prime,
            "seal": "0x",
            "vrf_output": block_input["entropy"],
            "epoch_mark": epoch_mark,
            "tickets_mark": tickets_mark,
        }

        post_state_output = deep_clone(current_state)
        post_state_output["gamma_a"] = [
            {"id": bytes_to_hex(t["randomness"]), "attempt": t["index"]}
            for t in post_state_output["gamma_a"]
        ]

        self.state = current_state
        return {"header": header, "post_state": post_state_output} 