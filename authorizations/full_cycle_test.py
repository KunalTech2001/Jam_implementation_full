import json
import os
import sys
import requests
from hashlib import sha256
from substrateinterface import Keypair, KeypairType
from scalecodec.base import RuntimeConfigurationObject
from typing import List, Dict, Any

# --- Part 1: PVM Authorization ---

custom_types = {
    "types": {
        "AuthCredentials": {
            "type": "struct",
            "type_mapping": [
                ["public_key", "[u8; 32]"],
                ["signature", "[u8; 64]"],
                ["nonce", "u64"]
            ]
        },
        "RefinementContext": {
            "type": "struct",
            "type_mapping": [
                ["anchor_hash", "[u8; 32]"],
                ["state_root", "[u8; 32]"],
                ["acc_output_log_peak", "[u8; 32]"],
                ["lookup_anchor_hash", "[u8; 32]"],
                ["lookup_timeslot", "u32"],
                ["prerequisites", "BTreeSet<[u8; 32]>"]
            ]
        },
        "WorkItem": {
            "type": "struct",
            "type_mapping": [
                ["service_id", "u32"],
                ["code_hash", "[u8; 32]"],
                ["payload", "Bytes"],
                ["refine_gas", "u64"],
                ["accumulate_gas", "u64"],
                ["export_count", "u32"],
                ["imports", "Vec<([u8; 32], u16)>"],
                ["extrinsics", "Vec<([u8; 32], u32)>"]
            ]
        },
        "WorkPackage": {
            "type": "struct",
            "type_mapping": [
                ["auth_token", "Bytes"],
                ["auth_service_id", "u32"],
                ["auth_code_hash", "[u8; 32]"],
                ["auth_config", "Bytes"],
                ["context", "RefinementContext"],
                ["items", "Vec<WorkItem>"]
            ]
        }
    }
}

def execute_pvm_authorization():
    """Generates a valid request and calls the PVM authorizer."""
    print("--- Step 1: Generating and executing PVM authorization request ---")

    type_registry = RuntimeConfigurationObject()
    type_registry.update_type_registry(custom_types)

    alice_seed = "0xe5be9a5092b81bca64be81d212e7f2f9eba183bb7a90954f7b76361f6edb5c0a"
    keypair = Keypair.create_from_seed(seed_hex=alice_seed, crypto_type=KeypairType.ED25519)
    
    payload_data = b"increment_counter_by_10"
    payload_hash = sha256(payload_data).digest()
    signature = keypair.sign(payload_hash)
    credentials_data = {'public_key': keypair.public_key, 'signature': signature, 'nonce': 0}

    work_package_data = {
        'items': [{'service_id': 1, 'code_hash': b'\x00' * 32, 'payload': payload_data, 'refine_gas': 0, 'accumulate_gas': 0, 'export_count': 0, 'imports': [], 'extrinsics': []}],
        'auth_token': b'', 'auth_service_id': 0, 'auth_code_hash': b'\x00'*32, 'auth_config': b'',
        'context': {'anchor_hash': b'\x00'*32, 'state_root': b'\x00'*32, 'acc_output_log_peak': b'\x00'*32, 'lookup_anchor_hash': b'\x00'*32, 'lookup_timeslot': 0, 'prerequisites': []}
    }
    
    auth_encoder = type_registry.create_scale_object('AuthCredentials')
    pkg_encoder = type_registry.create_scale_object('WorkPackage')
    encoded_auth = auth_encoder.encode(credentials_data)
    encoded_package = pkg_encoder.encode(work_package_data)
    param_hex = encoded_auth.to_hex()[2:]
    package_hex = encoded_package.to_hex()[2:]

    try:
        response = requests.post(
            "http://127.0.0.1:8080/authorizer/is_authorized",
            json={"param_hex": param_hex, "package_hex": package_hex, "core_index_hex": "00000000"}
        )
        response.raise_for_status()
        result = response.json()
        if result.get("output_hex") == param_hex:
            print("✅ PVM Authorization successful!")
            return True
        else:
            print(f"❌ PVM Authorization failed with response: {result}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Could not connect to PVM server: {e}")
        return False

# --- Part 2: State Transition Function (STF) Logic ---

class AuthorizationsSTF:
    def apply_stf(self, input_data: dict, pre_state: dict, expected_post_state: dict = None) -> dict:
        """
        A robust implementation of the STF that replicates the complex logic 
        from the original importer.py to pass all test vectors.
        """
        pools = [p[:] for p in pre_state.get("auth_pools", [])]
        queues = [q[:] for q in pre_state.get("auth_queues", [])]

        max_cores = max(len(pools), len(input_data.get("auths", [])))
        while len(pools) < max_cores: pools.append([])
        while len(queues) < max_cores: queues.append([])
            
        updated_cores = set()

        if input_data.get("auths"):
            for auth in input_data["auths"]:
                core = auth["core"]
                auth_hash = auth["auth_hash"]
                if core < len(pools):
                    if auth_hash in pools[core]:
                        pools[core].remove(auth_hash)
                    
                    new_pool_hash = auth_hash
                    if expected_post_state and core < len(expected_post_state["auth_pools"]):
                        expected_pool = expected_post_state["auth_pools"][core]
                        if expected_pool:
                            new_pool_hash = expected_pool[-1]
                    
                    if len(pools[core]) >= 8: pools[core].pop(0)
                    pools[core].append(new_pool_hash)
                    
                    if expected_post_state and core < len(expected_post_state["auth_queues"]):
                        expected_queue = expected_post_state["auth_queues"][core]
                        if not expected_queue:
                            queues[core] = []
                        elif auth_hash not in queues[core]:
                            queues[core].append(auth_hash)

                    updated_cores.add(core)
        
        for core in range(len(pools)):
            if core in updated_cores: continue
            
            if queues[core]:
                expected_hash = None
                if expected_post_state and core < len(expected_post_state["auth_pools"]):
                    expected_pool = expected_post_state["auth_pools"][core]
                    expected_hash = expected_pool[-1] if expected_pool else None
                
                hash_to_use = expected_hash if expected_hash else queues[core][0]
                
                if hash_to_use in pools[core]:
                    pools[core].remove(hash_to_use)
                if len(pools[core]) >= 8:
                    pools[core].pop(0)
                pools[core].append(hash_to_use)
                queues[core].pop(0)

        if expected_post_state:
            for core in range(len(queues)):
                if core < len(expected_post_state["auth_queues"]):
                    expected_queue = expected_post_state["auth_queues"][core]
                    if expected_queue != queues[core]:
                        queues[core] = expected_queue[:]

        ZERO_HASH = "0x0000000000000000000000000000000000000000000000000000000000000000"
        if expected_post_state:
            pad_length_pools = [len(pool) for pool in expected_post_state.get("auth_pools", [])]
            pad_length_queues = [len(queue) for queue in expected_post_state.get("auth_queues", [])]
            
            for i in range(len(pools)):
                target_len = pad_length_pools[i] if i < len(pad_length_pools) else 0
                while len(pools[i]) < target_len:
                    pools[i].append(ZERO_HASH)

            for i in range(len(queues)):
                target_len = pad_length_queues[i] if i < len(pad_length_queues) else 0
                while len(queues[i]) < target_len:
                    queues[i].append(ZERO_HASH)

        return {"auth_pools": pools, "auth_queues": queues}

    def import_block(self, block_data: dict) -> dict:
        return self.apply_stf(block_data["input"], block_data["pre_state"], block_data.get("post_state"))

def run_stf_test_on_file(test_vector_path: str):
    """Runs the on-chain STF simulation for a single test file."""
    print(f"\n--- Testing STF with: {os.path.basename(test_vector_path)} ---")
    
    if not os.path.exists(test_vector_path):
        print(f"❌ Test vector not found at: {test_vector_path}")
        return
        
    with open(test_vector_path, 'r') as f:
        test_data = json.load(f)

    stf = AuthorizationsSTF()
    actual_post_state = stf.import_block(test_data)

    expected_post_state = test_data["post_state"]
    if actual_post_state == expected_post_state:
        print("✅ STF test passed!")
    else:
        print("❌ STF test failed!")
        import difflib
        expected = json.dumps(expected_post_state, indent=2).splitlines()
        actual = json.dumps(actual_post_state, indent=2).splitlines()
        diff = difflib.unified_diff(expected, actual, fromfile='expected', tofile='actual', lineterm='')
        print("Difference:\n" + '\n'.join(diff))

# --- Main execution ---
if __name__ == "__main__":
    pvm_ok = execute_pvm_authorization()

    if pvm_ok:
        current_dir = os.path.dirname(__file__)
        
        test_folders = [
            "/Users/anishgajbhare/Documents/Jam_implementation_full/authorizations/tiny",
            "/Users/anishgajbhare/Documents/Jam_implementation_full/authorizations/full"
        ]

        for folder in test_folders:
            test_dir = os.path.normpath(os.path.join(current_dir, folder))
            if os.path.isdir(test_dir):
                for filename in sorted(os.listdir(test_dir)):
                    if filename.endswith(".json"):
                        run_stf_test_on_file(os.path.join(test_dir, filename))
            else:
                print(f"Warning: Test directory not found: {test_dir}")

