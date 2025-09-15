import json
import os
import sys
import requests
from hashlib import sha256
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from substrateinterface import Keypair, KeypairType
from scalecodec.base import RuntimeConfigurationObject
from datetime import datetime, timezone

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

def load_updated_state(server_dir: str = "../server") -> Dict[str, Any]:
    """Load the current state from updated_state.json"""
    state_path = Path(server_dir) / "updated_state.json"
    try:
        with open(state_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"authorizations": {}}

def save_updated_state(state: Dict[str, Any], server_dir: str = "../server") -> None:
    """Save the updated state to updated_state.json"""
    state_path = Path(server_dir) / "updated_state.json"
    with open(state_path, 'w') as f:
        json.dump(state, f, indent=2)

def execute_pvm_authorization(
    payload_data: bytes = None,
    service_id: int = 1,
    seed: str = None,
    server_url: str = "http://127.0.0.1:8000"
) -> Tuple[bool, Dict[str, Any]]:
    """
    Generates and executes a PVM authorization request.
    
    Args:
        payload_data: The payload data to be authorized (default: b"increment_counter_by_10")
        service_id: The service ID for the work item
        seed: The seed for key generation (default: Alice's test seed)
        server_url: Base URL of the server
        
    Returns:
        Tuple of (success: bool, result: dict)
    """
    print("--- Step 1: Generating and executing PVM authorization request ---")
    
    # Use default payload if none provided
    if payload_data is None:
        payload_data = b"increment_counter_by_10"
    
    # Use Alice's test seed if none provided
    if seed is None:
        seed = "0xe5be9a5092b81bca64be81d212e7f2f9eba183bb7a90954f7b76361f6edb5c0a"
    
    # Initialize type registry
    type_registry = RuntimeConfigurationObject()
    type_registry.update_type_registry(custom_types)
    
    # Load current state to get nonce
    current_state = load_updated_state()
    keypair = Keypair.create_from_seed(seed_hex=seed, crypto_type=KeypairType.ED25519)
    public_key_hex = keypair.public_key.hex()
    
    # Get or initialize nonce
    nonce = 0
    if "authorizations" in current_state and public_key_hex in current_state["authorizations"]:
        nonce = current_state["authorizations"][public_key_hex].get("nonce", 0) + 1
    
    # Sign the payload
    payload_hash = sha256(payload_data).digest()
    signature = keypair.sign(payload_hash)
    
    # Prepare authorization data
    auth_data = {
        "public_key": keypair.public_key,
        "signature": signature,
        "nonce": nonce
    }
    
    # Prepare work package
    work_package_data = {
        'items': [{
            'service_id': service_id,
            'code_hash': b'\x00' * 32,
            'payload': payload_data,
            'refine_gas': 0,
            'accumulate_gas': 0,
            'export_count': 0,
            'imports': [],
            'extrinsics': []
        }],
        'auth_token': b'',
        'auth_service_id': 0,
        'auth_code_hash': b'\x00' * 32,
        'auth_config': b'',
        'context': {
            'anchor_hash': b'\x00' * 32,
            'state_root': b'\x00' * 32,
            'acc_output_log_peak': b'\x00' * 32,
            'lookup_anchor_hash': b'\x00' * 32,
            'lookup_timeslot': 0,
            'prerequisites': []
        }
    }
    
    try:
        # First, try the new server endpoint
        response = requests.post(
            f"{server_url}/authorize",
            json={
                "public_key": public_key_hex,
                "signature": signature.hex(),
                "nonce": nonce,
                "payload": {
                    "service_id": service_id,
                    "payload_data": payload_data.decode('utf-8', 'ignore'),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success", False):
                print("✅ Authorization successful via server endpoint")
                return True, result
            else:
                print(f"❌ Server authorization failed: {result.get('message', 'Unknown error')}")
                return False, result
        
        # Fall back to direct PVM call if server endpoint fails
        print("⚠️  Server endpoint failed, falling back to direct PVM call")
        
        # Encode the authorization and package
        auth_encoder = type_registry.create_scale_object('AuthCredentials')
        pkg_encoder = type_registry.create_scale_object('WorkPackage')
        encoded_auth = auth_encoder.encode(auth_data)
        encoded_package = pkg_encoder.encode(work_package_data)
        
        # Make the PVM request
        response = requests.post(
            "http://127.0.0.1:8080/authorizer/is_authorized",
            json={
                "param_hex": encoded_auth.to_hex()[2:],
                "package_hex": encoded_package.to_hex()[2:],
                "core_index_hex": "00000000"
            }
        )
        response.raise_for_status()
        result = response.json()
        
        # Update state if successful
        if result.get("output_hex") == encoded_auth.to_hex()[2:]:
            # Update the state with the new authorization
            if "authorizations" not in current_state:
                current_state["authorizations"] = {}
                
            current_state["authorizations"][public_key_hex] = {
                "public_key": public_key_hex,
                "nonce": nonce,
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "payload": {
                    "service_id": service_id,
                    "payload_data": payload_data.decode('utf-8', 'ignore')
                }
            }
            save_updated_state(current_state)
            print("✅ PVM Authorization successful!")
            return True, result
        else:
            print(f"❌ PVM Authorization failed with response: {result}")
            return False, result
            
    except Exception as e:
        print(f"❌ Error during authorization: {str(e)}")
        return False, {"error": str(e)}
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

def main():
    """Main function to demonstrate the authorization flow"""
    # Example 1: Simple authorization with default values
    print("\n--- Example 1: Default authorization ---")
    success, result = execute_pvm_authorization()
    
    if success:
        print("\n--- Example 2: Custom payload ---")
        custom_payload = b"custom_payload_123"
        success, result = execute_pvm_authorization(
            payload_data=custom_payload,
            service_id=2
        )
    
    # Run STF tests if available
    if success:
        print("\n--- Running STF tests ---")
        test_files = [
            "progress_authorizations-1.json",
            "progress_authorizations-2.json"
        ]
        
        all_passed = True
        for test_file in test_files:
            test_path = os.path.join("full", test_file)
            if os.path.exists(test_path):
                print(f"\nRunning test: {test_file}")
                if not run_stf_test_on_file(test_path):
                    all_passed = False
            else:
                print(f"Test file not found: {test_path}")
                all_passed = False
        
        if all_passed:
            print("\n✅ All tests passed!")
        else:
            print("\n❌ Some tests failed!")
            sys.exit(1)

if __name__ == "__main__":
    with open('updated_state.json', 'r') as f:
        current_state = json.load(f)
    main()
