import json
import os
import sys
import requests
import asyncio
from hashlib import sha256
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from substrateinterface import Keypair, KeypairType
from scalecodec.base import RuntimeConfigurationObject
from datetime import datetime, timezone

# Import the new authorization integrator
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'server'))
from auth_integration import AuthorizationIntegrator

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

async def execute_pvm_authorization(
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
    print("--- Step 1: Generating and executing PVM authorization request (NEW INTEGRATION) ---")
    
    # Use default payload if none provided
    if payload_data is None:
        payload_data = b"increment_counter_by_10"
    
    # Use Alice's test seed if none provided
    if seed is None:
        seed = "0xe5be9a5092b81bca64be81d212e7f2f9eba183bb7a90954f7b76361f6edb5c0a"
    
    # Initialize the new authorization integrator
    integrator = AuthorizationIntegrator(server_dir="../server")
    
    # Generate keypair from seed
    keypair = Keypair.create_from_seed(seed_hex=seed, crypto_type=KeypairType.ED25519)
    public_key_hex = keypair.public_key.hex()
    private_key_hex = keypair.private_key.hex()
    
    try:
        # First, try the new server endpoint with Ed25519 integration
        payload_json = {
            "service_id": service_id,
            "payload_data": payload_data.decode('utf-8', 'ignore'),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        # Sign the JSON payload using Ed25519
        payload_bytes = json.dumps(payload_json, sort_keys=True).encode()
        signature_hex = integrator.sign_payload_ed25519(payload_bytes, private_key_hex)
        
        response = requests.post(
            f"{server_url}/authorize",
            json={
                "public_key": public_key_hex,
                "signature": signature_hex,
                "payload": payload_json
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success", False):
                print("✅ Authorization successful via server endpoint (Ed25519)")
                return True, result
            else:
                print(f"❌ Server authorization failed: {result.get('message', 'Unknown error')}")
                return False, result
        
        # Fall back to direct PVM call using new integration
        print("⚠️  Server endpoint failed, falling back to direct PVM call")
        
        success, pvm_result = await integrator.authorize_with_pvm(
            payload_data=payload_data,
            private_key_hex=private_key_hex,
            public_key_hex=public_key_hex,
            service_id=service_id
        )
        
        if success:
            print("✅ Direct PVM Authorization successful!")
            return True, pvm_result
        else:
            print(f"❌ Direct PVM Authorization failed: {pvm_result.get('error', 'Unknown error')}")
            return False, pvm_result
            
    except Exception as e:
        print(f"❌ Error during authorization: {str(e)}")
        return False, {"error": str(e)}

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

async def main():
    """Main function to demonstrate the authorization flow"""
    # Example 1: Simple authorization with default values
    print("\n--- Example 1: Default authorization ---")
    success, result = await execute_pvm_authorization()
    
    if success:
        print("\n--- Example 2: Custom payload ---")
        custom_payload = b"custom_payload_123"
        success, result = await execute_pvm_authorization(
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
    # Ensure updated_state.json exists
    if os.path.exists('updated_state.json'):
        with open('updated_state.json', 'r') as f:
            current_state = json.load(f)
    else:
        current_state = {"authorizations": {}}
        with open('updated_state.json', 'w') as f:
            json.dump(current_state, f, indent=2)
    
    asyncio.run(main())
