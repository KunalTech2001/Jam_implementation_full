from hashlib import sha256
from scalecodec.base import RuntimeConfigurationObject
# Import the Keypair class and the KeypairType enum
from substrateinterface import Keypair, KeypairType

# 1. Define the custom types for our JAM structures
# This is the FULL definition, matching the JAM specification.
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

def main():
    print("--- Generating signed authorization request ---")

    type_registry = RuntimeConfigurationObject()
    type_registry.update_type_registry(custom_types)

    # 1. Generate an ED25519 keypair from a raw seed to match the Rust server.
    # The create_from_uri function doesn't support ed25519, so we use the seed directly.
    alice_seed = "0xe5be9a5092b81bca64be81d212e7f2f9eba183bb7a90954f7b76361f6edb5c0a"
    keypair = Keypair.create_from_seed(seed_hex=alice_seed, crypto_type=KeypairType.ED25519)
    print(f"Using Public Key (hex): {keypair.public_key.hex()}")

    # 2. Define the payload for our work package
    payload_data = b"increment_counter_by_10"
    payload_hash = sha256(payload_data).digest()
    print(f"Payload Hash: {payload_hash.hex()}")

    # 3. Sign the payload hash with the keypair
    signature = keypair.sign(payload_hash)
    print(f"Signature: {signature.hex()}")

    # 4. Create the data for the structures
    credentials_data = {
        'public_key': keypair.public_key,
        'signature': signature,
        'nonce': 0,
    }

    work_package_data = {
        'auth_token': b'\x00',
        'auth_service_id': 0,
        'auth_code_hash': b'\x00' * 32,
        'auth_config': b'\x00',
        'context': {
            'anchor_hash': b'\x00' * 32,
            'state_root': b'\x00' * 32,
            'acc_output_log_peak': b'\x00' * 32,
            'lookup_anchor_hash': b'\x00' * 32,
            'lookup_timeslot': 0,
            'prerequisites': []
        },
        'items': [
            {
                'service_id': 1,
                'code_hash': b'\x00' * 32,
                'payload': payload_data,
                'refine_gas': 1000000,
                'accumulate_gas': 100000,
                'export_count': 0,
                'imports': [],
                'extrinsics': []
            }
        ]
    }

    # 5. Encode the data
    auth_encoder = type_registry.create_scale_object('AuthCredentials')
    pkg_encoder = type_registry.create_scale_object('WorkPackage')

    encoded_auth = auth_encoder.encode(credentials_data)
    encoded_package = pkg_encoder.encode(work_package_data)

    param_hex = encoded_auth.to_hex()[2:]
    package_hex = encoded_package.to_hex()[2:]

    print("\n--- Generated API Inputs ---")
    print(f"param_hex: {param_hex}")
    print(f"package_hex: {package_hex}")

    # 6. Generate the full curl command
    curl_command = f"""
curl -X POST -H "Content-Type: application/json" \\
-d '{{
    "param_hex": "{param_hex}",
    "package_hex": "{package_hex}",
    "core_index_hex": "00000000"
}}' \\
http://127.0.0.1:8080/authorizer/is_authorized
"""
    print("\n--- Ready to Execute ---")
    print("Copy and paste the following command into a new terminal:")
    print(curl_command)


if __name__ == "__main__":
    main()

