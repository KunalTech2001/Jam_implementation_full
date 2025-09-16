#!/usr/bin/env python3
"""
Generate valid Ed25519 keypair and signature for testing JAM authorization.
"""

import json
from auth_integration import authorization_processor

def generate_test_data():
    # Create Ed25519 keypair
    public_key, private_key = authorization_processor.create_ed25519_keypair()
    
    # Create test payload
    test_payload = {
        "service_id": 1,
        "core": 0,
        "slot": 100,
        "work_package": {
            "items": [{
                "service_id": 1,
                "payload_data": "Hello JAM Authorization!",
                "gas_limit": 1000000
            }]
        }
    }
    
    # Sign the payload
    signature = authorization_processor.sign_payload(test_payload, private_key)
    
    # Create the complete request payload
    request_payload = {
        "public_key": public_key,
        "signature": signature,
        "payload": test_payload
    }
    
    print("=== JAM Authorization Test Data ===")
    print(f"Public Key: {public_key}")
    print(f"Private Key: {private_key}")
    print(f"Signature: {signature}")
    print()
    print("=== Complete Request Payload ===")
    print(json.dumps(request_payload, indent=2))
    print()
    print("=== cURL Command ===")
    curl_command = f"""curl -X POST http://127.0.0.1:8000/authorize \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(request_payload)}'"""
    print(curl_command)
    
    return request_payload

if __name__ == "__main__":
    generate_test_data()
