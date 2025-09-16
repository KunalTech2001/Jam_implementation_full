#!/usr/bin/env python3
"""
Complete Integration Test for JAM Authorization with PVM

This script demonstrates the complete integration flow:
1. Server (@server) receives authorization request
2. Authorization component (@authorizations) processes using STF logic
3. PVM (@jam_pvm) verifies signatures and authorization
4. State synchronization across all components

Usage:
1. Start PVM: cd jam_pvm && cargo run
2. Start Server: cd server && python server.py
3. Run this test: python test_complete_integration.py
"""

import asyncio
import json
import httpx
from auth_integration import authorization_processor

async def test_complete_integration():
    """Test complete authorization flow with payload."""
    
    print("🚀 Starting Complete JAM Authorization Integration Test")
    print("=" * 60)
    
    # Step 1: Create Ed25519 keypair for testing
    print("1. Creating Ed25519 keypair...")
    public_key, private_key = authorization_processor.create_ed25519_keypair()
    print(f"   Public Key: {public_key}")
    print(f"   Private Key: {private_key[:16]}...")
    
    # Step 2: Create test payload
    print("\n2. Creating test payload...")
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
        },
        "timestamp": "2024-01-01T00:00:00Z"
    }
    print(f"   Payload: {json.dumps(test_payload, indent=2)}")
    
    # Step 3: Sign the payload
    print("\n3. Signing payload with Ed25519...")
    signature = authorization_processor.sign_payload(test_payload, private_key)
    print(f"   Signature: {signature[:32]}...")
    
    # Step 4: Test direct authorization processor
    print("\n4. Testing authorization processor directly...")
    input_data = {
        "public_key": public_key,
        "signature": signature,
        "payload": test_payload,
        "slot": test_payload["slot"],
        "auths": [{
            "core": test_payload["core"],
            "auth_hash": authorization_processor.create_auth_hash(
                public_key, test_payload, test_payload["slot"]
            )
        }]
    }
    
    try:
        result = await authorization_processor.process_authorization(input_data)
        print(f"   ✅ Direct Processing: {result['success']}")
        print(f"   Message: {result['message']}")
        print(f"   PVM Authorized: {result.get('pvm_authorized', 'N/A')}")
        
        if result.get('pvm_response'):
            print(f"   PVM Response: {result['pvm_response']}")
            
    except Exception as e:
        print(f"   ❌ Direct Processing Failed: {e}")
    
    # Step 5: Test server endpoint
    print("\n5. Testing server endpoint...")
    server_url = "http://127.0.0.1:8000"
    
    authorization_request = {
        "public_key": public_key,
        "signature": signature,
        "payload": test_payload,
        "nonce": None  # Will be calculated automatically
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{server_url}/authorize",
                json=authorization_request,
                timeout=30.0
            )
            
            if response.status_code == 200:
                server_result = response.json()
                print(f"   ✅ Server Authorization: {server_result['success']}")
                print(f"   Message: {server_result['message']}")
                
                if 'auth_output' in server_result:
                    print(f"   Auth Output: {server_result['auth_output'][:32]}...")
                    
            else:
                print(f"   ❌ Server Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
    except httpx.ConnectError:
        print("   ⚠️  Server not running. Start with: cd server && python server.py")
    except Exception as e:
        print(f"   ❌ Server Test Failed: {e}")
    
    # Step 6: Check state synchronization
    print("\n6. Checking state synchronization...")
    try:
        state = authorization_processor.load_state()
        auth_record = state.get("authorizations", {}).get(public_key, {})
        
        if auth_record:
            print(f"   ✅ Authorization Record Found")
            print(f"   Nonce: {auth_record.get('nonce', 'N/A')}")
            print(f"   PVM Authorized: {auth_record.get('pvm_authorized', 'N/A')}")
            print(f"   Last Updated: {auth_record.get('last_updated', 'N/A')}")
        else:
            print(f"   ⚠️  No authorization record found for {public_key[:16]}...")
            
        print(f"   Auth Pools: {len(state.get('auth_pools', []))}")
        print(f"   Auth Queues: {len(state.get('auth_queues', []))}")
        print(f"   PVM Integration: {state.get('pvm_integration', False)}")
        
    except Exception as e:
        print(f"   ❌ State Check Failed: {e}")
    
    print("\n" + "=" * 60)
    print("🎯 Integration Test Complete!")
    print("\nTo test the complete flow:")
    print("1. Start PVM: cd jam_pvm && cargo run")
    print("2. Start Server: cd server && python server.py") 
    print("3. Send authorization requests to http://127.0.0.1:8000/authorize")
    print("\nPayload format:")
    print(json.dumps({
        "public_key": "your_ed25519_public_key_hex",
        "signature": "your_ed25519_signature_hex", 
        "payload": {
            "service_id": 1,
            "core": 0,
            "slot": 100,
            "work_package": {"items": []}
        }
    }, indent=2))

if __name__ == "__main__":
    asyncio.run(test_complete_integration())
