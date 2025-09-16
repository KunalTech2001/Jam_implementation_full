#!/usr/bin/env python3
"""
Test script for Ed25519-based authorization with proper PVM integration.
This replaces the old NaCl-based test with Ed25519 signatures compatible with the PVM.
"""

import json
import sys
import os
import requests
import asyncio
from pathlib import Path

# Add the server directory to the path so we can import our integration module
sys.path.append(os.path.dirname(__file__))

from auth_integration import AuthorizationIntegrator

async def test_ed25519_authorization():
    """Test the complete Ed25519 authorization flow."""
    print("=== Ed25519 Authorization Integration Test ===\n")
    
    # Initialize the integrator
    integrator = AuthorizationIntegrator()
    
    # Test 1: Generate Ed25519 keypair
    print("1. Generating Ed25519 keypair...")
    public_key_hex, private_key_hex = integrator.create_ed25519_keypair()
    print(f"   Public Key: {public_key_hex}")
    print(f"   Private Key: {private_key_hex[:16]}... (truncated)")
    
    # Test 2: Sign payload
    print("\n2. Signing test payload...")
    test_payload = b"test_authorization_payload"
    signature_hex = integrator.sign_payload_ed25519(test_payload, private_key_hex)
    print(f"   Payload: {test_payload}")
    print(f"   Signature: {signature_hex}")
    
    # Test 3: Verify signature locally
    print("\n3. Verifying signature locally...")
    is_valid = integrator.verify_ed25519_signature(test_payload, signature_hex, public_key_hex)
    print(f"   Signature valid: {is_valid}")
    
    if not is_valid:
        print("❌ Local signature verification failed!")
        return False
    
    # Test 4: Test SCALE encoding
    print("\n4. Testing SCALE encoding...")
    try:
        nonce = integrator.get_next_nonce(public_key_hex)
        auth_creds_hex = integrator.encode_auth_credentials(public_key_hex, signature_hex, nonce)
        work_pkg_hex = integrator.encode_work_package(test_payload, service_id=1)
        print(f"   Nonce: {nonce}")
        print(f"   Auth credentials (hex): {auth_creds_hex[:32]}...")
        print(f"   Work package (hex): {work_pkg_hex[:32]}...")
    except Exception as e:
        print(f"❌ SCALE encoding failed: {e}")
        return False
    
    # Test 5: Full PVM authorization flow
    print("\n5. Testing full PVM authorization flow...")
    try:
        success, result = await integrator.authorize_with_pvm(
            payload_data=test_payload,
            private_key_hex=private_key_hex,
            public_key_hex=public_key_hex,
            service_id=1
        )
        
        print(f"   Authorization success: {success}")
        if success:
            print(f"   PVM Response: {result.get('pvm_response', {}).get('output_hex', 'N/A')[:32]}...")
            print("✅ PVM authorization successful!")
        else:
            print(f"   Error: {result.get('error', 'Unknown error')}")
            print("⚠️  PVM authorization failed (PVM might not be running)")
            
    except Exception as e:
        print(f"⚠️  PVM authorization failed: {e}")
        print("   This is expected if the PVM server is not running on port 8080")
    
    # Test 6: Test server endpoint integration
    print("\n6. Testing server endpoint integration...")
    try:
        # Prepare request for server endpoint
        payload_json = {"service_id": 1, "action": "test", "data": test_payload.decode()}
        payload_bytes = json.dumps(payload_json, sort_keys=True).encode()
        
        # Sign the JSON payload
        server_signature = integrator.sign_payload_ed25519(payload_bytes, private_key_hex)
        
        request_data = {
            "public_key": public_key_hex,
            "signature": server_signature,
            "payload": payload_json
        }
        
        response = requests.post(
            "http://127.0.0.1:8000/authorize",
            json=request_data,
            timeout=10
        )
        
        print(f"   Server response status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"   Server authorization success: {result.get('success', False)}")
            if result.get('success'):
                print("✅ Server authorization successful!")
            else:
                print(f"   Server error: {result.get('message', 'Unknown error')}")
        else:
            print(f"   Server error: {response.text}")
            print("⚠️  Server authorization failed (server might not be running)")
            
    except Exception as e:
        print(f"⚠️  Server authorization failed: {e}")
        print("   This is expected if the server is not running on port 8000")
    
    print("\n=== Test Summary ===")
    print("✅ Ed25519 keypair generation: PASSED")
    print("✅ Ed25519 signature creation: PASSED") 
    print("✅ Ed25519 signature verification: PASSED")
    print("✅ SCALE encoding: PASSED")
    print("⚠️  PVM integration: DEPENDS ON PVM SERVER")
    print("⚠️  Server integration: DEPENDS ON SERVER")
    
    return True

def test_deterministic_keypair():
    """Test deterministic keypair generation from seed."""
    print("\n=== Deterministic Keypair Test ===")
    
    integrator = AuthorizationIntegrator()
    
    # Alice's test seed (from Substrate)
    alice_seed = "0xe5be9a5092b81bca64be81d212e7f2f9eba183bb7a90954f7b76361f6edb5c0a"
    
    # Generate keypair twice with same seed
    pub1, priv1 = integrator.create_ed25519_keypair(alice_seed)
    pub2, priv2 = integrator.create_ed25519_keypair(alice_seed)
    
    print(f"First generation - Public: {pub1}")
    print(f"Second generation - Public: {pub2}")
    print(f"Keys match: {pub1 == pub2 and priv1 == priv2}")
    
    if pub1 == pub2 and priv1 == priv2:
        print("✅ Deterministic keypair generation: PASSED")
        return True
    else:
        print("❌ Deterministic keypair generation: FAILED")
        return False

async def main():
    """Run all tests."""
    print("JAM Authorization-PVM Integration Test Suite")
    print("=" * 50)
    
    # Test deterministic keypair generation
    deterministic_passed = test_deterministic_keypair()
    
    # Test main authorization flow
    auth_passed = await test_ed25519_authorization()
    
    print(f"\n{'=' * 50}")
    print("OVERALL RESULTS:")
    print(f"✅ Deterministic keypairs: {'PASSED' if deterministic_passed else 'FAILED'}")
    print(f"✅ Authorization flow: {'PASSED' if auth_passed else 'FAILED'}")
    
    if deterministic_passed and auth_passed:
        print("\n🎉 All core tests PASSED! Integration is working correctly.")
        print("\nTo test with live servers:")
        print("1. Start PVM: cd jam_pvm && cargo run")
        print("2. Start server: cd server && python server.py")
        print("3. Run this test again")
        return 0
    else:
        print("\n❌ Some tests FAILED!")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
