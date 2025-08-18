#!/usr/bin/env python3
"""
Quick test script for the JAM Safrole Integration Server

This script tests the basic functionality of the server endpoints.
"""

import requests
import json
import time
import sys

SERVER_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint."""
    try:
        response = requests.get(f"{SERVER_URL}/health")
        print(f"✅ Health check: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Status: {data['status']}")
            print(f"   Safrole initialized: {data['safrole_initialized']}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_root():
    """Test the root endpoint."""
    try:
        response = requests.get(f"{SERVER_URL}/")
        print(f"✅ Root endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Message: {data['message']}")
            print(f"   Version: {data['version']}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Root endpoint failed: {e}")
        return False

def test_initialize():
    """Test the initialize endpoint."""
    try:
        # Load sample data
        with open('sample_data.json', 'r') as f:
            sample_data = json.load(f)
        
        response = requests.post(f"{SERVER_URL}/initialize", json=sample_data)
        print(f"✅ Initialize endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Success: {data['success']}")
            print(f"   Message: {data['message']}")
            if data['data']:
                print(f"   Current slot: {data['data']['current_slot']}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Initialize endpoint failed: {e}")
        return False

def test_state():
    """Test the state endpoint."""
    try:
        response = requests.get(f"{SERVER_URL}/state")
        print(f"✅ State endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Success: {data['success']}")
            print(f"   Current slot (tau): {data['data']['tau']}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ State endpoint failed: {e}")
        return False

def test_process_block():
    """Test the process-block endpoint."""
    try:
        # Create a test block
        test_block = {
            "input": {
                "slot": 2,
                "entropy": "0x9d3f7e438f2b1c5a6e8d9f0a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0",
                "extrinsic": []
            },
            "pre_state": {}  # Will be ignored since we're already initialized
        }
        
        response = requests.post(f"{SERVER_URL}/process-block", json=test_block)
        print(f"✅ Process-block endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Success: {data['success']}")
            print(f"   New slot: {data['data']['current_slot']}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Process-block endpoint failed: {e}")
        return False

def test_reset():
    """Test the reset endpoint."""
    try:
        response = requests.post(f"{SERVER_URL}/reset")
        print(f"✅ Reset endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Success: {data['success']}")
            print(f"   Message: {data['message']}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Reset endpoint failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing JAM Safrole Integration Server")
    print("=" * 50)
    
    # Check if server is running
    if not test_health():
        print("\n❌ Server is not running or not accessible")
        print("   Start the server with: python app.py")
        return
    
    print("\n" + "=" * 50)
    
    # Test root endpoint
    test_root()
    
    print("\n" + "=" * 50)
    
    # Test initialize
    if test_initialize():
        print("\n" + "=" * 50)
        
        # Test state retrieval
        test_state()
        
        print("\n" + "=" * 50)
        
        # Test block processing
        test_process_block()
        
        print("\n" + "=" * 50)
        
        # Test reset
        test_reset()
    
    print("\n" + "=" * 50)
    print("🎉 Testing completed!")
    
    # Final health check
    print("\nFinal health check:")
    test_health()

if __name__ == "__main__":
    main()
