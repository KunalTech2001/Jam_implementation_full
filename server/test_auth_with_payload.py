import json
import sys
import os
import requests
from nacl.signing import SigningKey
from nacl.encoding import HexEncoder

def load_payload(payload_file):
    """Load payload from a JSON file."""
    try:
        with open(payload_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading payload file: {e}")
        sys.exit(1)

def test_authorization(payload_file):
    # Load the payload from file
    payload = load_payload(payload_file)
    
    # Check if we need to generate a new keypair
    if 'private_key' not in payload:
        print("No private key found in payload. Generating a new keypair...")
        signing_key = SigningKey.generate()
        verify_key = signing_key.verify_key
        payload['private_key'] = signing_key.encode(encoder=HexEncoder).decode('utf-8')
        payload['public_key'] = verify_key.encode(encoder=HexEncoder).decode('utf-8')
        print(f"Generated new keypair. Public key: {payload['public_key']}")
    else:
        # Load the keypair from the payload
        try:
            signing_key = SigningKey(payload['private_key'], encoder=HexEncoder)
            verify_key = signing_key.verify_key
            payload['public_key'] = verify_key.encode(encoder=HexEncoder).decode('utf-8')
        except Exception as e:
            print(f"Error loading private key: {e}")
            sys.exit(1)
    
    # Prepare the request data
    request_data = {
        "public_key": payload['public_key'],
        "signature": "",  # Will be set after signing
        "payload": payload.get('data', {})
    }
    
    # Convert payload to JSON string with sorted keys for consistent signing
    message = json.dumps(request_data['payload'], sort_keys=True).encode()
    
    # Sign the message
    signature = signing_key.sign(message).signature
    request_data['signature'] = signature.hex()
    
    # Add nonce if present in payload (though server now manages it server-side)
    if 'nonce' in payload:
        request_data['nonce'] = payload['nonce']
    
    print("Sending request with:")
    print(f"Public Key: {request_data['public_key']}")
    print(f"Signature: {request_data['signature']}")
    print(f"Payload: {json.dumps(request_data['payload'], indent=2)}")
    
    # Send the request
    try:
        response = requests.post(
            "http://127.0.0.1:8000/authorize",
            json=request_data,
            timeout=10
        )
        print(f"\nStatus Code: {response.status_code}")
        try:
            print("Response:", json.dumps(response.json(), indent=2))
        except:
            print("Response:", response.text)
            
        # Save the updated payload with any generated keys
        with open(payload_file, 'w') as f:
            json.dump(payload, f, indent=2)
            
    except Exception as e:
        print(f"\nError making request: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <payload_file.json>")
        sys.exit(1)
        
    payload_file = sys.argv[1]
    if not os.path.exists(payload_file):
        print(f"Payload file not found: {payload_file}")
        print("Creating a new payload file with default values...")
        default_payload = {
            "data": {
                "action": "authorize",
                "data": "test_data"
            }
        }
        with open(payload_file, 'w') as f:
            json.dump(default_payload, f, indent=2)
        print(f"Created new payload file: {payload_file}")
        print("Please edit the file and run the script again.")
        sys.exit(0)
        
    test_authorization(payload_file)
