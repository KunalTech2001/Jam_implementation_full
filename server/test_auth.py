import json
import base64
import requests
from nacl.signing import SigningKey
from nacl.encoding import Base64Encoder, HexEncoder

def test_authorization():
    # Generate a new keypair
    signing_key = SigningKey.generate()
    verify_key = signing_key.verify_key
    
    # Prepare the payload
    payload = {
        "action": "authorize",
        "data": "test_data"
    }
    
    # Convert payload to JSON string with sorted keys
    message = json.dumps(payload, sort_keys=True).encode()
    
    # Sign the message
    signature = signing_key.sign(message).signature
    
    # Prepare the request
    request_data = {
        "public_key": verify_key.encode(encoder=HexEncoder).decode('utf-8'),
        "signature": signature.hex(),
        "payload": payload
    }
    
    print("Sending request with:")
    print(f"Public Key: {request_data['public_key']}")
    print(f"Signature: {request_data['signature']}")
    print(f"Payload: {json.dumps(payload, indent=2)}")
    
    # Send the request
    try:
        response = requests.post(
            "http://127.0.0.1:8000/authorize",
            json=request_data,
            timeout=10
        )
        print(f"\nStatus Code: {response.status_code}")
        print("Response:", response.json())
    except Exception as e:
        print(f"\nError making request: {str(e)}")

if __name__ == "__main__":
    test_authorization()