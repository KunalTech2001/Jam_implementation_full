import json
import nacl.encoding
import nacl.signing
from nacl.exceptions import BadSignatureError

# Generate a new key pair for testing
signing_key = nacl.signing.SigningKey.generate()
verify_key = signing_key.verify_key

# Convert to hex strings for storage/transmission
private_key_hex = signing_key.encode(encoder=nacl.encoding.HexEncoder).decode('utf-8')
public_key_hex = verify_key.encode(encoder=nacl.encoding.HexEncoder).decode('utf-8')

print(f"Generated key pair:")
print(f"Private key (keep this secret!): {private_key_hex}")
print(f"Public key: {public_key_hex}")

# Create a test payload
payload = {
    "action": "authorize",
    "data": "test"
}

# Convert payload to a string with sorted keys for consistent signing
message = json.dumps(payload, sort_keys=True).encode()
print(f"\nMessage to sign: {message}")

# Sign the message - in NaCl, the signature is prepended to the message
signed_message = signing_key.sign(message)

# Get just the signature part (first 64 bytes)
signature_bytes = signed_message.signature
signature_hex = signature_bytes.hex()

print(f"\nSignature: {signature_hex}")
print(f"Message with signature: {signed_message.hex()}")

# Now verify the signature
try:
    # Reconstruct the signed message by combining signature and message
    reconstructed_signed = signature_bytes + message
    verify_key.verify(reconstructed_signed)
    print("\nSignature verification SUCCEEDED")
    
    # Prepare the test request
    test_request = {
        "public_key": public_key_hex,
        "signature": signature_hex,
        "payload": payload
    }
    
    print("\nTest request JSON:")
    print(json.dumps(test_request, indent=2))
    
except BadSignatureError:
    print("\nSignature verification FAILED")
