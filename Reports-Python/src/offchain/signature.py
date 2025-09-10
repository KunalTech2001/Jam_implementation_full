"""
Handles cryptographic signing and verification using pynacl.
Assumes Ed25519 keys.

To use:
pip install pynacl
"""

import json
import base64
from nacl.signing import SigningKey, VerifyKey
from nacl.exceptions import BadSignatureError

def encode_text(text: str) -> bytes:
    """Helper to encode text as UTF-8 bytes."""
    return text.encode('utf-8')

def sign_message(message_object: dict, private_key_bytes: bytes) -> str:
    """
    Signs a message using a private key.
    :param message_object: The object to be signed (will be stringified).
    :param private_key_bytes: The Ed25519 private key (32-byte seed or 64-byte expanded).
    :return: The base64 encoded signature.
    """
    try:
        message_string = json.dumps(message_object, separators=(',', ':'), sort_keys=True)
        message_bytes = encode_text(message_string)
        signing_key = SigningKey(private_key_bytes)
        signature = signing_key.sign(message_bytes).signature
        return base64.b64encode(signature).decode('utf-8')
    except Exception as error:
        print("Error signing message:", error)
        raise RuntimeError("Failed to sign message.")

def verify_signature(message_object: dict, signature_base64: str, public_key_bytes: bytes) -> bool:
    """
    Verifies a signature against a message and public key.
    :param message_object: The original message object that was signed.
    :param signature_base64: The base64 encoded signature.
    :param public_key_bytes: The Ed25519 public key (32 bytes).
    :return: True if the signature is valid, False otherwise.
    """
    try:
        message_string = json.dumps(message_object, separators=(',', ':'), sort_keys=True)
        message_bytes = encode_text(message_string)
        signature_bytes = base64.b64decode(signature_base64)
        verify_key = VerifyKey(public_key_bytes)
        verify_key.verify(message_bytes, signature_bytes)
        return True
    except BadSignatureError:
        return False
    except Exception as error:
        print("Error verifying signature:", error)
        return False

def generate_key_pair():
    """
    Generates a new Ed25519 key pair.
    :return: A dict containing 'public_key' and 'private_key' as bytes.
    """
    signing_key = SigningKey.generate()
    verify_key = signing_key.verify_key
    return {
        'public_key': verify_key.encode(),
        'private_key': signing_key.encode()
    }

def public_key_to_base64(public_key_bytes: bytes) -> str:
    """
    Converts a public key (bytes) to a base64 string.
    """
    return base64.b64encode(public_key_bytes).decode('utf-8')

def base64_to_public_key(public_key_base64: str) -> bytes:
    """
    Converts a base64 string public key to bytes.
    """
    return base64.b64decode(public_key_base64)