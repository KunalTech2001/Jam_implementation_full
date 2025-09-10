# server/gen_sign.py
import os, sys, json
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
REPORTS_SRC = os.path.join(PROJECT_ROOT, 'Reports-Python', 'src')
if REPORTS_SRC not in sys.path:
    sys.path.append(REPORTS_SRC)

from offchain.signature import generate_key_pair, sign_message, public_key_to_base64

# Keep these in sync with your curl payload
SLOT = 69
ANCHOR_BLOCK_NUMBER = 69

# Generate keypair first so we can include the public key in currentGuarantors
kp = generate_key_pair()
pub_b64 = public_key_to_base64(kp["public_key"])

signable = {
    "workPackage": {
        "authorizationToken": "0x",
        "authorizationServiceDetails": {
            "h": "auth.service.local",
            "u": "/authorize",
            "f": "authorizeWork"
        },
        "context": "example-context",
        "workItems": [
            {
                "id": "item-1",
                "programHash": "0x00",
                "inputData": "0x",
                "gasLimit": 1000
            }
        ]
    },
    "refinementContext": {
        "anchorBlockRoot": "0x0000000000000000000000000000000000000000000000000000000000000000",
        "anchorBlockNumber": ANCHOR_BLOCK_NUMBER,
        "beefyMmrRoot": "0x0000000000000000000000000000000000000000000000000000000000000000",
        "currentSlot": 1,
        "currentEpoch": 0,
        "currentGuarantors": [pub_b64],
        "previousGuarantors": []
    },
    "pvmOutput": "0x",
    "gasUsed": 0,
    "availabilitySpec": None,
    "coreIndex": 0,
    "slot": SLOT,
    "dependencies": []
}

sig_b64 = sign_message(signable, kp["private_key"])

print("guarantorPublicKey:", pub_b64)
print("guarantorSignature:", sig_b64)