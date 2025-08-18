"""
Helper utility functions for JAM protocol implementation.
"""

import json
import copy
import requests
import sys


PADDING_VALIDATOR = {
    "bandersnatch": "0x" + "0" * 64,
    "ed25519": "0x" + "0" * 64,
    "bls": "0x" + "0" * 288,
    "metadata": "0x" + "0" * 256,
}


def hex_to_bytes(hex_str):
    """Convert hex string to bytes."""
    if not hex_str:
        return b""
    if hex_str.startswith("0x"):
        hex_str = hex_str[2:]
    return bytes.fromhex(hex_str)


def bytes_to_hex(b):
    """Convert bytes to hex string."""
    return "0x" + b.hex()


def deep_clone(obj):
    """Create a deep copy of an object."""
    return copy.deepcopy(obj)


def deep_equal(a, b):
    """Compare two objects for deep equality."""
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def get_epoch_and_slot_phase(timeslot, epoch_length):
    """Calculate epoch and slot phase from timeslot."""
    return timeslot // epoch_length, timeslot % epoch_length


def process_validator_keys_for_offenders(keys, offenders):
    """Process validator keys, replacing offenders with padding validators."""
    offender_set = set(offenders)
    return [
        deep_clone(PADDING_VALIDATOR) if k["ed25519"] in offender_set else deep_clone(k)
        for k in keys
    ]


def z(sequence):
    """Zigzag function for processing sequences."""
    result = []
    left, right = 0, len(sequence) - 1
    while left <= right:
        if left <= right:
            result.append(sequence[left])
        if left < right:
            result.append(sequence[right])
        left += 1
        right -= 1
    return result


def get_gamma_z_from_rust_server(public_keys, offenders):
    """Get gamma_z from Rust server."""
    processed_keys = process_validator_keys_for_offenders(public_keys, offenders)
    bandersnatch_keys = [
        (
            "0x" + "0" * 64
            if k["bandersnatch"]
            == "0x1ecc3686b60ee3b84b6c7d321d70d5c06e9dac63a4d0a79d731b17c0d04d030d"
            else k["bandersnatch"]
        )
        for k in processed_keys
    ]
    try:
        response = requests.post(
            "http://127.0.0.1:3000/compose_gamma_z",
            json={"public_keys": bandersnatch_keys},
        )
        response.raise_for_status()
        return response.json()["gamma_z"]
    except requests.exceptions.RequestException as e:
        print(f"Could not reach Rust server to compose gamma_z: {e}", file=sys.stderr)
        raise Exception("Failed to get gamma_z") 