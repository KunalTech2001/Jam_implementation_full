"""
Fallback condition implementation for JAM protocol.

This module implements the fallback mechanism for gamma_s calculation
as described in the JAM Graypaper Equation 6.26.
"""

from hashlib import blake2b


def hex_to_bytes(hex_str):
    """Convert hex string to bytes."""
    if hex_str.startswith('0x'):
        hex_str = hex_str[2:]
    return bytes.fromhex(hex_str)


def le_bytes_to_int(b):
    """Convert little-endian bytes to integer."""
    return int.from_bytes(b, 'little')


def calculate_fallback_gamma_s(post_state):
    """
    Calculates the gamma_s for the post_state based on the JAM protocol's fallback mechanism.
    Corresponds to Equation 6.26 in the JAM Graypaper.
    """
    E = post_state.get('E', 600)
    
    entropy_bytes = hex_to_bytes(post_state['eta'][2])
    new_validator_set = post_state['kappa']
    validator_set_size = len(new_validator_set)

    if validator_set_size == 0:
        return []

    new_gamma_s_keys = []

    for i in range(E):
        index_bytes = i.to_bytes(4, 'little')
        
        concatenated_input = entropy_bytes + index_bytes
        
        hash_result = blake2b(concatenated_input, digest_size=32).digest()
        
        random_index_bytes = hash_result[:4]
        random_index = le_bytes_to_int(random_index_bytes)
        
        validator_index = random_index % validator_set_size
        selected_validator = new_validator_set[validator_index]
        
        new_gamma_s_keys.append(selected_validator['bandersnatch'])
        
    return new_gamma_s_keys 