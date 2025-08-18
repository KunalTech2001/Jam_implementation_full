"""
Cryptographic bridge utilities for JAM protocol.
"""

from hashlib import blake2b


class CryptoBridge:
    """Cryptographic bridge for JAM protocol operations."""
    
    @staticmethod
    def calculate_eta_update(eta0, vrf_output):
        """Calculate eta update using Blake2b hash."""
        combined_input = eta0 + vrf_output
        return blake2b(combined_input, digest_size=32).digest() 