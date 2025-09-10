from utils.validator import validate_required, validate_type
import re
import hashlib

class WorkDigest:
    """
    Represents a Work-Digest (D), a cryptographic digest of the Work-Report.
    Used for uniqueness and efficient lookup.
    """

    def __init__(self, hash: str):
        validate_required(hash, 'WorkDigest Hash')
        validate_type(hash, 'WorkDigest Hash', str)
        # Basic hash format validation (e.g., hex string, specific length)
        if not re.fullmatch(r'[0-9a-fA-F]{64}', hash):
            # Optionally warn or raise here
            pass
        self.hash = hash

    def to_object(self) -> dict:
        """
        Converts the WorkDigest to a plain dict for serialization.
        """
        return {
            'hash': self.hash,
        }

    @staticmethod
    def sha256_hash(data: str) -> str:
        """
        Utility to compute SHA256 hash as a hex string.
        """
        return hashlib.sha256(data.encode('utf-8')).hexdigest()