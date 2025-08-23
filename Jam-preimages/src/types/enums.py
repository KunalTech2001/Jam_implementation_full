from enum import IntEnum


class PreimageErrorCode(IntEnum):
    """Error codes for preimage validation."""
    preimage_unneeded = 0
    preimages_not_sorted_unique = 1
