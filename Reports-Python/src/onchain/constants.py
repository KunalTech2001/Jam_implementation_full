"""
Defines constants for on-chain logic and validation.
"""

ONCHAIN_CONSTANTS = {
    "SUPER_MAJORITY_THRESHOLD_NUMERATOR": 2,  # For 2/3 super-majority
    "SUPER_MAJORITY_THRESHOLD_DENOMINATOR": 3,
    "REPORT_TIMEOUT_SLOTS": 100,  # Reports older than this might be considered timed out
    "MAX_DEPENDENCIES": 10,  # Maximum number of dependencies a report can have
    "MAX_WORK_REPORT_GAS": 200000,  # Max gas for a WorkReport (too_high_work_report_gas)
    "MIN_SERVICE_ITEM_GAS": 10,  # Min gas for a service item (service_item_gas_too_low)
    "MAX_CORE_INDEX": 1023,  # Example max core index (bad_core_index)
    "MAX_VALIDATOR_INDEX": 1023,  # Example max validator index (bad_validator_index)
    "ANCHOR_MAX_AGE_SLOTS": 50,  # Max age for context anchor (anchor_not_recent)
    "RECENT_HISTORY_LOOKUP_SLOTS": 200,  # How far back to check for duplicate packages or dependencies
}