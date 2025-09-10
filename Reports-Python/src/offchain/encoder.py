"""
Handles Reed-Solomon erasure coding for data packaging and availability specification.
This is a simplified mock/conceptual outline. A full implementation would involve
a dedicated Reed-Solomon library.
"""

import hashlib
import json

from models.availability_spec import AvailabilitySpec
from models.work_digest import WorkDigest

def sha256_hash(data: str) -> str:
    return hashlib.sha256(data.encode('utf-8')).hexdigest()

def encode_for_availability(report, data_fragments: int = 4, parity_fragments: int = 2) -> AvailabilitySpec:
    """
    Mocks the Reed-Solomon encoding process.
    In a real scenario, this would take the report data, split it into
    data fragments, generate parity fragments, and return hashes of all fragments.
    :param report: The WorkReport to encode.
    :param data_fragments: The desired number of data fragments.
    :param parity_fragments: The desired number of parity fragments.
    :return: AvailabilitySpec
    """
    total_fragments = data_fragments + parity_fragments
    report_string = json.dumps(report.to_signable_object(), sort_keys=True)

    fragment_hashes = []
    for i in range(total_fragments):
        fragment_content = report_string[i * 10:(i + 1) * 10] + f"_fragment_{i}"
        fragment_hashes.append(sha256_hash(fragment_content))

    return AvailabilitySpec(total_fragments, data_fragments, fragment_hashes)

def generate_work_digest(report) -> WorkDigest:
    """
    Generates a WorkDigest for a given WorkReport.
    :param report: The WorkReport to generate a digest for.
    :return: WorkDigest
    """
    report_string = json.dumps(report.to_signable_object(), sort_keys=True)
    digest_hash = sha256_hash(report_string)
    return WorkDigest(digest_hash)