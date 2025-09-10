"""
Simulates the processing of an E_G (Guarantees) extrinsic.
Handles validation and state updates for new Work-Report submissions.
"""

import math
from offchain.signature import verify_signature, base64_to_public_key
# from ...offchain.encoder import generate_work_digest
from onchain.constants import ONCHAIN_CONSTANTS
from utils.errors import ProtocolError

# from ...models.work_report import WorkReport
# from ...models.work_digest import WorkDigest
# from ..state import OnchainState

def validate_work_report(report, onchain_state, current_slot, current_block_digests):
    public_key_bytes = base64_to_public_key(report.guarantor_public_key)
    if not verify_signature(report.to_signable_object(), report.guarantor_signature, public_key_bytes):
        raise ProtocolError('bad_signature: Work-Report signature is invalid.')

    context = report.refinement_context
    if current_slot - context.anchor_block_number > ONCHAIN_CONSTANTS["ANCHOR_MAX_AGE_SLOTS"]:
        raise ProtocolError('anchor_not_recent: Context anchor block is too old.')

    # authorization_service_details is modeled as a dict with keys 'h', 'u', 'f'
    service_details = report.work_package.authorization_service_details
    service_id = (
        service_details.get('u')
        if isinstance(service_details, dict)
        else getattr(service_details, 'u', None)
    )
    if not service_id:
        raise ProtocolError('bad_service_details: Missing service identifier "u".')
    if service_id not in (onchain_state.global_state.service_registry or {}):
        # raise ProtocolError("bad_service_id: Work result service identifier has no associated account in state.")
        pass

    expected_code_hash = (onchain_state.global_state.service_registry or {}).get(service_id, {}).get('codeHash')
    if (
        expected_code_hash and
        getattr(report.work_package.work_items[0], "program_hash", None) != expected_code_hash
    ):
        raise ProtocolError("bad_code_hash: Work result code hash doesn't match expected for service.")

    current_guarantors = context.current_guarantors
    previous_guarantors = context.previous_guarantors
    report_slot = report.slot
    current_epoch = context.current_epoch
    report_epoch = report_slot // ONCHAIN_CONSTANTS["REPORT_TIMEOUT_SLOTS"]

    is_guarantor_assigned = False
    if report_epoch == current_epoch and report.guarantor_public_key in current_guarantors:
        is_guarantor_assigned = True
    if (
        not is_guarantor_assigned and
        report_epoch == current_epoch - 1 and
        report.guarantor_public_key in previous_guarantors
    ):
        is_guarantor_assigned = True

    if not is_guarantor_assigned:
        raise ProtocolError(
            'wrong_assignment / not_authorized: Unexpected guarantor for work report core or not authorized.'
        )

    if (onchain_state.global_state.core_status or {}).get(report.core_index) == 'engaged':
        raise ProtocolError(f"core_engaged: Core {report.core_index} is not available.")

    if report.slot > current_slot:
        raise ProtocolError('future_report_slot: Report refers to a slot in the future.')
    if current_slot - report.slot > ONCHAIN_CONSTANTS["REPORT_TIMEOUT_SLOTS"]:
        raise ProtocolError('report_before_last_rotation: Report guarantee slot is too old.')

    if len(report.dependencies) > ONCHAIN_CONSTANTS["MAX_DEPENDENCIES"]:
        raise ProtocolError('too_many_dependencies: Work report has too many dependencies.')
    for dep_hash in report.dependencies:
        is_dependency_met = (
            dep_hash in onchain_state.ξ or
            dep_hash in onchain_state.ρ or
            any(d.hash == dep_hash for d in current_block_digests)
        )
        if not is_dependency_met:
            raise ProtocolError(
                f"dependency_missing / segment_root_lookup_invalid: Dependency {dep_hash} not found."
            )

    if report.gas_used > ONCHAIN_CONSTANTS["MAX_WORK_REPORT_GAS"]:
        raise ProtocolError('too_high_work_report_gas: Work report per core gas is too high.')
    if any(
        item.gas_limit < ONCHAIN_CONSTANTS["MIN_SERVICE_ITEM_GAS"]
        for item in report.work_package.work_items
    ):
        raise ProtocolError('service_item_gas_too_low: Accumulate gas is below the service minimum.')

    report_digest = report.generate_digest()  # You must implement this method or import generate_work_digest
    if report_digest.hash in onchain_state.ξ:
        raise ProtocolError('duplicate_package_in_recent_history: Package was already finalized.')

def process_guarantee_extrinsic(
    report,
    onchain_state,
    current_slot,
    current_block_digests=None
):
    if current_block_digests is None:
        current_block_digests = []

    print(
        f"[E_G] Processing Work-Report from {report.guarantor_public_key} for core {report.core_index} at slot {report.slot}"
    )

    try:
        validate_work_report(report, onchain_state, current_slot, current_block_digests)
    except Exception as error:
        print(f"[E_G] Report validation failed: {str(error)}")
        report_digest = report.generate_digest()
        onchain_state.ψ_B[report_digest.hash] = {
            'reason': str(error),
            'disputed_by': {'system_validation'}
        }
        offender_record = onchain_state.ψ_O.get(report.guarantor_public_key)
        if offender_record:
            offender_record['dispute_count'] += 1
            offender_record['last_dispute_slot'] = current_slot
        else:
            onchain_state.ψ_O[report.guarantor_public_key] = {
                'dispute_count': 1,
                'last_dispute_slot': current_slot
            }
        return False

    report_digest = report.generate_digest()
    digest_hash = report_digest.hash
    report_entry = onchain_state.ρ.get(digest_hash)

    if not report_entry:
        report_entry = {
            'report': report,
            'received_signatures': {report.guarantor_public_key},
            'submission_slot': current_slot
        }
        onchain_state.ρ[digest_hash] = report_entry
        print(f"[E_G] New report {digest_hash} added to pending (ρ).")
    else:
        if report.guarantor_public_key in report_entry['received_signatures']:
            print(
                f"[E_G] Duplicate signature from {report.guarantor_public_key} for report {digest_hash}. Ignoring."
            )
            return False
        report_entry['received_signatures'].add(report.guarantor_public_key)
        print(
            f"[E_G] Added signature from {report.guarantor_public_key} for report {digest_hash}. Total signatures: {len(report_entry['received_signatures'])}"
        )

    total_guarantors = (
        len(report.refinement_context.current_guarantors) +
        len(report.refinement_context.previous_guarantors)
    )
    required_signatures = math.ceil(
        (total_guarantors * ONCHAIN_CONSTANTS["SUPER_MAJORITY_THRESHOLD_NUMERATOR"]) /
        ONCHAIN_CONSTANTS["SUPER_MAJORITY_THRESHOLD_DENOMINATOR"]
    )

    if len(report_entry['received_signatures']) >= required_signatures:
        print(
            f"[E_G] Report {digest_hash} reached 2/3 super-majority ({len(report_entry['received_signatures'])}/{total_guarantors}). Moving to accumulation queue (ω)."
        )
        del onchain_state.ρ[digest_hash]
        onchain_state.ω[digest_hash] = {'report': report, 'status': 'ready'}
        return True
    else:
        print(
            f"[E_G] Report {digest_hash} needs more signatures ({len(report_entry['received_signatures'])}/{required_signatures})."
        )

    if current_slot - report_entry['submission_slot'] > ONCHAIN_CONSTANTS["REPORT_TIMEOUT_SLOTS"]:
        print(f"[E_G] Report {digest_hash} timed out. Removing from pending (ρ).")
        del onchain_state.ρ[digest_hash]
        onchain_state.ψ_B[digest_hash] = {
            'reason': 'timed_out',
            'disputed_by': {'system_timeout'}
        }
        return False

    return True