from typing import Any

from utils.errors import ProtocolError
# from ..state import OnchainState
# from ...models.work_report import WorkReport

def process_dispute_extrinsic(
    dispute: dict,
    onchain_state: 'OnchainState',
    current_slot: int
) -> None:
    disputed_digest_hash = dispute['disputedDigestHash']
    disputer_public_key = dispute['disputerPublicKey']
    reason = dispute['reason']
    print(f"[E_D] Processing dispute for digest: {disputed_digest_hash} by {disputer_public_key} at slot {current_slot}. Reason: {reason}")

    if not onchain_state.get_report_by_digest(disputed_digest_hash):
        print(f"[E_D] Attempted to dispute non-existent or already finalized/disputed report: {disputed_digest_hash}")
        return

    disputed_report = None

    # Remove from pending reports (ρ)
    if disputed_digest_hash in onchain_state.ρ:
        disputed_report = onchain_state.ρ[disputed_digest_hash].report
        del onchain_state.ρ[disputed_digest_hash]
        print(f"[E_D] Removed report {disputed_digest_hash} from pending (ρ).")

    # Remove from accumulation queue (ω)
    if disputed_digest_hash in onchain_state.ω:
        disputed_report = onchain_state.ω[disputed_digest_hash].report
        del onchain_state.ω[disputed_digest_hash]
        print(f"[E_D] Removed report {disputed_digest_hash} from accumulation queue (ω).")

    # Handle late dispute from history (ξ)
    if disputed_digest_hash in onchain_state.ξ and not disputed_report:
        disputed_report = onchain_state.ξ[disputed_digest_hash]
        print(f"[E_D] Late dispute for finalized report {disputed_digest_hash}. It remains in history but guarantor will be penalized.")

    if not disputed_report:
        raise ProtocolError(f"Attempted to dispute a report ({disputed_digest_hash}) that was not found in active state or history.")

    # Record in ψ_B (bad reports)
    existing_bad_report = onchain_state.ψ_B.get(disputed_digest_hash)
    if existing_bad_report:
        if isinstance(existing_bad_report['disputed_by'], set):
            existing_bad_report['disputed_by'].add(disputer_public_key)
        elif isinstance(existing_bad_report['disputed_by'], list):
            existing_bad_report['disputed_by'].append(disputer_public_key)
        else:
            existing_bad_report['disputed_by'] = {disputer_public_key}
    else:
        new_bad_report = {
            'reason': reason,
            'disputed_by': {disputer_public_key},
        }
        onchain_state.ψ_B[disputed_digest_hash] = new_bad_report
    print(f"[E_D] Recorded report {disputed_digest_hash} in bad reports (ψ_B).")

    # Update ψ_O (offenders)
    guarantor_public_key = getattr(disputed_report, 'guarantor_public_key', None)
    offender_record = onchain_state.ψ_O.get(guarantor_public_key)
    if offender_record:
        offender_record['dispute_count'] += 1
        offender_record['last_dispute_slot'] = current_slot
    else:
        new_offender = {
            'dispute_count': 1,
            'last_dispute_slot': current_slot,
        }
        onchain_state.ψ_O[guarantor_public_key] = new_offender
    print(f"[E_D] Updated offender record for guarantor {guarantor_public_key}.")