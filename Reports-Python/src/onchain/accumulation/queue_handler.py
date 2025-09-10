"""
Handles the accumulation queue (ω), including topological sorting (Q function)
and orchestrating the Ψ_A PVM execution and state integration.
"""

from collections import defaultdict, deque
from typing import Dict, List, Set, Any

from .pvm_simulator import simulate_psi_a_pvm
from .state_integrator import apply_delta
from ..state import OnchainState

# Types for the accumulation queue
class AccumulationQueueEntry:
    def __init__(self, report, status: str):
        self.report = report
        self.status = status  # 'pending', 'ready', or 'processing'

AccumulationQueue = Dict[str, AccumulationQueueEntry]

def topological_sort(accumulation_queue: AccumulationQueue) -> List[str]:
    print("[Q] Performing topological sort on accumulation queue...")

    graph = defaultdict(set)
    in_degree = defaultdict(int)
    reports_map = {}

    for digest_hash, entry in accumulation_queue.items():
        reports_map[digest_hash] = entry.report
        graph[digest_hash] = set()
        in_degree[digest_hash] = 0

    for digest_hash, entry in accumulation_queue.items():
        for dep_hash in getattr(entry.report, "dependencies", []):
            if dep_hash in accumulation_queue:
                graph[dep_hash].add(digest_hash)
                in_degree[digest_hash] += 1

    queue = deque()
    for digest_hash, degree in in_degree.items():
        if degree == 0:
            queue.append(digest_hash)

    sorted_order = []
    while queue:
        current_digest = queue.popleft()
        sorted_order.append(current_digest)

        for neighbor_digest in graph[current_digest]:
            in_degree[neighbor_digest] -= 1
            if in_degree[neighbor_digest] == 0:
                queue.append(neighbor_digest)

    if len(sorted_order) != len(accumulation_queue):
        print("[Q] Cyclic dependency detected or unresolved dependencies.")

    print("[Q] Topological sort completed. Order:", sorted_order)
    return sorted_order

def process_accumulation_queue(onchain_state: OnchainState, current_slot: int) -> None:
    print(f"[Accumulation] Processing accumulation queue (ω) at slot {current_slot}...")

    reports_to_accumulate_digests = topological_sort(onchain_state.ω)

    for digest_hash in reports_to_accumulate_digests:
        entry = onchain_state.ω.get(digest_hash)

        if not entry or entry.status != 'ready':
            continue

        report = entry.report
        print(f"[Accumulation] Accumulating report {digest_hash} (Core: {getattr(report, 'core_index', None)})...")

        # Update status
        entry.status = 'processing'
        onchain_state.ω[digest_hash] = entry

        try:
            for work_item in getattr(report.work_package, "work_items", []):
                state_delta = simulate_psi_a_pvm(work_item, onchain_state.global_state)

                # Ensure apply_delta returns a valid GlobalState (dict)
                updated_state = apply_delta(onchain_state.global_state, state_delta)
                onchain_state.global_state = {
                    'accounts': updated_state.get('accounts', {}),
                    'core_status': updated_state.get('core_status', {}),
                    'service_registry': updated_state.get('service_registry', {}),
                }

            # Move report to ξ
            del onchain_state.ω[digest_hash]
            onchain_state.ξ[digest_hash] = report
            print(f"[Accumulation] Report {digest_hash} accumulated and moved to ξ.")

        except Exception as error:
            print(f"[Accumulation] Failed to accumulate report {digest_hash}: {str(error)}")

            del onchain_state.ω[digest_hash]
            onchain_state.ψ_B[digest_hash] = {
                'reason': f'accumulation_failed: {str(error)}',
                'disputed_by': {'system_accumulation'},
            }

            key = getattr(report, "guarantor_public_key", None)
            offender = onchain_state.ψ_O.get(key)
            if offender:
                offender.dispute_count += 1
                offender.last_dispute_slot = current_slot
            else:
                onchain_state.ψ_O[key] = {
                    'dispute_count': 1,
                    'last_dispute_slot': current_slot,
                }

    print("[Accumulation] Accumulation queue processing finished.")