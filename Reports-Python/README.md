### Python Code

```python
# filepath: /Users/hatim/Projects/JAM/Reports/src/onchain/accumulation/queue_handler.py
from collections import defaultdict, deque
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Types for the accumulation queue
class WorkReport:
    def __init__(self, core_index, dependencies, work_package):
        self.core_index = core_index
        self.dependencies = dependencies
        self.work_package = work_package

class WorkDigest:
    pass  # Define as needed

class OnchainState:
    def __init__(self):
        self.ω = {}  # Accumulation queue
        self.ξ = {}  # Processed reports
        self.ψ_B = {}  # Failed accumulations
        self.ψ_O = {}  # Offenders
        self.global_state = {}  # Global state

def simulate_psi_a_pvm(work_item, global_state):
    # Simulate the PVM execution (placeholder)
    return {}  # Return a state delta

def apply_delta(global_state, state_delta):
    # Apply the state delta to the global state (placeholder)
    return global_state  # Return updated global state

def topological_sort(accumulation_queue):
    logging.info("[Q] Performing topological sort on accumulation queue...")

    graph = defaultdict(set)
    in_degree = defaultdict(int)
    reports_map = {}

    for digest_hash, entry in accumulation_queue.items():
        reports_map[digest_hash] = entry['report']
        in_degree[digest_hash] = 0

    for digest_hash, entry in accumulation_queue.items():
        for dep_hash in entry['report'].dependencies:
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
        logging.warning("[Q] Cyclic dependency detected or unresolved dependencies.")

    logging.info("[Q] Topological sort completed. Order: %s", sorted_order)
    return sorted_order

def process_accumulation_queue(onchain_state, current_slot):
    logging.info("[Accumulation] Processing accumulation queue (ω) at slot %d...", current_slot)

    reports_to_accumulate_digests = topological_sort(onchain_state.ω)

    for digest_hash in reports_to_accumulate_digests:
        entry = onchain_state.ω.get(digest_hash)

        if not entry or entry['status'] != 'ready':
            continue

        report = entry['report']
        logging.info("[Accumulation] Accumulating report %s (Core: %d)...", digest_hash, report.core_index)

        # Update status
        entry['status'] = 'processing'
        onchain_state.ω[digest_hash] = entry

        try:
            for work_item in report.work_package['work_items']:
                state_delta = simulate_psi_a_pvm(work_item, onchain_state.global_state)

                # Ensure apply_delta returns a valid global state
                updated_state = apply_delta(onchain_state.global_state, state_delta)
                onchain_state.global_state = {
                    'accounts': updated_state.get('accounts', {}),
                    'core_status': updated_state.get('core_status', {}),
                    'service_registry': updated_state.get('service_registry', {}),
                }

            # Move report to ξ
            del onchain_state.ω[digest_hash]
            onchain_state.ξ[digest_hash] = report
            logging.info("[Accumulation] Report %s accumulated and moved to ξ.", digest_hash)

        except Exception as error:
            logging.error("[Accumulation] Failed to accumulate report %s: %s", digest_hash, str(error))

            del onchain_state.ω[digest_hash]
            onchain_state.ψ_B[digest_hash] = {
                'reason': f'accumulation_failed: {str(error)}',
                'disputed_by': {'system_accumulation'},
            }

            key = report.guarantor_public_key
            offender = onchain_state.ψ_O.get(key)
            if offender:
                offender['dispute_count'] += 1
                offender['last_dispute_slot'] = current_slot
            else:
                onchain_state.ψ_O[key] = {
                    'dispute_count': 1,
                    'last_dispute_slot': current_slot,
                }

    logging.info("[Accumulation] Accumulation queue processing finished.")
```

### Key Changes and Considerations:
1. **Data Structures**: Python uses dictionaries and lists instead of TypeScript's `Map` and `Set`.
2. **Logging**: Python's `logging` module is used for logging instead of `console.log`.
3. **Error Handling**: Python uses `try-except` for error handling.
4. **Type Annotations**: Python does not enforce types like TypeScript, but you can use type hints for better clarity.
5. **Function Definitions**: Functions are defined using `def` in Python, and the syntax is slightly different from TypeScript.

Make sure to adapt the placeholder functions (`simulate_psi_a_pvm` and `apply_delta`) to your actual logic as needed.