import os
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
"""
Handles the accumulation queue (ω), including topological sorting (Q function)
and orchestrating the Ψ_A PVM execution and state integration.
"""
import json

from models.work_report import WorkReport
from models.work_package import WorkPackage
from models.work_item import WorkItem
from models.refinement_context import RefinementContext
from models.availability_spec import AvailabilitySpec
from onchain.__init__ import OnchainState, process_guarantee_extrinsic as real_process_guarantee_extrinsic
from onchain.state import GlobalState

# Equivalent to the TS hydrateMap function
def hydrate_map(obj):
    if obj is None:
        return obj
    if isinstance(obj, list):
        return [hydrate_map(x) for x in obj]
    if isinstance(obj, dict):
        # If _isSet or _isMap are used to indicate special objects, handle them here
        if obj.get('_isSet'):
            return set(hydrate_map(x) for x in obj['values'])
        if obj.get('_isMap'):
            return {k: hydrate_map(v) for k, v in obj['entries']}
        # Optionally, if every key is non-numeric, you might choose to convert to a dict or a Map.
        return {k: hydrate_map(v) for k, v in obj.items()}
    return obj

def initialize_state(pre_state):
    state = OnchainState()
    if pre_state:
        if 'ρ' in pre_state:
            state.ρ = hydrate_map(pre_state['ρ'])
        if 'ω' in pre_state:
            state.ω = hydrate_map(pre_state['ω'])
        if 'ξ' in pre_state:
            state.ξ = hydrate_map(pre_state['ξ'])
        if 'ψ_B' in pre_state:
            state.ψ_B = hydrate_map(pre_state['ψ_B'])
        if 'ψ_O' in pre_state:
            state.ψ_O = hydrate_map(pre_state['ψ_O'])
        if 'globalState' in pre_state:
            gs = pre_state['globalState']
            state.global_state = GlobalState(
                accounts=gs.get('accounts', {}),
                core_status=hydrate_map(gs.get('coreStatus', {})),
                service_registry=hydrate_map(gs.get('serviceRegistry', {}))
            )
    return state

def process_guarantee_extrinsic(extrinsic, state, slot, expected_error=None, post_state=None):
    if not extrinsic or 'guarantees' not in extrinsic or not extrinsic['guarantees']:
        print("WARNING: No guarantees found in extrinsic, skipping processing")
        return "No guarantees to process"
        
    guarantee = extrinsic['guarantees'][0]
    if 'report' not in guarantee:
        print("WARNING: No report in guarantee, skipping")
        return "No report in guarantee"
        
    report = guarantee['report']
    if report is None:
        print("WARNING: Report is None, skipping processing")
        return "Report is None"
        
    error = None
    try:
        print(f"DEBUG: Processing guarantee with report: {report}")
        real_process_guarantee_extrinsic(report, state, slot)
    except Exception as e:
        error = f"Error processing guarantee: {str(e)}"
        print(f"ERROR: {error}")
        import traceback
        traceback.print_exc()
        if expected_error and expected_error in str(e):
            return error
        raise
        
    return error

def deep_equal(a, b):
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)

def load_vector(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def compare_states(state, post_state):
    expected_state = initialize_state(post_state).to_plain_object()
    final_state = state.to_plain_object()
    return deep_equal(final_state, expected_state)

def map_input_to_extrinsic(input_data):
    # Deep copy via JSON roundtrip similar to the TS version
    extrinsic = json.loads(json.dumps(input_data))
    if 'guarantees' not in extrinsic:
        return extrinsic
        
    for guarantee in extrinsic['guarantees']:
        if not isinstance(guarantee, dict) or 'report' not in guarantee:
            print("WARNING: Invalid guarantee format, skipping")
            continue
            
        r = guarantee['report']
        if not r:
            print("WARNING: Report is None, skipping")
            continue
            
        try:
            # Handle workPackage
            wp = r.get('workPackage')
            if not wp:
                print("WARNING: Missing 'workPackage' key in report")
                continue
                
            # Build WorkPackage from the input data
            work_items = []
            for wi in wp.get('workItems', []):
                work_items.append(WorkItem(
                    wi.get('id'),
                    wi.get('programHash'),
                    wi.get('inputData'),
                    wi.get('gasLimit')
                ))
                
            package = WorkPackage(
                wp.get('authorizationToken'),
                wp.get('authorizationServiceDetails'),
                wp.get('context'),
                work_items
            )
            
            # Build RefinementContext
            ctx = r.get('refinementContext', {})
            ref_ctx = RefinementContext(
                ctx.get('anchorBlockRoot'),
                ctx.get('anchorBlockNumber'),
                ctx.get('beefyMmrRoot'),
                ctx.get('currentSlot'),
                ctx.get('currentEpoch'),
                ctx.get('currentGuarantors', []),
                ctx.get('previousGuarantors', [])
            )
            
            # Handle availability spec
            availability_spec = None
            if r.get('availabilitySpec'):
                aspec = r.get('availabilitySpec', {})
                availability_spec = AvailabilitySpec(
                    aspec.get('totalFragments'),
                    aspec.get('dataFragments'),
                    aspec.get('fragmentHashes')
                )
                
            # Create the work report
            guarantee['report'] = WorkReport(
                package,
                ref_ctx,
                r.get('pvmOutput'),
                r.get('gasUsed'),
                availability_spec,
                r.get('guarantorSignature'),
                r.get('guarantorPublicKey'),
                r.get('coreIndex'),
                r.get('slot'),
                r.get('dependencies', [])
            )
            
        except Exception as e:
            print(f"WARNING: Error processing guarantee: {e}")
            import traceback
            traceback.print_exc()
            continue
            
    return extrinsic

def run_vector(vector_path):
    print(f"Processing test vector: {vector_path}")
    vector = load_vector(vector_path)
    if not vector or 'pre_state' not in vector:
        print(f"{os.path.basename(vector_path)}: FAIL (Invalid vector, missing 'pre_state')")
        return
    state = initialize_state(vector['pre_state'])
    slot = 0
    # Optionally adjust the slot if lookup_anchor_slot provided in vector input
    try:
        lookup_slot = vector.get('input', {}).get('guarantees', [{}])[0].get('report', {}).get('context', {}).get('lookup_anchor_slot')
        if lookup_slot is not None:
            slot = lookup_slot + 65
    except Exception:
        pass

    error = None
    try:
        extrinsic = map_input_to_extrinsic(vector['input'])
        process_guarantee_extrinsic(extrinsic, state, slot, vector.get('expected_error'), vector.get('post_state'))
    except Exception as e:
        error = str(e)

    base = os.path.basename(vector_path)
    if 'expected_error' in vector and vector['expected_error']:
        if error and vector['expected_error'] in error:
            print(f"{base}: PASS (expected error)")
        else:
            print(f"{base}: FAIL (unexpected error/result)")
            if error:
                print(f"  -> Threw: {error}")
            else:
                print(f"  -> Expected error '{vector['expected_error']}', but none thrown.")
    else:
        if 'post_state' in vector and compare_states(state, vector['post_state']):
            print(f"{base}: PASS")
        else:
            print(f"{base}: FAIL (state mismatch)")

def inspect_vector(vector_path):
    try:
        vector = load_vector(vector_path)
        print(f"--- Inspection of {os.path.basename(vector_path)} ---")
        print("File exists: True")
        print("Vector content:")
        print(json.dumps(vector, indent=2))
        print(f"Has 'pre_state': {'pre_state' in vector}")
        print(f"Has 'input': {'input' in vector}")
        print(f"Has 'post_state': {'post_state' in vector}")
        print(f"Has 'expected_error': {'expected_error' in vector}")
    except Exception as e:
        print(f"Error inspecting file: {e}")

def main():

    print("DEBUG: main() entered")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    updated_state_path = os.path.abspath(
        os.path.join(script_dir, '..', '..', 'server', 'updated_state.json')
    )

    if not os.path.exists(updated_state_path):
        print(f"❌ Could not find updated_state.json at {updated_state_path}")
        return

    with open(updated_state_path, "r") as f:
        updated_state = json.load(f)
    if isinstance(updated_state, list) and len(updated_state) > 0 and isinstance(updated_state[0], dict):
        updated_state = updated_state[0]

    pre_state = updated_state.get("pre_state")
    input_data = updated_state.get("input")
    print("DEBUG: Loaded updated_state.json content")
    print("DEBUG: pre_state:", json.dumps(pre_state, indent=2) if pre_state else "None")
    print("DEBUG: input_data:", json.dumps(input_data, indent=2) if input_data else "None")
    if input_data and "extrinsic" in input_data:
        for key in ["guarantees", "assurances", "tickets", "preimages", "disputes"]:
            if key in input_data["extrinsic"]:
                input_data[key] = input_data["extrinsic"][key]

    if not pre_state or not input_data:
        print("❌ updated_state.json missing 'pre_state' or 'input'")
        return
    if "guarantees" not in input_data or not input_data["guarantees"]:
        print("❌ No guarantees found in input_data, skipping processing.")
        return

    print("DEBUG: Loaded pre_state and input from updated_state.json")
    state = initialize_state(pre_state)
    print("DEBUG: Initialized state")

    slot = 0
    try:
        print("DEBUG: Attempting to extract lookup_slot from input_data")
        guarantees = input_data.get('guarantees', [])
        if guarantees and guarantees[0] and guarantees[0].get('report'):
            report = guarantees[0]['report']
            if isinstance(report, dict):
                context = report.get('context', {}) if isinstance(report, dict) else {}
                lookup_slot = context.get('lookup_anchor_slot')
                if lookup_slot is not None:
                    slot = int(lookup_slot) + 65
                    print(f"DEBUG: Set slot to {slot} based on lookup_anchor_slot {lookup_slot}")
    except Exception as e:
        print(f"DEBUG: Exception in slot lookup: {e}")
        import traceback
        traceback.print_exc()

    try:
        extrinsic = map_input_to_extrinsic(input_data)
        print("DEBUG: Mapped input to extrinsic")
        process_guarantee_extrinsic(extrinsic, state, slot)
        print("DEBUG: Processed guarantee extrinsic")
    except Exception as e:
        print(f"❌ Exception during processing: {e}")
        return

    # Prepare post_state
    post_state = state.to_plain_object()
    print("DEBUG: post_state to write:", json.dumps(post_state, indent=2))

    # Write post_state back to updated_state.json
    with open(updated_state_path, 'r+') as f:
        try:
            data = json.load(f)
        except Exception:
            data = {}
        data['post_state'] = post_state
        f.seek(0)
        json.dump(data, f, indent=2)
        f.truncate()
    print(" Reports component wrote post_state to updated_state.json")

if __name__ == '__main__':
    main()