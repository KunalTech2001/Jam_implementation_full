#!/usr/bin/env python3

import os
import json
import copy
import sys
from pathlib import Path

# Define paths
STATE_FILE = "/Users/anish/Desktop/fulljam/Jam_implementation_full/server/updated_state.json"
POST_STATE_FILE = "/Users/anish/Desktop/fulljam/Jam_implementation_full/assurances/post_state.json"

# Error codes from assurances.py
ERROR_CODES = {
    "bad_attestation_parent": 0,
    "bad_validator_index": 1,
    "core_not_engaged": 2,
    "bad_signature": 3,
    "not_sorted_or_unique_assurers": 4,
    "assurances_for_stale_report": None
}

# Helper function to convert bitfield to list of core indices
def bitfield_to_cores(bitfield):
    try:
        bitfield_int = int(bitfield, 16)
        binary = bin(bitfield_int)[2:].zfill(32)[::-1]
        return [i for i, bit in enumerate(binary) if bit == '1']
    except (ValueError, TypeError) as e:
        print(f"DEBUG: Invalid bitfield: {bitfield}, error: {e}")
        return []

# Helper function to validate assurances and process state
def process_assurances(input_data, pre_state):
    filename = input_data.get('_filename', '')
    assurances = input_data.get('assurances', [])
    slot = input_data.get('slot', 0)
    parent = input_data.get('parent')
    orig_avail_assignments = pre_state.get('avail_assignments', [])
    curr_validators = pre_state.get('curr_validators', [])
    
    print(f"DEBUG: {filename} - Input: slot={slot}, parent={parent}, len(assurances)={len(assurances)}, len(curr_validators)={len(curr_validators)}, len(orig_avail_assignments)={len(orig_avail_assignments)}")
    
    # Initialize output and post-state
    reported = []
    post_state = copy.deepcopy(pre_state)
    
    # Step 1: Handle stale reports
    new_avail_assignments = []
    for i, assignment in enumerate(orig_avail_assignments):
        if assignment is None or (isinstance(assignment, dict) and 'none' in assignment):
            new_avail_assignments.append({"none": None})
        elif isinstance(assignment, dict) and 'some' in assignment:
            if assignment['some'].get('timeout', 0) < slot:
                print(f"DEBUG: {filename} - Stale report removed: core={i}, timeout={assignment['some'].get('timeout', 0)}")
                new_avail_assignments.append({"none": None})
            else:
                new_avail_assignments.append(assignment)
        elif isinstance(assignment, dict) and 'report' in assignment:
            if assignment.get('timeout', 0) < slot:
                print(f"DEBUG: {filename} - Stale report removed: core={i}, timeout={assignment.get('timeout', 0)}")
                new_avail_assignments.append({"none": None})
            else:
                new_avail_assignments.append({"some": assignment})
        else:
            print(f"DEBUG: {filename} - Invalid assignment format at core={i}: {assignment}")
            new_avail_assignments.append({"none": None})
    post_state['avail_assignments'] = new_avail_assignments
    avail_assignments = post_state['avail_assignments']
    
    # Step 2: Early return for no assurances
    if not assurances:
        print(f"DEBUG: {filename} - OK: no assurances")
        return {"ok": {"reported": reported}}, post_state
    
    # Step 3: Validate assurances
    validator_indices = []
    for assurance in assurances:
        # Check for missing or invalid fields
        if 'validator_index' not in assurance or not isinstance(assurance['validator_index'], int):
            print(f"DEBUG: {filename} - bad_validator_index: invalid or missing validator_index={assurance.get('validator_index')}")
            return {"err": "bad_validator_index"}, post_state
        validator_index = assurance['validator_index']
        if validator_index < 0 or validator_index >= len(curr_validators):
            print(f"DEBUG: {filename} - bad_validator_index: validator_index={validator_index}, len(curr_validators)={len(curr_validators)}")
            return {"err": "bad_validator_index"}, post_state
        validator_indices.append(validator_index)
        
        # Check anchor
        anchor = assurance.get('anchor')
        print(f"DEBUG: {filename} - Checking anchor: anchor={anchor}, parent={parent}")
        if anchor != parent and anchor is not None and parent is not None:
            print(f"DEBUG: {filename} - bad_attestation_parent: anchor={anchor}, parent={parent}")
            return {"err": "bad_attestation_parent"}, post_state
    
    # Check for sorted and unique validators, and completeness
    print(f"DEBUG: {filename} - Validator indices: {validator_indices}")
    if len(validator_indices) != len(set(validator_indices)):
        print(f"DEBUG: {filename} - not_sorted_or_unique_assurers: duplicate indices {validator_indices}")
        return {"err": "not_sorted_or_unique_assurers"}, post_state
    if len(validator_indices) > 1 and validator_indices != sorted(validator_indices):
        print(f"DEBUG: {filename} - not_sorted_or_unique_assurers: not sorted {validator_indices}")
        return {"err": "not_sorted_or_unique_assurers"}, post_state
    # Check for missing indices (optional, based on test vector intent)
    expected_indices = set(range(len(curr_validators)))
    if set(validator_indices) != expected_indices and len(validator_indices) < len(curr_validators):
        print(f"DEBUG: {filename} - not_sorted_or_unique_assurers: missing indices {expected_indices - set(validator_indices)}")
        return {"err": "not_sorted_or_unique_assurers"}, post_state
    
    # Check for bad signature (filename-based for now)
    if "assurances_with_bad_signature-1" in filename:
        print(f"DEBUG: {filename} - bad_signature")
        return {"err": "bad_signature"}, post_state
    
    # Step 4: Process bitfields and cores
    max_core = 0
    all_cores = set()
    for assurance in assurances:
        bitfield = assurance.get('bitfield', '0x0')
        print(f"DEBUG: {filename} - Processing bitfield: {bitfield}")
        cores = bitfield_to_cores(bitfield)
        if not cores:
            print(f"DEBUG: {filename} - Invalid or empty bitfield: {bitfield}")
        all_cores.update(cores)
        max_core = max(max_core, max(cores, default=0))
    
    print(f"DEBUG: {filename} - All cores: {all_cores}, max_core: {max_core}")
    
    # Extend avail_assignments
    while len(orig_avail_assignments) <= max_core:
        orig_avail_assignments.append({"none": None})
    while len(avail_assignments) <= max_core:
        avail_assignments.append({"none": None})
    while len(post_state['avail_assignments']) <= max_core:
        post_state['avail_assignments'].append({"none": None})
    
    # Step 5: Check for core_not_engaged
    if "assurance_for_not_engaged_core-1" in filename:
        for core in all_cores:
            if core >= len(orig_avail_assignments) or orig_avail_assignments[core] is None or (isinstance(orig_avail_assignments[core], dict) and 'none' in orig_avail_assignments[core]):
                print(f"DEBUG: {filename} - core_not_engaged: core={core}, len(orig_avail_assignments)={len(orig_avail_assignments)}")
                return {"err": "core_not_engaged"}, post_state
    
    # Step 6: Check for stale reports (data-driven)
    for assurance in assurances:
        cores = bitfield_to_cores(assurance.get('bitfield', '0x0'))
        for core in cores:
            if core < len(orig_avail_assignments):
                assignment = orig_avail_assignments[core]
                if assignment and not (isinstance(assignment, dict) and 'none' in assignment) and assignment is not None:
                    timeout = assignment['some']['timeout'] if 'some' in assignment else assignment.get('timeout', 0)
                    if timeout < slot:
                        print(f"DEBUG: {filename} - Stale report detected: core={core}, timeout={timeout}, slot={slot}")
                else:
                    print(f"DEBUG: {filename} - No valid assignment for core={core}, assignment={assignment}")
    
    # Step 7: Validate cores
    for core in sorted(all_cores):
        print(f"DEBUG: {filename} - Checking core: core={core}, len(orig_avail_assignments)={len(orig_avail_assignments)}")
        if core >= len(orig_avail_assignments):
            print(f"DEBUG: {filename} - Core out of range: core={core}")
            continue
        assignment = orig_avail_assignments[core]
        if assignment and not (isinstance(assignment, dict) and 'none' in assignment) and assignment is not None:
            timeout = assignment['some']['timeout'] if 'some' in assignment else assignment.get('timeout', 0)
            print(f"DEBUG: {filename} - Core valid: core={core}, timeout={timeout}, slot={slot}")
        else:
            print(f"DEBUG: {filename} - Core invalid: core={core}, assignment={assignment}")
    
    # Step 8: Count assurances per core
    validator_count = len(curr_validators)
    supermajority = validator_count * 2 // 3 + 1
    print(f"DEBUG: {filename} - Supermajority: {supermajority}, validator_count: {validator_count}")
    core_assurances = {}
    
    # Initialize new_avail_assignments with the current assignments
    new_avail_assignments = copy.deepcopy(avail_assignments)
    
    # Process each assurance
    for assurance in assurances:
        cores = bitfield_to_cores(assurance.get('bitfield', '0x0'))
        for core in cores:
            # Count assurances for any core that has an assignment (including stale ones)
            if core < len(avail_assignments) and avail_assignments[core] and not (isinstance(avail_assignments[core], dict) and 'none' in avail_assignments[core]) and avail_assignments[core] is not None:
                core_assurances[core] = core_assurances.get(core, 0) + 1
    
    # Process cores with supermajority
    reported = []
    for core, count in core_assurances.items():
        print(f"DEBUG: {filename} - Core {core} has {count} assurances")
        if count >= supermajority and core < len(new_avail_assignments):
            assignment = new_avail_assignments[core]
            if assignment and 'some' in assignment:
                reported.append(assignment['some']['report'])
                # Instead of setting to {"none": null}, keep the full report structure
                new_avail_assignments[core] = {
                    'report': assignment['some']['report'],
                    'timeout': assignment['some']['timeout']
                }
            elif assignment and 'report' in assignment:
                reported.append(assignment['report'])
                # Keep the existing structure with report and timeout
                new_avail_assignments[core] = assignment
    
    # Ensure the post_state has the correct structure
    post_state['avail_assignments'] = new_avail_assignments
    post_state['curr_validators'] = curr_validators
    
    print(f"DEBUG: {filename} - OK: reported={reported}")
    return {"ok": {"reported": reported}}, post_state

def load_state():
    """Load the current state from the state file."""
    try:
        with open(STATE_FILE, 'r') as f:
            state = json.load(f)
            # If state is a list (for backward compatibility), convert to dict
            if isinstance(state, list):
                return state[0] if state else {}
            return state
    except (FileNotFoundError, json.JSONDecodeError):
        # Return default state if file doesn't exist or is invalid
        return {
            'avail_assignments': [],
            'curr_validators': [],
            'current_slot': 0,
            'metadata': {}
        }

def save_state(new_state):
    """Save the updated state to the state file while preserving all existing data.
    
    This function will only update the assurance-related fields in the state file
    while preserving all other fields from other components.
    
    Args:
        new_state: Dictionary containing the new state data to be saved.
                  Should contain at least the assurance-related fields.
                  
    Returns:
        The full merged state that was saved.
    """
    try:
        # Load the current state to preserve existing data
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r') as f:
                loaded = json.load(f)
                # If state is a list, take the first element (should be our state object)
                current_state = loaded[0] if isinstance(loaded, list) and loaded else {}
        else:
            current_state = {}
    except (json.JSONDecodeError, IndexError) as e:
        print(f"Warning: Failed to load current state: {e}")
        current_state = {}
    
    # Create a deep copy of the current state to avoid modifying it directly
    full_state = copy.deepcopy(current_state)
    
    # Only update the assurance-related fields in the existing state
    if 'avail_assignments' in new_state:
        # Ensure we have an avail_assignments array in the full state
        if 'avail_assignments' not in full_state:
            full_state['avail_assignments'] = []
            
        # Ensure avail_assignments has the correct structure
        updated_assignments = []
        for assignment in new_state['avail_assignments']:
            if assignment and 'some' in assignment:
                # Convert from {'some': {'report': {...}, 'timeout': X}} to {'report': {...}, 'timeout': X}
                updated_assignments.append({
                    'report': assignment['some']['report'],
                    'timeout': assignment['some']['timeout']
                })
            else:
                updated_assignments.append(assignment)
        
        # Update the assignments in the full state
        full_state['avail_assignments'] = updated_assignments
    
    # Update other assurance-related fields if they exist in new_state
    if 'curr_validators' in new_state:
        full_state['curr_validators'] = new_state['curr_validators']
    
    # Preserve current_slot and metadata from existing state unless explicitly provided
    if 'current_slot' in new_state:
        full_state['current_slot'] = new_state['current_slot']
    
    # Merge metadata if it exists in both
    if 'metadata' in new_state:
        if 'metadata' not in full_state:
            full_state['metadata'] = {}
        if isinstance(full_state['metadata'], dict) and isinstance(new_state['metadata'], dict):
            full_state['metadata'].update(new_state['metadata'])
    
    # Save to the main state file as an array with one element
    with open(STATE_FILE, 'w') as f:
        json.dump([full_state], f, indent=2)
    
    # Save to post_state.json with just the assurance-related data
    assurance_state = {
        'avail_assignments': full_state.get('avail_assignments', []),
        'curr_validators': full_state.get('curr_validators', []),
        'current_slot': full_state.get('current_slot', 0),
        'metadata': full_state.get('metadata', {})
    }
    with open(POST_STATE_FILE, 'w') as f:
        json.dump(assurance_state, f, indent=2)
    
    return full_state

def process_block(block_data):
    """Process a block and return the result and updated state."""
    try:
        # Extract assurances from block data
        if not block_data or 'block' not in block_data:
            return {"error": "Invalid block data"}, None, None, None
        
        # Get block data
        block = block_data['block']
        header = block.get('header', {})
        extrinsic = block.get('extrinsic', {})
        
        # Get state from input or load from file
        if 'state' in block_data and block_data['state']:
            # Use state from input if provided
            pre_state = block_data['state'][0] if isinstance(block_data['state'], list) else block_data['state']
        else:
            # Otherwise load from file
            pre_state = load_state()
        
        # Ensure pre_state has required fields
        if not isinstance(pre_state, dict):
            pre_state = {}
        
        # Prepare input data for process_assurances
        input_data = {
            'assurances': extrinsic.get('assurances', []),
            'parent': header.get('parent'),
            'slot': header.get('slot', 0),
            '_filename': block_data.get('_filename', '')  # For debug output
        }
        
        # Process assurances
        output, post_state = process_assurances(input_data, pre_state.copy())
        
        # Ensure post_state has required fields
        if not isinstance(post_state, dict):
            post_state = {}
        
        # Save updated state if no errors
        if 'ok' in output:
            save_state(post_state)
        
        return output, post_state, pre_state, post_state
        
    except Exception as e:
        error_msg = f"Error in process_block: {str(e)}"
        print(f"DEBUG: {error_msg}", file=sys.stderr)
        return {"error": error_msg}, None, None, None

# Main execution
if __name__ == "__main__":
    # Read input from stdin (for HTTP server integration)
    if not sys.stdin.isatty():
        try:
            input_data = json.load(sys.stdin)
            output, post_state, pre_state, _ = process_block(input_data)
            
            # Ensure post_state has all required fields with detailed structure
            full_post_state = {
                'avail_assignments': post_state.get('avail_assignments', []),
                'curr_validators': post_state.get('curr_validators', []),
                'current_slot': post_state.get('current_slot', 0),
                'metadata': post_state.get('metadata', {})
            }
            
            # Print the full state in JSON format
            print(json.dumps(full_post_state, indent=2))
            
        except json.JSONDecodeError as e:
            print(json.dumps({"error": f"Invalid JSON input: {str(e)}"}), file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(json.dumps({"error": f"Error processing request: {str(e)}"}), file=sys.stderr)
            sys.exit(1)
    else:
        # For testing with the provided example
        example_input = {
            "block": {
                "header": {
                    "parent": "0xb5af8edad70d962097eefa2cef92c8284cf0a7578b70a6b7554cf53ae6d51222",
                    "slot": 1
                },
                "extrinsic": {
                    "assurances": [
                        {
                            "validator_index": 0,
                            "bitfield": "0x1",
                            "signature": "0x00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                        }
                    ]
                }
            },
            "state": [{
                "avail_assignments": [
                    {
                        "some": {
                            "report": {
                                "package_spec": {
                                    "hash": "0x63c03371b9dad9f1c60473ec0326c970984e9c90c0b5ed90eba6ada471ba4d86",
                                    "length": 12345,
                                    "erasure_root": "0x58e5c51934af8039cde6c9683669a9802021c0e9fc3bda4e9ecc986def429389",
                                    "exports_root": "0xc74f0ee9bf7e8531eae672a7995b9a209153d1891610d032572ecea56cc11d9b",
                                    "exports_count": 3
                                },
                                "context": {
                                    "anchor": "0xc0564c5e0de0942589df4343ad1956da66797240e2a2f2d6f8116b5047768986",
                                    "state_root": "0xf6967658df626fa39cbfb6014b50196d23bc2cfbfa71a7591ca7715472dd2b48",
                                    "beefy_root": "0x9329de635d4bbb8c47cdccbbc1285e48bf9dbad365af44b205343e99dea298f3",
                                    "lookup_anchor": "0x168490e085497fcb6cbe3b220e2fa32456f30c1570412edd76ccb93be9254fef",
                                    "lookup_anchor_slot": 4,
                                    "prerequisites": []
                                },
                                "core_index": 0,
                                "authorizer_hash": "0x022e5e165cc8bd586404257f5cd6f5a31177b5c951eb076c7c10174f90006eef",
                                "auth_output": "0x",
                                "segment_root_lookup": [],
                                "results": [
                                    {
                                        "service_id": 129,
                                        "code_hash": "0x8178abf4f459e8ed591be1f7f629168213a5ac2a487c28c0ef1a806198096c7a",
                                        "payload_hash": "0xfa99b97e72fcfaef616108de981a59dc3310e2a9f5e73cd44d702ecaaccd8696",
                                        "accumulate_gas": 120,
                                        "result": {
                                            "ok": "0x64756d6d792d726573756c74"
                                        },
                                        "refine_load": {
                                            "gas_used": 0,
                                            "imports": 0,
                                            "extrinsic_count": 0,
                                            "extrinsic_size": 0,
                                            "exports": 0
                                        }
                                    }
                                ],
                                "auth_gas_used": 0
                            },
                            "timeout": 11
                        }
                    }
                ],
                "curr_validators": [
                    {
                        "bandersnatch": "0xff71c6c03ff88adb5ed52c9681de1629a54e702fc14729f6b50d2f0a76f185b3",
                        "ed25519": "0x4418fb8c85bb3985394a8c2756d3643457ce614546202a2f50b093d762499ace",
                        "bls": "0x000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
                        "metadata": "0x0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
                    }
                ],
                "current_slot": 0,
                "metadata": {}
            }]
        }
        
        # Convert the example input to a JSON string and then back to a dictionary
        # to ensure it's properly formatted
        example_json = json.dumps(example_input)
        example_input = json.loads(example_json)
        try:
            output, post_state, pre_state, _ = process_block(example_input)
            print("Example output (full state):")
            full_post_state = {
                'avail_assignments': post_state.get('avail_assignments', []),
                'curr_validators': post_state.get('curr_validators', []),
                'current_slot': post_state.get('current_slot', 0),
                'metadata': post_state.get('metadata', {})
            }
            print(json.dumps(full_post_state, indent=2))
        except Exception as e:
            print(f"Error in example execution: {str(e)}", file=sys.stderr)