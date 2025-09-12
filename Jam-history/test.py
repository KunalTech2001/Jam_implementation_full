import os
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

from normalize import normalize
from history_stf import HistorySTF
from jam_types import Input, State, BetaBlock, MMR, Reported
from state_utils import load_updated_state, save_updated_state, extract_input_from_payload


def create_input_from_dict(data: Dict[str, Any]) -> Input:
    # Safely extract work packages with validation
    work_packages = []
    for wp in data.get('work_packages', []):
        try:
            work_packages.append(Reported(
                hash=wp['hash'],
                exports_root=wp.get('exports_root', '0x')
            ))
        except (KeyError, TypeError) as e:
            print(f"⚠️ Warning: Invalid work package format: {e}")
            continue
    
    # Get required fields with validation
    header_hash = data.get('header_hash')
    if not header_hash:
        raise ValueError("header_hash is required in input data")
        
    parent_state_root = data.get('parent_state_root')
    if not parent_state_root:
        raise ValueError("parent_state_root is required in input data")
        
    # Provide a default for accumulate_root if not present
    accumulate_root = data.get('accumulate_root', '0x' + '0' * 64)
    
    return Input(
        header_hash=header_hash,
        parent_state_root=parent_state_root,
        accumulate_root=accumulate_root,
        work_packages=work_packages
    )


def create_state_from_dict(data: Dict[str, Any]) -> State:
    """
    Create a State object from a dictionary.
    
    Args:
        data: Dictionary containing state data with a 'beta' key
        
    Returns:
        State: A State object with the parsed beta blocks
    """
    # Handle case where data is already a State object
    if hasattr(data, 'beta') and isinstance(data.beta, list):
        return data
        
    # Handle case where data is a dict with 'beta' key
    if isinstance(data, dict):
        beta_data = data.get('pre_state', {}).get('beta', []) if 'pre_state' in data else data.get('beta', [])
        if not isinstance(beta_data, list):
            beta_data = [beta_data] if beta_data else []
            
        beta_blocks = []
        for block_data in beta_data:
            if not isinstance(block_data, dict):
                continue
                
            try:
                mmr_data = block_data.get('mmr', {})
                if not isinstance(mmr_data, dict):
                    mmr_data = {}
                    
                mmr = MMR(
                    peaks=mmr_data.get('peaks', []),
                    count=mmr_data.get('count', 0)
                )
                
                reported_data = block_data.get('reported', [])
                if not isinstance(reported_data, list):
                    reported_data = [reported_data] if reported_data else []
                    
                reported = []
                for r in reported_data:
                    if not isinstance(r, dict):
                        continue
                    try:
                        reported.append(Reported(
                            hash=r.get('hash', ''),
                            exports_root=r.get('exports_root', '')
                        ))
                    except Exception as e:
                        print(f"⚠️ Warning: Invalid reported item format: {e}")
                        continue
                
                beta_block = BetaBlock(
                    header_hash=block_data.get('header_hash', ''),
                    state_root=block_data.get('state_root', ''),
                    mmr=mmr,
                    reported=reported,
                    timestamp=block_data.get('timestamp', 0)
                )
                beta_blocks.append(beta_block)
                
            except Exception as e:
                print(f"⚠️ Warning: Error processing beta block: {e}")
                continue
        
        return State(beta=beta_blocks)
    
    # Return empty state if data format is not recognized
    return State(beta=[])


def state_to_dict(state: State) -> Dict[str, Any]:
  
    beta_list = []
    for block in state.beta:
        mmr_dict = {
            'peaks': block.mmr.peaks
        }
        if block.mmr.count is not None:
            mmr_dict['count'] = block.mmr.count
            
        reported_list = [
            {'hash': r.hash, 'exports_root': r.exports_root}
            for r in block.reported
        ]
        
        block_dict = {
            'header_hash': block.header_hash,
            'state_root': block.state_root,
            'mmr': mmr_dict,
            'reported': reported_list
        }
        beta_list.append(block_dict)
    
    return {'beta': beta_list}


def green(msg: str) -> None:
    
    print(f'\033[32m✓ {msg}\033[0m')


def red(msg: str) -> None:
   
    print(f'\033[31m✗ {msg}\033[0m')


def parse_curl_payload() -> Optional[Dict[str, Any]]:
    """Parse the curl payload from command line arguments or stdin."""
    try:
        payload_str = None
        
        # Check if payload is passed as a command line argument
        if len(sys.argv) > 1 and sys.argv[1] == '--payload':
            payload_str = sys.argv[2]
        # Otherwise, try to read from stdin
        elif not sys.stdin.isatty():
            payload_str = sys.stdin.read().strip()
        
        if not payload_str:
            return None
            
        # Parse the JSON payload
        payload = json.loads(payload_str)
        print(f"ℹ️  Raw payload received: {json.dumps(payload, indent=2)}")
        return payload
        
    except json.JSONDecodeError as e:
        print(f"❌ Failed to parse JSON payload: {e}")
        print(f"Payload content: {payload_str}" if 'payload_str' in locals() else "No payload content available")
        return None
    except Exception as e:
        print(f"❌ Error reading payload: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    script_dir = Path(__file__).parent
    results_dir = script_dir / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # Try to get payload from curl command
    payload = parse_curl_payload()
    
    # Path to the updated_state.json file
    updated_state_path = (script_dir / ".." / "server" / "updated_state.json").resolve()
    
    if not updated_state_path.exists():
        print(f"❌ State file not found: {updated_state_path}")
        print("The jam_history component requires the state file to run.")
        return
        
    print("\n ✅ ✅ ✅  Jam_History Component is running successfully... ✅ ✅ ✅ ")
    
    # Load current state from updated_state.json
    state_data = load_updated_state(updated_state_path)
    if not state_data:
        print("❌ Failed to load state data from updated_state.json")
        return
        
    # Extract pre_state from the loaded data
    if 'pre_state' in state_data:
        pre_state = state_data['pre_state']
    else:
        # If no pre_state in the loaded data, use the entire state_data
        pre_state = state_data
    
    # Get input data from payload - it's required now
    if not payload:
        print("❌ Error: No payload provided.")
        return
        
    # Debug: Print the full payload structure for inspection
    print(f"ℹ️  Payload structure: {json.dumps(payload, indent=2)}")
    
    # The payload comes directly with the required fields, no need for 'block' or 'header' nesting
    header_hash = payload.get('header_hash')
    parent_state_root = payload.get('parent_state_root')
    accumulate_root = payload.get('accumulate_root', "0x" + "0" * 64)
    work_packages = payload.get('work_packages', [])
    
    # Validate required fields
    if not header_hash:
        print("❌ Error: 'header_hash' is required in payload")
        return
        
    if not parent_state_root:
        print("❌ Error: 'parent_state_root' is required in payload")
        return
        
    print(f"ℹ️  Using header_hash: {header_hash}")
    print(f"ℹ️  Using parent_state_root: {parent_state_root}")
    print(f"ℹ️  Using accumulate_root: {accumulate_root}")
    print(f"ℹ️  Found {len(work_packages)} work packages")
    
    input_data = {
        'header_hash': header_hash,
        'parent_state_root': parent_state_root,
        'accumulate_root': accumulate_root,
        'work_packages': work_packages
    }
    
    print(f"ℹ️  Extracted input data: {json.dumps(input_data, indent=2)}")
    
    # Create current state from pre_state
    current_state = create_state_from_dict(pre_state)
    
    # Print the latest beta block for debugging
    if current_state.beta:
        latest_beta = current_state.beta[-1]
        print(f"ℹ️  Latest beta block: {json.dumps({
            'header_hash': latest_beta.header_hash,
            'state_root': latest_beta.state_root,
            'mmr': {'peaks': latest_beta.mmr.peaks, 'count': latest_beta.mmr.count},
            'reported': [{'hash': r.hash, 'exports_root': r.exports_root} for r in latest_beta.reported],
            'timestamp': getattr(latest_beta, 'timestamp', 0)
        }, indent=2)}")
    
    # Create input object for state transition
    try:
        input_obj = create_input_from_dict(input_data)
        print(f"✅ Created input object: {input_obj}")
        
        # Perform state transition
        stf = HistorySTF()
        transition_result = stf.transition(current_state, input_obj)
        
        # Extract post_state from the transition result
        if not isinstance(transition_result, dict) or 'postState' not in transition_result:
            raise ValueError("Invalid transition result format. Expected dict with 'postState' key")
            
        post_state = transition_result['postState']
        
        # Convert post_state to dict for JSON serialization
        post_state_dict = state_to_dict(post_state)
        
        # Save the post_state to the server's updated_state.json
        server_state_path = script_dir.parent / 'server' / 'updated_state.json'
        
        # Load the existing state to preserve other fields
        try:
            with open(server_state_path, 'r') as f:
                server_state = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            server_state = {}
        
        # Update the beta blocks in the server state
        server_state['beta'] = post_state_dict['beta']
        
        # Save the updated state back to server's updated_state.json
        with open(server_state_path, 'w') as f:
            json.dump(server_state, f, indent=2)
        
        print(f"✅ State transition successful. Updated state saved to {server_state_path}")
        
        # Also save to results for reference
        output_path = results_dir / 'latest_result.json'
        with open(output_path, 'w') as f:
            json.dump(post_state_dict, f, indent=2)
        print(f"✅ Results also saved to {output_path} for reference")
        
        # Print only the post_state as requested
        print("\n=== POST_STATE ===")
        print(json.dumps(post_state_dict, indent=2))
        print("=== END POST_STATE ===\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during state transition: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    payload = parse_curl_payload()
    
    # Load the pre_state from updated_state.json
    updated_state_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'server', 'updated_state.json')
    try:
        with open(updated_state_path, 'r') as f:
            state_dict = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"ℹ️  {updated_state_path} not found or invalid, starting with empty state")
        state_dict = {'beta': []}
    
    # Get the latest beta block from state
    latest_beta = state_dict.get('beta', [{}])[-1] if state_dict.get('beta') else {}
    
    # Extract input data and work packages from payload
    input_data, work_packages = extract_input_from_payload(payload)
    
    # Debug output
    print("ℹ️  Latest beta block:", json.dumps(latest_beta, indent=2))
    print("ℹ️  Extracted work packages:", json.dumps(work_packages, indent=2))
    
    # If header_hash is not in the payload, try to generate it from the block header
    if not input_data.get('header_hash') and 'block' in payload and 'header' in payload['block']:
        # Create a hash of the header as a fallback
        import hashlib
        header_str = json.dumps(payload['block']['header'], sort_keys=True).encode()
        input_data['header_hash'] = '0x' + hashlib.sha256(header_str).hexdigest()
        print(f"ℹ️  Generated header_hash: {input_data['header_hash']}")
    
    # Validate required fields
    required_fields = ['header_hash', 'parent_state_root']
    missing_fields = [field for field in required_fields if not input_data.get(field)]
    if missing_fields:
        print(f"❌ Error: Missing required fields in payload: {', '.join(missing_fields)}")
        print("Payload structure:", json.dumps(payload, indent=2))
        return
    
    print(f"ℹ️  Extracted input data: {json.dumps(input_data, indent=2)}")
    
    # Ensure accumulate_root has a default value if not provided
    if 'accumulate_root' not in input_data or not input_data['accumulate_root']:
        input_data['accumulate_root'] = '0x' + '0' * 64
            
    print("ℹ️  Using input data from curl payload")
    
    try:
        print("🔄 Jam_History is processing input...")
        print(f"Input data: {json.dumps(input_data, indent=2)}")
        
        # Create Input object from input_data
        input_obj = create_input_from_dict(input_data)
        
        # Get current state
        current_state = create_state_from_dict(state_dict)
        
        # Perform state transition
        result = HistorySTF.transition(current_state, input_obj)
        
        # Create the new beta block
        new_beta_block = {
            'header_hash': input_data['header_hash'],
            'mmr': {
                'peaks': [latest_beta.get('mmr', {}).get('peaks', [])[-1]] if latest_beta and latest_beta.get('mmr', {}).get('peaks') else ['0x' + '0' * 64],
                'count': (latest_beta.get('mmr', {}).get('count', 0) + 1) if latest_beta else 1
            },
            'state_root': '0x' + '0' * 64,  # This should be calculated properly in a real implementation
            'reported': work_packages,  # Add the work packages to reported
            'timestamp': int(time.time())
        }
        
        # Prepare the complete post state with beta blocks
        complete_post_state = {
            **state_dict,  # Keep all existing state
            'beta': state_dict.get('beta', []) + [new_beta_block]
        }
        
        # Enhanced state serialization to handle various object types
        def state_to_serializable(obj):
            if obj is None:
                return None
            # Handle State objects
            if hasattr(obj, '__dict__'):
                obj = obj.__dict__
            # Handle dictionaries
            if isinstance(obj, dict):
                return {k: state_to_serializable(v) for k, v in obj.items()}
            # Handle lists and tuples
            if isinstance(obj, (list, tuple, set)):
                return [state_to_serializable(item) for item in obj]
            # Handle common types that are JSON serializable
            if isinstance(obj, (str, int, float, bool)):
                return obj
            # Convert any remaining objects to string representation
            return str(obj)
        
        # Convert complete_post_state to serializable format
        serializable_post_state = state_to_serializable(complete_post_state)
        
        # Save the updated state to output_state.json
        output_path = os.path.join(os.path.dirname(updated_state_path), 'output_state.json')
        with open(output_path, 'w') as f:
            json.dump(serializable_post_state, f, indent=2)
        print(f"💾 Updated state saved to {output_path}")
        
        # Also update the main updated_state.json with the new beta block
        with open(updated_state_path, 'w') as f:
            json.dump(serializable_post_state, f, indent=2)
        print(f"💾 Updated state saved to {updated_state_path}")
        
        # Save result to results directory
        result_path = results_dir / 'latest_result.json'
        result_data = {
            'input': input_data,
            'pre_state': state_to_serializable(current_state),
            'post_state': serializable_post_state,
            'new_beta_block': new_beta_block,
            'timestamp': int(time.time())
        }
        with open(result_path, 'w') as f:
            json.dump(result_data, f, indent=2)
            
        # Only print the post_state to terminal
        print("\n" + "="*50)
        print("POST_STATE OUTPUT:")
        print("="*50)
        print(json.dumps({'post_state': serializable_post_state.get('beta', [])}, indent=2))
        print("="*50 + "\n")
        
    except Exception as e:
        print(f"❌ Error during state transition: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
