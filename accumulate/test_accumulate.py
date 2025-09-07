import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

# Add parent directory to path to import accumulate_component
sys.path.append(str(Path(__file__).parent.parent))
from accumulate_component import accumulate

def load_input_from_server() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load input payload from server.py and pre_state from updated_state.json
    
    Returns:
        tuple: (pre_state, input_data) - The pre_state and input data for processing
    """
    # Path to server directory
    server_dir = Path(__file__).parent.parent / 'server'
    updated_state_path = server_dir / 'updated_state.json'
    
    # Load pre_state from updated_state.json
    with open(updated_state_path, 'r') as f:
        state_data = json.load(f)
        pre_state = state_data[0] if isinstance(state_data, list) and len(state_data) > 0 else {}
    
    # Load server.py content
    server_path = server_dir / 'server.py'
    with open(server_path, 'r') as f:
        server_content = f.read()
    
    # Try to extract payload from server.py
    payload_match = re.search(r'payload\s*=\s*({.*?})\s*$', server_content, re.DOTALL | re.MULTILINE)
    
    if payload_match:
        try:
            # Try to parse the JSON payload
            input_data = json.loads(payload_match.group(1))
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse payload from server.py: {e}")
            input_data = {"slot": 0, "reports": []}
    else:
        # Fallback to default if no payload found
        print("Warning: No payload found in server.py, using default input")
        input_data = {
            "slot": 0,
            "reports": []
        }
    
    return pre_state, input_data

def save_updated_state(post_state: Dict[str, Any]) -> None:
    """
    Save the updated state back to updated_state.json, preserving existing data
    
    Args:
        post_state: The post-state to save
    """
    server_dir = Path(__file__).parent.parent / 'server'
    updated_state_path = server_dir / 'updated_state.json'
    
    # Load existing state
    with open(updated_state_path, 'r') as f:
        state_data = json.load(f)
    
    # Update the state with post_state, preserving other fields
    if isinstance(state_data, list) and len(state_data) > 0:
        existing_state = state_data[0]
        # Update only the fields that exist in post_state
        for key, value in post_state.items():
            if isinstance(value, dict) and key in existing_state and isinstance(existing_state[key], dict):
                existing_state[key].update(value)
            else:
                existing_state[key] = value
    else:
        state_data = [post_state]
    
    # Save back to file
    with open(updated_state_path, 'w') as f:
        json.dump(state_data, f, indent=2)

def run_accumulate() -> None:
    """Run the accumulate component with input from server and update state"""
    print("Starting Accumulate Component")
    print("=" * 50)
    
    try:
        # Load input and pre_state
        pre_state, input_data = load_input_from_server()
        
        print("\nPre-state:")
        print(json.dumps(pre_state, indent=2))
        
        print("\nInput:")
        print(json.dumps(input_data, indent=2))
        
        # Process the input
        result = accumulate(pre_state, input_data)
        
        print("\nResult:")
        print(json.dumps(result, indent=2))
        
        if isinstance(result, dict) and 'ok' in result and result['ok'] == "0x0000000000000000000000000000000000000000000000000000000000000000":
            # Extract post_state from result if available, otherwise use the input pre_state
            post_state = result.get('post_state', pre_state)
            
            # Save the updated state
            save_updated_state(post_state)
            
            print("\n✅ Successfully processed and updated state")
        else:
            print("\n❌ Processing failed, state not updated")
    
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        raise

def run_test_case(test_name: str) -> Optional[Dict[str, Any]]:
    """
    Run a test case from the tiny directory
    
    Args:
        test_name: Name of the test case (without .json extension)
        
    Returns:
        The result of the accumulate function or None if there was an error
    """
    print(f"\n{'='*50}")
    print(f"Running test case: {test_name}")
    print(f"{'='*50}")
    
    try:
        # Load test case from tiny directory
        test_file = Path(__file__).parent / 'tiny' / f"{test_name}.json"
        with open(test_file, 'r') as f:
            test_data = json.load(f)
        
        pre_state = test_data.get('pre', {})
        input_data = test_data.get('input', {})
        
        print("\nPre-state:")
        print(json.dumps(pre_state, indent=2))
        
        print("\nInput:")
        print(json.dumps(input_data, indent=2))
        
        # Process the input
        result = accumulate(pre_state, input_data)
        
        print("\nResult:")
        print(json.dumps(result, indent=2))
        
        if isinstance(result, dict) and 'ok' in result and result['ok'] == "0x0000000000000000000000000000000000000000000000000000000000000000":
            print("\n✅ Test case completed successfully!")
            return result
        else:
            print("\n❌ Test case failed")
            return None
    
    except Exception as e:
        print(f"\n❌ Error running test case: {str(e)}")
        raise

def main() -> None:
    """Main entry point for the script"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the Accumulate component')
    parser.add_argument('--test', type=str, help='Run a specific test case')
    args = parser.parse_args()
    
    if args.test:
        run_test_case(args.test)
    else:
        run_accumulate()

if __name__ == "__main__":
    main()
