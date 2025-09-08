import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import copy

# Add parent directory to path to import accumulate_component
sys.path.append(str(Path(__file__).parent.parent))
from accumulate_component import process_immediate_report, load_updated_state, save_updated_state, process_immediate_report_from_server

def load_input_from_server() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load input payload from server.py and pre_state from updated_state.json
    
    Returns:
        tuple: (pre_state, input_data) - The pre_state and input data for processing
    """
    # Load pre_state from updated_state.json
    pre_state = load_updated_state()
    
    # In a real implementation, this would come from the server's HTTP request
    # For now, we'll use a default payload with the current slot + 1
    current_slot = pre_state.get("slot", 0) + 1 if pre_state else 1
    
    input_data = {
        "slot": current_slot,
        "reports": [
            {
                "core_index": 0,  # Default core index
                "prerequisites": [],
                # Add other report fields as needed
            }
        ]
    }
    
    return pre_state, input_data

def run_immediate_report_processing() -> None:
    """
    Main function to run immediate report processing
    """
    # Process the immediate report
    post_state = process_immediate_report_from_server()
    
    if post_state is not None:
        # Print only the post_state as JSON output
        print(json.dumps(post_state, indent=2))
    else:
        print("Error: Failed to process immediate report", file=sys.stderr)
        sys.exit(1)

def main() -> None:
    """Main entry point for the script"""
    run_immediate_report_processing()

if __name__ == "__main__":
    main()
