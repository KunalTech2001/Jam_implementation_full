import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import copy

# Add parent directory to path to import accumulate_component
sys.path.append(str(Path(__file__).parent.parent))
from accumulate_component import process_immediate_report, load_updated_state, save_updated_state, process_immediate_report_from_server

def load_input_from_server() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    
    # Load pre_state from updated_state.json
    pre_state = load_updated_state()
    
    
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

def wait_for_server_input() -> bool:
    """Wait for input from the server"""
    print("\n" + "="*50)
    print("Accumulate Component")
    print("="*50)
    print("Waiting for input from server...")
    
    # In a real implementation, this would wait for an actual server request
    # For now, we'll simulate a small delay
    for _ in range(3):
        print(".", end="", flush=True)
        time.sleep(0.5)
    print("\n" + "-"*50)
    return True

def display_changes(pre_state: Dict[str, Any], post_state: Dict[str, Any]) -> None:
    """Display the changes made during processing"""
    # print(f"\n{'='*20} UPDATES {'='*20}")
    
    # Slot Number
    old_slot = pre_state.get('slot', 0)
    new_slot = post_state.get('slot', 0)
    
   

def run_immediate_report_processing() -> None:
    """
    Main function to run immediate report processing
    """
    # Load pre_state to show before/after
    pre_state = load_updated_state()
    
    # Process the immediate report
    post_state = process_immediate_report_from_server()
    
    if post_state is not None:
        # Display the changes in terminal
        display_changes(pre_state, post_state)
        
        # Get the server's updated_state.json path
        server_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'server'))
        server_state_file = os.path.join(server_dir, 'updated_state.json')
        print(f"Updated Post_state saved to: {server_state_file}")
        
  

def main() -> None:
    """Main entry point for the script"""
    # Wait for server input before proceeding
    if not wait_for_server_input():
        print("No input received from server. Exiting.", file=sys.stderr)
        sys.exit(1)
        
    run_immediate_report_processing()

if __name__ == "__main__":
    main()
