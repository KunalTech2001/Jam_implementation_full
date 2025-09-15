#!/usr/bin/env python3
"""
Main entry point for the jam-preimages component.

This script processes preimages and updates the state accordingly.
It can be run directly or imported as a module.
"""
import os
import sys
import json
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def process_input_file(input_path):
    """Process the input file and update the state with its contents."""
    try:
        with open(input_path, 'r') as f:
            input_data = json.load(f)
        
        # Get the path to the state file
        state_file = os.path.join(
            os.path.dirname(__file__), 
            "..", 
            "server", 
            "updated_state.json"
        )
        updated_state_path = os.path.normpath(state_file)
        
        # Load the current state
        from process_updated_state import load_state_from_updated_state
        state_data = load_state_from_updated_state(updated_state_path) or {}
        
        # Update the state with the new preimages
        if 'preimages' in input_data and input_data['preimages']:
            state_data['input'] = {
                'preimages': input_data['preimages']
            }
            
            # Save the updated state
            with open(updated_state_path, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            return True
        return False
    except Exception as e:
        print(json.dumps({"error": f"Failed to process input file: {str(e)}"}, indent=2))
        return False

def main():
    """Main entry point for the jam-preimages component."""
    try:
        # Set up argument parsing
        parser = argparse.ArgumentParser(description='Process preimages and update state.')
        parser.add_argument('--input', type=str, help='Path to input JSON file')
        args = parser.parse_args()
        
        # Process input file if provided
        if args.input:
            if not os.path.exists(args.input):
                print(json.dumps({"error": f"Input file not found: {args.input}"}, indent=2))
                sys.exit(1)
            process_input_file(args.input)
        
        # Import and run the main processing function
        from process_updated_state import main as process_updated_state
        process_updated_state()
        
    except ImportError as e:
        print(json.dumps({"error": f"Import error: {str(e)}"}, indent=2))
        sys.exit(1)
    except Exception as e:
        print(json.dumps({"error": f"Unexpected error: {str(e)}"}, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()
