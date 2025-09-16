
"""
Main entry point for the jam-preimages component.

"""
import os
import sys
import json
import argparse
import logging
import traceback
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
        
        # Create a default state structure if it doesn't exist
        default_state = {
            "input": {"preimages": []},
            "pre_state": {"accounts": {}},
            "post_state": {"accounts": []},
            "statistics": {}
        }
        
        # Load the current state or use default
        state_data = {}
        if os.path.exists(updated_state_path):
            try:
                with open(updated_state_path, 'r') as f:
                    state_data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(json.dumps({"warning": f"Error loading state file, using default: {str(e)}"}, indent=2))
                state_data = default_state
        else:
            state_data = default_state
        
        # Ensure all required top-level keys exist
        for key in ["input", "pre_state", "post_state", "statistics"]:
            if key not in state_data:
                state_data[key] = default_state[key]
                
        # Ensure pre_state.accounts is a dict
        if not isinstance(state_data.get("pre_state", {}).get("accounts"), dict):
            state_data["pre_state"]["accounts"] = {}
        
        # Update the state with the new preimages
        if 'preimages' in input_data and isinstance(input_data['preimages'], list):
            # Ensure input exists and is a dict with a preimages list
            if 'input' not in state_data or not isinstance(state_data['input'], dict):
                state_data['input'] = {'preimages': []}
                
            # Replace the preimages list
            state_data['input']['preimages'] = input_data['preimages']
            
            # Ensure pre_state has the same structure as post_state
            if 'post_state' not in state_data or not isinstance(state_data['post_state'], dict):
                state_data['post_state'] = {"accounts": []}
            
            # Save the updated state
            os.makedirs(os.path.dirname(updated_state_path), exist_ok=True)
            with open(updated_state_path, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            return True
            
        return False
    except Exception as e:
        error_msg = f"Failed to process input file: {str(e)}\n{traceback.format_exc()}"
        print(json.dumps({"error": error_msg}, indent=2))
        return False

def main():
    """Main entry point for the jam-preimages component."""
    try:
        # Set up argument parsing
        parser = argparse.ArgumentParser(description='Process preimages and update state.')
        parser.add_argument('--input', type=str, help='Path to input JSON file')
        parser.add_argument('--debug', action='store_true', help='Enable debug logging')
        args = parser.parse_args()
        
        # Set up logging
        log_level = logging.DEBUG if args.debug else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('preimage_processor.log')
            ]
        )
        logger = logging.getLogger(__name__)
        
        # Process input file if provided
        if args.input:
            if not os.path.exists(args.input):
                error_msg = f"Input file not found: {args.input}"
                logger.error(error_msg)
                print(json.dumps({"error": error_msg}, indent=2))
                sys.exit(1)
                
            logger.info(f"Processing input file: {args.input}")
            if not process_input_file(args.input):
                error_msg = f"Failed to process input file: {args.input}"
                logger.error(error_msg)
                print(json.dumps({"error": error_msg}, indent=2))
                sys.exit(1)
        
        # Import and run the main processing function
        logger.info("Starting preimage processing")
        try:
            from process_updated_state import main as process_updated_state
            process_updated_state()
            logger.info("Preimage processing completed successfully")
        except Exception as e:
            error_msg = f"Error in process_updated_state: {str(e)}\n{traceback.format_exc()}"
            logger.error(error_msg)
            print(json.dumps({"error": error_msg}, indent=2))
            sys.exit(1)
        
    except Exception as e:
        error_msg = f"Unexpected error in main: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        print(json.dumps({"error": error_msg}, indent=2))
        sys.exit(1)

if __name__ == "__main__":
    main()
