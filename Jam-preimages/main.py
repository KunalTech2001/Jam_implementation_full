

#!/usr/bin/env python3
"""
Main entry point for Jam-preimages component.

This script processes the updated_state.json file to handle preimage data.
"""
import os
import sys

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Run the preimage processing."""
    try:
        # Import here to avoid circular imports
        from process_updated_state import main as process_updated_state
        process_updated_state()
    except ImportError as e:
        print(f"Error: {e}")
        print("Make sure all dependencies are installed and the project structure is correct.")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
