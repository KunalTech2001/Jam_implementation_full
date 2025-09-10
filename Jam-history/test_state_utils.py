import sys
import os
from pathlib import Path

# Add the parent directory to the path so we can import from Jam-history
sys.path.append(str(Path(__file__).parent.parent))

from Jam_history.state_utils import load_updated_state, save_updated_state

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    state_file = os.path.join(BASE_DIR, "..", "server", "updated_state.json")
    updated_state_path = os.path.normpath(state_file)
    # Path to the updated_state.json file
    # updated_state_path = Path('/Users/anish/Desktop/fulljam/Jam_implementation_full/server/updated_state.json')
    
    # Test loading the state
    print("Loading state from:", updated_state_path)
    state = load_updated_state(updated_state_path)
    
    if state is None:
        print("Failed to load state. The file might not exist or is malformed.")
        return
    
    print("\nSuccessfully loaded state:")
    print(f"Number of beta blocks: {len(state.beta)}")
    
    if state.beta:
        print("\nFirst beta block:")
        block = state.beta[0]
        print(f"  Header hash: {block.header_hash}")
        print(f"  State root: {block.state_root}")
        print(f"  MMR peaks: {len(block.mmr.peaks)} peaks")
        print(f"  Reported items: {len(block.reported)}")
    
    # Test saving the state
    print("\nSaving state back to file...")
    if save_updated_state(updated_state_path, state):
        print("Successfully saved state back to file.")
    else:
        print("Failed to save state back to file.")

if __name__ == "__main__":
    main()
