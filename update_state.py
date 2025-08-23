import json
import os
from pathlib import Path

def update_accounts_in_state():
    # Path to the updated_state.json file
    state_file = Path("/Users/anish/Desktop/fulljam/Jam_implementation_full/server/updated_state.json")
    
    # Check if the file exists
    if not state_file.exists():
        print(f"Error: {state_file} does not exist.")
        return False
    
    try:
        # Read the current state
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        # Define the new account information
        new_account = {
            "id": 42,
            "data": {
                "service": {
                    "code_hash": "0x6470fd21983eae8d706f1edd5e2dc5afe095980f8fb7bd4ebfd33550d8730246",
                    "balance": 20219,
                    "min_item_gas": 10,
                    "min_memo_gas": 10,
                    "bytes": 19999,
                    "deposit_offset": 0,
                    "items": 2,
                    "creation_slot": 0,
                    "last_accumulation_slot": 0,
                    "parent_service": 0
                }
            }
        }
        
        # Update the accounts array in the state
        if "accounts" not in state:
            state["accounts"] = []
        
        # Check if account with ID 42 already exists
        account_exists = False
        for i, account in enumerate(state["accounts"]):
            if account.get("id") == 42:
                # Update existing account
                state["accounts"][i] = new_account
                account_exists = True
                break
        
        # If account doesn't exist, add it
        if not account_exists:
            state["accounts"].append(new_account)
        
        # Create a backup of the original file
        backup_file = state_file.with_suffix(".json.bak")
        if not backup_file.exists():
            with open(backup_file, 'w') as f:
                json.dump(state, f, indent=2)
        
        # Write the updated state back to the file
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"Successfully updated {state_file}")
        return True
        
    except Exception as e:
        print(f"Error updating state: {e}")
        return False

if __name__ == "__main__":
    update_accounts_in_state()
