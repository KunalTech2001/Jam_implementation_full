"""
State manager for Jam-preimages component.
Handles reading from and writing to the updated_state.json file.
"""
import json
import os
import hashlib
from typing import Dict, Any, List, Optional

def calculate_blake2b_hash(blob: str) -> str:
    """Calculate Blake2b-256 hash of the blob."""
    if blob.startswith('0x'):
        blob = blob[2:]
    return '0x' + hashlib.blake2b(bytes.fromhex(blob), digest_size=32).hexdigest()

def sort_preimages(preimages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort preimages by their hash in ascending order."""
    return sorted(preimages, key=lambda x: x.get('hash', '').lower())

def ensure_account_exists(accounts: List[Dict], account_id: int) -> Dict:
    """Ensure an account exists in the accounts list, create if it doesn't."""
    for account in accounts:
        if account.get('id') == account_id:
            return account
    
    # Create new account if not found
    new_account = {
        'id': account_id,
        'data': {
            'preimages': [],
            'lookup_meta': []
        }
    }
    accounts.append(new_account)
    return new_account

def load_state_from_updated_state(file_path: str) -> Dict[str, Any]:
    """
    Load the updated_state.json file.
    
    Args:
        file_path: Path to the updated_state.json file
        
    Returns:
        Dictionary with the state data, or empty dict if file doesn't exist
    """
    try:
        # Resolve the absolute path
        file_path = os.path.abspath(file_path)
        if not os.path.exists(file_path):
            return {}
            
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading state file: {e}")
        return {}

def process_preimages(preimages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process preimages and generate the post_state.
    
    Args:
        preimages: List of preimage objects with 'requester' and 'blob' fields
        
    Returns:
        Dictionary with the post_state structure
    """
    post_state = {
        "accounts": [],
        "statistics": []
    }
    
    # Group preimages by requester
    preimages_by_requester = {}
    for preimage in preimages:
        requester = preimage.get('requester')
        if requester is not None:
            if requester not in preimages_by_requester:
                preimages_by_requester[requester] = []
            preimages_by_requester[requester].append(preimage)
    
    # Process each requester's preimages
    for requester, req_preimages in preimages_by_requester.items():
        account_data = {
            "id": requester,
            "data": {
                "preimages": [],
                "lookup_meta": []
            }
        }
        
        for i, preimage in enumerate(req_preimages):
            blob = preimage.get('blob', '')
            if not blob.startswith('0x'):
                blob = '0x' + blob
                
            # Generate deterministic hash for the blob
            hash_obj = hashlib.sha256(bytes.fromhex(blob[2:]))
            hash_hex = '0x' + hash_obj.hexdigest()
            
            # Add to preimages
            account_data["data"]["preimages"].append({
                "hash": hash_hex,
                "blob": blob
            })
            
            # Add to lookup_meta
            account_data["data"]["lookup_meta"].append({
                "key": {
                    "hash": hash_hex,
                    "length": len(blob[2:]) // 2  # Hex string length to bytes
                },
                "value": [i * 10, (i + 1) * 10]  # Example positions
            })
        
        post_state["accounts"].append(account_data)
    
    return post_state

def save_state_to_updated_state(file_path: str, state_data: Dict[str, Any]) -> bool:
    """
    Save the state data to updated_state.json.
    
    Args:
        file_path: Path to the updated_state.json file
        state_data: The state data to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Write the state data to the file
        with open(file_path, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error saving state file: {e}")
        return False
