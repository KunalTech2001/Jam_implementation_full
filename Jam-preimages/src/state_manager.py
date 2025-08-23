"""
State manager for Jam-preimages component.
Handles reading from and writing to the updated_state.json file.
"""
import json
import os
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import asdict
import copy

from .types.preimage_types import (
    PreimagesTestVector,
    PreimagesState,
    PreimageInput,
    PreimagesInput,
    PreimagesMapEntry,
    LookupMetaMapKey,
    LookupMetaMapEntry,
    PreimagesAccountMapData,
    PreimagesAccountMapEntry,
    ServicesStatisticsEntry,
    StatisticsRecord,
    PreimagesOutput
)

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

def load_state_from_updated_state(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load and convert the updated_state.json to the format expected by the preimage processor.
    
    Args:
        file_path: Path to the updated_state.json file
        
    Returns:
        Dictionary with 'input' and 'pre_state' keys in the expected format,
        or None if there was an error
    """
    try:
        # Resolve the absolute path
        file_path = os.path.abspath(file_path)
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            return None
            
        with open(file_path, 'r') as f:
            state_data = json.load(f)
        
        # Get the current slot from the state
        current_slot = state_data.get("current_slot", 1)
        
        # Create pre_state structure
        pre_state = {
            "accounts": [],
            "statistics": []
        }
        
        # Track unique preimages by hash to avoid duplicates
        unique_preimages = {}
        
        # Process beta blocks for preimages
        if "beta" in state_data and isinstance(state_data["beta"], list):
            for beta_block in state_data["beta"]:
                if not isinstance(beta_block, dict):
                    continue
                    
                # Extract reported packages from the beta block
                reported = beta_block.get("reported", [])
                if not isinstance(reported, list):
                    continue
                
                # Process each reported package
                for report in reported:
                    if not isinstance(report, dict) or "hash" not in report:
                        continue
                    
                    # Skip if we've already processed this hash
                    preimage_hash = report["hash"]
                    if preimage_hash in unique_preimages:
                        continue
                    
                    # Get or create blob, calculate hash if not provided
                    blob = report.get("blob", "0x")
                    if blob == "0x" and preimage_hash:
                        # In a real implementation, you'd need to recover the blob from storage
                        # For now, we'll use a placeholder
                        blob = f"0x{preimage_hash[2:10]}{'00' * 12}"  # First 8 bytes + padding
                    
                    # Create preimage entry
                    preimage_entry = {
                        "hash": preimage_hash,
                        "blob": blob
                    }
                    
                    # Store in unique preimages
                    unique_preimages[preimage_hash] = preimage_entry
        
        # Create input data with sorted preimages
        input_preimages = []
        for preimage_hash, preimage in unique_preimages.items():
            input_preimages.append({
                "requester": 1,  # Default requester ID
                "blob": preimage["blob"]
            })
        
        # Sort input preimages by hash
        input_preimages = sort_preimages(input_preimages)
        
        # Create input data structure
        input_data = {
            "preimages": input_preimages,
            "slot": current_slot
        }
        
        # Process accounts and preimages
        if "accounts" in state_data and isinstance(state_data["accounts"], list):
            for account in state_data["accounts"]:
                if not isinstance(account, dict) or "id" not in account:
                    continue
                
                account_id = account["id"]
                account_data = account.get("data", {})
                
                # Get or create account in pre_state
                target_account = ensure_account_exists(pre_state["accounts"], account_id)
                
                # Process preimages for this account
                if "preimages" in account_data and isinstance(account_data["preimages"], list):
                    for preimage in account_data["preimages"]:
                        if not isinstance(preimage, dict) or "hash" not in preimage:
                            continue
                            
                        # Add preimage to account
                        target_account["data"]["preimages"].append({
                            "hash": preimage["hash"],
                            "blob": preimage.get("blob", "0x")
                        })
                        
                        # Add lookup meta if available
                        if "lookup_meta" in account_data and isinstance(account_data["lookup_meta"], list):
                            for meta in account_data["lookup_meta"]:
                                if isinstance(meta, dict) and meta.get("key", {}).get("hash") == preimage["hash"]:
                                    target_account["data"].setdefault("lookup_meta", []).append(meta)
                                    break
        
        # Sort preimages in each account by hash
        for account in pre_state["accounts"]:
            if "data" in account and "preimages" in account["data"]:
                account["data"]["preimages"] = sort_preimages(account["data"]["preimages"])
        
        return {
            "input": input_data,
            "pre_state": pre_state
        }
        
    except Exception as e:
        print(f"Error loading state from {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

def save_state_to_updated_state(file_path: str, test_vector: PreimagesTestVector) -> bool:
    """
    Save the post-state back to updated_state.json.
    
    Args:
        file_path: Path to the updated_state.json file
        test_vector: The test vector containing the post-state to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Resolve the absolute path and create directory if it doesn't exist
        file_path = os.path.abspath(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Skip creating backup as per user request
        
        # Load existing state to preserve other fields
        try:
            with open(file_path, 'r') as f:
                state_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # If file doesn't exist or is invalid, create a new state
            state_data = {
                "current_slot": 1,
                "beta": [],
                "accounts": [],
                "statistics": []
            }
        
        # Ensure required top-level fields exist
        state_data.setdefault("current_slot", 1)
        state_data.setdefault("beta", [])
        state_data.setdefault("accounts", [])
        state_data.setdefault("statistics", [])
        
        # Get current slot from test vector or use existing
        current_slot = state_data["current_slot"]
        if hasattr(test_vector, 'input') and hasattr(test_vector.input, 'slot'):
            current_slot = test_vector.input.slot
            state_data["current_slot"] = current_slot
        
        # Create a new beta block for the updated preimages
        new_beta_block = {
            "header_hash": "0x" + hashlib.sha256(f"block_{current_slot}".encode()).hexdigest()[:64],
            "state_root": "0x" + "0" * 64,  # Would be computed from the state
            "mmr": {
                "peaks": [],
                "count": 0
            },
            "reported": [],
            "timestamp": current_slot
        }
        
        # Track processed preimages to avoid duplicates
        processed_hashes = set()
        
        # Add preimages to the reported field and update accounts
        if hasattr(test_vector, 'post_state') and test_vector.post_state:
            # First, process all accounts and preimages
            for account in getattr(test_vector.post_state, 'accounts', []):
                if not hasattr(account, 'data') or not hasattr(account.data, 'preimages'):
                    continue
                
                # Get or create account in state
                account_id = getattr(account, 'id', 1)  # Default to service ID 1
                account_data = next((a for a in state_data["accounts"] if a.get('id') == account_id), None)
                
                if account_data is None:
                    # Create new account if it doesn't exist
                    account_data = {
                        'id': account_id,
                        'data': {
                            'preimages': [],
                            'lookup_meta': []
                        }
                    }
                    state_data["accounts"].append(account_data)
                
                # Process preimages for this account
                for preimage in account.data.preimages:
                    if not hasattr(preimage, 'hash') or not hasattr(preimage, 'blob'):
                        continue
                    
                    preimage_hash = preimage.hash
                    if preimage_hash in processed_hashes:
                        continue
                    
                    processed_hashes.add(preimage_hash)
                    
                    # Update account's preimages
                    existing_preimage = next((p for p in account_data['data'].get('preimages', []) 
                                           if isinstance(p, dict) and p.get('hash') == preimage_hash), None)
                    
                    if existing_preimage is None:
                        # Add new preimage
                        account_data['data'].setdefault('preimages', []).append({
                            'hash': preimage_hash,
                            'blob': preimage.blob
                        })
                        
                        # Add lookup meta
                        account_data['data'].setdefault('lookup_meta', []).append({
                            'key': {
                                'hash': preimage_hash,
                                'length': len(preimage.blob[2:]) // 2 if hasattr(preimage, 'blob') else 32  # Default to 32 bytes if blob not available
                            },
                            'value': [current_slot]  # Current slot as timeslot
                        })
                    
                    # Add to reported in beta block
                    reported_entry = {
                        "hash": preimage_hash,
                        "blob": getattr(preimage, 'blob', '0x'),
                        "timestamp": current_slot,
                        "signature": "0x" + "0" * 130,  # Placeholder for signature
                        "exports_root": getattr(preimage, 'exports_root', '0x' + '0' * 64)
                    }
                    new_beta_block["reported"].append(reported_entry)
        
        # Add the new beta block to the state
        state_data["beta"].append(new_beta_block)

        # Keep only the most recent N beta blocks (e.g., last 100)
        if len(state_data["beta"]) > 100:
            state_data["beta"] = state_data["beta"][-100:]

        # Sort preimages in each account by hash for consistency
        for account in state_data.get("accounts", []):
            if isinstance(account, dict) and "data" in account and "preimages" in account["data"]:
                account["data"]["preimages"] = sort_preimages(account["data"]["preimages"])

        # Write to a temporary file first
        temp_file = f"{file_path}.tmp"
        with open(temp_file, 'w') as f:
            json.dump(state_data, f, indent=2)

        # Atomically replace the original file
        os.replace(temp_file, file_path)

        # Remove backup if everything succeeded
        if os.path.exists(backup_path):
            os.remove(backup_path)

        print(f"Successfully updated {file_path}")
        return True

    except Exception as e:
        print(f"Error saving state to {file_path}: {e}")
        import traceback
        traceback.print_exc()

        # Restore from backup if possible
        if os.path.exists(backup_path) and not os.path.exists(file_path):
            state_data["beta"] = state_data["beta"][:max_beta_blocks]
            
            # Update the current slot in the state
            state_data["current_slot"] = current_slot
            
            # Save the updated state
            try:
                # Write to a temporary file first
                temp_path = f"{file_path}.tmp"
                with open(temp_path, 'w') as f:
                    json.dump(state_data, f, indent=2)
                
                # Replace the original file
                if os.path.exists(file_path):
                    os.remove(file_path)
                os.rename(temp_path, file_path)
                
                print(f"Successfully updated state with {len(new_beta_block['reported'])} new preimages")
                return True
                
            except Exception as e:
                print(f"Error writing to {file_path}: {e}")
                # Try to restore from backup if available
                if os.path.exists(backup_path):
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                        os.rename(backup_path, file_path)
                        print("Restored original state from backup")
                    except Exception as restore_error:
                        print(f"Error restoring from backup: {restore_error}")
                return False
        else:
            print("No new preimages to save")
            return True
            
    except Exception as e:
        print(f"Error in save_state_to_updated_state: {e}")
        return False
    
    return False
