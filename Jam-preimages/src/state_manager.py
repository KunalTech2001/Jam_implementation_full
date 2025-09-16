"""
State manager for Jam-preimages component.
Handles reading from and writing to the updated_state.json file.
"""
import json
import os
import hashlib
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('preimage_processor.log')
    ]
)
logger = logging.getLogger(__name__)

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
    Process a list of preimages and return the updated state.
    
    Args:
        preimages: List of preimage dictionaries, each containing 'requester' and 'blob' fields.
        
    Returns:
        Dict containing the updated state with accounts and their preimages.
    """
    # Initialize default post state
    post_state = {"accounts": [], "statistics": []}
    logger = logging.getLogger(__name__)
    
    # Validate input
    if not isinstance(preimages, list):
        logger.warning(f"Expected preimages to be a list, got {type(preimages)}")
        return post_state
    
    logger.info(f"Processing {len(preimages)} preimages")
    
    # Group preimages by requester
    preimages_by_requester = {}
    for idx, preimage in enumerate(preimages):
        try:
            if not isinstance(preimage, dict):
                logger.warning(f"Skipping invalid preimage at index {idx}: not a dictionary")
                continue
                
            requester = preimage.get('requester')
            if requester is None:
                logger.warning(f"Skipping preimage at index {idx}: missing 'requester' field")
                continue
                
            if not isinstance(requester, (int, str)):
                logger.warning(f"Skipping preimage at index {idx}: invalid requester type {type(requester)}")
                continue
                
            if requester not in preimages_by_requester:
                preimages_by_requester[requester] = []
            preimages_by_requester[requester].append(preimage)
            
        except Exception as e:
            logger.error(f"Error processing preimage at index {idx}: {str(e)}")
            logger.debug(f"Preimage data: {preimage}")
            continue
    
    logger.info(f"Grouped preimages into {len(preimages_by_requester)} requesters")
    
    # Process preimages for each requester
    for requester, req_preimages in preimages_by_requester.items():
        if not req_preimages:
            logger.warning(f"No valid preimages for requester {requester}")
            continue
            
        logger.debug(f"Processing {len(req_preimages)} preimages for requester {requester}")
            
        account_data = {
            "id": str(requester),  # Ensure ID is string for JSON compatibility
            "data": {
                "preimages": [], 
                "lookup_meta": []
            }
        }
        
        for i, preimage in enumerate(req_preimages):
            try:
                # Get and validate blob
                blob = preimage.get('blob', '')
                if not isinstance(blob, str):
                    logger.warning(f"Skipping invalid blob at index {i} for requester {requester}: not a string")
                    continue
                    
                # Ensure blob has 0x prefix
                if not blob.startswith('0x'):
                    blob = '0x' + blob
                    
                # Skip empty blobs
                if len(blob) <= 2:  # Only has 0x prefix
                    logger.warning(f"Skipping empty blob at index {i} for requester {requester}")
                    continue
                    
                # Remove any whitespace from blob
                clean_blob = blob.strip()
                
                # Skip if blob is not a valid hex string (after removing 0x)
                try:
                    if len(clean_blob) > 2:  # Only try to validate if there's more than just 0x
                        bytes.fromhex(clean_blob[2:])
                except ValueError as e:
                    logger.warning(f"Skipping invalid hex blob at index {i} for requester {requester}: {str(e)}")
                    continue
                    
                # Calculate hash
                try:
                    hash_obj = hashlib.sha256(bytes.fromhex(clean_blob[2:]))
                    hash_hex = '0x' + hash_obj.hexdigest()
                    
                    # Calculate blob length in bytes (hex string length / 2)
                    blob_length = len(clean_blob[2:]) // 2
                    
                    # Add to preimages
                    account_data["data"]["preimages"].append({
                        "hash": hash_hex,
                        "blob": clean_blob
                    })
                    
                    # Add to lookup_meta with position information
                    account_data["data"]["lookup_meta"].append({
                        "key": {
                            "hash": hash_hex,
                            "length": blob_length
                        },
                        "value": [i * 10, (i + 1) * 10]  # Example positions
                    })
                    
                except (ValueError, IndexError) as e:
                    logger.error(f"Error processing blob at index {i} for requester {requester}: {str(e)}")
                    logger.debug(f"Blob data: {blob}")
                    continue
                    
            except Exception as e:
                logger.error(f"Unexpected error processing preimage at index {i} for requester {requester}: {str(e)}")
                logger.debug(f"Preimage data: {preimage}", exc_info=True)
                continue
        
        # Only add account if it has valid preimages
        if account_data["data"]["preimages"]:
            post_state["accounts"].append(account_data)
            logger.debug(f"Added {len(account_data['data']['preimages'])} preimages for requester {requester}")
        else:
            logger.warning(f"No valid preimages found for requester {requester}")
    
    logger.info(f"Processed preimages for {len(post_state['accounts'])} accounts")
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
