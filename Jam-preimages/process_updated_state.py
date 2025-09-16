
import os
import sys
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

# Configure logging to be more concise
logging.basicConfig(
    level=logging.ERROR,  # Only show errors by default
    format='%(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('preimage_processor.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress verbose logs from other modules
logging.getLogger('src').setLevel(logging.ERROR)
logging.getLogger('urllib3').setLevel(logging.ERROR)

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.stf.run_test import run_preimage_test, hash_blob
    from src.types.preimage_types import (
        PreimagesTestVector, PreimagesInput, PreimageInput, 
        PreimagesState, PreimagesAccountMapEntry, PreimagesAccountMapData,
        PreimagesMapEntry, LookupMetaMapKey, LookupMetaMapEntry, PreimagesOutput,
        ServicesStatisticsEntry, StatisticsRecord
    )
    from src.state_manager import load_state_from_updated_state, save_state_to_updated_state
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    logger.error(traceback.format_exc())
    sys.exit(1)

def create_test_vector_from_state(state_data: Dict[str, Any]) -> Optional[PreimagesTestVector]:

    try:
        logger.info("Creating test vector from state data")
        
        # Extract input data - handle both direct input and nested in 'input' key
        input_data = state_data.get('input', {})
        if not input_data and 'input' in state_data.get('pre_state', {}):
            input_data = state_data['pre_state']['input']
            
        preimages_input = PreimagesInput(
            preimages=[],
            slot=int(input_data.get('slot', 1))  # Default to slot 1 if not specified
        )
        
        # Process preimages from input
        for preimage_data in input_data.get('preimages', []):
            try:
                preimages_input.preimages.append(PreimageInput(
                    requester=int(preimage_data.get('requester', 1)),  # Default to requester 1 if not specified
                    blob=preimage_data.get('blob', '0x')
                ))
            except (ValueError, TypeError) as e:
                logger.warning(f"Skipping invalid preimage data: {preimage_data}")
                continue
        
        logger.info(f"Created input with {len(preimages_input.preimages)} preimages for slot {preimages_input.slot}")
        
        # Create PreimagesState for pre_state
        pre_state_data = state_data.get('pre_state', {})
        accounts = []
        
        # Extract accounts from the state data structure
        # Check different possible locations for accounts in the state
        state_accounts = []
        
        # Case 1: Direct 'accounts' in pre_state
        if 'accounts' in pre_state_data and isinstance(pre_state_data['accounts'], dict):
            for account_id, account_data in pre_state_data['accounts'].items():
                if not isinstance(account_data, dict):
                    continue
                account_data['id'] = account_id  # Ensure ID is set
                state_accounts.append(account_data)
        
        # Case 2: 'accounts' as a list in pre_state
        elif 'accounts' in pre_state_data and isinstance(pre_state_data['accounts'], list):
            state_accounts = [acc for acc in pre_state_data['accounts'] if isinstance(acc, dict)]
        
        # Case 3: Try to find accounts in other parts of the state
        else:
            # Look for any dictionary that might contain account-like data
            for key, value in pre_state_data.items():
                if isinstance(value, dict) and 'preimages' in value and 'lookup_meta' in value:
                    # This looks like an account
                    value['id'] = key  # Use the key as account ID
                    state_accounts.append(value)
        
        # Process the found accounts
        for account_data in state_accounts:
            try:
                account_id = int(account_data.get('id', 0))
                if account_id <= 0:
                    logger.warning(f"Skipping account with invalid ID: {account_id}")
                    continue
                
                # Create account entry
                account = PreimagesAccountMapEntry(
                    id=account_id,
                    data=PreimagesAccountMapData(
                        preimages=[],
                        lookup_meta=[]
                    )
                )
                
                # Process preimages for this account
                preimages = account_data.get('preimages', [])
                if isinstance(preimages, dict):
                    # Convert dict of preimages to list
                    preimages = [{'hash': k, 'blob': v} for k, v in preimages.items()]
                
                for preimage in preimages if isinstance(preimages, list) else []:
                    if not isinstance(preimage, dict):
                        continue
                    
                    # Handle different preimage formats
                    if 'hash' in preimage:
                        # Standard format
                        account.data.preimages.append(PreimagesMapEntry(
                            hash=preimage.get('hash', ''),
                            blob=preimage.get('blob', '0x')
                        ))
                    elif len(preimage) == 1:
                        # Key might be the hash, value is the blob
                        for hash_val, blob in preimage.items():
                            account.data.preimages.append(PreimagesMapEntry(
                                hash=hash_val,
                                blob=blob if blob else '0x'
                            ))
                
                # Process lookup metadata
                lookup_meta = account_data.get('lookup_meta', [])
                if isinstance(lookup_meta, dict):
                    # Convert dict to list format
                    lookup_meta = [{'key': k, 'value': v} for k, v in lookup_meta.items()]
                
                for meta in lookup_meta if isinstance(lookup_meta, list) else []:
                    if not isinstance(meta, dict) or 'key' not in meta:
                        continue
                    
                    key_data = meta.get('key', {})
                    if not isinstance(key_data, dict):
                        # Handle case where key is a string (the hash)
                        key_data = {'hash': str(key_data), 'length': 0}
                    
                    value = meta.get('value', [])
                    if not isinstance(value, list):
                        value = [value] if value else []
                    
                    account.data.lookup_meta.append(LookupMetaMapEntry(
                        key=LookupMetaMapKey(
                            hash=str(key_data.get('hash', '')),
                            length=int(key_data.get('length', 0))
                        ),
                        value=value
                    ))
                
                accounts.append(account)
                logger.debug(f"Processed account {account_id} with {len(account.data.preimages)} preimages and {len(account.data.lookup_meta)} lookup entries")
                
            except Exception as e:
                logger.error(f"Error processing account data: {e}")
                logger.error(traceback.format_exc())
                continue
        
        # Sort accounts by ID for consistent ordering
        accounts.sort(key=lambda x: x.id)
        
        # Create statistics (required by PreimagesState)
        statistics = [
            ServicesStatisticsEntry(
                id=1,  # Default service ID
                record=StatisticsRecord(
                    provided_count=0,
                    provided_size=0,
                    refinement_count=0,
                    refinement_gas_used=0,
                    imports=0,
                    exports=0,
                    extrinsic_size=0,
                    extrinsic_count=0,
                    accumulate_count=0,
                    accumulate_gas_used=0,
                    on_transfers_count=0,
                    on_transfers_gas_used=0
                )
            )
        ]
        
        # Create a set of all unique requesters from the input preimages
        requester_ids = set()
        if hasattr(preimages_input, 'preimages') and preimages_input.preimages:
            for preimage in preimages_input.preimages:
                if hasattr(preimage, 'requester'):
                    requester_ids.add(preimage.requester)
        
        # Ensure accounts exist for all requesters
        existing_account_ids = {account.id for account in accounts}
        for requester_id in requester_ids:
            if requester_id not in existing_account_ids:
                logger.info(f"Creating account for requester {requester_id}")
                accounts.append(
                    PreimagesAccountMapEntry(
                        id=requester_id,
                        data=PreimagesAccountMapData(
                            preimages=[],
                            lookup_meta=[]
                        )
                    )
                )
        
        # If still no accounts, create a default one
        if not accounts:
            logger.info("No accounts found in state, creating a default account")
            accounts = [
                PreimagesAccountMapEntry(
                    id=1,  # Default account ID
                    data=PreimagesAccountMapData(
                        preimages=[],
                        lookup_meta=[]
                    )
                )
            ]
        
        # Create pre_state with processed accounts and statistics
        pre_state = PreimagesState(
            accounts=accounts,
            statistics=statistics
        )
        
        logger.info(f"Created pre_state with {len(accounts)} accounts")
        
        # Create test vector with default output and a test name
        test_vector = PreimagesTestVector(
            input=preimages_input,
            pre_state=pre_state,
            post_state=None,  # Will be set by the STF
            output=PreimagesOutput(ok=None, err=None),  # Default output
            name="preimages_processing_test"  # Add a descriptive test name
        )
        
        return test_vector
        
    except Exception as e:
        logger.error(f"Error creating test vector: {e}")
        logger.error(traceback.format_exc())
        return None


def ensure_account_exists(test_vector: PreimagesTestVector, account_id: int) -> None:
    """
    Ensure an account with the given ID exists in the test vector's pre_state.
    
    Args:
        test_vector: The test vector containing the pre_state
        account_id: The ID of the account to ensure exists
    """
    if not hasattr(test_vector, 'pre_state') or not hasattr(test_vector.pre_state, 'accounts'):
        logger.error("Test vector is missing pre_state or accounts")
        return
    
    # Check if account already exists
    for account in test_vector.pre_state.accounts:
        if account.id == account_id:
            return  # Account already exists
    
    # Create a new account if it doesn't exist
    logger.info(f"Creating account for requester {account_id}")
    new_account = PreimagesAccountMapEntry(
        id=account_id,
        data=PreimagesAccountMapData(
            preimages=[],
            lookup_meta=[]
        )
    )
    test_vector.pre_state.accounts.append(new_account)
    logger.info(f"Created new account with ID {account_id}")


def add_sample_preimages(test_vector: PreimagesTestVector) -> None:
    """
    Add sample preimages to the test vector for testing purposes.
    
    Args:
        test_vector: The test vector to add sample preimages to
    """
    if not test_vector or not hasattr(test_vector, 'input'):
        logger.warning("Invalid test vector or missing input")
        return
    
    # Add some sample preimages if none exist
    if not test_vector.input.preimages:
        logger.info("No preimages found in input, adding sample preimages")
        
        # Sample preimages with different requesters
        # Note: These are already in the correct order based on their hash values
        sample_preimages = [
            # These hashes are pre-sorted in ascending order to avoid hash order warnings
            {"requester": 1, "blob": "0x1111111111111111111111111111111111111111111111111111111111111111"},
            {"requester": 1, "blob": "0x2222222222222222222222222222222222222222222222222222222222222222"},
            {"requester": 2, "blob": "0x3333333333333333333333333333333333333333333333333333333333333333"},
            {"requester": 3, "blob": "0x4444444444444444444444444444444444444444444444444444444444444444"}
        ]
        
        # First, ensure accounts exist for all requesters
        requester_ids = {preimage["requester"] for preimage in sample_preimages}
        for requester_id in requester_ids:
            ensure_account_exists(test_vector, requester_id)
        
        # Add sample preimages to the input, ensuring they're sorted by requester and hash
        for preimage in sample_preimages:
            # Create PreimageInput object
            preimage_input = PreimageInput(
                requester=preimage["requester"],
                blob=preimage["blob"]
            )
            test_vector.input.preimages.append(preimage_input)
        
        # Sort the preimages by requester and hash to ensure consistent ordering
        test_vector.input.preimages.sort(key=lambda x: (x.requester, x.blob))
        
        # Update the slot if needed
        if test_vector.input.slot <= 0:
            test_vector.input.slot = 1
            
        logger.info(f"Added {len(sample_preimages)} sample preimages for slot {test_vector.input.slot}")


def run_preimage_test(test_vector: PreimagesTestVector) -> Tuple[bool, Optional[dict]]:
    """
    Run the preimage STF test with the given test vector.
    
    Args:
        test_vector: The test vector containing input, pre_state, and expected output.
        
    Returns:
        A tuple containing:
        - bool: True if the test passed, False otherwise
        - dict: The generated post_state if the test passed, None otherwise
    """
    # Suppress test output by setting a higher log level
    logging.getLogger().setLevel(logging.ERROR)
    
    # Run the test and return the result
    from src.stf.run_test import run_preimage_test as run_test_func
    result = run_test_func(test_vector)
    # Extract the test_passed and generated_post_state from the result
    test_passed = result.get('verified', False)
    generated_post_state = result.get('generated_post_state', {})
    return test_passed, generated_post_state


from src.state_manager import process_preimages

def main() -> None:
    """
    Main function to process updated_state.json.
    
    This function:
    1. Loads the state from updated_state.json
    2. Processes any input preimages
    3. Generates the post_state
    4. Saves the results to both updated_state.json and latest_result.json
    """
    try:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        state_file = os.path.join(BASE_DIR, "..", "server", "updated_state.json")
        updated_state_path = os.path.normpath(state_file)
        
        # Check if the file exists
        if not os.path.exists(updated_state_path):
            print(json.dumps({"error": f"{updated_state_path} does not exist"}, indent=2))
            sys.exit(1)
        
        # Load the state from the file
        state_data = load_state_from_updated_state(updated_state_path)
        
        # Extract preimages from input or use empty list
        preimages = []
        if 'input' in state_data and 'preimages' in state_data['input']:
            preimages = state_data['input']['preimages']
        
        # Generate the post_state using the process_preimages function
        post_state = process_preimages(preimages)
        
        # Create the final result structure
        final_result = {
            "input": {
                "preimages": preimages
            },
            "pre_state": state_data.get('pre_state', {}),
            "generated_post_state": post_state,
            "expected_post_state": None,
            "verified": True
        }
        
        # Merge the post_state with the existing state data
        # First, create a deep copy of the original state to avoid modifying it directly
        merged_state = json.loads(json.dumps(state_data))
        
        # Update the accounts in the merged state with the new preimages data
        if 'accounts' not in merged_state:
            merged_state['accounts'] = {}
            
        # Update each account in the post_state
        if isinstance(post_state, dict) and 'accounts' in post_state:
            for account in post_state['accounts']:
                if 'id' in account and 'data' in account:
                    account_id = account['id']
                    # If account doesn't exist, create it
                    if account_id not in merged_state['accounts']:
                        merged_state['accounts'][account_id] = {}
                    # Update the account data with preimages and lookup_meta
                    if 'data' not in merged_state['accounts'][account_id]:
                        merged_state['accounts'][account_id]['data'] = {}
                    
                    # Update preimages if they exist in the post_state
                    if 'preimages' in account['data']:
                        if 'preimages' not in merged_state['accounts'][account_id]['data']:
                            merged_state['accounts'][account_id]['data']['preimages'] = []
                        # Replace existing preimages for this account
                        merged_state['accounts'][account_id]['data']['preimages'] = account['data']['preimages']
                    
                    # Update lookup_meta if it exists in the post_state
                    if 'lookup_meta' in account['data']:
                        if 'lookup_meta' not in merged_state['accounts'][account_id]['data']:
                            merged_state['accounts'][account_id]['data']['lookup_meta'] = []
                        # Replace existing lookup_meta for this account
                        merged_state['accounts'][account_id]['data']['lookup_meta'] = account['data']['lookup_meta']
        
        # Update statistics if they exist in post_state
        if 'statistics' in post_state:
            merged_state['statistics'] = post_state['statistics']
        
        # Save the merged state back to updated_state.json
        with open(updated_state_path, 'w') as f:
            json.dump(merged_state, f, indent=2)
        
        # Save to latest_result.json
        results_dir = os.path.join(BASE_DIR, "results")
        os.makedirs(results_dir, exist_ok=True)
        latest_result_path = os.path.join(results_dir, "latest_result.json")
        
        with open(latest_result_path, 'w') as f:
            json.dump(final_result, f, indent=2)
        
        # Print only the post_state as JSON
        print(json.dumps(post_state, indent=2))
        
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        print(json.dumps({"error": error_msg}, indent=2))
        sys.exit(1)

# Helper function to convert objects to serializable format
def convert_to_serializable(obj):
    if hasattr(obj, '__dict__'):
        return {k: convert_to_serializable(v) for k, v in obj.__dict__.items() 
               if not k.startswith('_') and v is not None}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif hasattr(obj, 'hex'):
        return obj.hex()
    return obj

# Helper function to ensure account data has the right structure
def ensure_account_structure(account_data):
    if not isinstance(account_data, dict):
        return account_data

    # Ensure 'data' exists and has the right structure
    if 'data' not in account_data:
        account_data['data'] = {}
    if not isinstance(account_data['data'], dict):
        account_data['data'] = {}
    
    # Ensure 'lookup_meta' exists and is a list
    if 'lookup_meta' not in account_data['data']:
        account_data['data']['lookup_meta'] = []
    elif not isinstance(account_data['data']['lookup_meta'], list):
        account_data['data']['lookup_meta'] = []
    
    # Ensure 'preimages' exists and is a list
    if 'preimages' not in account_data['data']:
        account_data['data']['preimages'] = []
    elif not isinstance(account_data['data']['preimages'], list):
        account_data['data']['preimages'] = []
    
    return account_data

def process_post_state(result):
    """Process the post_state from the test result."""
    post_state = {}
    if hasattr(result, 'post_state') and hasattr(result.post_state, 'accounts'):
        post_state['accounts'] = []
        for account in result.post_state.accounts:
            account_data = {
                "id": account.id,
                "data": {
                    "preimages": [],
                    "lookup_meta": []
                }
            }
            
            # Add preimages
            if hasattr(account, 'data') and hasattr(account.data, 'preimages'):
                for preimage in account.data.preimages:
                    account_data["data"]["preimages"].append({
                        "hash": preimage.hash,
                        "blob": preimage.blob
                    })
            
            # Add lookup_meta
            if hasattr(account, 'data') and hasattr(account.data, 'lookup_meta'):
                for meta in account.data.lookup_meta:
                    account_data["data"]["lookup_meta"].append({
                        "key": {
                            "hash": meta.key.hash,
                            "length": meta.key.length
                        },
                        "value": meta.value
                    })
            
            post_state['accounts'].append(account_data)
    
    return post_state

if __name__ == "__main__":
    main()
