#!/usr/bin/env python3
"""
Process updated_state.json for preimage data.

This script reads from the server's updated_state.json file, processes it
using the preimage STF, and saves the results back to updated_state.json.
"""
import os
import sys
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

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
    """
    Create a PreimagesTestVector from the state data.
    
    Args:
        state_data: Dictionary containing 'input' and 'pre_state' data
        
    Returns:
        PreimagesTestVector or None if there was an error
    """
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
    logger.debug(f"Created new account with ID {account_id}")

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
        
        # Log the order of preimages and accounts for debugging
        logger.info(f"Current accounts in pre_state: {[acc.id for acc in test_vector.pre_state.accounts]}")
        for i, preimage in enumerate(test_vector.input.preimages):
            logger.debug(f"Preimage {i}: requester={preimage.requester}, blob_hash={hash_blob(preimage.blob) if hasattr(preimage, 'blob') else 'N/A'}")

def main() -> None:
    """
    Main function to process updated_state.json.
    
    This function:
    1. Loads the state from updated_state.json
    2. Creates a test vector from the state
    3. Adds sample preimages if needed
    4. Runs the preimage STF test
    5. Saves the updated state back to updated_state.json
    """
    try:
        # Path to updated_state.json (relative to the project root)
        project_root = Path(__file__).resolve().parent
        updated_state_path = project_root.parent / "server" / "updated_state.json"
        
        logger.info(f"Starting preimage processing for {updated_state_path}")
        
        # Ensure the updated_state.json exists or create a default one
        if not updated_state_path.exists():
            logger.warning(f"{updated_state_path} does not exist, creating a default state")
            updated_state_path.parent.mkdir(parents=True, exist_ok=True)
            with open(updated_state_path, 'w') as f:
                json.dump({
                    "current_slot": 1,
                    "beta": [],
                    "accounts": [],
                    "statistics": []
                }, f, indent=2)
        
        # Load state from updated_state.json
        logger.info(f"Loading state from {updated_state_path}")
        state_data = load_state_from_updated_state(str(updated_state_path))
        if not state_data:
            logger.error("Failed to load state data")
            return
        
        # Create test vector from state data
        logger.info("Creating test vector from state data")
        test_vector = create_test_vector_from_state(state_data)
        if not test_vector:
            logger.error("Failed to create test vector from state data")
            return
        
        # Add sample preimages if needed (for testing)
        add_sample_preimages(test_vector)
        
        # Run the preimage test
        logger.info("Running preimage STF test...")
        
        # Log test vector details before running the test
        logger.debug(f"Test vector name: {getattr(test_vector, 'name', 'unnamed')}")
        logger.debug(f"Input preimages: {len(test_vector.input.preimages) if hasattr(test_vector, 'input') and hasattr(test_vector.input, 'preimages') else 'N/A'}")
        logger.debug(f"Pre-state accounts: {len(test_vector.pre_state.accounts) if hasattr(test_vector, 'pre_state') and hasattr(test_vector.pre_state, 'accounts') else 'N/A'}")
        
        try:
            result = run_preimage_test(test_vector)
            
            if not result:
                logger.error("Test returned None")
                return
                
            if not hasattr(result, 'post_state'):
                logger.error("Test result has no post_state attribute")
                logger.debug(f"Result attributes: {dir(result)}")
                
                # If there's an error message, log it
                if hasattr(result, 'error'):
                    logger.error(f"Test error: {result.error}")
                    
                return
                
        except Exception as e:
            logger.error(f"Exception during test execution: {e}")
            logger.error(traceback.format_exc())
            return
        
        # Save the updated state back to updated_state.json
        logger.info("Saving updated state...")
        if save_state_to_updated_state(str(updated_state_path), result):
            logger.info(f"Successfully updated {updated_state_path}")
            
            # Save the full test results to a file for reference
            results_dir = project_root / "test_results"
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # Find the next available results file
            results_index = 1
            while True:
                results_path = results_dir / f"preimage_test_results_{results_index:03d}.json"
                if not results_path.exists():
                    break
                results_index += 1
            
            # Save detailed results
            with open(results_path, 'w') as f:
                import json
                from dataclasses import asdict
                
                results = {
                    "input": asdict(result.input) if hasattr(result, 'input') else {},
                    "pre_state": asdict(result.pre_state) if hasattr(result, 'pre_state') else {},
                    "post_state": asdict(result.post_state) if hasattr(result, 'post_state') else {},
                    "output": asdict(result.output) if hasattr(result, 'output') else {}
                }
                
                # Ensure JSON serialization
                def default_serializer(obj):
                    if hasattr(obj, '__dict__'):
                        return obj.__dict__
                    elif hasattr(obj, 'hex'):
                        return obj.hex()
                    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
                
                json.dump(results, f, indent=2, default=default_serializer)
                
            logger.info(f"Test results saved to {results_path}")
            
            # Print a summary
            input_count = len(result.input.preimages) if hasattr(result, 'input') and hasattr(result.input, 'preimages') else 0
            pre_state_accounts = len(result.pre_state.accounts) if hasattr(result, 'pre_state') and hasattr(result.pre_state, 'accounts') else 0
            post_state_accounts = len(result.post_state.accounts) if hasattr(result, 'post_state') and hasattr(result.post_state, 'accounts') else 0
            
            print("\n=== Preimage Processing Summary ===")
            print(f"Input preimages: {input_count}")
            print(f"Pre-state accounts: {pre_state_accounts}")
            print(f"Post-state accounts: {post_state_accounts}")
            print(f"Results saved to: {results_path}")
            print("================================\n")
            
        else:
            logger.error("Failed to save updated state")
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
