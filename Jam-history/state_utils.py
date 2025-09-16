import json
import os
import copy
from pathlib import Path
from typing import Dict, Any, Optional, List, Union, Tuple
from jam_types import State, BetaBlock, MMR, Reported, Input
from history_stf import keccak256

def extract_input_from_payload(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict]]:

    if not payload or 'block' not in payload:
        return {}, []
        
    header = payload['block'].get('header', {})
    
    # Extract work packages from header or extrinsic
    work_packages = header.get('work_packages', [])
    if not work_packages and 'extrinsic' in payload['block']:
        work_packages = [
            {
                'hash': wp['report']['workPackage']['hash'] if 'hash' in wp['report']['workPackage'] else None,
                'exports_root': wp['report'].get('package_spec', {}).get('exports_root')
            }
            for wp in payload['block']['extrinsic'].get('guarantees', [])
            if wp.get('report', {}).get('workPackage')
        ]
    
    input_data = {
        'header_hash': header.get('header_hash'),
        'parent_state_root': header.get('parent_state_root'),
        'accumulate_root': header.get('accumulate_root'),
        'work_packages': work_packages
    }
    
    return input_data, work_packages

def load_updated_state(file_path: Union[str, Dict[str, Any]]) -> dict:
    
    # Default empty state
    state_data = {}
    
    if isinstance(file_path, dict):
        state_data = file_path
    else:
        try:
            with open(file_path, 'r') as f:
                state_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            state_data = {}
    
    # Initialize beta state if not exists
    if 'beta' not in state_data:
        state_data['beta'] = []
    
    # Get the latest beta block or create a default one
    latest_beta = state_data['beta'][-1] if state_data.get('beta') else {
        'header_hash': '0x' + '0' * 64,
        'mmr': {'peaks': []},
        'state_root': '0x' + '0' * 64,
        'reported': []
    }
    
    # Prepare default input based on latest beta block
    default_input = {
        'header_hash': latest_beta.get('header_hash', '0x' + '0' * 64),
        'parent_state_root': latest_beta.get('state_root', '0x' + '0' * 64),
        'accumulate_root': '0x' + '0' * 64,
        'work_packages': latest_beta.get('reported', [])
    }
    
    # Initialize beta_blocks as empty list
    beta_blocks = []
    
    # If file_path is already a dict (from payload), use it directly
    if isinstance(file_path, dict):
        state_data = file_path
    else:
        # Otherwise, try to read from file
        try:
            with open(file_path, 'r') as f:
                state_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading state file: {e}")
            state_data = {}
    
    # Initialize with default input values
    input_data = default_input.copy()
    
    # Try to get input data from different possible locations
    if 'input' in state_data:
        # If input is directly in state_data
        input_data.update(state_data['input'])
    elif 'block' in state_data and 'header' in state_data['block']:
        # If we have a block header in the payload
        header = state_data['block']['header']
        input_data.update({
            'header_hash': header.get('header_hash', input_data['header_hash']),
            'parent_state_root': header.get('parent_state_root', input_data['parent_state_root']),
            'accumulate_root': header.get('accumulate_root', input_data['accumulate_root']),
            'work_packages': header.get('work_packages', input_data['work_packages'])
        })
        
        # Initialize beta_blocks from pre_state if available
        if 'pre_state' in state_data and 'beta' in state_data['pre_state']:
            beta_blocks = state_data['pre_state']['beta']
        
        # If no beta blocks found, try to create from block header if available
        if not beta_blocks and 'block' in state_data and 'header' in state_data['block']:
            header = state_data['block']['header']
            mmr_peaks = []
            if 'header_hash' in header and 'parent_state_root' in header:
                # Create a hash from header_hash and parent_state_root for the MMR peak
                header_hash = header['header_hash'][2:] if header['header_hash'].startswith('0x') else header['header_hash']
                state_root = header['parent_state_root'][2:] if header['parent_state_root'].startswith('0x') else header['parent_state_root']
                mmr_input = bytes.fromhex(header_hash + state_root)
                mmr_peaks = [keccak256(mmr_input)]
            
            beta_blocks = [{
                'header_hash': header.get('header_hash', '0x' + '00' * 32),
                'state_root': header.get('parent_state_root', '0x' + '00' * 32),
                'mmr': {
                    'peaks': mmr_peaks,
                    'count': len(mmr_peaks)
                },
                'reported': header.get('work_packages', [])
            }]
        
    return {
        'input': input_data,
        'pre_state': {
            'beta': beta_blocks
        }
    }

def save_updated_state(file_path: str, state_data: dict, new_beta_block: Optional[Dict] = None) -> bool:
  
    try:
        # Load existing state if file exists
        existing_state = {}
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                existing_state = json.load(f)
        
        # Initialize beta list if it doesn't exist
        if 'beta' not in existing_state:
            existing_state['beta'] = []
        
        # Add new beta block if provided
        if new_beta_block:
            existing_state['beta'].append(new_beta_block)
        
        # Update other state data
        for key, value in state_data.items():
            if key != 'beta':  # Don't overwrite beta with empty list
                existing_state[key] = value
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Write the updated state to file
        with open(file_path, 'w') as f:
            json.dump(existing_state, f, indent=2)
            
        return True
    except Exception as e:
        print(f"Error saving updated state: {e}")
        return False
