import json
from pathlib import Path
from typing import Dict, Any, Optional
from jam_types import State, BetaBlock, MMR, Reported
from history_stf import keccak256

def load_updated_state(file_path: str) -> dict:
    """
    Load and parse the updated_state.json file, extracting the required fields
    for jam_history input and pre_state.
    
    Args:
        file_path: Path to the updated_state.json file
        
    Returns:
        dict: Dictionary containing 'input' and 'pre_state' keys
    """
    # Default hardcoded values
    default_input = {
        'header_hash': '0x37fe33f96de26f26d8c256e3b5baed0fbc28e1824f1844e0e1417b5342863740',
        'parent_state_root': '0x17e901a3e6e660fcb31214ccaed62b0d43df9d5e91c31609cf2acf5844d7f625',
        'accumulate_root': '0x8720b97ddd6acc0f6eb66e095524038675a4e4067adc10ec39939eaefc47d842',
        'work_packages': [
            {
                'hash': '0x303026af983b91393c6b31e972263759f72c5e7656c00b267e9b61292252c125',
                'exports_root': '0x3d16ed66bae11acbdccc7015b4ee5c1bf87c864a7c930f78f59368280551f60d'
            }
        ]
    }
    
    # Initialize beta_blocks as empty list
    
    try:
        with open(file_path, 'r') as f:
            state_data = json.load(f)
            
        # Get the most recent block from recent_blocks if available
        recent_blocks = state_data.get('recent_blocks', {}).get('history', [])
        
        # Initialize with default input values
        input_data = default_input.copy()
        
        # Override with values from recent_blocks if available
        if recent_blocks:
            latest_block = recent_blocks[-1]
            if 'header_hash' in latest_block:
                input_data['header_hash'] = latest_block['header_hash']
            if 'state_root' in latest_block:
                input_data['parent_state_root'] = latest_block['state_root']
            if 'beefy_root' in latest_block:
                input_data['accumulate_root'] = latest_block['beefy_root']
            if 'reported' in latest_block:
                input_data['work_packages'] = latest_block['reported']
        
        # Initialize beta_blocks as empty list
        beta_blocks = []
        
        # Try to get beta blocks from pre_state first
        if 'pre_state' in state_data and 'beta' in state_data['pre_state'] and state_data['pre_state']['beta']:
            beta_blocks = state_data['pre_state']['beta']
        # If no pre_state.beta, try to convert recent_blocks to beta format
        elif recent_blocks:
            # Convert recent_blocks to beta blocks format if no pre_state.beta exists
            beta_blocks = []
            for block in recent_blocks:
                # Create a simple MMR with a single peak for each block
                mmr_peaks = []
                if block.get('header_hash') and block.get('state_root'):
                    # Create a hash from header_hash and state_root for the MMR peak
                    mmr_input = f"{block['header_hash']}{block['state_root']}".encode()
                    mmr_peaks.append(keccak256(mmr_input).hex())
                
                beta_block = {
                    'header_hash': block.get('header_hash', '0x' + '00' * 32),
                    'state_root': block.get('state_root', '0x' + '00' * 32),
                    'mmr': {
                        'peaks': mmr_peaks,
                        'count': len(mmr_peaks)
                    },
                    'reported': block.get('reported', [])
                }
                beta_blocks.append(beta_block)
        
        return {
            'input': input_data,
            'pre_state': {
                'beta': beta_blocks
            }
        }
        
    except Exception as e:
        print(f"Error loading updated state: {e}")
        # Return empty beta blocks on error
        return {
            'input': default_input,
            'pre_state': {
                'beta': []
            }
        }

def save_updated_state(file_path: str, state_data: dict) -> bool:
    """
    Save the current state to the updated_state.json file.
    
    Args:
        file_path: Path to save the updated_state.json file
        state_data: Dictionary containing 'input' and 'pre_state' data
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load existing data to preserve other fields
        existing_data = {}
        try:
            with open(file_path, 'r') as f:
                existing_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            existing_data = {}
        
        # Update the pre_state with the new beta blocks
        if 'pre_state' not in existing_data:
            existing_data['pre_state'] = {}
            
        # Update the beta blocks in pre_state
        if 'beta' in state_data:
            existing_data['pre_state']['beta'] = state_data['beta']
        
        # Preserve the input data if it exists in the existing data
        if 'input' in existing_data:
            # Update the input data with any new values from state_data
            if 'input' in state_data:
                existing_data['input'].update(state_data['input'])
        elif 'input' in state_data:
            existing_data['input'] = state_data['input']
        
        # Update recent_blocks with the new beta blocks
        if 'beta' in existing_data.get('pre_state', {}):
            beta_blocks = existing_data['pre_state']['beta']
            
            # Initialize recent_blocks if it doesn't exist
            if 'recent_blocks' not in existing_data:
                existing_data['recent_blocks'] = {'history': []}
            
            # Convert beta blocks to recent_blocks format
            recent_blocks = []
            for block in beta_blocks:
                recent_block = {
                    'header_hash': block.get('header_hash', '0x' + '00' * 32),
                    'state_root': block.get('state_root', '0x' + '00' * 32),
                    'beefy_root': '0x' + '00' * 32,  # Default beefy_root if not available
                    'reported': block.get('reported', [])
                }
                recent_blocks.append(recent_block)
            
            # Keep only the most recent 8 blocks (if needed)
            existing_data['recent_blocks']['history'] = recent_blocks[-8:]
        
        # Save back to file
        with open(file_path, 'w') as f:
            json.dump(existing_data, f, indent=2)
            
        return True
        
    except Exception as e:
        print(f"Error saving updated state: {e}")
        return False
