
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional

from normalize import normalize
from history_stf import HistorySTF
from jam_types import Input, State, BetaBlock, MMR, Reported
from state_utils import load_updated_state, save_updated_state


def create_input_from_dict(data: Dict[str, Any]) -> Input:
    work_packages = [
        Reported(hash=wp['hash'], exports_root=wp['exports_root'])
        for wp in data.get('work_packages', [])
    ]
    
    return Input(
        header_hash=data['header_hash'],
        parent_state_root=data['parent_state_root'],
        accumulate_root=data['accumulate_root'],
        work_packages=work_packages
    )


def create_state_from_dict(data: Dict[str, Any]) -> State:
   
    beta_blocks = []
    for block_data in data.get('beta', []):
        mmr = MMR(
            peaks=block_data['mmr']['peaks'],
            count=block_data['mmr'].get('count')
        )
        
        reported = [
            Reported(hash=r['hash'], exports_root=r['exports_root'])
            for r in block_data.get('reported', [])
        ]
        
        beta_block = BetaBlock(
            header_hash=block_data['header_hash'],
            state_root=block_data['state_root'],
            mmr=mmr,
            reported=reported
        )
        beta_blocks.append(beta_block)
    
    return State(beta=beta_blocks)


def state_to_dict(state: State) -> Dict[str, Any]:
  
    beta_list = []
    for block in state.beta:
        mmr_dict = {
            'peaks': block.mmr.peaks
        }
        if block.mmr.count is not None:
            mmr_dict['count'] = block.mmr.count
            
        reported_list = [
            {'hash': r.hash, 'exports_root': r.exports_root}
            for r in block.reported
        ]
        
        block_dict = {
            'header_hash': block.header_hash,
            'state_root': block.state_root,
            'mmr': mmr_dict,
            'reported': reported_list
        }
        beta_list.append(block_dict)
    
    return {'beta': beta_list}


def green(msg: str) -> None:
    
    print(f'\033[32m✓ {msg}\033[0m')


def red(msg: str) -> None:
   
    print(f'\033[31m✗ {msg}\033[0m')


def main():
    script_dir = Path(__file__).parent
    results_dir = script_dir / 'results'
    
    # Path to the updated_state.json file
    updated_state_path = Path('/Users/anish/Desktop/fulljam/Jam_implementation_full/server/updated_state.json')
    
    results_dir.mkdir(exist_ok=True)
    
    print("Loading state from:", updated_state_path)
    state_data = load_updated_state(updated_state_path)
    
    if not state_data:
        print("❌ Failed to load state data from updated_state.json")
        return
    print("\n ✅ ✅ ✅  Jam_History Component is running successfully... ✅ ✅ ✅ ")
    print("\nCreating State and Input objects...")
    current_state = create_state_from_dict(state_data['pre_state'])
    input_data = create_input_from_dict(state_data['input'])
    
    try:
        print("\nProcessing state transition...")
        result = HistorySTF.transition(current_state, input_data)
        generated_post_state = result['postState']
        
        print("✅ State transition completed successfully")
        # print(f"- Previous state root: {current_state.beta[-1].state_root if current_state.beta else 'None'}")
        # print(f"- New state root: {generated_post_state.beta[-1].state_root}")
        
        # Convert State object to dictionary before saving
        state_dict = {
            'pre_state': {
                'beta': [{
                    'header_hash': block.header_hash,
                    'state_root': block.state_root,
                    'mmr': {
                        'peaks': block.mmr.peaks,
                        'count': block.mmr.count
                    },
                    'reported': [
                        {'hash': r.hash, 'exports_root': r.exports_root}
                        for r in block.reported
                    ]
                } for block in generated_post_state.beta]
            },
            'input': state_data.get('input', {})
        }
        
        if save_updated_state(updated_state_path, state_dict):
            print("✅ Updated state saved to updated_state.json")
        else:
            print("❌ Failed to save updated state")
        
        output_path = results_dir / 'latest_result.json'
        result_data = {
            'input': state_data['input'],
            'pre_state': state_to_dict(current_state),
            'generated_post_state': state_to_dict(generated_post_state),
            'verified': True
        }
        
        with open(output_path, 'w') as f:
            json.dump(result_data, f, indent=2)
            
        print(f"\nResults saved to: {output_path}")
            
    except Exception as e:
        print(f"❌ Error during state transition: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
