
import hashlib
from typing import Optional, Tuple
from Crypto.Hash import keccak

from jam_types import Input, State, MMR, BetaBlock


def keccak256(data: bytes) -> str:
   
    try:
       
        keccak_hash = keccak.new(digest_bits=256)
        keccak_hash.update(data)
        return '0x' + keccak_hash.hexdigest()
    except Exception as e:
    
        print(f'CRITICAL: Keccak-256 hash function failed: {e}')
        print(' Keccak-256, not SHA-256. Tests will fail.')
        print('Falling back to SHA-256 - test failures!')
        
      
        hash_obj = hashlib.sha256()
        hash_obj.update(data)
        return '0x' + hash_obj.hexdigest()


def mmr_append(prev: Optional[MMR], leaf: str) -> MMR:
   
    peaks = prev.peaks.copy() if prev else []
    count = prev.count if prev and prev.count is not None else 0
    
    hash_val = leaf
    height = 0
    
   
    while (count & (1 << height)) != 0:
        if height < len(peaks) and peaks[height] is not None:
          
            peak_bytes = bytes.fromhex(peaks[height][2:])  
            hash_bytes = bytes.fromhex(hash_val[2:])      
            combined = peak_bytes + hash_bytes
            hash_val = keccak256(combined)
            peaks[height] = None
        height += 1
    
    #
    while len(peaks) <= height:
        peaks.append(None)
    peaks[height] = hash_val
    
    return MMR(peaks=peaks, count=count + 1)


def update_mmr(prev: Optional[MMR], leaf: str) -> MMR:
  
    return mmr_append(prev, leaf)


def compute_state_root(peaks: list) -> str:
    """
    Compute the state root from MMR peaks by hashing them together.
    If there are no peaks, return a zero hash.
    """
    if not peaks:
        return '0x' + '00' * 32
    
    # Sort and dedupe peaks to ensure consistent ordering
    unique_peaks = sorted(list(set(peaks)))
    
    # If there's only one peak, return it directly
    if len(unique_peaks) == 1:
        return unique_peaks[0]
    
    # Hash all peaks together
    hasher = hashlib.sha256()
    for peak in unique_peaks:
        # Remove '0x' prefix if present and decode hex to bytes
        peak_bytes = bytes.fromhex(peak[2:]) if peak.startswith('0x') else peak.encode()
        hasher.update(peak_bytes)
    
    return '0x' + hasher.hexdigest()


class HistorySTF:
   
    
    @staticmethod
    def transition(pre_state: State, input_data: Input) -> dict:
     
        beta = []
        for block in pre_state.beta:
            new_block = BetaBlock(
                header_hash=block.header_hash,
                state_root=block.state_root,
                mmr=MMR(
                    peaks=block.mmr.peaks.copy(),
                    count=block.mmr.count
                ),
                reported=[
                    type(r)(hash=r.hash, exports_root=r.exports_root) 
                    for r in block.reported
                ]
            )
            beta.append(new_block)
        
        
        if len(beta) > 0:
         
            total_count = len(beta)
            global_mmr = MMR(peaks=[], count=total_count)
            
          
            for i in range(len(beta)):
              
                if i == len(beta) - 1:
                    global_mmr = MMR(
                        peaks=beta[i].mmr.peaks.copy(),
                        count=beta[i].mmr.count or total_count
                    )
        else:
            global_mmr = MMR(peaks=[], count=0)
        
      
        global_mmr = update_mmr(global_mmr, input_data.accumulate_root)
        
     
        if len(beta) > 0:
            beta[-1].state_root = input_data.parent_state_root
        
        # Ensure we have valid peaks (filter out None values and ensure they're strings)
        valid_peaks = [p for p in global_mmr.peaks if p is not None]
        
        # Create the new block with the current state root
        new_block = BetaBlock(
            header_hash=input_data.header_hash,
            mmr=MMR(peaks=valid_peaks, count=global_mmr.count),
            state_root=input_data.accumulate_root,  # Use accumulate_root as the initial state root
            reported=input_data.work_packages.copy()
        )
        
        # Compute the new state root based on the new block's data
        # Ensure we have at least an empty string if no peaks are available
        peaks_str = ''.join(valid_peaks) if valid_peaks else ''
        state_root_data = f"{new_block.header_hash}{peaks_str}".encode()
        state_root = '0x' + hashlib.sha256(state_root_data).hexdigest()
        new_block.state_root = state_root
        
        beta.append(new_block)
        while len(beta) > 8:
            beta.pop(0)
        
        return {
            'postState': State(beta=beta),
            'output': None
        }
