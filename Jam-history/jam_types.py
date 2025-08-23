

from typing import List, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class MMR:
   
    peaks: List[Optional[str]]
    count: Optional[int] = None


@dataclass
class Reported:
  
    hash: str
    exports_root: str


@dataclass
class BetaBlock:
  
    header_hash: str
    state_root: str
    mmr: MMR
    reported: List[Reported]


@dataclass
class Input:
   
    header_hash: str
    parent_state_root: str
    accumulate_root: str
    work_packages: List[Reported]


@dataclass
class State:
    beta: List[BetaBlock]
