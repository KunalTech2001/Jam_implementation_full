from typing import List, Dict, Optional, Union
from dataclasses import dataclass


@dataclass
class StatisticsRecord:
    """Statistics record for service activity."""
    provided_count: int
    provided_size: int
    refinement_count: int
    refinement_gas_used: int
    imports: int
    exports: int
    extrinsic_size: int
    extrinsic_count: int
    accumulate_count: int
    accumulate_gas_used: int
    on_transfers_count: int
    on_transfers_gas_used: int


@dataclass
class LookupMetaMapKey:
    """Key for lookup meta map entry."""
    hash: str
    length: int


@dataclass
class LookupMetaMapEntry:
    """Entry in lookup meta map."""
    key: LookupMetaMapKey
    value: List[int]  # List of timeslots


@dataclass
class PreimagesMapEntry:
    """Entry in preimages map."""
    hash: str
    blob: str


@dataclass
class PreimagesAccountMapData:
    """Data for preimages account map."""
    preimages: List[PreimagesMapEntry]
    lookup_meta: List[LookupMetaMapEntry]


@dataclass
class PreimagesAccountMapEntry:
    """Entry in preimages account map."""
    id: int  # ServiceId
    data: PreimagesAccountMapData


@dataclass
class ServicesStatisticsEntry:
    """Entry for services statistics."""
    id: int
    record: StatisticsRecord


@dataclass
class PreimagesState:
    """State for preimages."""
    accounts: List[PreimagesAccountMapEntry]
    statistics: List[ServicesStatisticsEntry]


@dataclass
class PreimageInput:
    """Input for a single preimage."""
    requester: int
    blob: str


@dataclass
class PreimagesInput:
    """Input for preimages processing."""
    preimages: List[PreimageInput]
    slot: int


@dataclass
class PreimagesOutput:
    """Output for preimages processing."""
    ok: Optional[None] = None
    err: Optional[str] = None


@dataclass
class PreimagesTestVector:
    """Test vector for preimages."""
    input: PreimagesInput
    pre_state: PreimagesState
    output: PreimagesOutput
    post_state: PreimagesState
    name: Optional[str] = None
