import json
import os
from typing import Dict, Any
from ..types.preimage_types import (
    PreimagesTestVector, PreimagesInput, PreimagesState, PreimagesOutput,
    PreimageInput, PreimagesAccountMapEntry, PreimagesAccountMapData,
    PreimagesMapEntry, LookupMetaMapEntry, LookupMetaMapKey,
    ServicesStatisticsEntry, StatisticsRecord
)


def load_test_vector(folder: str, file_name: str) -> PreimagesTestVector:
    """Load a test vector from JSON file."""
    # Get the directory of this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels to reach the project root, then to test-vectors
    file_path = os.path.join(current_dir, "../../test-vectors", folder, file_name)
    
    with open(file_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    
    return _parse_test_vector(raw_data)


def _parse_test_vector(data: Dict[str, Any]) -> PreimagesTestVector:
    """Parse raw JSON data into PreimagesTestVector."""
    # Parse input
    input_data = data["input"]
    preimages = [
        PreimageInput(requester=p["requester"], blob=p["blob"])
        for p in input_data["preimages"]
    ]
    input_obj = PreimagesInput(preimages=preimages, slot=input_data["slot"])
    
    # Parse pre_state
    pre_state = _parse_state(data["pre_state"])
    
    # Parse output
    output_data = data["output"]
    output_obj = PreimagesOutput(
        ok=output_data.get("ok"),
        err=output_data.get("err")
    )
    
    # Parse post_state
    post_state = _parse_state(data["post_state"])
    
    return PreimagesTestVector(
        input=input_obj,
        pre_state=pre_state,
        output=output_obj,
        post_state=post_state
    )


def _parse_state(state_data: Dict[str, Any]) -> PreimagesState:
    """Parse state data into PreimagesState."""
    # Parse accounts
    accounts = []
    for acc_data in state_data["accounts"]:
        # Parse preimages
        preimages = [
            PreimagesMapEntry(hash=p["hash"], blob=p["blob"])
            for p in acc_data["data"]["preimages"]
        ]
        
        # Parse lookup_meta
        lookup_meta = []
        for lm_data in acc_data["data"]["lookup_meta"]:
            key = LookupMetaMapKey(
                hash=lm_data["key"]["hash"],
                length=lm_data["key"]["length"]
            )
            lookup_meta.append(LookupMetaMapEntry(
                key=key,
                value=lm_data["value"]
            ))
        
        data = PreimagesAccountMapData(
            preimages=preimages,
            lookup_meta=lookup_meta
        )
        
        accounts.append(PreimagesAccountMapEntry(
            id=acc_data["id"],
            data=data
        ))
    
    # Parse statistics
    statistics = []
    for stat_data in state_data["statistics"]:
        record = StatisticsRecord(
            provided_count=stat_data["record"]["provided_count"],
            provided_size=stat_data["record"]["provided_size"],
            refinement_count=stat_data["record"]["refinement_count"],
            refinement_gas_used=stat_data["record"]["refinement_gas_used"],
            imports=stat_data["record"]["imports"],
            exports=stat_data["record"]["exports"],
            extrinsic_size=stat_data["record"]["extrinsic_size"],
            extrinsic_count=stat_data["record"]["extrinsic_count"],
            accumulate_count=stat_data["record"]["accumulate_count"],
            accumulate_gas_used=stat_data["record"]["accumulate_gas_used"],
            on_transfers_count=stat_data["record"]["on_transfers_count"],
            on_transfers_gas_used=stat_data["record"]["on_transfers_gas_used"]
        )
        
        statistics.append(ServicesStatisticsEntry(
            id=stat_data["id"],
            record=record
        ))
    
    return PreimagesState(accounts=accounts, statistics=statistics)
