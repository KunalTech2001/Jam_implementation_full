import json
import os
import sys
import pytest
from copy import deepcopy
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server.server import process_blockchain, init_empty_stats, process_pvm_state
from server.server import load_updated_state, acc_save_state

# Sample test data
SAMPLE_BLOCK = {
    "slot": 12345,
    "author_index": 0,
    "extrinsic": {
        "tickets": [],
        "preimages": [],
        "guarantees": [{"signatures": [{"validator_index": 0}]}],
        "assurances": [{"validator_index": 0}],
        "pvm_operations": [
            {
                "service_id": "test_service",
                "operation": "test_op",
                "accumulate": {
                    "data": "test_data",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
        ]
    }
}

def create_test_state(num_validators=1):
    """Create a test state with the given number of validators."""
    return {
        "slot": 12344,
        "vals_curr_stats": init_empty_stats(num_validators),
        "vals_last_stats": init_empty_stats(num_validators),
        "curr_validators": [{"id": i, "pubkey": f"validator_{i}"} for i in range(num_validators)],
        "pvm_state": {
            "last_processed_slot": 12344,
            "active_services": {},
            "accumulated_items": []
        }
    }

def test_init_empty_stats():
    """Test that init_empty_stats creates the correct structure."""
    stats = init_empty_stats(2)
    assert len(stats) == 2
    assert all(set(stat.keys()) == {
        'blocks', 'tickets', 'pre_images', 'pre_images_size',
        'guarantees', 'assurances', 'pvm_operations', 'pvm_errors', 'pvm_last_operation'
    } for stat in stats)

def test_process_pvm_state_basic():
    """Test basic PVM state processing."""
    state = create_test_state()
    input_data = {
        "slot": 12345,
        "extrinsic": {
            "pvm_operations": [{"service_id": "test_service", "operation": "test_op"}]
        }
    }
    
    updated_state, responses = process_pvm_state(input_data, state)
    
    assert updated_state["pvm_state"]["last_processed_slot"] == 12345
    assert "test_service" in updated_state["pvm_state"]["active_services"]
    assert updated_state["pvm_state"]["active_services"]["test_service"]["operation_count"] == 1

def test_process_blockchain_with_pvm():
    """Test blockchain processing with PVM operations."""
    state = create_test_state()
    block = deepcopy(SAMPLE_BLOCK)
    
    # Process block
    result, new_state = process_blockchain(block, state, False)
    
    # Check results
    assert "ok" in result
    assert new_state["slot"] == 12345
    assert new_state["vals_curr_stats"][0]["pvm_operations"] == 1
    assert "test_service" in new_state["pvm_state"]["active_services"]
    assert new_state["pvm_state"]["active_services"]["test_service"]["operation_count"] == 1

def test_pvm_error_handling():
    """Test that PVM errors are properly handled and tracked."""
    state = create_test_state()
    # Create a malformed operation that will cause an error
    block = deepcopy(SAMPLE_BLOCK)
    # Use a dictionary key that will cause a KeyError when processed
    block["extrinsic"]["pvm_operations"][0] = {"invalid_structure": True}
    
    result, new_state = process_blockchain(block, state, False)
    
    assert "ok" in result  # Block processing should still succeed
    assert new_state["vals_curr_stats"][0]["pvm_errors"] == 1
    assert "pvm_errors" in new_state
    assert len(new_state.get("pvm_errors", [])) > 0

if __name__ == "__main__":
    # Run tests
    pytest.main(["-v", "test_pvm_integration.py"])
