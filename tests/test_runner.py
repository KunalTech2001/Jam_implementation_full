"""
Test runner for JAM protocol test vectors.

This module contains the logic for running test vectors against the JAM protocol implementation.
"""

import json
import os
import sys
from typing import List, Dict, Any

from src.jam.core.safrole_manager import SafroleManager
from src.jam.utils.helpers import deep_clone, deep_equal


def strip_internal_keys(state: Dict[str, Any]) -> Dict[str, Any]:
    """Remove internal keys from state for comparison."""
    internal_keys = ["E", "Y", "e", "m", "N"]
    return {k: v for k, v in state.items() if k not in internal_keys}


def initialize_set(s):
    """Initialize a set if it's None."""
    return s if s is not None else []


def run_test_vector(test_vector_file: str, index: int, total: int, is_full: bool = False) -> bool:
    """Run a single test vector."""
    directory = "full" if is_full else "tiny"
    test_vector_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "Jam",
        "jam-test-vectors",
        "stf",
        "safrole",
        directory,
        test_vector_file,
    )
    print(test_vector_path)
    print(
        f"\n--- Running test vector {index + 1}/{total}: {test_vector_file} ({directory}) ---"
    )

    try:
        with open(test_vector_path, "r") as f:
            test_vector = json.load(f)

        for key in ["lambda", "kappa", "gamma_k", "iota", "gamma_a", "post_offenders"]:
            test_vector["pre_state"][key] = initialize_set(
                test_vector["pre_state"].get(key)
            )

        manager = SafroleManager(test_vector["pre_state"], test_vector_path)

        result_output, post_state = None, None
        try:
            result = manager.process_block(test_vector["input"])
            result_output = {
                "ok": {
                    "epoch_mark": result["header"]["epoch_mark"],
                    "tickets_mark": result["header"]["tickets_mark"],
                }
            }
            post_state = strip_internal_keys(result["post_state"])
        except Exception as e:
            print(f"Caught error: {e}")
            result_output = {"err": str(e)}
            post_state = strip_internal_keys(test_vector["pre_state"])

        expected_output = test_vector["output"]
        expected_post_state = strip_internal_keys(test_vector["post_state"])

        for key in ["lambda", "kappa", "gamma_k", "iota", "gamma_a", "post_offenders"]:
            expected_post_state[key] = initialize_set(expected_post_state.get(key))
        if expected_post_state.get("gamma_s") is None:
            expected_post_state["gamma_s"] = {"keys": []}

        output_matches = deep_equal(result_output, expected_output)
        post_state_matches = deep_equal(post_state, expected_post_state)

        if output_matches and post_state_matches:
            print("--------------------------------")
            print("Output:    ", result_output)
            print(f"✅ Success! Test vector {index + 1} passed: {test_vector_file}")
            return True
        else:
            print(f"❌ FAILED! Test vector {index + 1}: {test_vector_file}")
            if not output_matches:
                print(f"--- Output Mismatch in {test_vector_file} ---")
                print("Expected:", json.dumps(expected_output, indent=2))
                print("Actual:", json.dumps(result_output, indent=2))
            if not post_state_matches:
                print(f"--- Post-State Mismatch in {test_vector_file} ---")
                print("Expected:", json.dumps(expected_post_state, indent=2))
                print("Actual:", json.dumps(post_state, indent=2))
            return False

    except Exception as e:
        print(f"❌ Error running test vector {index + 1} ({test_vector_file}): {e}")
        return False


def run_all_tests(test_files: List[str], is_full: bool = False, fail_fast: bool = False) -> bool:
    """Run all test vectors."""
    all_passed = True
    for idx, test_file in enumerate(test_files):
        try:
            result = run_test_vector(test_file, idx, len(test_files), is_full=is_full)
            if not result:
                all_passed = False
                if fail_fast:
                    print(f"Fail-fast: stopping at {test_file}")
                    break
        except Exception as e:
            print(f"Error running {test_file}: {e}")
            all_passed = False
            if fail_fast:
                print(f"Fail-fast: stopping at {test_file}")
                break
    
    if all_passed:
        print("\nAll test vectors passed! ✅")
    else:
        print("\nSome test vectors failed. ❌")
    
    return all_passed 