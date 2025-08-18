#!/usr/bin/env python3
"""
JAM Protocol Test Runner

Main script for running JAM protocol test vectors.
This script provides a command-line interface for running test vectors
against the JAM protocol implementation.
"""

import argparse
import sys
from typing import List

from tests.test_runner import run_all_tests


TINY_TEST_VECTOR_FILES = [
    "enact-epoch-change-with-no-tickets-1.json",
    "enact-epoch-change-with-no-tickets-2.json",
    "enact-epoch-change-with-no-tickets-3.json",
    "enact-epoch-change-with-no-tickets-4.json",
    "enact-epoch-change-with-padding-1.json",
    "publish-tickets-no-mark-1.json",
    "publish-tickets-no-mark-2.json",
    "publish-tickets-no-mark-3.json",
    "publish-tickets-no-mark-4.json",
    "publish-tickets-no-mark-5.json",
    "publish-tickets-no-mark-6.json",
    "publish-tickets-no-mark-7.json",
    "publish-tickets-no-mark-8.json",
    "publish-tickets-no-mark-9.json",
    "publish-tickets-with-mark-1.json",
    "publish-tickets-with-mark-2.json",
    "publish-tickets-with-mark-3.json",
    "publish-tickets-with-mark-4.json",
    "publish-tickets-with-mark-5.json",
    "skip-epochs-1.json",
    "skip-epoch-tail-1.json",
]

FULL_TEST_VECTOR_FILES = [
    "enact-epoch-change-with-no-tickets-1.json",
     "enact-epoch-change-with-no-tickets-2.json",
    "enact-epoch-change-with-no-tickets-3.json",
    "enact-epoch-change-with-no-tickets-4.json",
    "enact-epoch-change-with-padding-1.json",
    "publish-tickets-no-mark-1.json",
    "publish-tickets-no-mark-2.json",
    "publish-tickets-no-mark-3.json",
    "publish-tickets-no-mark-4.json",
    "publish-tickets-no-mark-5.json",
    "publish-tickets-no-mark-6.json",
    "publish-tickets-no-mark-7.json",
    "publish-tickets-no-mark-8.json",
    "publish-tickets-no-mark-9.json",
    "publish-tickets-with-mark-1.json",
    "publish-tickets-with-mark-2.json",
    "publish-tickets-with-mark-3.json",
    "publish-tickets-with-mark-4.json",
    "publish-tickets-with-mark-5.json",
    "skip-epochs-1.json",
    "skip-epoch-tail-1.json"
]


def main():
    """Main entry point for the JAM protocol test runner."""
    parser = argparse.ArgumentParser(description="Run JAM protocol test vectors.")
    parser.add_argument(
        "--full", action="store_true", help="Run full test vectors instead of tiny ones."
    )
    parser.add_argument(
        "--fail-fast", action="store_true", help="Stop on first failure."
    )
    args = parser.parse_args()

    if args.full:
        test_files = FULL_TEST_VECTOR_FILES
        is_full = True
        print("Running FULL test vectors...")
    else:
        test_files = TINY_TEST_VECTOR_FILES
        is_full = False
        print("Running TINY test vectors...")

    success = run_all_tests(test_files, is_full=is_full, fail_fast=args.fail_fast)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 