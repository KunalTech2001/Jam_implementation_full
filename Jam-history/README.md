# JAM History STF - Python Implementation

This is a Python conversion of the TypeScript JAM History State Transition Function implementation. It maintains all the same functionality and logic as the original TypeScript version.

## Project Structure

- `types.py` - Type definitions (dataclasses equivalent to TypeScript interfaces)
- `history_stf.py` - Main HistorySTF implementation with Keccak-256 and MMR functions
- `normalize.py` - Utility for normalizing objects by removing 'count' fields
- `test.py` - Test runner that processes test vectors from the `tiny/` directory
- `requirements.txt` - Python dependencies
- `tiny/` - Test vectors (JSON files)
- `results/` - Test output results (created when tests run)

## Setup

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Run the tests:
```bash
python test.py
```

## Features

- **Keccak-256 Hashing**: Uses pycryptodome for reliable Keccak-256 implementation (JAM protocol compatible)
- **MMR (Merkle Mountain Range)**: Implements JAM protocol MMR append function
- **State Transition Function**: Processes beta blocks and maintains global MMR state
- **Test Vector Support**: Runs against provided test vectors in the `tiny/` directory
- **Result Verification**: Compares output against expected results and reports pass/fail

## Dependencies

- `pycryptodome`: For Keccak-256 hash function (JAM protocol requirement)

## Test Vectors

The test vectors are located in the `tiny/` directory and contain:
- Input data for state transitions
- Expected post-state results
- Pre-state data (when applicable)

Results are written to the `results/` directory with verification status.
