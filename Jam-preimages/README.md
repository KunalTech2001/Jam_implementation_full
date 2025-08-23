# JAM Preimages STF Tests - Python Implementation

This is a Python conversion of the TypeScript JAM Preimages STF (State Transition Function) test suite.

## Project Structure

```
Jam-preimages/
├── src/
│   ├── __init__.py
│   ├── index.py              # Main test runner
│   ├── stf/
│   │   ├── __init__.py
│   │   └── run_test.py       # Core preimage test logic
│   ├── types/
│   │   ├── __init__.py
│   │   ├── enums.py          # Error code enumerations
│   │   └── preimage_types.py # Data structures and types
│   └── utils/
│       ├── __init__.py
│       └── json_loader.py    # Test vector JSON loading utilities
├── test-vectors/             # Test data (tiny/ and full/ subdirectories)
├── main.py                   # Entry point script
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## Features

- **Complete Python conversion** of the original TypeScript implementation
- **Type-safe data structures** using Python dataclasses
- **BLAKE2b-256 hashing** using Python's built-in hashlib
- **Comprehensive test validation** including:
  - Duplicate hash detection
  - Sorting validation for requesters and hashes
  - Solicited preimage verification
  - State transition validation

## Requirements

- Python 3.6 or higher (for built-in BLAKE2b support)
- No external dependencies required

## Usage

Run the test suite:

```bash
python main.py
```

Or run directly from the src directory:

```bash
python src/index.py
```

## Test Vectors

The test suite processes JSON test vectors from the `test-vectors/` directory:

- `tiny/` - Basic test cases
- `full/` - Comprehensive test cases

Each test vector includes:
- Input preimages and slot information
- Pre-state and expected post-state
- Expected output (success or error)

## Error Codes

- `preimage_unneeded` (0) - Preimage not requested or already provided
- `preimages_not_sorted_unique` (1) - Invalid ordering or duplicate preimages

## Implementation Details

This Python implementation maintains exact functional equivalence with the original TypeScript version:

1. **State Processing**: Deep copies state for mutation-safe processing
2. **Hash Validation**: Uses BLAKE2b-256 for preimage hash computation  
3. **Sorting Requirements**: Enforces sorting of requesters and hashes
4. **Statistics Updates**: Tracks provided_count and provided_size metrics
5. **Error Handling**: Maps string error codes to numeric enum values

## Test Results

The test runner provides detailed output including:
- ✅ Passed tests
- ❌ Failed tests with detailed error information
- 🧪 Total test count and summary statistics
