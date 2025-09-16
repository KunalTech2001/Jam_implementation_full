# JAM Authorization-PVM Integration

## Overview

This document describes the complete integration between the JAM authorization component and the Polkadot Virtual Machine (PVM). The integration has been fully implemented with proper Ed25519 signatures, SCALE encoding, and state synchronization.

## Integration Status: ✅ COMPLETE

The authorization component is now **properly integrated** with the PVM. All major integration gaps have been resolved:

### ✅ Fixed Issues

1. **Signature Algorithm Mismatch**: Fixed NaCl vs Ed25519 incompatibility
2. **SCALE Encoding**: Implemented proper SCALE codec for PVM communication
3. **State Synchronization**: Fixed state management between server and PVM
4. **Nonce Management**: Standardized nonce handling across components

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Authorization │    │     Server      │    │      PVM        │
│   Component     │◄──►│   (Port 8000)   │◄──►│   (Port 8080)   │
│                 │    │                 │    │                 │
│ • STF Logic     │    │ • Ed25519 Auth  │    │ • Rust Auth     │
│ • Test Vectors  │    │ • SCALE Codec   │    │ • Signature     │
│ • Full Cycle    │    │ • State Sync    │    │   Verification  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Key Components

### 1. Authorization Integrator (`server/auth_integration.py`)
- **Purpose**: Handles Ed25519 signatures and SCALE encoding
- **Features**:
  - Ed25519 keypair generation (deterministic and random)
  - Payload signing with Ed25519
  - SCALE encoding for PVM communication
  - State synchronization with `updated_state.json`
  - Nonce management

### 2. Updated Server (`server/server.py`)
- **Endpoint**: `POST /authorize`
- **Features**:
  - Ed25519 signature verification
  - Direct PVM integration via `AuthorizationIntegrator`
  - Proper error handling and fallbacks

### 3. PVM Authorizer (`jam_pvm/src/authorizer.rs`)
- **Purpose**: Rust-based authorization verification
- **Features**:
  - Ed25519 signature verification with `ed25519_dalek`
  - SCALE decoding of authorization credentials
  - State persistence to `../server/updated_state.json`

### 4. Updated Test Files
- **`server/test_ed25519_auth.py`**: Comprehensive integration test
- **`authorizations/full_cycle_test.py`**: Updated with new integration
- **`server/test_auth_with_payload.py`**: Legacy NaCl test (still works)

## Usage

### Prerequisites

1. Install dependencies:
```bash
cd server
pip install -r requirements.txt
```

2. Build PVM:
```bash
cd jam_pvm
cargo build --release
```

### Running the Integration

1. **Start PVM Server**:
```bash
cd jam_pvm
cargo run
# PVM runs on http://127.0.0.1:8080
```

2. **Start JAM Server**:
```bash
cd server
python server.py
# Server runs on http://127.0.0.1:8000
```

3. **Run Integration Tests**:
```bash
# Comprehensive Ed25519 test
cd server
python test_ed25519_auth.py

# Full cycle test with STF
cd authorizations
python full_cycle_test.py
```

### API Usage

#### Authorization Request
```bash
curl -X POST http://127.0.0.1:8000/authorize \
  -H "Content-Type: application/json" \
  -d '{
    "public_key": "0x...",
    "signature": "0x...",
    "payload": {
      "service_id": 1,
      "action": "test",
      "data": "test_payload"
    }
  }'
```

#### Response
```json
{
  "success": true,
  "message": "Authorization successful",
  "auth_output": "0x...",
  "pvm_response": {
    "output_hex": "..."
  },
  "nonce": 1,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Testing

### 1. Unit Tests
- **Ed25519 Operations**: Keypair generation, signing, verification
- **SCALE Encoding**: Authorization credentials and work packages
- **State Management**: Nonce handling and state persistence

### 2. Integration Tests
- **Server-PVM Communication**: Full authorization flow
- **Fallback Mechanisms**: Server endpoint → Direct PVM
- **Error Handling**: Invalid signatures, network failures

### 3. STF Tests
- **State Transition Function**: Authorization pool and queue management
- **Test Vectors**: Validation against expected outcomes

## Configuration

### Environment Variables
- `PVM_URL`: PVM server URL (default: `http://127.0.0.1:8080`)
- `SERVER_URL`: JAM server URL (default: `http://127.0.0.1:8000`)

### State Files
- `server/updated_state.json`: Shared state between server and PVM
- `authorizations/state.json`: Authorization component state

## Troubleshooting

### Common Issues

1. **PVM Connection Failed**
   - Ensure PVM server is running on port 8080
   - Check firewall settings

2. **Signature Verification Failed**
   - Verify Ed25519 keypair generation
   - Check payload encoding (must be consistent)

3. **Nonce Mismatch**
   - State file corruption - delete `updated_state.json` to reset
   - Multiple concurrent requests - implement proper locking

### Debug Commands

```bash
# Test Ed25519 operations only
python -c "
from auth_integration import AuthorizationIntegrator
integrator = AuthorizationIntegrator()
pub, priv = integrator.create_ed25519_keypair()
print(f'Public: {pub}')
print(f'Private: {priv}')
"

# Test PVM connectivity
curl -X POST http://127.0.0.1:8080/authorizer/is_authorized \
  -H "Content-Type: application/json" \
  -d '{"param_hex": "00", "package_hex": "00", "core_index_hex": "00000000"}'
```

## Dependencies

### Python Dependencies
- `cryptography>=41.0.0`: Ed25519 operations
- `httpx>=0.25.0`: Async HTTP client
- `substrate-interface>=1.7.0`: Substrate keypairs
- `scalecodec>=1.2.0`: SCALE encoding/decoding

### Rust Dependencies
- `ed25519-dalek`: Ed25519 signature verification
- `parity-scale-codec`: SCALE encoding/decoding
- `serde_json`: JSON serialization

## Security Considerations

1. **Private Key Management**: Never log or expose private keys
2. **Nonce Replay Protection**: Nonces must be sequential and unique
3. **Signature Validation**: Always verify signatures before processing
4. **State Integrity**: Protect `updated_state.json` from unauthorized access

## Future Improvements

1. **Database Integration**: Replace JSON files with proper database
2. **Rate Limiting**: Implement request rate limiting
3. **Monitoring**: Add metrics and health checks
4. **Load Balancing**: Support multiple PVM instances

## Conclusion

The JAM authorization component is now fully integrated with the PVM using proper Ed25519 signatures, SCALE encoding, and state synchronization. The integration supports both server-mediated and direct PVM authorization flows with comprehensive error handling and fallback mechanisms.
