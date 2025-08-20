# JAM Safrole Integration Server

A FastAPI-based REST server that integrates with the JAM protocol safrole component, providing state management and block processing capabilities.

## Overview

This server acts as a bridge between external systems and the JAM protocol's safrole component. It allows you to:

- Initialize the safrole manager with pre_state data
- Process blocks and update the protocol state
- Monitor the current state of the system
- Reset the manager when needed

## Features

- **RESTful API**: Clean HTTP endpoints for all operations
- **State Management**: Persistent safrole manager state across requests
- **Input Validation**: Pydantic models ensure data integrity
- **Error Handling**: Comprehensive error handling and logging
- **CORS Support**: Cross-origin request support for web applications
- **Health Monitoring**: Built-in health check endpoints

## API Endpoints

### Core Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Server information and status |
| `GET` | `/health` | Health check and safrole status |
| `POST` | `/initialize` | Initialize safrole manager with pre_state |
| `POST` | `/process-block` | Process a block and update state |
| `GET` | `/state` | Get current safrole manager state |
| `POST` | `/reset` | Reset safrole manager to uninitialized state |

### Data Models

#### BlockInput
```json
{
  "slot": 1,
  "entropy": "0x8c2e6d327dfaa6ff8195513810496949210ad20a96e2b0672a3e1b9335080801",
  "extrinsic": []
}
```

#### PreState
```json
{
  "tau": 0,
  "eta": ["0x...", "0x...", "0x...", "0x..."],
  "lambda": [...],
  "kappa": [...],
  "gamma_k": [...],
  "gamma_z": "0x...",
  "iota": [...],
  "gamma_a": [],
  "gamma_s": {...},
  "post_offenders": []
}
```

#### StateRequest
```json
{
  "input": { /* BlockInput */ },
  "pre_state": { /* PreState */ }
}
```

## Installation

### Prerequisites

- Python 3.8+
- Access to the JAM protocol source code

### Setup

1. **Navigate to the server directory:**
   ```bash
   cd server
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the JAM source is accessible:**
   The server expects the JAM source code to be in the `../src/` directory relative to the server folder.

## Usage

### Starting the Server

```bash
cd server
# python app.py
python3 -m server.app
```

The server will start on `http://localhost:8000` with auto-reload enabled.

### API Documentation

Once running, you can access:
- **Interactive API docs**: `http://localhost:8000/docs`
- **Alternative API docs**: `http://localhost:8000/redoc`

### Basic Workflow

1. **Initialize the safrole manager:**
   ```bash
   curl -X POST "http://localhost:8000/initialize" \
        -H "Content-Type: application/json" \
        -d @your_pre_state.json
   ```

2. **Process a block:**
   ```bash
   curl -X POST "http://localhost:8000/process-block" \
        -H "Content-Type: application/json" \
        -d @block_data.json
   ```

3. **Check current state:**
   ```bash
   curl "http://localhost:8000/state"
   ```

4. **Reset when needed:**
   ```bash
   curl -X POST "http://localhost:8000/reset"
   ```

### Client Example

A complete client example is provided in `client_example.py` that demonstrates the full workflow:

```bash
python client_example.py
```

## Configuration

### Environment Variables

The server can be configured using environment variables:

- `HOST`: Server host (default: `0.0.0.0`)
- `PORT`: Server port (default: `8000`)
- `LOG_LEVEL`: Logging level (default: `info`)

### Server Settings

The server is configured with:
- CORS enabled for all origins
- JSON response format
- Comprehensive error handling
- Request/response validation

## Architecture

### Components

1. **FastAPI Application**: Main server framework
2. **SafroleManager**: JAM protocol integration
3. **Pydantic Models**: Data validation and serialization
4. **Request Handlers**: API endpoint logic
5. **Error Handlers**: Global exception management

### State Management

The server maintains a global `safrole_manager` instance that:
- Persists across requests
- Can be reinitialized with new pre_state data
- Maintains the current JAM protocol state
- Handles all state transitions

### Data Flow

1. **Initialization**: Client sends pre_state → Server creates SafroleManager
2. **Block Processing**: Client sends block data → Server processes through safrole
3. **State Retrieval**: Client requests state → Server returns current state
4. **Reset**: Client requests reset → Server clears manager instance

## Error Handling

### HTTP Status Codes

- `200`: Success
- `400`: Bad request (validation errors, not initialized)
- `500`: Internal server error

### Error Response Format

```json
{
  "success": false,
  "message": "Error description",
  "error": "Detailed error information"
}
```

### Common Errors

- **Not Initialized**: Call `/initialize` before other operations
- **Validation Errors**: Check your JSON data format
- **Processing Errors**: Verify block data and state consistency

## Development

### Project Structure

```
server/
├── __init__.py          # Package initialization
├── app.py              # Main FastAPI application
├── client_example.py   # Example client implementation
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

### Adding New Endpoints

1. Define Pydantic models for request/response
2. Create the endpoint function with proper error handling
3. Add the route to the FastAPI app
4. Update this README

### Testing

The server includes comprehensive error handling and validation. Test with:

- Valid data to ensure success paths
- Invalid data to verify error handling
- Edge cases like uninitialized state

## Integration Examples

### Python Client

```python
import requests

# Initialize
response = requests.post("http://localhost:8000/initialize", json=data)
if response.status_code == 200:
    print("Initialized successfully")

# Process block
response = requests.post("http://localhost:8000/process-block", json=block_data)
if response.status_code == 200:
    result = response.json()
    print(f"Block processed: {result['data']['header']['slot']}")
```

### JavaScript/Node.js Client

```javascript
const axios = require('axios');

// Initialize
const initResponse = await axios.post('http://localhost:8000/initialize', data);
console.log('Initialized:', initResponse.data);

// Process block
const processResponse = await axios.post('http://localhost:8000/process-block', blockData);
console.log('Processed:', processResponse.data);
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Initialize with file
curl -X POST http://localhost:8000/initialize \
     -H "Content-Type: application/json" \
     -d @pre_state.json

# Get current state
curl http://localhost:8000/state
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure JAM source is in `../src/` directory
2. **Port Conflicts**: Change port in `app.py` if 8000 is busy
3. **CORS Issues**: Check browser console for CORS errors
4. **State Errors**: Verify pre_state data format matches expected schema

### Debug Mode

Enable debug logging by setting `LOG_LEVEL=debug` in the environment or modifying the logging configuration in `app.py`.

### Logs

The server logs all operations to stdout. Check the console output for:
- Initialization messages
- Block processing details
- Error information
- Request/response summaries

## Contributing

1. Follow the existing code style
2. Add proper error handling for new features
3. Update documentation for API changes
4. Test with various input scenarios

## License

This server is part of the JAM protocol implementation and follows the same licensing terms.

## Support

For issues related to:
- **Server functionality**: Check logs and error messages
- **JAM protocol**: Refer to the main JAM documentation
- **API usage**: Review the client examples and API docs

---

**Note**: This server is designed to work with the JAM protocol safrole component. Ensure you have the correct version of the JAM source code and understand the protocol's requirements before deployment.
