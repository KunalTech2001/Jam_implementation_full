# JAM Python Implementation

A Python implementation of the JAM (Joint Accumulator Mechanism) protocol, featuring a safrole component for state management and block processing.

## Project Structure

```
Jam_python/
├── config/                 # Configuration files
├── docs/                   # Documentation
├── examples/               # Basic usage examples
├── server/                 # 🆕 JAM Safrole Integration Server
│   ├── app.py             # FastAPI server application
│   ├── client_example.py  # Example client implementation
│   ├── test_server.py     # Server testing script
│   ├── sample_data.json   # Sample data for testing
│   ├── start_server.sh    # Server startup script
│   ├── requirements.txt   # Server dependencies
│   └── README.md          # Server documentation
├── src/                    # Core JAM implementation
│   └── jam/
│       ├── core/          # Core protocol logic
│       │   └── safrole_manager.py
│       ├── protocols/     # Protocol implementations
│       └── utils/         # Utility functions
├── tests/                  # Test suite
├── main.py                 # Main application entry point
├── requirements.txt        # Project dependencies
└── setup.py               # Package setup
```

## Features

- **Core JAM Protocol**: Implementation of the Joint Accumulator Mechanism
- **Safrole Component**: State management and block processing
- **Integration Server**: 🆕 REST API server for external integration
- **Crypto Bridge**: Cryptographic operations and verification
- **Fallback Conditions**: Protocol fallback mechanisms
- **Comprehensive Testing**: Unit and integration tests

## 🆕 JAM Safrole Integration Server

The project now includes a **FastAPI-based REST server** that provides a clean interface to the JAM protocol's safrole component. This server allows external systems to:

- Initialize the safrole manager with pre_state data
- Process blocks and update protocol state
- Monitor current system state
- Reset the manager when needed

### Quick Start with Server

1. **Navigate to server directory:**
   ```bash
   cd server
   ```

2. **Install server dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Start the server:**
   ```bash
   python app.py
   # Or use the startup script:
   ./start_server.sh
   ```

4. **Access API documentation:**
   - Interactive docs: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc

5. **Test the server:**
   ```bash
   python test_server.py
   ```

### Server API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Server information |
| `GET` | `/health` | Health check |
| `POST` | `/initialize` | Initialize safrole manager |
| `POST` | `/process-block` | Process a block |
| `GET` | `/state` | Get current state |
| `POST` | `/reset` | Reset manager |

For detailed server documentation, see [server/README.md](server/README.md).

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd Jam_python
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install server dependencies (optional):**
   ```bash
   cd server
   pip install -r requirements.txt
   cd ..
   ```

## Usage

### Basic JAM Protocol Usage

```python
from src.jam.core.safrole_manager import SafroleManager

# Initialize with pre_state
manager = SafroleManager(pre_state_data)

# Process a block
result = manager.process_block(block_input)
```

### Server Integration Usage

```python
import requests

# Initialize safrole manager
response = requests.post("http://localhost:8000/initialize", json=pre_state_data)

# Process blocks
response = requests.post("http://localhost:8000/process-block", json=block_data)

# Get current state
response = requests.get("http://localhost:8000/state")
```

## Examples

### Running Examples

```bash
# Basic usage examples
python examples/basic_usage.py

# Server client example
cd server
python client_example.py
```

### Testing

```bash
# Run all tests
python -m pytest tests/

# Test the server
cd server
python test_server.py
```

## Configuration

The project uses YAML configuration files in the `config/` directory. See `config/config.yaml` for available options.

## Development

### Project Structure

- **`src/jam/core/`**: Core protocol implementation
- **`src/jam/protocols/`**: Protocol-specific implementations
- **`src/jam/utils/`**: Utility functions and crypto bridge
- **`server/`**: Integration server and API
- **`tests/`**: Comprehensive test suite

### Adding New Features

1. Follow the existing code structure
2. Add appropriate tests
3. Update documentation
4. Ensure compatibility with the integration server

## API Reference

### Core Classes

- **`SafroleManager`**: Main JAM protocol manager
- **`CryptoBridge`**: Cryptographic operations
- **`FallbackCondition`**: Protocol fallback logic

### Server Models

- **`BlockInput`**: Block input data structure
- **`PreState`**: Protocol pre-state structure
- **`StateRequest`**: Complete state request
- **`StateResponse`**: Standardized response format

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the same terms as the JAM protocol specification.

## Support

For questions and support:
- Check the documentation in `docs/`
- Review the examples in `examples/`
- Test with the integration server
- Open an issue for bugs or feature requests

---

**Note**: The integration server requires the JAM source code to be accessible in the `src/` directory. Ensure proper setup before running the server. 