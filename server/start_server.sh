#!/bin/bash

# JAM Safrole Integration Server Startup Script

echo "🚀 Starting JAM Safrole Integration Server..."
echo "=============================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed or not in PATH"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Please run this script from the server directory"
    echo "   cd server"
    echo "   ./start_server.sh"
    exit 1
fi

# Check if requirements are installed
if [ ! -d "venv" ] && [ ! -d "../venv" ]; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
fi

# Check if JAM source is accessible
if [ ! -d "../src/jam" ]; then
    echo "❌ JAM source code not found in ../src/jam"
    echo "   Make sure you're running from the correct directory"
    exit 1
fi

echo "✅ Dependencies and source code verified"
echo "🌐 Starting server on http://localhost:8000"
echo "📚 API documentation will be available at:"
echo "   - http://localhost:8000/docs"
echo "   - http://localhost:8000/redoc"
echo ""
echo "Press Ctrl+C to stop the server"
echo "=============================================="

# Start the server
python3 app.py
