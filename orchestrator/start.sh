#!/bin/bash
# =============================================================================
# Automatos AI - Local Development Start Script
# =============================================================================
# This script starts the backend in development mode with hot reload
# Usage: ./start.sh [port]
# =============================================================================

set -e

# Get port from argument or use default
PORT=${1:-8000}

# Create necessary directories
mkdir -p logs vector_stores projects exports

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies: the lock the image installs, every pin with its
# hashes (PRD-262 US-022). requirements.in is its input, not an install list.
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install --require-hashes -r requirements.lock

# Start the server with hot reload
echo "🚀 Starting Automatos AI backend on port $PORT..."
echo "   - Hot reload: ENABLED"
echo "   - Access: http://localhost:$PORT"
echo "   - API docs: http://localhost:$PORT/docs"
echo ""
echo "Press CTRL+C to stop"
echo ""

python -m uvicorn main:app --host 0.0.0.0 --port $PORT --reload
