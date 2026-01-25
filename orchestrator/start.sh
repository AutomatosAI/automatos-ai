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

# Set NLTK data directory
export NLTK_DATA=/usr/local/nltk_data

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

# Install/update dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download NLTK data if not present
echo "📚 Checking NLTK data..."
python -c "import nltk; import os; os.makedirs('/usr/local/nltk_data', exist_ok=True); nltk.download('punkt', quiet=True, download_dir='/usr/local/nltk_data'); nltk.download('stopwords', quiet=True, download_dir='/usr/local/nltk_data')" 2>/dev/null || true

# Start the server with hot reload
echo "🚀 Starting Automatos AI backend on port $PORT..."
echo "   - Hot reload: ENABLED"
echo "   - Access: http://localhost:$PORT"
echo "   - API docs: http://localhost:$PORT/docs"
echo ""
echo "Press CTRL+C to stop"
echo ""

python -m uvicorn main:app --host 0.0.0.0 --port $PORT --reload
