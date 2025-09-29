#!/bin/bash

# Local build script for Automatos frontend
set -e

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log "🚀 Starting Frontend Local Build Process"

# Create environment file
log "Creating .env.local with required environment variables..."
cat > .env.local << EOF
NEXT_PUBLIC_API_URL=https://api.automatos.app
NEXT_PUBLIC_WS_URL=wss://api.automatos.app/ws
NEXT_PUBLIC_API_KEY=test_api_key_for_backend_validation_2025
EOF
info "✅ Environment configuration created"

# Install dependencies
log "Installing dependencies..."
npm ci --legacy-peer-deps || npm install --legacy-peer-deps
info "✅ Dependencies installed"

# Clean .next cache
log "Cleaning Next.js cache..."
rm -rf .next
info "✅ Cache cleaned"

# Build the application
log "Building the application..."
npm run build
info "✅ Build completed"

# Start the application
log "Starting the application..."
npm start
