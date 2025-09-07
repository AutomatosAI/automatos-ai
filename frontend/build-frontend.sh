#!/bin/bash

# Frontend Build Script - Repeatable Docker Compose Build
# Run this on the frontend server (can be same or different server)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

log "🚀 Starting Frontend Build Process"

# Apply API configuration fix
log "Applying API configuration fix..."
if [ -f lib/api.ts ]; then
    cp lib/api.ts lib/api.ts.backup
    # Fix the double /api issue
    sed -i 's/prefix: overrides.prefix || (globalThis as any)?.process?.env?.NEXT_PUBLIC_API_PREFIX || "\/api"/prefix: overrides.prefix !== undefined ? overrides.prefix : ((globalThis as any)?.process?.env?.NEXT_PUBLIC_API_PREFIX !== undefined ? (globalThis as any)?.process?.env?.NEXT_PUBLIC_API_PREFIX : "\/api")/g' lib/api.ts
    info "✅ API configuration fix applied"
else
    error "❌ lib/api.ts not found"
    exit 1
fi

# Create correct environment file
log "Creating environment configuration..."
cat > .env.local << EOF
NEXT_PUBLIC_API_URL=https://ui.automatos.app
NEXT_PUBLIC_API_PREFIX=
NEXT_PUBLIC_API_KEY=change_me
NEXT_PUBLIC_AUTH_TOKEN=
NEXT_PUBLIC_TENANT_ID=
EOF
info "✅ Environment configuration created"

# Create docker-compose for frontend
log "Creating frontend docker-compose..."
cat > docker-compose.frontend.yml << EOF
version: '3.8'

services:
  frontend:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: automatos_frontend
    environment:
      - NEXT_PUBLIC_API_URL=https://ui.automatos.app
      - NEXT_PUBLIC_API_PREFIX=
      - NEXT_PUBLIC_API_KEY=change_me
      - NODE_ENV=production
    ports:
      - "3000:3000"
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:3000"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
EOF

# Stop existing containers
log "Stopping existing frontend containers..."
docker compose -f docker-compose.frontend.yml down || true

# Clean up old images
log "Cleaning up old images..."
docker system prune -f

# Build frontend
log "Building frontend..."
docker compose -f docker-compose.frontend.yml build --no-cache

# Start frontend
log "Starting frontend..."
docker compose -f docker-compose.frontend.yml up -d

# Verify service
log "Verifying frontend service..."
sleep 20
docker compose -f docker-compose.frontend.yml ps

# Health check
log "Running health check..."
if curl -f http://localhost:3000 >/dev/null 2>&1; then
    log "✅ Frontend health check passed"
else
    error "❌ Frontend health check failed"
    docker compose -f docker-compose.frontend.yml logs
    exit 1
fi

log "🎉 Frontend build completed successfully!"
log "Frontend running on: http://localhost:3000"
