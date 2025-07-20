#!/bin/bash

# Simple Frontend Deployment Script
# Deploys Next.js frontend with optional Redis cache

set -e

echo "🚀 Starting Automatos AI Frontend Deployment..."

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Load environment variables
if [ -f .env ]; then
    echo "📋 Loading environment variables from .env"
    export $(cat .env | grep -v '^#' | xargs)
else
    echo "⚠️  No .env file found. Using default values."
fi

# Create required directories
echo "📁 Creating required directories..."
mkdir -p logs

# Set up Next.js configuration for standalone output
echo "⚙️  Configuring Next.js for production build..."
if [ ! -f next.config.js ]; then
    cat > next.config.js << 'EOF'
/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  experimental: {
    outputFileTracingRoot: undefined,
  },
}

module.exports = nextConfig
EOF
fi

# Build and start frontend
echo "🔨 Building and starting frontend..."
docker-compose down --remove-orphans
docker-compose build --no-cache
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be healthy..."
sleep 30

# Check service health
echo "🔍 Checking service health..."
docker-compose ps

# Test frontend endpoint
echo "🧪 Testing frontend endpoint..."
if curl -f http://localhost:${FRONTEND_PORT:-3000} > /dev/null 2>&1; then
    echo "✅ Frontend is healthy"
else
    echo "❌ Frontend health check failed"
    docker-compose logs frontend
fi

# Test Redis cache if enabled
if docker-compose ps | grep -q redis_cache; then
    echo "🧪 Testing Redis cache..."
    if docker-compose exec -T redis_cache redis-cli --no-auth-warning -a "${REDIS_CACHE_PASSWORD:-redis_cache_123}" ping > /dev/null 2>&1; then
        echo "✅ Redis cache is healthy"
    else
        echo "❌ Redis cache health check failed"
    fi
fi

echo "🎉 Frontend deployment completed!"
echo ""
echo "📊 Service URLs:"
echo "   Frontend: http://localhost:${FRONTEND_PORT:-3000}"
echo "   API Base URL: ${API_BASE_URL:-http://localhost:8001}"
echo ""
echo "📝 Useful commands:"
echo "   View logs: docker-compose logs -f"
echo "   Stop services: docker-compose down"
echo "   Restart services: docker-compose restart"
echo "   Rebuild: docker-compose build --no-cache"
echo "   Enable Redis cache: docker-compose --profile cache up -d" 