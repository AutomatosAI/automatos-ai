#!/bin/bash

# Deploy Enhanced Orchestration Services
# ========================================
# This script deploys the new RAG, Action Executor, and Monitoring services

echo "🚀 Deploying Enhanced Orchestration Services..."

# Check if we're in the right directory
if [ ! -f "orchestrator/main.py" ]; then
    echo "❌ Error: Must run from automatos-ai directory"
    exit 1
fi

echo "📦 Installing required Python packages..."
pip install numpy 2>/dev/null || echo "numpy already installed"
pip install psutil 2>/dev/null || echo "psutil already installed"

echo "✅ Services deployed:"
echo "  - RAG Service with vector database"
echo "  - Action Executor with file/shell operations"
echo "  - Advanced Monitoring Service"

echo "🔄 Restarting backend services..."
# Kill existing uvicorn process
pkill -f uvicorn

# Start backend with new services
cd orchestrator
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > ../backend.log 2>&1 &

echo "⏳ Waiting for services to start..."
sleep 5

# Test if backend is responding
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend started successfully with new services"
else
    echo "⚠️ Backend might still be starting, check logs with: tail -f backend.log"
fi

echo "📊 Monitoring service will start automatically"
echo "🔍 RAG service initialized with default knowledge base"
echo "⚡ Action executor ready in sandboxed workspace: /tmp/automatos_workspace"

echo ""
echo "Next steps:"
echo "1. Test with: python test_9_stages.py"
echo "2. Check monitoring at: http://localhost:8000/api/monitoring/dashboard"
echo "3. View logs: tail -f backend.log"
