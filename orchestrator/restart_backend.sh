#!/bin/bash
# Restart backend with environment-agnostic credential lookup

echo "🔄 Restarting backend..."

cd ~/automatos-ai/orchestrator

# Kill existing backend
pkill -9 -f uvicorn
sleep 2

# Start backend (use whichever ENVIRONMENT you prefer now!)
export ENVIRONMENT=development  # or production - doesn't matter now!
uvicorn main:app --host 0.0.0.0 --port 8000 > ../backend.log 2>&1 &

echo "✅ Backend restarted (PID: $!)"
echo ""
echo "Monitor startup: tail -f ../backend.log"
echo ""
echo "The credential lookup now works regardless of ENVIRONMENT setting!"

