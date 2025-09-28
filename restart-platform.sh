#!/bin/bash

# Log files
BACKEND_LOG="/root/automatos-ai/backend.log"
FRONTEND_LOG="/root/automatos-ai/frontend.log"
DEBUG_LOG="/root/automatos-ai/debug.log"

# Function to timestamp log entries
timestamp() {
  date "+%Y-%m-%d %H:%M:%S"
}

# Function to log messages
log() {
  echo "$(timestamp) - $1" | tee -a "$DEBUG_LOG"
}

# Save environment variables for the frontend
export NODE_ENV=${NODE_ENV:-production}
export NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL:-"http://localhost:8000"}
export LOG_LEVEL=${LOG_LEVEL:-"info"}
export LOG_FILE_PATH="/root/automatos-ai/frontend.log"

# Print startup banner
echo "======================================================"
echo "🚀 Automatos AI Platform - Restart Script"
echo "======================================================"
log "Starting restart procedure"

# Create fresh log files
> "$BACKEND_LOG"
> "$FRONTEND_LOG"
> "$DEBUG_LOG"

echo "$(timestamp) - Log files initialized" >> "$DEBUG_LOG"

# Stop any running services
log "Stopping existing services"
pkill -f "npm start" || true
pkill -f "next" || true
pkill -f uvicorn || true
sleep 3

# Clear frontend cache if needed
if [ "$CLEAR_CACHE" = "true" ]; then
  log "Clearing frontend cache"
  cd /root/automatos-ai/frontend
  rm -rf .next node_modules/.cache
  log "Cache cleared"
fi

# Start backend
log "Starting backend service"
cd /root/automatos-ai/orchestrator
log "Running: uvicorn main:app --host 0.0.0.0 --port 8000"
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > "$BACKEND_LOG" 2>&1 &
BACKEND_PID=$!
log "Backend started with PID: $BACKEND_PID"

# Wait for backend to initialize
sleep 5
log "Checking if backend is running"

# Verify backend is running
if ps -p $BACKEND_PID > /dev/null; then
  log "✅ Backend running successfully"
else
  log "❌ Backend failed to start - check $BACKEND_LOG for details"
  tail -n 20 "$BACKEND_LOG" >> "$DEBUG_LOG"
  exit 1
fi

# Test backend API connection
log "Testing backend API connection"
curl -s http://localhost:8000/health > /dev/null
if [ $? -eq 0 ]; then
  log "✅ Backend API responding"
else
  log "❌ Backend API not responding - check $BACKEND_LOG for details"
  exit 1
fi

# Build frontend if needed
if [ "$REBUILD_FRONTEND" = "true" ]; then
  log "Building frontend"
  cd /root/automatos-ai/frontend
  npm run build >> "$DEBUG_LOG" 2>&1
  log "Frontend build complete"
fi

# Start frontend
log "Starting frontend service"
cd /root/automatos-ai/frontend
log "Running: npm start"
NODE_OPTIONS="--max-old-space-size=4096" nohup npm start > "$FRONTEND_LOG" 2>&1 &
FRONTEND_PID=$!
log "Frontend started with PID: $FRONTEND_PID"

# Wait for frontend to initialize
sleep 10
log "Checking if frontend is running"

# Verify frontend is running
if ps -p $FRONTEND_PID > /dev/null; then
  log "✅ Frontend running successfully"
else
  log "❌ Frontend failed to start - check $FRONTEND_LOG for details"
  tail -n 20 "$FRONTEND_LOG" >> "$DEBUG_LOG"
  exit 1
fi

# Run API diagnostic tool
log "Running API diagnostic tests"
cd /root/automatos-ai/frontend
node scripts/api-debug.js >> "$DEBUG_LOG" 2>&1
API_TEST_RESULT=$?

if [ $API_TEST_RESULT -eq 0 ]; then
  log "✅ API diagnostic tests passed"
else
  log "⚠️ Some API diagnostic tests failed - check $DEBUG_LOG for details"
fi

# Save PID files for future reference
echo $BACKEND_PID > /root/automatos-ai/backend.pid
echo $FRONTEND_PID > /root/automatos-ai/frontend.pid

# Final success message
log "✅ Automatos AI Platform restart process completed!"
log "🔗 Frontend: http://localhost:3000 or https://ui.automatos.app"
log "🔗 Backend API: http://localhost:8000 or http://206.81.0.227:8000"

# Output success to console with color
echo -e "\e[32m✅ Platform restarted!\e[0m"
echo "🔗 Frontend: http://localhost:3000 or https://ui.automatos.app"
echo "🔗 Backend API: http://localhost:8000 or http://206.81.0.227:8000"
echo ""
echo "📝 Logs:"
echo "  - Frontend: $FRONTEND_LOG"
echo "  - Backend: $BACKEND_LOG"
echo "  - Debug: $DEBUG_LOG"
echo ""
echo "💡 To monitor logs in real-time:"
echo "  - Frontend: tail -f $FRONTEND_LOG"
echo "  - Backend: tail -f $BACKEND_LOG"
echo "  - Debug: tail -f $DEBUG_LOG"
echo ""
echo "⚠️ To stop services: pkill -f \"npm start\" && pkill -f \"next\" && pkill -f uvicorn"
echo ""
