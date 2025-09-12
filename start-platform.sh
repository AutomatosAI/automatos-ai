
#!/bin/bash

echo "🚀 Starting AutomatOS AI Platform..."
echo ""

# Start backend orchestrator in background
echo "📡 Starting Backend Orchestrator on port 8000..."
cd /home/ubuntu/automatos_ai_orchestrator
python main.py &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend
echo "🌐 Starting Frontend on port 3000..."
cd /home/ubuntu/automatos_ai_frontend/frontend
yarn dev &
FRONTEND_PID=$!

echo ""
echo "✅ Platform Started Successfully!"
echo "🔗 Frontend: http://localhost:3000"
echo "🔗 Backend API: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop both services..."

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down platform..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ Platform stopped."
    exit 0
}

# Set trap to cleanup on exit
trap cleanup SIGINT SIGTERM

# Wait for processes
wait

