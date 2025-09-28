#!/bin/bash

# Log files
LOG_FILE=/root/automatos-ai/frontend.log

# Clear the log file
echo "" > 

# Stop any running frontend services
echo "Stopping existing frontend services..."
pkill -f "npm start" || true
pkill -f "next" || true
sleep 3

# Clear frontend cache 
echo "Clearing frontend cache for clean restart..."
cd /root/automatos-ai/frontend
rm -rf .next/cache

# Set environment variables for debugging
export NODE_ENV=development
export LOG_LEVEL=debug

# Start frontend in development mode
echo "Starting frontend in development mode with debugging..."
cd /root/automatos-ai/frontend
NODE_OPTIONS="--max-old-space-size=4096" npm run dev >  2>&1 &
FRONTEND_PID=
echo "Frontend started with PID: "

# Save PID file for future reference
echo  > /root/automatos-ai/frontend.pid

# Wait a moment
sleep 2

# Output success
echo "✅ Frontend restarted in debug mode!"
echo "🔗 Frontend: http://localhost:3000 or https://ui.automatos.app"
echo ""
echo "📝 Logs: "
echo ""
echo "💡 To monitor logs in real-time:"
echo "  - Frontend: tail -f "
echo ""
echo "⚠️ To stop services: pkill -f \"npm start\" && pkill -f \"next\""
echo ""
