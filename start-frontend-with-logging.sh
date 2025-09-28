#!/bin/bash

# Kill any existing frontend processes
pkill -f 'next\|npm' || true
sleep 2

# Clear Next.js cache
cd /root/automatos-ai/frontend
rm -rf .next node_modules/.cache

# Clear and set up log file
cd /root/automatos-ai
rm -f frontend.log
touch frontend.log
chmod 666 frontend.log

# Build the frontend
cd /root/automatos-ai/frontend
echo "Building frontend..."
npm run build >> ../frontend.log 2>&1

# Check build status
if [ 0 -ne 0 ]; then
    echo "Build failed. Check /root/automatos-ai/frontend.log for details."
    exit 1
fi

# Start frontend with enhanced logging
echo "Starting frontend..."
NODE_OPTIONS='--inspect' npm start >> ../frontend.log 2>&1 &

# Display startup information
echo "Frontend started with enhanced logging. Check /root/automatos-ai/frontend.log for details."
echo "Allow a few seconds for startup to complete..."
sleep 5
echo "Recent log entries:"
tail -n 30 /root/automatos-ai/frontend.log
