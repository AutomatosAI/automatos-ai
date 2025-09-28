#!/bin/bash

# Stop any existing Node.js processes
pkill -9 -f 'next' || true
sleep 2

# Clear cache
rm -rf .next node_modules/.cache

# Set up fresh log file
cd ..
> frontend.log
chmod 666 frontend.log
cd frontend

# Start Next.js with debug flags
export NODE_OPTIONS=--trace-warnings
export DEBUG=*

echo 'Starting with logging redirected to ../frontend.log'
exec npm start >> ../frontend.log 2>&1
