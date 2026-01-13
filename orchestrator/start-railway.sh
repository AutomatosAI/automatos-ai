#!/bin/bash
# =============================================================================
# Railway Startup Script
# =============================================================================
# This script ensures libmagic is available before starting the app
# Works for both Railpack and Nixpacks
# =============================================================================

# Set NLTK data path
export NLTK_DATA=/usr/local/nltk_data

# Fix libmagic for python-magic - MUST be done before any Python imports
# Try multiple locations and methods to find libmagic
set +e  # Don't exit on error for libmagic setup

if [ -f "/usr/lib/libmagic.so.1" ]; then
    # Symlink exists from build phase (Nixpacks)
    export LD_LIBRARY_PATH="/usr/lib:${LD_LIBRARY_PATH:-}"
elif [ -f "/usr/lib/x86_64-linux-gnu/libmagic.so.1" ]; then
    # Debian/Ubuntu standard location (Railpack)
    export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
elif [ -d "/nix/store" ]; then
    # Nixpacks: Find libmagic in /nix/store and create symlink
    LIBMAGIC=$(find /nix/store -name 'libmagic.so*' 2>/dev/null | head -1)
    if [ -n "$LIBMAGIC" ]; then
        mkdir -p /usr/lib
        ln -sf "$LIBMAGIC" /usr/lib/libmagic.so.1 2>/dev/null || true
        export LD_LIBRARY_PATH="/usr/lib:${LD_LIBRARY_PATH:-}"
    fi
fi

# Verify libmagic is accessible (for debugging)
if [ -n "$LD_LIBRARY_PATH" ]; then
    echo "LD_LIBRARY_PATH set to: $LD_LIBRARY_PATH"
fi

set -e  # Re-enable exit on error for app startup

# Get port from Railway (defaults to 8000)
PORT=${PORT:-8000}

# Start the application
# Use workers for production (Railway), reload for dev (local)
if [ "${ENVIRONMENT:-production}" = "development" ]; then
    exec uvicorn main:app --host 0.0.0.0 --port "$PORT" --reload
else
    exec uvicorn main:app --host 0.0.0.0 --port "$PORT" --workers 4
fi
