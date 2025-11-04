#!/bin/bash
# Find the old encryption key and make development use it

echo "🔍 Finding existing encryption key..."
echo "======================================"

cd ~/automatos-ai

# Search for .credential_key files
echo ""
echo "Searching for .credential_key files:"
find . -name ".credential_key*" -type f 2>/dev/null

# Check if key exists in common locations
echo ""
echo "Checking common locations:"
if [ -f "orchestrator/.credential_key" ]; then
    echo "✓ Found: orchestrator/.credential_key"
    KEY_FILE="orchestrator/.credential_key"
elif [ -f ".credential_key" ]; then
    echo "✓ Found: .credential_key (root)"
    KEY_FILE=".credential_key"
else
    echo "❌ No .credential_key file found"
    echo ""
    echo "The old key might be in an environment variable or backup."
    echo "Check these locations:"
    echo "  - Any .env files"
    echo "  - Backup directories"
    echo "  - /root/ directory"
    find /root -name ".credential_key*" -type f 2>/dev/null | head -5
    exit 1
fi

echo ""
echo "📋 Found encryption key at: $KEY_FILE"
echo ""
echo "Option 1: Set as environment variable (recommended for dev)"
echo "Add this to your startup script or .bashrc:"
echo ""
KEY_VALUE=$(cat "$KEY_FILE")
echo "export CREDENTIAL_ENCRYPTION_KEY='$KEY_VALUE'"
echo ""
echo "Option 2: Make sure it's in orchestrator/.credential_key"
if [ "$KEY_FILE" != "orchestrator/.credential_key" ]; then
    echo "Copy it: cp $KEY_FILE orchestrator/.credential_key"
fi
echo ""
echo "After fixing, restart backend:"
echo "  cd ~/automatos-ai/orchestrator"
echo "  pkill -9 -f uvicorn"
echo "  uvicorn main:app --host 0.0.0.0 --port 8000 > ../backend.log 2>&1 &"

