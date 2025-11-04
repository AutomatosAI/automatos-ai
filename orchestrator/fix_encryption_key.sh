#!/bin/bash
# Fix encryption key mismatch in dev environment

echo "🔧 Fixing Encryption Key Mismatch for Dev Environment"
echo "======================================================"

cd ~/automatos-ai/orchestrator

# Step 1: Clear problematic encrypted credentials
echo ""
echo "Step 1: Clearing encrypted credentials from database..."
PGPASSWORD="${POSTGRES_PASSWORD:-automatos_dev_pass}" psql -h 127.0.0.1 -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-orchestrator_db}" -f clear_credentials.sql

# Step 2: Remove old encryption key file (if exists)
echo ""
echo "Step 2: Removing old encryption key file..."
if [ -f .credential_key ]; then
    mv .credential_key .credential_key.backup_$(date +%Y%m%d_%H%M%S)
    echo "✓ Old key backed up"
else
    echo "✓ No old key file found"
fi

# Step 3: Restart backend to generate new key
echo ""
echo "Step 3: Restarting backend..."
pkill -9 -f uvicorn
sleep 2
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > ../backend.log 2>&1 &
echo "✓ Backend restarted (PID: $!)"

echo ""
echo "✅ Fix complete!"
echo ""
echo "Next steps:"
echo "1. Wait 5 seconds for backend to start"
echo "2. Go to Settings > Credentials in the UI"
echo "3. Re-add your API keys (OpenAI, Anthropic, etc.)"
echo "4. Test that everything works"
echo ""
echo "To monitor startup: tail -f ../backend.log"

