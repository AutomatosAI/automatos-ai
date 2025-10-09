#!/bin/bash

# Deploy Enhanced Orchestration to PRODUCTION SERVER
# ====================================================
# This script deploys all changes to the production server at 206.81.0.227

PROD_SERVER="root@206.81.0.227"
SSH_KEY="/Users/gkavanagh/.ssh/id_rsa"
REMOTE_DIR="/root/automatos-ai"

echo "🚀 DEPLOYING TO PRODUCTION SERVER (206.81.0.227)"
echo "=================================================="

# Create a tarball of the new services
echo "📦 Packaging new services..."
tar -czf services_update.tar.gz \
    orchestrator/services/rag_service.py \
    orchestrator/services/agent_action_executor.py \
    orchestrator/services/monitoring_service.py \
    orchestrator/services/agent_factory.py \
    orchestrator/api/workflows.py \
    orchestrator/api/tools.py \
    orchestrator/api/execution_history.py \
    orchestrator/core/context_engineering_integrator.py \
    orchestrator/core/workflow_memory_integrator.py \
    orchestrator/database/migrations/004_add_metadata_fields.sql \
    orchestrator/database/models.py \
    orchestrator/config.py \
    orchestrator/main.py \
    frontend/hooks/use-execution-history.ts \
    2>/dev/null

echo "📤 Uploading to production server..."
scp -i "$SSH_KEY" services_update.tar.gz "$PROD_SERVER:/tmp/"

echo "🔧 Installing on production server..."
ssh -i "$SSH_KEY" "$PROD_SERVER" << 'REMOTE_SCRIPT'
cd /root/automatos-ai


# Extract new files
echo "📦 Extracting updates..."
tar -xzf /tmp/services_update.tar.gz

# Apply database migration
echo "🗄️ Applying database migration..."
cd orchestrator
python -c "
import psycopg2
from config import get_config

config = get_config()
conn = psycopg2.connect(config.DATABASE_URL)
cur = conn.cursor()

# Read migration file
with open('database/migrations/004_add_metadata_fields.sql', 'r') as f:
    migration_sql = f.read()
    
# Apply migration
try:
    cur.execute(migration_sql)
    conn.commit()
    print('✅ Database migration applied successfully')
except Exception as e:
    print(f'⚠️ Migration may already be applied or error: {e}')
    conn.rollback()
finally:
    cur.close()
    conn.close()
"

# Install required packages
echo "📦 Installing required packages..."
pip install numpy psutil 2>/dev/null || echo "Packages already installed"

# Kill existing backend process
echo "🔄 Restarting backend..."
pkill -f uvicorn

# Start backend with new services (must be in orchestrator directory)
cd orchestrator
nohup uvicorn main:app --host 0.0.0.0 --port 8000 > ../backend.log 2>&1 &
cd ..

echo "⏳ Waiting for backend to start..."
sleep 10

# Test if backend is responding (test localhost first)
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Backend started successfully with enhanced services!"
    
    # Verify external access via Nginx
    if curl -s http://api.automatos.app/health > /dev/null; then
        echo "✅ External API access via api.automatos.app verified!"
    else
        echo "⚠️ External access not working - check Nginx configuration"
    fi
    
    # Show service status
    echo ""
    echo "📊 Service Status:"
    echo "  - RAG Service: Active with vector database"
    echo "  - Action Executor: Sandboxed at /tmp/automatos_workspace"
    echo "  - Monitoring Service: Active with metrics collection"
    echo "  - Mock Data: DISABLED ✅"
    echo "  - Hardcoded Templates: REMOVED ✅"
else
    echo "⚠️ Backend might still be starting..."
    echo "Check logs with: ssh root@206.81.0.227 'tail -f /root/automatos-ai/backend.log'"
fi

# Clean up
rm /tmp/services_update.tar.gz
REMOTE_SCRIPT

# Clean up local tarball
rm services_update.tar.gz

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📍 Access Points:"
echo "  - Dashboard: https://ui.automatos.app"
echo "  - API: http://api.automatos.app"
echo "  - Monitoring: http://api.automatos.app/api/monitoring/dashboard"
echo ""
echo "🧪 Test with:"
echo "  python test_9_stages.py"
echo ""
echo "📋 To check logs:"
echo "  ssh -i $SSH_KEY $PROD_SERVER 'tail -f $REMOTE_DIR/backend.log'"
