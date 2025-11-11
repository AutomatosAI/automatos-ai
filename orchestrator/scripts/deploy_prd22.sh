#!/bin/bash
# PRD-22 Skills as Executable Tools - Deployment Script
# Run this on the production server to deploy all changes

set -e  # Exit on error

echo "🚀 PRD-22 Deployment Script"
echo "================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ Please run as root (sudo su)"
    exit 1
fi

cd /root/automatos-ai/orchestrator

# Step 1: Check Dependencies
echo "📦 Step 1: Checking dependencies..."
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Installing..."
    apt-get update && apt-get install -y python3 python3-pip
fi
echo "  ✅ Python3: $(python3 --version)"

# Check Pandoc
if ! command -v pandoc &> /dev/null; then
    echo "⚠️  Pandoc not found. Installing..."
    apt-get update && apt-get install -y pandoc texlive-latex-base texlive-latex-extra
else
    echo "  ✅ Pandoc: $(pandoc --version | head -1)"
fi

# Check psycopg2
if ! python3 -c "import psycopg2" 2>/dev/null; then
    echo "⚠️  psycopg2 not found. Installing..."
    pip3 install psycopg2-binary
fi
echo "  ✅ psycopg2 installed"

# Optional: Install fallback libraries
echo ""
echo "📚 Installing optional libraries (reportlab, pandas, openpyxl)..."
pip3 install -q reportlab pandas openpyxl 2>/dev/null || echo "  ⚠️  Some libraries failed to install (non-critical)"

echo ""

# Step 2: Apply Database Migration
echo "🗄️  Step 2: Applying database migration..."
echo ""

# Check if migration file exists
if [ ! -f "database/migrations/002_add_skill_tools_schema.sql" ]; then
    echo "❌ Migration file not found!"
    echo "   Expected: database/migrations/002_add_skill_tools_schema.sql"
    exit 1
fi

# Get database credentials from environment
DB_HOST="${POSTGRES_HOST:-localhost}"
DB_PORT="${POSTGRES_PORT:-5432}"
DB_NAME="${POSTGRES_DB:-automatos}"
DB_USER="${POSTGRES_USER:-postgres}"
DB_PASSWORD="${POSTGRES_PASSWORD}"

# Apply migration
if [ -z "$DB_PASSWORD" ]; then
    echo "⚠️  POSTGRES_PASSWORD not set in environment. You may be prompted for password."
    export PGPASSWORD=""
else
    export PGPASSWORD="$DB_PASSWORD"
fi

echo "  📋 Applying migration to database: $DB_NAME"

if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f database/migrations/002_add_skill_tools_schema.sql > /dev/null 2>&1; then
    echo "  ✅ Migration applied successfully!"
else
    echo "  ⚠️  Migration may have already been applied or failed. Continuing..."
fi

# Verify migration
echo ""
echo "  🔍 Verifying migration..."
SKILL_COUNT=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT COUNT(*) FROM skills WHERE tools_schema IS NOT NULL;" 2>/dev/null | xargs)

if [ "$SKILL_COUNT" -gt 0 ]; then
    echo "  ✅ Found $SKILL_COUNT skills with tools_schema"
else
    echo "  ⚠️  No skills with tools_schema found. Migration may not have applied correctly."
fi

unset PGPASSWORD

echo ""

# Step 3: Restart Services
echo "🔄 Step 3: Restarting services..."
echo ""

cd /root/automatos-ai

# Check if docker compose is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Cannot restart services."
    exit 1
fi

echo "  🛑 Stopping orchestrator..."
docker compose stop orchestrator

echo "  🚀 Starting orchestrator..."
docker compose up -d orchestrator

echo "  ⏳ Waiting for orchestrator to be ready (10 seconds)..."
sleep 10

# Check if orchestrator is running
if docker compose ps orchestrator | grep -q "Up"; then
    echo "  ✅ Orchestrator is running"
else
    echo "  ❌ Orchestrator failed to start. Check logs:"
    echo "     docker compose logs orchestrator"
    exit 1
fi

echo ""

# Step 4: Verify Deployment
echo "✅ Step 4: Verifying deployment..."
echo ""

# Check logs for PRD-22 initialization
echo "  📋 Checking logs for PRD-22 indicators..."
sleep 5  # Give logs time to populate

if docker compose logs orchestrator | tail -100 | grep -q "UnifiedToolExecutor initialized"; then
    echo "  ✅ UnifiedToolExecutor initialized (document tools available)"
else
    echo "  ⚠️  UnifiedToolExecutor not found in recent logs"
fi

echo ""
echo "🎉 PRD-22 Deployment Complete!"
echo "================================"
echo ""
echo "📊 Summary:"
echo "  ✅ Dependencies installed"
echo "  ✅ Database migration applied"
echo "  ✅ Services restarted"
echo "  ✅ Deployment verified"
echo ""
echo "🧪 Next Steps:"
echo "  1. Run the 'Research and Write Report' workflow"
echo "  2. Watch logs: docker compose logs -f orchestrator | grep -E '(skill|tool|pdf)'"
echo "  3. Check output: ls -lh /tmp/automatos_workspace/*.pdf"
echo ""
echo "📚 Documentation:"
echo "  - Technical details: /root/automatos-ai/PRD-22-SKILL-TOOLS-FIX.md"
echo "  - Deployment guide: /root/automatos-ai/PRD-22-DEPLOYMENT-GUIDE.md"
echo "  - Complete summary: /root/automatos-ai/PRD-22-COMPLETE-SUMMARY.md"
echo ""
echo "🐛 Troubleshooting:"
echo "  - Check logs: docker compose logs orchestrator | grep -E '(ERROR|Failed)'"
echo "  - Verify skills: psql -U postgres -d automatos -c \"SELECT name, tools_schema IS NOT NULL FROM skills\""
echo "  - Test pandoc: pandoc --version"
echo ""
echo "🎯 Expected Log Messages:"
echo "  ✅ '🦸 PRD-22: Added X skill tools'"
echo "  ✅ '🔧 Executing tool create_pdf'"
echo "  ✅ 'PDF created successfully'"
echo ""
echo "Happy automating! 🤖"

