#!/bin/bash
# Database Initialization Script - Shell Version
# ==============================================

echo "🗄️  AUTOMATOS AI DATABASE INITIALIZATION"
echo "========================================"

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Check if variables are set
if [ -z "$POSTGRES_DB" ] || [ -z "$POSTGRES_USER" ] || [ -z "$POSTGRES_PASSWORD" ]; then
    echo "❌ Error: Database credentials not found in .env"
    echo "Please ensure POSTGRES_DB, POSTGRES_USER, and POSTGRES_PASSWORD are set"
    exit 1
fi

# Set default host and port if not provided
POSTGRES_HOST=${POSTGRES_HOST:-localhost}
POSTGRES_PORT=${POSTGRES_PORT:-5432}

echo "📍 Database: $POSTGRES_DB"
echo "🖥️  Host: $POSTGRES_HOST:$POSTGRES_PORT"
echo "👤 User: $POSTGRES_USER"
echo ""

# Check if schema file exists
SCHEMA_FILE="orchestrator/database/init_complete_schema.sql"
if [ ! -f "$SCHEMA_FILE" ]; then
    echo "❌ Schema file not found: $SCHEMA_FILE"
    exit 1
fi

echo "📋 Initializing database with schema..."
echo ""

# Execute the schema
PGPASSWORD=$POSTGRES_PASSWORD psql \
    -h $POSTGRES_HOST \
    -p $POSTGRES_PORT \
    -U $POSTGRES_USER \
    -d $POSTGRES_DB \
    -f $SCHEMA_FILE \
    -v ON_ERROR_STOP=1

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ DATABASE INITIALIZATION SUCCESSFUL!"
    
    # Quick verification
    echo ""
    echo "📊 Verifying tables..."
    
    PGPASSWORD=$POSTGRES_PASSWORD psql \
        -h $POSTGRES_HOST \
        -p $POSTGRES_PORT \
        -U $POSTGRES_USER \
        -d $POSTGRES_DB \
        -c "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public' ORDER BY table_name;" \
        -t | head -20
    
    echo ""
    echo "✅ Database is ready for use!"
    echo ""
    echo "Next steps:"
    echo "  1. Test configuration: python test_config.py"
    echo "  2. Test task decomposition: python test_real_decomposition.py"
    echo "  3. Start API server: python orchestrator/main.py"
else
    echo ""
    echo "❌ DATABASE INITIALIZATION FAILED!"
    echo "Check the errors above and fix any issues"
    exit 1
fi

