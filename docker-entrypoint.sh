#!/bin/bash
# =============================================================================
# Automatos AI - Backend Entrypoint Script
# =============================================================================
# This script:
#   1. Waits for PostgreSQL to be ready
#   2. Loads seed data (if not already loaded)
#   3. Starts the backend application
# =============================================================================

set -e

echo "========================================="
echo "Automatos AI Backend Starting..."
echo "========================================="

# =============================================================================
# Function: Wait for PostgreSQL
# =============================================================================
wait_for_postgres() {
    echo "⏳ Waiting for PostgreSQL to be ready..."
    
    max_attempts=30
    attempt=0
    
    until pg_isready -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" > /dev/null 2>&1; do
        attempt=$((attempt + 1))
        if [ $attempt -ge $max_attempts ]; then
            echo "❌ PostgreSQL did not become ready in time"
            exit 1
        fi
        echo "   Attempt $attempt/$max_attempts - waiting..."
        sleep 2
    done
    
    echo "✅ PostgreSQL is ready!"
}

# =============================================================================
# Function: Load Seed Data
# =============================================================================
load_seed_data() {
    echo ""
    echo "📦 Checking seed data..."
    
    # Set PGPASSWORD for psql commands
    export PGPASSWORD="$POSTGRES_PASSWORD"
    
    # Check if seed data is already loaded
    CREDENTIAL_COUNT=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT COUNT(*) FROM credential_types;" 2>/dev/null || echo "0")
    
    # Clean up whitespace
    CREDENTIAL_COUNT=$(echo "$CREDENTIAL_COUNT" | tr -d ' ')
    
    if [ "$CREDENTIAL_COUNT" -gt 0 ] 2>/dev/null; then
        echo "✅ Seed data already loaded ($CREDENTIAL_COUNT credential types found)"
        unset PGPASSWORD
        return 0
    fi
    
    echo "📥 Loading seed data..."
    
    # Run seed data loader
    if python database/load_seed_data.py; then
        echo "✅ Seed data loaded successfully!"
    else
        echo "⚠️  Warning: Seed data loading failed (will continue anyway)"
    fi
    
    # Unset PGPASSWORD
    unset PGPASSWORD
}

# =============================================================================
# Function: Check Database Connection
# =============================================================================
check_database() {
    echo ""
    echo "🔍 Verifying database connection..."
    
    # Set PGPASSWORD for psql commands
    export PGPASSWORD="$POSTGRES_PASSWORD"
    
    if psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "\dt" > /dev/null 2>&1; then
        TABLE_COUNT=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" | tr -d ' ')
        echo "✅ Database connected! ($TABLE_COUNT tables found)"
    else
        echo "❌ Database connection failed"
        exit 1
    fi
    
    # Unset PGPASSWORD for security
    unset PGPASSWORD
}

# =============================================================================
# Main Execution
# =============================================================================

# Wait for services
wait_for_postgres

# Check database
check_database

# Load seed data (idempotent)
load_seed_data

echo ""
echo "========================================="
echo "🚀 Starting Backend Application"
echo "========================================="
echo "   API: http://0.0.0.0:8000"
echo "   Docs: http://0.0.0.0:8000/docs"
echo "   Environment: $ENVIRONMENT"
echo "========================================="
echo ""

# Execute the CMD from Dockerfile (uvicorn command)
exec "$@"

