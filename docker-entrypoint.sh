#!/bin/bash
# =============================================================================
# Automatos AI - Backend Entrypoint Script
# =============================================================================
# This script is the single wait-migrate-seed lifecycle (PRD-176 F051):
#   1. Waits for PostgreSQL to be ready
#   2. Verifies the database connection
#   3. Runs Alembic migrations (fails closed — a bad migration aborts startup)
#   4. Loads seed data (idempotent, if not already loaded)
#   5. Starts the backend application
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
# Function: Run Database Migrations (PRD-176 F051)
# =============================================================================
# Brings the schema to head via Alembic. This is the single owner of schema
# lifecycle for a fresh clone: `alembic upgrade heads` replays every revision
# against the (initdb-populated or empty) database.
#
# Fails CLOSED: unlike seed loading, a failed migration exits non-zero so the
# app never starts against a half-built schema. Alembic is idempotent — on an
# already-migrated database `upgrade heads` is a no-op.
run_migrations() {
    echo ""
    echo "🗄️  Running database migrations (alembic upgrade heads)..."

    if alembic upgrade heads; then
        echo "✅ Migrations applied (schema at head)"
    else
        echo "❌ Migration failed — aborting startup (will not run on a half-built schema)"
        exit 1
    fi
}

# =============================================================================
# Function: Load Seed Data
# =============================================================================
load_seed_data() {
    echo ""
    echo "📦 Checking seed data..."
    
    # Set PGPASSWORD for psql commands
    export PGPASSWORD="$POSTGRES_PASSWORD"
    
    # No shell-level "already loaded" gate: every section of the loader is
    # idempotent on its own (credential types insert-if-missing, models/skills/
    # personas/categories upsert, marketplace catalog checks by name/slug).
    # The old credential_types>0 early-return silently skipped every LATER
    # section (marketplace catalog, packages) on any pre-seeded database.
    echo "📥 Loading seed data (idempotent)..."
    
    # Run seed data loader AS A MODULE — script-mode sets sys.path[0] to the
    # script's own dir, so its `from config import config` (line 23) can never
    # resolve. (The old `database/` path also never existed in the image, so
    # this step silently failed on every boot since PRD-176; fail-open hid
    # both bugs. PRD-209 local-run finding.)
    if python -m core.database.load_seed_data; then
        echo "✅ Seed data loaded successfully!"
    else
        echo "⚠️  Warning: Seed data loading failed (will continue anyway)"
    fi

    # Unset PGPASSWORD
    unset PGPASSWORD
}

# =============================================================================
# Function: Ensure the local-edition workspace exists (PRD-209)
# =============================================================================
# In local mode every anonymous request resolves to DEFAULT_WORKSPACE_ID; a
# boot that "succeeds" without that row is a shell that 500s on first use.
# Same idempotent shape as the CI seed (scripts/init_test_db.py). FAILS CLOSED
# in local mode — SaaS never enters this branch (AUTH_EDITION defaults saas).
ensure_local_workspace() {
    if [ "${AUTH_EDITION:-saas}" != "local" ] || [ -z "${DEFAULT_WORKSPACE_ID:-}" ]; then
        return 0
    fi
    echo ""
    echo "🏠 Ensuring local workspace ${DEFAULT_WORKSPACE_ID} exists..."
    export PGPASSWORD="$POSTGRES_PASSWORD"
    if psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1 -c \
        "INSERT INTO workspaces (id, name, slug, is_personal, is_active) VALUES ('${DEFAULT_WORKSPACE_ID}', 'Local Workspace', 'local', TRUE, TRUE) ON CONFLICT (id) DO NOTHING;"; then
        echo "✅ Local workspace present"
        unset PGPASSWORD
    else
        echo "❌ Could not create the local workspace — refusing to start a shell instance"
        unset PGPASSWORD
        exit 1
    fi
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

# Fresh (empty) database? Build the CI-proven schema and stamp at heads
# (PRD-209: replaces the stale init_complete_schema.sql snapshot — see
# scripts/init_fresh_db.py for why the migration forest cannot replay from
# empty). Fails CLOSED: a half-initialized database must never serve. Existing
# databases (alembic_version present) skip straight to incremental migrations.
init_fresh_if_empty() {
    export PGPASSWORD="$POSTGRES_PASSWORD"
    HAS_VERSION=$(psql -h "$POSTGRES_HOST" -p "$POSTGRES_PORT" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -tc "SELECT to_regclass('alembic_version');" 2>/dev/null | tr -d ' ')
    unset PGPASSWORD
    if [ -z "$HAS_VERSION" ]; then
        echo ""
        echo "🆕 No alembic_version — initializing fresh database (CI-proven schema + stamp)..."
        if python -m scripts.init_fresh_db; then
            echo "✅ Fresh database initialized"
        else
            echo "❌ Fresh-database initialization failed — refusing to start"
            exit 1
        fi
    fi
}
init_fresh_if_empty

# Run migrations (fail-closed) — the single owner of schema lifecycle
run_migrations

# Load seed data (idempotent)
load_seed_data

# Local edition: the anonymous workspace must exist (fail-closed; no-op in saas)
ensure_local_workspace

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

