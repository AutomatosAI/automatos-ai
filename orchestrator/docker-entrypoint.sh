#!/bin/bash
# =============================================================================
# Automatos AI - Backend Entrypoint Script
# =============================================================================
# The single wait-migrate-seed lifecycle (PRD-176 F051):
#   1. Waits for PostgreSQL to be ready
#   2. Verifies the database connection
#   3. Initializes an empty database, then runs Alembic migrations (fail closed)
#   4. Loads seed data (idempotent)
#   5. Ensures the local-edition workspace and operator exist (local only)
#
# The image carries this script as its ENTRYPOINT. How it runs:
#
#   docker-entrypoint.sh migrate
#       Runs the lifecycle, then exits 0 without starting the app. Idempotent.
#       A failed seed fails it (exit 1). This is what a Kubernetes migration
#       Job runs before app pods start.
#
#   docker-entrypoint.sh <command...>   (the image CMD, or any other command)
#       AUTOMATOS_MIGRATE_ON_BOOT=true: runs the lifecycle, then the command;
#       a failed seed only warns. Set by envs/api.defaults, so `docker compose
#       up` does this.
#       Unset or anything else: runs the command directly, touching nothing.
#       This is the image default, so Railway (which builds this image and
#       relies on the CMD's own `alembic upgrade heads`) boots unchanged.
#       Note the image CMD itself runs `alembic upgrade heads` before uvicorn.
#       Where a migration Job owns the schema (Kubernetes), override the app
#       container's command to start uvicorn directly, so several replicas
#       don't race Alembic with each other or with the Job.
#
# Database connection: POSTGRES_HOST/PORT/USER/PASSWORD/DB when POSTGRES_HOST
# is set (compose); otherwise DATABASE_URL (Railway, Kubernetes secrets).
# =============================================================================

set -e

# =============================================================================
# Database connection helpers
# =============================================================================
# pg_isready and psql against whichever connection the environment provides.
# The password travels in PGPASSWORD for the POSTGRES_* form and inside the
# URL for the DATABASE_URL form; it is never passed on the command line.
require_database_settings() {
    if [ -z "${POSTGRES_HOST:-}" ] && [ -z "${DATABASE_URL:-}" ]; then
        echo "❌ No database configured: set POSTGRES_HOST (with POSTGRES_PORT/USER/PASSWORD/DB) or DATABASE_URL"
        exit 1
    fi
}

pg_ready() {
    if [ -n "${POSTGRES_HOST:-}" ]; then
        pg_isready -h "$POSTGRES_HOST" -p "${POSTGRES_PORT:-5432}" -U "$POSTGRES_USER"
    else
        pg_isready -d "$DATABASE_URL"
    fi
}

db_psql() {
    if [ -n "${POSTGRES_HOST:-}" ]; then
        PGPASSWORD="${POSTGRES_PASSWORD:-}" psql -h "$POSTGRES_HOST" -p "${POSTGRES_PORT:-5432}" -U "$POSTGRES_USER" -d "$POSTGRES_DB" "$@"
    else
        psql "$DATABASE_URL" "$@"
    fi
}

# =============================================================================
# Function: Wait for PostgreSQL
# =============================================================================
wait_for_postgres() {
    echo "⏳ Waiting for PostgreSQL to be ready..."

    max_attempts=30
    attempt=0

    until pg_ready > /dev/null 2>&1; do
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
# Function: Check Database Connection
# =============================================================================
check_database() {
    echo ""
    echo "🔍 Verifying database connection..."

    if db_psql -c "\dt" > /dev/null 2>&1; then
        TABLE_COUNT=$(db_psql -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" | tr -d ' ')
        echo "✅ Database connected! ($TABLE_COUNT tables found)"
    else
        echo "❌ Database connection failed"
        exit 1
    fi
}

# =============================================================================
# Function: Initialize an empty database
# =============================================================================
# Fresh (empty) database? Build the CI-proven schema and stamp at heads
# (PRD-209: replaces the stale init_complete_schema.sql snapshot — see
# scripts/init_fresh_db.py for why the migration forest cannot replay from
# empty). Fails CLOSED: a half-initialized database must never serve. Existing
# databases (alembic_version present) skip straight to incremental migrations,
# and init_fresh_db itself refuses a database that already has tables.
init_fresh_if_empty() {
    HAS_VERSION=$(db_psql -tc "SELECT to_regclass('alembic_version');" 2>/dev/null | tr -d ' ')
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

# =============================================================================
# Function: Run Database Migrations (PRD-176 F051)
# =============================================================================
# Brings the schema to head via Alembic — the single owner of the schema
# lifecycle. Fails CLOSED: unlike seed loading, a failed migration exits
# non-zero so the app never starts against a half-built schema. Alembic is
# idempotent — on an already-migrated database `upgrade heads` is a no-op.
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
    elif [ "${LIFECYCLE_MODE:-boot}" = "migrate" ]; then
        # A migration Job that reports success on an unseeded database hides
        # the failure until the first request; fail it instead.
        echo "❌ Seed data loading failed — migrate must not report success on an unseeded database"
        exit 1
    else
        echo "⚠️  Warning: Seed data loading failed (will continue anyway)"
    fi
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
    # A brand-new install starts Auto-led onboarding: the row carries an explicit
    # not_started document (PRD-222's veteran backfill matches only stage-less rows
    # older than PRD-222 — see prd222_veteran_skip_backfill).
    # Values reach SQL as psql variables (:'name' quotes them), never by pasting
    # them into the statement. psql only interpolates variables in SQL it reads
    # itself, not in -c, so the statements come in on stdin.
    if db_psql -v ON_ERROR_STOP=1 -v workspace_id="$DEFAULT_WORKSPACE_ID" <<'SQL'
INSERT INTO workspaces (id, name, slug, is_personal, is_active, onboarding) VALUES (:'workspace_id', 'Local Workspace', 'local', TRUE, TRUE, '{"stage": "not_started", "stages": {}, "segment": {}}'::jsonb) ON CONFLICT (id) DO NOTHING;
SQL
    then
        echo "✅ Local workspace present"
    else
        echo "❌ Could not create the local workspace — refusing to start a shell instance"
        exit 1
    fi
    # The single local operator (users id 1 — api/chat.py's own fallback).
    # Idempotent; PRD-233 S6 makes name/email editable in Settings → Profile.
    if db_psql -v ON_ERROR_STOP=1 -v operator_email="${LOCAL_OPERATOR_EMAIL:-local@automatos.local}" >/dev/null <<'SQL'
INSERT INTO users (id, username, email, name, is_active) VALUES (1, 'local', :'operator_email', 'Local Operator', TRUE) ON CONFLICT (id) DO NOTHING;
SELECT setval(pg_get_serial_sequence('users','id'), GREATEST((SELECT max(id) FROM users), 1));
SQL
    then
        echo "✅ Local operator user present"
    else
        echo "❌ Could not seed the local operator user — refusing to start a shell instance"
        exit 1
    fi
}

# =============================================================================
# The lifecycle, in order: wait -> check -> init -> migrate -> seed -> local
# =============================================================================
run_lifecycle() {
    echo "========================================="
    echo "Automatos AI Backend Starting..."
    echo "========================================="

    require_database_settings
    wait_for_postgres
    check_database
    init_fresh_if_empty
    run_migrations
    load_seed_data
    ensure_local_workspace
}

# =============================================================================
# Main Execution
# =============================================================================
if [ "${1:-}" = "migrate" ]; then
    LIFECYCLE_MODE=migrate
    run_lifecycle
    echo ""
    echo "✅ Migrate complete — database is initialized, migrated and seeded"
    exit 0
fi

if [ "${AUTOMATOS_MIGRATE_ON_BOOT:-}" = "true" ]; then
    run_lifecycle
    echo ""
    echo "========================================="
    echo "🚀 Starting Backend Application"
    echo "========================================="
    echo "   API: http://0.0.0.0:8000"
    echo "   Docs: http://0.0.0.0:8000/docs"
    echo "   Environment: $ENVIRONMENT"
    echo "========================================="
    echo ""
fi

# Execute the CMD from Dockerfile (uvicorn command)
exec "$@"
