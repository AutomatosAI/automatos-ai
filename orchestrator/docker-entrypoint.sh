#!/bin/bash
# =============================================================================
# Automatos AI - Backend Entrypoint Script
# =============================================================================
# The only owner of migrations (PRD-176 F051, PRD-262 US-029). The boot steps:
#   1. Waits for PostgreSQL to be ready
#   2. Verifies the database connection
#   3. Initializes an empty database, then runs Alembic migrations (fail closed)
#   4. Loads seed data (idempotent)
#   5. Ensures the local-edition workspace and operator exist (local only)
# Steps 1-3 are the migration step; 4-5 run only where a flag below asks.
#
# The image carries this script as its ENTRYPOINT. How it runs:
#
#   docker-entrypoint.sh migrate
#       Runs all five steps, then exits 0 without starting the app. Idempotent.
#       A failed seed fails it (exit 1). This is what a Kubernetes migration
#       Job runs before app pods start. It ignores AUTOMATOS_SKIP_MIGRATIONS.
#
#   docker-entrypoint.sh <server command>   (the image CMD: uvicorn)
#       The boot steps run first, then the server. Each flag has one meaning:
#       AUTOMATOS_SKIP_MIGRATIONS=true: none of the boot steps run. The Helm
#         api pods set it, because the migration Job owns the schema there.
#       AUTOMATOS_MIGRATE_ON_BOOT=true: the migration step, then the seed
#         loader and the local workspace and operator; a failed seed only
#         warns. Set by envs/api.defaults, so `docker compose up` does this.
#       Neither set (Railway): the migration step only. The seed loader never
#         runs there: its package seed is a full refresh, which would
#         overwrite the packages curated in production.
#       The image CMD runs no alembic of its own: it execs uvicorn.
#
#   docker-entrypoint.sh <any other command>   (a one-off)
#       Runs the command as given, with no boot step: `docker compose run
#       backend <cmd>` and CI's `docker run <image> python -c …` neither
#       migrate nor seed. A server command starts uvicorn: directly, as
#       `python -m uvicorn`, or through `sh -c` (after an optional `exec`).
#
#   AUTOMATOS_LIVENESS_WATCHDOG=true (envs/api.defaults): the command runs as
#   this script's child, and a server that stops serving is stopped so the
#   container exits and restarts (the liveness watchdog, below).
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

# libpq tools accept postgresql:// or postgres:// only. Drop a SQLAlchemy driver
# suffix (postgresql+psycopg2://, postgresql+asyncpg://), which the app accepts
# but pg_isready and psql reject.
libpq_url() {
    printf '%s' "$DATABASE_URL" | sed -E 's#^(postgres(ql)?)\+[A-Za-z0-9_]+://#\1://#'
}

pg_ready() {
    if [ -n "${POSTGRES_HOST:-}" ]; then
        pg_isready -h "$POSTGRES_HOST" -p "${POSTGRES_PORT:-5432}" -U "$POSTGRES_USER"
    else
        pg_isready -d "$(libpq_url)"
    fi
}

db_psql() {
    if [ -n "${POSTGRES_HOST:-}" ]; then
        PGPASSWORD="${POSTGRES_PASSWORD:-}" psql -h "$POSTGRES_HOST" -p "${POSTGRES_PORT:-5432}" -U "$POSTGRES_USER" -d "$POSTGRES_DB" "$@"
    else
        psql "$(libpq_url)" "$@"
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
# and init_fresh_db itself refuses a database that already has tables. A build
# that was interrupted leaves alembic_version behind WITH its incomplete-build
# marker (scripts/generate_schema_baseline.py); that database is not "existing",
# so it goes back to init_fresh_db, which finishes the build.
init_fresh_if_empty() {
    DB_STATE=$(db_psql -tc "SELECT CASE WHEN to_regclass('automatos_fresh_init_incomplete') IS NOT NULL THEN 'interrupted' WHEN to_regclass('alembic_version') IS NULL THEN 'empty' ELSE 'existing' END;" 2>/dev/null | tr -d ' ')
    if [ "$DB_STATE" = "existing" ]; then
        return 0
    fi
    echo ""
    if [ "$DB_STATE" = "interrupted" ]; then
        echo "♻️  A previous fresh-database initialization was interrupted — resuming it..."
    else
        echo "🆕 No alembic_version — initializing fresh database (CI-proven schema + stamp)..."
    fi
    if python -m scripts.init_fresh_db; then
        echo "✅ Fresh database initialized"
    else
        echo "❌ Fresh-database initialization failed — refusing to start"
        exit 1
    fi
}

# =============================================================================
# Function: Run Database Migrations (PRD-176 F051)
# =============================================================================
# Brings the schema to head via Alembic — the single owner of the schema
# lifecycle, run once per boot (the image CMD runs no alembic). Fails CLOSED:
# unlike seed loading, a failed migration exits non-zero so the app never
# starts against a half-built schema. Alembic is idempotent — on an
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
# The boot steps, in order: wait -> check -> init -> migrate -> seed -> local
# =============================================================================
# The migration step: wait -> check -> init -> migrate.
run_migration_step() {
    echo "========================================="
    echo "Automatos AI Backend Starting..."
    echo "========================================="

    require_database_settings
    wait_for_postgres
    check_database
    init_fresh_if_empty
    run_migrations
}

# The full lifecycle: the migration step, then the seeds and the local edition's
# workspace and operator (`migrate`, and AUTOMATOS_MIGRATE_ON_BOOT=true).
run_lifecycle() {
    run_migration_step
    load_seed_data
    ensure_local_workspace
}

# =============================================================================
# A server command, or a one-off (PRD-262 US-029)
# =============================================================================
# The boot steps run before the server only. A server command starts uvicorn:
# `uvicorn …`, `python -m uvicorn …`, or `sh -c '[exec] uvicorn …'` (the image
# CMD is shell form, which Docker runs as `/bin/sh -c '<CMD>'`).
SERVER_PROGRAM=uvicorn

# Prints the program a command starts. Call it in a subshell, $(…): it splits a
# `sh -c` script into words, with globbing off, to read its first one.
command_program() {
    case "${1##*/}" in
        sh|bash|dash)
            if [ "${2:-}" = "-c" ]; then
                set -f
                # shellcheck disable=SC2086  # the script's words, split on purpose
                set -- ${3:-}
                if [ "${1:-}" = "exec" ]; then
                    shift
                fi
                command_program "$@"
                return
            fi
            ;;
        python|python3|python3.*)
            if [ "${2:-}" = "-m" ]; then
                printf '%s' "${3:-}"
                return
            fi
            ;;
    esac
    printf '%s' "${1##*/}"
}

is_server_command() {
    [ "$(command_program "$@")" = "$SERVER_PROGRAM" ]
}

# Before a server command. AUTOMATOS_SKIP_MIGRATIONS=true: nothing.
# AUTOMATOS_MIGRATE_ON_BOOT=true: the full lifecycle. Neither: the migration
# step only, so Railway migrates and never runs the seed loader.
run_boot_steps() {
    if [ "${AUTOMATOS_SKIP_MIGRATIONS:-}" = "true" ]; then
        return 0
    fi
    if [ "${AUTOMATOS_MIGRATE_ON_BOOT:-}" = "true" ]; then
        run_lifecycle
    else
        run_migration_step
    fi
    echo ""
    echo "========================================="
    echo "🚀 Starting Backend Application"
    echo "========================================="
    echo "   API: http://0.0.0.0:${PORT:-8000}"
    echo "   Docs: http://0.0.0.0:${PORT:-8000}/docs"
    echo "   Environment: ${ENVIRONMENT:-}"
    echo "========================================="
    echo ""
}

# =============================================================================
# Liveness watchdog (F416)
# =============================================================================
# uvicorn's --reload supervisor (and its --workers one before 0.30) outlives a worker
# the kernel kills: the container stays "running" and unhealthy with nothing
# serving, and Docker's restart policy never applies to an unhealthy container
# (night 14: ~5 h dark after an OOM kill). With AUTOMATOS_LIVENESS_WATCHDOG=true
# the entrypoint runs the command as its child and probes it with the
# healthcheck script (docker-healthcheck.sh) every AUTOMATOS_LIVENESS_INTERVAL
# seconds. Once the server has served, AUTOMATOS_LIVENESS_RETRIES failed probes
# in a row stop it (TERM, then KILL after AUTOMATOS_LIVENESS_GRACE seconds) and
# the entrypoint exits non-zero, so the container exits and its restart policy
# brings it back. A server that exits by itself passes its exit status on. The
# watchdog arms at the first passing probe, so a slow boot (F228) is Docker's
# start_period to judge, never a restart. Off by default (the command is
# exec'd as before, as Railway runs it); envs/api.defaults turns it on.
LIVENESS_DEFAULT_INTERVAL_SECONDS=30
LIVENESS_DEFAULT_RETRIES=3
LIVENESS_DEFAULT_GRACE_SECONDS=10
LIVENESS_STOP_POLL_SECONDS=1
LIVENESS_EXIT_CODE=1

probe_server() {
    "${AUTOMATOS_HEALTHCHECK_CMD:-docker-healthcheck.sh}" > /dev/null 2>&1
}

# TERM the server, then KILL it if it is still there after the grace period.
stop_server() {
    grace=${AUTOMATOS_LIVENESS_GRACE:-$LIVENESS_DEFAULT_GRACE_SECONDS}
    kill -TERM "$SERVER_PID" 2>/dev/null || return 0
    waited=0
    while kill -0 "$SERVER_PID" 2>/dev/null; do
        if [ "$waited" -ge "$grace" ]; then
            kill -KILL "$SERVER_PID" 2>/dev/null || true
            return 0
        fi
        sleep "$LIVENESS_STOP_POLL_SECONDS"
        waited=$((waited + LIVENESS_STOP_POLL_SECONDS))
    done
}

watch_liveness() {
    interval=${AUTOMATOS_LIVENESS_INTERVAL:-$LIVENESS_DEFAULT_INTERVAL_SECONDS}
    retries=${AUTOMATOS_LIVENESS_RETRIES:-$LIVENESS_DEFAULT_RETRIES}
    armed=false
    failures=0
    while kill -0 "$SERVER_PID" 2>/dev/null; do
        sleep "$interval"
        if probe_server; then
            [ "$armed" = true ] || echo "🩺 Liveness watchdog armed (the backend serves)"
            armed=true
            failures=0
        elif [ "$armed" = true ]; then
            failures=$((failures + 1))
            echo "⚠️  Liveness probe failed ($failures/$retries)"
            if [ "$failures" -ge "$retries" ]; then
                echo "❌ The backend stopped serving — stopping it so the container exits and restarts"
                : > "$LIVENESS_MARK" || true
                stop_server
                return 0
            fi
        fi
    done
}

# Wait for the server's own exit status. A forwarded signal interrupts `wait`
# (status > 128) while the server is still stopping: wait again.
wait_for_server() {
    while :; do
        wait "$SERVER_PID" && return 0
        status=$?
        if [ "$status" -le 128 ] || ! kill -0 "$SERVER_PID" 2>/dev/null; then
            return "$status"
        fi
    done
}

run_supervised() {
    LIVENESS_MARK="${TMPDIR:-/tmp}/automatos-liveness.$$"
    rm -f "$LIVENESS_MARK"
    "$@" &
    SERVER_PID=$!
    trap 'kill -TERM "$SERVER_PID" 2>/dev/null || true' TERM INT
    watch_liveness &
    WATCHDOG_PID=$!
    server_status=0
    wait_for_server || server_status=$?
    kill "$WATCHDOG_PID" 2>/dev/null || true
    if [ -e "$LIVENESS_MARK" ]; then
        rm -f "$LIVENESS_MARK"
        exit "$LIVENESS_EXIT_CODE"
    fi
    exit "$server_status"
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

if is_server_command "$@"; then
    run_boot_steps
fi

if [ "${AUTOMATOS_LIVENESS_WATCHDOG:-}" = "true" ]; then
    run_supervised "$@"
fi

# Execute the command: the image CMD (uvicorn), or a one-off
exec "$@"
