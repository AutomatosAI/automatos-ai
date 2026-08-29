#!/usr/bin/env bash
# PRD-176 Headline B — fresh-clone deployability smoke test.
#
# The open-core thesis bar: `git clone && docker compose up` yields a working
# local instance with NO external credentials. This script proves it end to end:
# it provides ONLY the required local secrets (POSTGRES_PASSWORD, REDIS_PASSWORD,
# API_KEY) and NO external SaaS credentials (no Clerk, no real S3/AWS, no LLM
# keys), brings the compose stack up, and asserts the backend `/health` returns
# HTTP 200.
#
# It exercises together: the repointed initdb mount (F009), the wait-migrate-seed
# entrypoint (F051), local MinIO via S3_ENDPOINT_URL (F089), and the local-safe
# railway.internal defaults (F068).
#
# W5 (auth decoupling) HAS LANDED: AUTH_EDITION=local forces the no-login posture
# and, together with US-001 (entrypoint exec bit), US-002 (stamped initdb), and
# US-003 (local is the compose default), a no-credential boot now reaches a green
# /health. This script sets AUTH_EDITION=local AND DEFAULT_WORKSPACE_ID (the local
# workspace validate_auth_edition() hard-requires). The lane is now HARD in CI
# (no continue-on-error, PRD-209 US-004); still non-required in branch protection
# until the owner flips it (Q2).
#
# Exit 0 = /health returned 200 with no external creds. Exit 1 = boot failed.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

COMPOSE_FILE="docker-compose.yml"
HEALTH_URL="http://localhost:${API_PORT:-8000}/health"
MAX_WAIT_SECONDS="${SMOKE_MAX_WAIT_SECONDS:-300}"
POLL_INTERVAL_SECONDS=5

# ---------------------------------------------------------------------------
# Local-only secrets. These are throwaway values for an ephemeral local stack —
# NOT real credentials, and never reused. No external SaaS credential is set:
# CLERK_*, AWS_*, OPENAI_API_KEY, ANTHROPIC_API_KEY are all intentionally absent.
# ---------------------------------------------------------------------------
export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-smoke_pg_pw}"
export REDIS_PASSWORD="${REDIS_PASSWORD:-smoke_redis_pw}"
export API_KEY="${API_KEY:-smoke_api_key}"
# W5 flag: run the local no-login edition (auth-optional).
export AUTH_EDITION="${AUTH_EDITION:-local}"
export REQUIRE_AUTH="${REQUIRE_AUTH:-false}"
# validate_auth_edition() hard-requires DEFAULT_WORKSPACE_ID in local mode — the
# CI-seed convention value (matches envs/api.defaults + the entrypoint seed). The
# backend also receives it from compose env_file; exporting it keeps this script
# self-sufficient and documents the local-mode contract.
export DEFAULT_WORKSPACE_ID="${DEFAULT_WORKSPACE_ID:-00000000-0000-0000-0000-0000000000c1}"

echo "========================================="
echo "PRD-176 fresh-clone smoke test"
echo "  health url : ${HEALTH_URL}"
echo "  max wait   : ${MAX_WAIT_SECONDS}s"
echo "  auth       : AUTH_EDITION=${AUTH_EDITION} REQUIRE_AUTH=${REQUIRE_AUTH}"
echo "  workspace  : DEFAULT_WORKSPACE_ID=${DEFAULT_WORKSPACE_ID}"
echo "  external creds: NONE (no Clerk / AWS / LLM keys)"
echo "========================================="

# Bring the stack up on just the core boot services (backend gates the frontend).
echo "Starting compose stack (postgres, redis, minio, backend)..."
if ! docker compose -f "$COMPOSE_FILE" up -d --build postgres redis minio backend; then
  echo "FAIL: docker compose up failed" >&2
  docker compose -f "$COMPOSE_FILE" logs --tail 100 || true
  docker compose -f "$COMPOSE_FILE" down -v || true
  exit 1
fi

# Poll /health until 200 or timeout.
echo "Waiting for backend /health to return 200..."
elapsed=0
status=""
while [ "$elapsed" -lt "$MAX_WAIT_SECONDS" ]; do
  status="$(curl -s -o /dev/null -w '%{http_code}' "$HEALTH_URL" 2>/dev/null || echo '000')"
  if [ "$status" = "200" ]; then
    echo "OK: /health returned 200 after ${elapsed}s (no external credentials)"
    docker compose -f "$COMPOSE_FILE" down -v || true
    exit 0
  fi
  echo "  ...${elapsed}s: /health -> ${status}"
  sleep "$POLL_INTERVAL_SECONDS"
  elapsed=$((elapsed + POLL_INTERVAL_SECONDS))
done

echo "FAIL: /health did not return 200 within ${MAX_WAIT_SECONDS}s (last: ${status})" >&2
echo "----- postgres logs (tail) — initdb/schema errors surface here -----" >&2
docker compose -f "$COMPOSE_FILE" logs --tail 200 postgres >&2 || true
echo "----- backend logs (tail) -----" >&2
docker compose -f "$COMPOSE_FILE" logs --tail 150 backend >&2 || true
docker compose -f "$COMPOSE_FILE" down -v || true
exit 1
