#!/bin/bash
# Launcher for the PRD-150→153 chain. Detached-safe (nohup): survives the IDE.
#
# Postgres on 5432 IS load-bearing — not for data, but because
# tests/test_82c_wiring.py blocks forever on psycopg2.connect when no
# Postgres is present (documented in .github/workflows/test.yml; thread
# timeouts can't interrupt a blocking C connect, so the PRD-150 per-story
# full-suite runs would wedge). An EMPTY Postgres satisfies the suite
# (main CI is green against a fresh service container); the dev compose
# Postgres additionally lets P150-S2 apply its migration. Docker itself is
# only a hard need for PRD-153's boot gate, but it's how this machine
# serves Postgres — so both come up before ignition.
set -uo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO"

echo "[launcher] $(date '+%F %H:%M:%S') starting"

# 1. Docker daemon
if ! docker info >/dev/null 2>&1; then
  echo "[launcher] docker daemon down — starting Docker Desktop"
  open -a Docker 2>/dev/null || open -a "Docker Desktop" 2>/dev/null || true
fi
for i in $(seq 1 36); do
  docker info >/dev/null 2>&1 && break
  sleep 5
done
if ! docker info >/dev/null 2>&1; then
  echo "[launcher] FATAL: docker daemon did not come up within 3 minutes"
  exit 1
fi
echo "[launcher] docker up"

# 2. Postgres on 5432 (test_82c_wiring wedges the whole suite without it)
pg_up() { nc -z localhost 5432 >/dev/null 2>&1; }
if ! pg_up; then
  echo "[launcher] postgres :5432 not reachable — starting db services from compose"
  SVCS=$(docker compose config --services 2>/dev/null | grep -E '^(postgres|db|redis|qdrant)' | tr '\n' ' ')
  if [ -n "${SVCS// /}" ]; then
    docker compose up -d $SVCS
  else
    echo "[launcher] no obvious db services in root compose — bringing up full dev stack"
    docker compose up -d
  fi
  for i in $(seq 1 36); do
    pg_up && break
    sleep 5
  done
fi
if ! pg_up; then
  echo "[launcher] FATAL: postgres :5432 never became reachable"
  exit 1
fi
echo "[launcher] postgres reachable on :5432"

# 3. Hand over (launcher's nc check replaces the container-name grep, which
# false-fails when postgres is served outside docker).
export RALPH_SKIP_DOCKER_CHECK=1
echo "[launcher] igniting chain"
exec ./scripts/ralph/overnight-prd150-153.sh
