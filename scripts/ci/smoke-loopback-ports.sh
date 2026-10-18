#!/usr/bin/env bash
# PRD-262 US-005 — every port the stack publishes listens on loopback.
#
# smoke-fresh-clone.sh runs this once the backend is ready, on a stack booted the way
# a fresh clone boots it: no .env and no BIND_ADDRESS, so every ports: entry's
# ${BIND_ADDRESS:-127.0.0.1} default applies. It
#   1. starts the frontend from this checkout, unless it is already running;
#   2. requires `docker compose port frontend 3000` to print 127.0.0.1:3000. Before
#      US-005 it printed 0.0.0.0:3000, and anyone on the network could chat with the
#      backend as the owner through the web app's chat proxy;
#   3. refuses any running service that publishes a port on another address.
# The static half is orchestrator/tests/test_prd234_s1a_compose_bind.py, which reads
# every ports: entry of every compose file.
#
# Usage: smoke-loopback-ports.sh [compose-file]     (default docker-compose.yml)
# Exit 0 = every published port is on 127.0.0.1. Exit 1 = one is not, or the
# frontend did not start, or the ports could not be read.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

COMPOSE_FILE="${1:-docker-compose.yml}"
LOOPBACK="127.0.0.1"
FRONTEND_EXPECTED="${LOOPBACK}:${FRONTEND_PORT:-3000}"

fail() {
  echo "FAIL: $*" >&2
  exit 1
}

# 1. The frontend publishes :3000. The backend gates it (depends_on), and the caller
#    has already seen /health/ready, so --no-deps leaves the running backend alone.
if [ -z "$(docker compose -f "$COMPOSE_FILE" ps --status running -q frontend 2>/dev/null)" ]; then
  echo "Starting the frontend (its published port is part of this check)..."
  if ! docker compose -f "$COMPOSE_FILE" up -d --build --no-deps frontend; then
    docker compose -f "$COMPOSE_FILE" logs --tail 100 frontend >&2 || true
    fail "the frontend did not start"
  fi
fi

# 2. The address the frontend publishes on.
frontend_addr="$(docker compose -f "$COMPOSE_FILE" port frontend 3000 2>&1)"
echo "docker compose port frontend 3000 -> ${frontend_addr}"
[ "$frontend_addr" = "$FRONTEND_EXPECTED" ] \
  || fail "the frontend publishes on '${frontend_addr}', expected ${FRONTEND_EXPECTED} (BIND_ADDRESS unset)"

# 3. Every published port of every running service, one per line:
#    "<service> <address> <published port> <container port>/<protocol>".
#    `ps --format json` prints one object per line on current Compose and one array on
#    older releases; `jq -s 'flatten'` reads both.
publishers="$(docker compose -f "$COMPOSE_FILE" ps --format json | jq -rs '
  flatten | .[] | .Service as $svc | (.Publishers // [])[]
  | select(.PublishedPort > 0)
  | "\($svc) \(.URL) \(.PublishedPort) \(.TargetPort)/\(.Protocol)"')" \
  || fail "could not read the published ports (docker compose ps --format json | jq)"
[ -n "$publishers" ] || fail "no published ports found, so the check would pass for nothing"

echo "Published ports:"
echo "$publishers" | sed 's/^/  /'
if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
  {
    echo "### Published ports (PRD-262 US-005: loopback only)"
    echo ""
    echo "| Service | Address | Container port |"
    echo "|---|---|---|"
    echo "$publishers" | awk '{ printf "| %s | %s:%s | %s |\n", $1, $2, $3, $4 }'
  } >> "$GITHUB_STEP_SUMMARY"
fi

exposed="$(echo "$publishers" | awk -v lo="$LOOPBACK" '$2 != lo')"
[ -z "$exposed" ] || fail "these ports listen beyond this machine (expected ${LOOPBACK}):
${exposed}"

echo "OK: every published port listens on ${LOOPBACK}"
