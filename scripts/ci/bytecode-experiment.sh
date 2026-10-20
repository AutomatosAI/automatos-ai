#!/usr/bin/env bash
# PRD-262 US-032 (§9-Q7): boot the two API builds bytecode-experiment.yml made, each in
# new containers against one seeded database, and time each to /health 200.
#
# The builds are $BYTECODE_REPO:with (as committed: the boot modules ship as bytecode)
# and $BYTECODE_REPO:without (the compile step removed), in the job's own registry. The
# datastores come from docker-compose.yml unchanged. The backend does too, with an
# override written outside the checkout: it runs the build under test (BYTECODE_IMAGE),
# with no source or script mounts, WEB_CONCURRENCY=1 and the volumes a self-hoster's
# backend has (docker-compose.yml's own list since PRD-262 US-035).
#
# One boot on the empty database seeds it, and a second settles the warm path: the seed
# fingerprints are stored (US-031) and the schema is at head (US-030). Then each build
# boots twice, in the order with, without, without, with, so a drift in the runner's
# speed weighs on both. Every boot is a new container (`up --force-recreate`), never
# `restart`, which would keep the .pyc the first boot wrote. scripts/ci/
# bytecode_experiment.py times each from the container's own start and appends it to
# $BYTECODE_OUT_DIR/boots.jsonl; each boot's backend log is kept beside it.
#
# Built for a throwaway CI runner: it refuses a host that already has Automatos
# containers or volumes, and tears the stack down with its volumes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

REPO="${BYTECODE_REPO:?BYTECODE_REPO names the registry repository the builds were pushed to}"
OUT_DIR="${BYTECODE_OUT_DIR:?BYTECODE_OUT_DIR names where the records go}"
BOOTS_FILE="${OUT_DIR}/boots.jsonl"
OVERRIDE="${RUNNER_TEMP:-/tmp}/bytecode-experiment-backend.yml"
TOOL="scripts/ci/bytecode_experiment.py"
HEALTH_URL="http://localhost:${API_PORT:-8000}/health"
CONTAINER="automatos_backend"
STACK_NAME_PATTERN='^automatos_'
ORDER=(with without without with)
# The core stack's override, where the backend mounts the workspaces volume writable.
CORE_OVERRIDE="docker-compose.core.yml"

compose() {
  docker compose -f docker-compose.yml -f "$OVERRIDE" "$@"
}

refuse_existing_stack() {
  local found
  found="$( { docker ps -a --format '{{.Names}}'; docker volume ls --format '{{.Name}}'; } | grep -E "$STACK_NAME_PATTERN" || true)"
  if [ -n "$found" ]; then
    echo "::error title=Bytecode experiment::this Docker host already has Automatos containers or volumes: ${found//$'\n'/ }" >&2
    exit 1
  fi
}

write_override() {
  cat > "$OVERRIDE" <<'YAML'
services:
  backend:
    image: ${BYTECODE_IMAGE}
    build: !reset null
    pull_policy: never
    environment:
      WEB_CONCURRENCY: "1"
    volumes: !override
      - backend_data:/app/data
      - backend_logs:/app/logs
      - workspaces_data:/workspaces:ro
      - ${AUTOMATOS_WORKSPACE_DIR:-./workspaces}:/workspaces/${DEFAULT_WORKSPACE_ID:-00000000-0000-0000-0000-0000000000c1}:ro
YAML
}

# Every backend here runs without the workspace worker, on a fresh workspaces volume. Its
# read-only mount of that volume, with your deliverables folder nested in it at
# /workspaces/<DEFAULT_WORKSPACE_ID>, needs that mount point to exist in the volume:
# Docker cannot create it on a read-only mount and refuses to start the container. In the
# default stack the worker, which mounts the volume writable and starts first, creates it.
# Here a one-off backend under docker-compose.core.yml (the volume writable, as in the core
# stack) creates it and exits; the boots keep the read-only mounts of the override.
make_workspace_mountpoint() {
  compose -f "$CORE_OVERRIDE" run --rm -T --no-deps --entrypoint true backend
}

# $1 = the build (with, without; empty for a boot that is not measured), $2 = the label.
boot() {
  export BYTECODE_IMAGE="${REPO}:${1:-with}"
  echo "Boot ${2}: a new backend container from ${BYTECODE_IMAGE}"
  compose up -d --no-deps --no-build --force-recreate backend
  python3 "$TOOL" boot --variant "$1" --label "$2" --container "$CONTAINER" --url "$HEALTH_URL" --out "$BOOTS_FILE"
  docker logs -t "$CONTAINER" > "${OUT_DIR}/backend-${2}.log" 2>&1 || true
}

finish() {
  local code=$?
  compose logs -t --no-color > "${OUT_DIR}/compose.log" 2>&1 || true
  compose down -v || true
  exit "$code"
}

main() {
  mkdir -p "$OUT_DIR"
  : > "$BOOTS_FILE"
  refuse_existing_stack
  local variant round=0
  for variant in with without; do
    docker pull --quiet "${REPO}:${variant}"
  done
  write_override
  export BYTECODE_IMAGE="${REPO}:with"
  trap finish EXIT
  compose up -d --wait --no-build postgres redis minio
  make_workspace_mountpoint
  boot "" seed
  boot "" settle
  for variant in "${ORDER[@]}"; do
    round=$((round + 1))
    boot "$variant" "${variant}-${round}"
  done
}

main "$@"
