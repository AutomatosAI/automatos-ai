#!/usr/bin/env bash
# PRD-262 US-040: the five-container core stack, and the six-container default an old .env keeps.
#
# smoke-images.yml's smoke-images-pr job runs this after its boot smoke, on the images that
# job built and tagged (AUTOMATOS_IMAGE_TAG), so make up pulls and builds nothing. Three
# stacks, one after the other, each with the real make and Docker:
#   1. the default from an .env written before W5 (the keys .env.example had then, the
#      three secrets filled as its QUICKSTART said, no AUTOMATOS_IMAGE_TAG, no
#      COMPOSE_FILE or COMPOSE_PROFILES): `make up` starts six containers, the workspace
#      worker among them;
#   2. on that running stack, `make up CORE=1`: the worker an earlier make up started is
#      stopped and removed, and five run;
#   3. after `down -v`, a fresh core stack: `make up CORE=1` on a checkout with no .env.
#      docker compose config shows the five services (and minio-volume-owner, the
#      one-shot that runs before MinIO and exits, PRD-262 US-041) and the backend's
#      workspaces volume mounted writable (docker-compose.core.yml); five containers
#      run; /health/ready answers 200; a workspace endpoint answers 503 "Workspace worker not running"
#      (orchestrator/core/workspace_client.py). Then the worker starts, the endpoint lists
#      the deliverables folder, a file planted there before the stack started keeps its
#      host owner, and the folders the worker makes there are the folder owner's. When uid
#      1000 cannot write the folder (this runner's user is another uid), the worker runs as
#      its owner, and its walk of the volume never enters it
#      (services/workspace-worker/entrypoint.sh).
#
# Built for a throwaway Docker host, such as a CI runner: its teardown removes the stack
# with its volumes, so it refuses to start on a host that has Automatos containers or
# volumes, or on a checkout with a .env. The deliverables folder is a temporary one
# (AUTOMATOS_WORKSPACE_DIR), so nothing in the checkout changes owner. Records land in
# $SMOKE_OUT_DIR.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_DIR="${SMOKE_OUT_DIR:-${RUNNER_TEMP:-/tmp}/smoke-core-stack}"
API_BASE="http://localhost:${API_PORT:-8000}"
MAX_WAIT_SECONDS="${SMOKE_MAX_WAIT_SECONDS:-300}"
BOOT_TOOL="scripts/ci/smoke_boot.py"
BOOTS_FILE="${OUT_DIR}/boots.jsonl"
ENV_FILE=".env"
CORE_FILES="docker-compose.yml:docker-compose.core.yml"
# docker-compose.yml's container_name values and named volumes all start with this.
STACK_NAME_PATTERN='^automatos_'
SIX="automatos_backend automatos_frontend automatos_minio automatos_postgres automatos_redis automatos_workspace_worker"
FIVE="automatos_backend automatos_frontend automatos_minio automatos_postgres automatos_redis"
# The core override's services: the five, and the one-shot that re-owns MinIO's volume and
# exits before MinIO starts (PRD-262 US-041), so it never counts as running.
CORE_SERVICES="backend frontend minio minio-volume-owner postgres redis"
# What the backend answers a workspace call with when the worker is not there
# (core/workspace_client.py WORKER_NOT_RUNNING).
NOT_RUNNING="Workspace worker not running or unreachable"
# The deliverables folders, outside the checkout: one for the first two stacks, a new one
# for the fresh core stack, with a file the worker must not re-own.
OLD_STACK_DIR="${RUNNER_TEMP:-/tmp}/smoke-core-deliverables-default"
WORKSPACE_DIR="${RUNNER_TEMP:-/tmp}/smoke-core-deliverables"
PROBE_NAME="us040-owner-probe.txt"
PROBE="${WORKSPACE_DIR}/${PROBE_NAME}"
# Compose interpolates the three secrets even to stop a stack; these stand in when .env
# is gone. Only the teardown and the config check use them.
UNUSED_SECRETS=(POSTGRES_PASSWORD=unused REDIS_PASSWORD=unused API_KEY=unused)
# The worker's user (services/workspace-worker/Dockerfile); a probe already owned by it
# could not show a re-own, so it is handed to another uid first.
WORKER_UID=1000
PROBE_OWNER="4242:4242"
# A folder the worker makes in the local workspace's root on first use
# (services/workspace-worker/workspace_manager.py DEFAULT_SUBDIRS).
WORKER_MADE="reports"

outcome="fail"

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

refuse_existing_stack() {
  local found
  found="$( { docker ps -a --format '{{.Names}}'; docker volume ls --format '{{.Name}}'; } 2>/dev/null \
    | grep -E "$STACK_NAME_PATTERN" | sort -u | tr '\n' ' ')"
  [ -z "$found" ] || fail "this Docker host already has Automatos containers or volumes (${found% }). The teardown would delete them, with their data: run it on a clean Docker host, as CI does"
  [ ! -e "$ENV_FILE" ] || fail "this checkout already has a .env, which this smoke would replace: run it on a clean checkout, as CI does"
}

# The stack's running containers, by name, sorted, on one line.
running() {
  docker ps --format '{{.Names}}' | grep -E "$STACK_NAME_PATTERN" | sort | tr '\n' ' ' | sed 's/ $//'
}

# $1 = what is expected, $2 = the stack it is.
expect_running() {
  local found
  found="$(running)"
  echo "running (${2}): ${found}"
  [ "$found" = "$1" ] || fail "${2}: expected ${1}, found ${found:-nothing}"
}

# docker compose on docker-compose.yml alone, where the worker has no profile, so it
# reaches all six; with stand-in secrets, which these commands never use.
whole_stack() {
  env "${UNUSED_SECRETS[@]}" docker compose -f docker-compose.yml "$@"
}

# Removes every container and volume of the stack, the worker's included.
teardown() {
  whole_stack down -v --remove-orphans > /dev/null 2>&1 || true
  rm -f "$ENV_FILE"
}

finish() {
  local code=$?
  if [ "$outcome" != "pass" ]; then
    whole_stack ps -a >&2 || true
    whole_stack logs --tail 120 backend workspace-worker >&2 || true
  fi
  whole_stack logs -t --no-color >> "${OUT_DIR}/compose.log" 2>&1 || true
  teardown
  exit "$code"
}

# $1 = the deliverables folder the stack mounts (AUTOMATOS_WORKSPACE_DIR, which make and
# compose both read), with the projects folder inside it.
use_deliverables() {
  export AUTOMATOS_WORKSPACE_DIR="$1"
  export LOCAL_PROJECTS_DIR="${1}/projects"
  mkdir -p "$LOCAL_PROJECTS_DIR" || fail "could not create ${LOCAL_PROJECTS_DIR}"
}

# The keys .env.example held before W5, as a self-hoster's .env carried them: the three
# secrets filled, as its QUICKSTART said; no AUTOMATOS_IMAGE_TAG, COMPOSE_FILE or
# COMPOSE_PROFILES. Frozen here, so a key a later .env.example adds cannot hide an
# upgrade that would lose the worker.
write_env_before_w5() {
  cat > "$ENV_FILE" <<EOF
POSTGRES_DB=orchestrator_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=$(openssl rand -hex 24)
POSTGRES_PORT=5432
REDIS_PASSWORD=$(openssl rand -hex 24)
REDIS_PORT=6379
API_PORT=8000
API_KEY=$(openssl rand -hex 24)
FRONTEND_PORT=3000
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000/ws
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
COMPOSIO_KEY=
FIRECRAWL_API_KEY=
NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY=
CLERK_SECRET_KEY=
CLERK_JWKS_URL=
DEFAULT_WORKSPACE_ID=
REQUIRE_AUTH=true
ENVIRONMENT=development
LOG_LEVEL=INFO
ADMINER_PORT=8080
EOF
  [ -s "$ENV_FILE" ] || fail "could not write the pre-W5 .env"
}

# 1. The default stack from an .env written before W5: six containers.
check_an_old_env_starts_six() {
  write_env_before_w5
  echo "make up with an .env written before W5 (AUTOMATOS_IMAGE_TAG=${AUTOMATOS_IMAGE_TAG})..."
  make up > "${OUT_DIR}/make-up-old-env.log" 2>&1 || { cat "${OUT_DIR}/make-up-old-env.log" >&2; fail "make up failed on an .env written before W5"; }
  expect_running "$SIX" "make up from an .env written before W5"
}

# 2. make up CORE=1 on the running six: the worker goes, five stay.
check_core_replaces_a_running_six() {
  echo "make up CORE=1 on the running six-container stack..."
  make up CORE=1 > "${OUT_DIR}/make-up-core-switch.log" 2>&1 || { cat "${OUT_DIR}/make-up-core-switch.log" >&2; fail "make up CORE=1 failed on a running six-container stack"; }
  expect_running "$FIVE" "make up CORE=1 after make up"
}

# The file the worker must leave alone, owned by someone other than the worker's user.
plant_owner_probe() {
  use_deliverables "$WORKSPACE_DIR"
  printf 'PRD-262 US-040: this file keeps its owner when the worker starts.\n' > "$PROBE" || fail "could not write ${PROBE}"
  if [ "$(stat -c '%u' "$PROBE")" = "$WORKER_UID" ]; then
    sudo -n chown "$PROBE_OWNER" "$PROBE" 2>/dev/null \
      || fail "the probe is owned by uid ${WORKER_UID}, the worker's user, so a re-own would not show: run this as another user, or with passwordless sudo"
  fi
  probe_owner="$(stat -c '%u:%g' "$PROBE")"
  folder_owner="$(stat -c '%u:%g' "$WORKSPACE_DIR")"
  echo "planted ${PROBE} (owner ${probe_owner}) in a folder owned by ${folder_owner}"
}

# docker compose itself says the override does what it should: five services and the
# MinIO one-shot, and the backend's workspaces volume writable, so Docker can create the
# deliverables folder's mount point inside it on a fresh volume.
check_core_config() {
  local config services writable
  config="$(env "${UNUSED_SECRETS[@]}" COMPOSE_FILE="$CORE_FILES" docker compose config --format json)" || fail "docker compose config failed with docker-compose.core.yml"
  services="$(printf '%s' "$config" | jq -r '.services | keys[]' | sort | tr '\n' ' ' | sed 's/ $//')"
  [ "$services" = "$CORE_SERVICES" ] || fail "the core override runs ${services}, not ${CORE_SERVICES}"
  writable="$(printf '%s' "$config" | jq -r '.services.backend.volumes[] | select(.target == "/workspaces") | (.read_only // false)')"
  [ "$writable" = "false" ] || fail "the core override leaves the backend's /workspaces volume read-only (${writable:-not mounted})"
}

# $1 = the expected HTTP status, $2 = text the body must hold, $3 = where the body goes.
call_workspace_files() {
  local status
  status="$(curl -sS --max-time 60 -o "$3" -w '%{http_code}' \
    -H "X-Workspace-ID: ${DEFAULT_WORKSPACE_ID}" \
    "${API_BASE}/api/workspaces/${DEFAULT_WORKSPACE_ID}/files?path=.")" || fail "GET /api/workspaces/…/files did not answer"
  echo "GET /api/workspaces/${DEFAULT_WORKSPACE_ID}/files -> ${status}: $(head -c 300 "$3")"
  [ "$status" = "$1" ] || fail "the workspace endpoint answered ${status}, expected ${1}"
  grep -qF "$2" "$3" || fail "the workspace endpoint's answer does not say '${2}'"
}

# 3. A fresh core stack: five containers, ready, the worker's error; then the worker.
check_a_fresh_core_stack() {
  plant_owner_probe
  check_core_config
  echo "make up CORE=1 on a fresh checkout (no .env, no volumes)..."
  make up CORE=1 > "${OUT_DIR}/make-up-core.log" 2>&1 || { cat "${OUT_DIR}/make-up-core.log" >&2; fail "make up CORE=1 failed"; }
  expect_running "$FIVE" "make up CORE=1"
  python3 "$BOOT_TOOL" --label core --container automatos_backend --url "ready=${API_BASE}/health/ready" \
    --timeout "$MAX_WAIT_SECONDS" --out "$BOOTS_FILE" || fail "/health/ready did not answer 200 within ${MAX_WAIT_SECONDS}s"
  call_workspace_files 503 "$NOT_RUNNING" "${OUT_DIR}/files-without-worker.json"
  echo "starting the worker on the core stack (naming it turns its profile on)..."
  COMPOSE_FILE="$CORE_FILES" docker compose up -d --no-build --wait --wait-timeout "$MAX_WAIT_SECONDS" workspace-worker \
    || fail "the workspace worker did not start and turn healthy on the core stack"
  call_workspace_files 200 "$PROBE_NAME" "${OUT_DIR}/files-with-worker.json"
  local owner
  owner="$(stat -c '%u:%g' "$PROBE")"
  [ "$owner" = "$probe_owner" ] \
    || fail "the worker re-owned ${PROBE} from ${probe_owner} to ${owner}: its entrypoint's walk entered the deliverables folder"
  echo "${PROBE} kept its owner, ${owner}"
  owner="$(stat -c '%u:%g' "${WORKSPACE_DIR}/${WORKER_MADE}")" || fail "the worker made no ${WORKER_MADE}/ in ${WORKSPACE_DIR}"
  [ "$owner" = "$folder_owner" ] \
    || fail "the worker made ${WORKSPACE_DIR}/${WORKER_MADE} as ${owner}, not as the folder's owner ${folder_owner}"
  echo "the worker made ${WORKER_MADE}/ there as the folder's owner, ${owner}"
}

main() {
  mkdir -p "$OUT_DIR" || exit 1
  : > "$BOOTS_FILE"
  [ -n "${AUTOMATOS_IMAGE_TAG:-}" ] || fail "AUTOMATOS_IMAGE_TAG must name the images to run (smoke-images-pr's tag)"
  export DEFAULT_WORKSPACE_ID="${DEFAULT_WORKSPACE_ID:-00000000-0000-0000-0000-0000000000c1}"
  # The .env holds the secrets: none from the shell.
  unset POSTGRES_PASSWORD REDIS_PASSWORD API_KEY COMPOSE_FILE COMPOSE_PROFILES
  refuse_existing_stack
  trap finish EXIT
  use_deliverables "$OLD_STACK_DIR"
  check_an_old_env_starts_six
  check_core_replaces_a_running_six
  whole_stack logs -t --no-color > "${OUT_DIR}/compose.log" 2>&1 || true
  teardown
  check_a_fresh_core_stack
  outcome="pass"
  echo "OK: an .env from before W5 started six containers; make up CORE=1 stopped the worker and ran five;" \
    "a fresh core stack answered /health/ready 200 and 'Workspace worker not running', and the worker" \
    "started later left the deliverables folder's owners alone and wrote there as its owner"
}

main "$@"
