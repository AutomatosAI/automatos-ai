#!/usr/bin/env bash
# PRD-176 Headline B — fresh-clone deployability smoke test, and since PRD-262 US-002
# the measurement behind the boot bars in PRD-262 §8.
#
# The open-core thesis bar: `git clone && docker compose up` yields a working
# local instance with NO external credentials. This script proves it end to end:
# it provides ONLY the required local secrets (POSTGRES_PASSWORD, REDIS_PASSWORD,
# API_KEY) and NO external SaaS credentials (no Clerk, no real S3/AWS, no LLM
# keys), brings the compose stack up, and asserts the backend `/health` and
# `/health/ready` and the frontend's `:3000` answer HTTP 200.
#
# It exercises together: the repointed initdb mount (F009), the wait-migrate-seed
# entrypoint (F051), local MinIO via S3_ENDPOINT_URL (F089), the local-safe
# railway.internal defaults (F068), and AUTH_EDITION=local with the
# DEFAULT_WORKSPACE_ID that validate_auth_edition() hard-requires (PRD-209 W5).
# The lane is HARD in CI (no continue-on-error, PRD-209 US-004); still
# non-required in branch protection until the owner flips it (Q2).
#
# It is built for a throwaway Docker host, such as a CI runner: its teardown removes the
# stack together with its volumes, and the compose file names those volumes
# (automatos_postgres_data, ...), so on a machine running its own Automatos stack it
# would delete that stack's database. It refuses to start on a host that already has
# Automatos containers or volumes, before it builds, pulls or tears anything down.
# SMOKE_IMAGES=pull (below) also empties the Docker image cache, which deletes every
# image on the host, Automatos's or not. Outside CI (CI=true) it refuses unless
# SMOKE_EMPTY_IMAGE_CACHE=true says the host's images may go.
#
# What it measures (PRD-262 US-002), every time from the container's own start
# (`docker inspect -f '{{.State.StartedAt}}'`), never from this script's sleeps,
# polling every second:
#   1. fresh boot on an empty database: backend start to /health 200 (liveness)
#      and to /health/ready 200 (readiness), each timed separately;
#   2. the frontend, started once the backend is healthy: :3000 must answer 200,
#      and the address `docker compose port frontend 3000` reports is recorded
#      (127.0.0.1:3000 since PRD-262 US-005, checked below);
#   3. two warm restarts, `docker compose restart backend` with the database kept,
#      timed the same way; the report gives their median.
# The step summary always shows (pass or fail): those times, Started → Healthy per
# service from Docker's start and health_status events, the /health/bootstrap
# stage table of each boot and the alembic runs per boot. scripts/ci/smoke_boot.py
# does the polling and scripts/ci/smoke_report.py the report; the records and logs
# land in $SMOKE_OUT_DIR, which the workflow uploads. No time is a bar yet: the lane fails only when a
# boot or the frontend never answers 200 (PRD-262 US-004 records the baselines).
#
# PRD-262 US-005: once the frontend answers, scripts/ci/smoke-loopback-ports.sh asserts
# that it, and every other published port, listens on 127.0.0.1 (`docker compose port
# frontend 3000` prints 127.0.0.1:3000).
#
# PRD-262 US-003 runs the same boot steps on images instead of a source build, for
# .github/workflows/smoke-images.yml. SMOKE_IMAGES picks where the app images come from:
#   (unset)   build them from this checkout (the fresh-clone lane, above), as make build
#             does: tagged local-<sha7> and labelled with HEAD (AUTOMATOS_REVISION);
#   pull      the published path a self-hoster takes (PRD-262 US-036): the Docker image
#             cache is emptied, then `make up` runs on this clean checkout. With no
#             AUTOMATOS_IMAGE_TAG (an empty one counts as none) make up picks the tag
#             itself; a weekly or manual run pins it. make up pulls and starts the whole
#             stack; it is timed, the bytes it brought in are counted, and its output is
#             kept and searched for registry rate limits (Docker Hub's 429). It runs in
#             the background, so the backend's and the frontend's boots are timed from
#             their own start while it brings them up, never from when it returns; it
#             must then exit 0;
#   prebuilt  the pull-request variant: the workflow built the three amd64 images and
#             tagged them with the GHCR names and AUTOMATOS_IMAGE_TAG, so compose
#             neither pulls nor builds them.
# The stack starts with `up --no-build` (make up's own, with `pull`), and
# scripts/ci/smoke_images.py asserts what booted: in the image modes the backend runs the
# production command (no --reload, SMOKE_BACKEND_WORKERS workers), and in every mode each
# app container runs its ghcr.io/automatosai/automatos-* image carrying the commit's
# revision label (SMOKE_EXPECT_REVISION; HEAD for a source build, whose tag must be
# local-*). With `pull`, each image's RepoDigests must also hold the index digest its tag
# resolves to on GHCR, so nothing was built locally, and the app images exist only under
# that tag; no image may carry a local-* tag or compose's own <project>-<service> name.
# The step summary adds make up's run, the first run from its start to :3000 200, and
# those checks.
#
# PRD-262 US-008, in the pull-request variant (`prebuilt`) only: once the worker count
# checks out, scripts/ci/smoke_worker_restart.py kills one uvicorn worker of the
# production-stage backend and needs uvicorn to start a replacement within 30 s, with
# /health 200 and no container restart. uvicorn 0.24 never replaced a dead worker.
#
# PRD-262 US-017, when it builds from this checkout: the build log is kept, and
# scripts/ci/build_context.py reads what the backend's build sent as its context off it.
# The workflow first seeds dummy uploads and caches where a self-hoster's checkout
# collects them, each as large as the budget, so one that reaches the context fails it.
#
# PRD-262 US-021, in every mode: the API image keeps only the botocore models of the
# services we call. Once the frontend and the images check out, and before the warm
# restarts, scripts/ci/smoke_document_presign.py uploads a document through the backend,
# which puts it in MinIO through that botocore, and the host fetches it back by the
# presigned URL the Documents page's download call answers with.
#
# PRD-262 US-035, in every mode: compose runs the backend's production stage, and hot
# reload is `make dev`'s. scripts/ci/smoke_production_backend.py checks, after the fresh
# boot, that no bind mount lays the checkout over /app, nothing runs --reload, and the
# process the entrypoint's watchdog signals (SERVER_PID) is uvicorn itself, with no sh in
# between (docker top). After the warm restarts it writes a key to envs/api.local alone,
# recreates the backend with `docker compose up -d --no-deps backend` as the night tooling
# does, and finds the key inside it (a checkout's own envs/api.local is kept and put back).
# Last, with an SSE client holding the board's event stream open, `docker compose stop
# backend` must finish in under STOP_MAX_SECONDS. The recreated backend is a new
# container, so the report reads the earlier one's log from the copy taken before it.
#
# Exit 0 = every boot answered 200 with no external creds, every published port is on
# loopback, the document came back, and the backend ran as the production stage should.
# Exit 1 = a boot failed, a port listens beyond this machine, the backend's build context
# is over its budget, the document did not come back from object storage, or a US-035
# check failed.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

COMPOSE_FILE="docker-compose.yml"
SMOKE_IMAGES="${SMOKE_IMAGES:-}"
# smoke_images.py's name for where the app images come from; a source build is `build`.
IMAGES_MODE="${SMOKE_IMAGES:-build}"
COMPOSE_ARGS=(-f "$COMPOSE_FILE")
# An empty AUTOMATOS_IMAGE_TAG would make compose run :edge rather than the tag make up
# wrote to .env, so an empty one is no tag (PRD-262 US-036).
[ -n "${AUTOMATOS_IMAGE_TAG:-}" ] || unset AUTOMATOS_IMAGE_TAG
# A source build, as make build does it (PRD-262 US-036): the images carry HEAD as their
# revision label and a local-<sha7> tag, never an official one.
if [ -z "$SMOKE_IMAGES" ]; then
  AUTOMATOS_REVISION="${AUTOMATOS_REVISION:-$(git rev-parse HEAD 2>/dev/null)}"
  export AUTOMATOS_REVISION
  export AUTOMATOS_IMAGE_TAG="${AUTOMATOS_IMAGE_TAG:-local-${AUTOMATOS_REVISION:0:7}}"
  SMOKE_EXPECT_REVISION="${SMOKE_EXPECT_REVISION:-$AUTOMATOS_REVISION}"
fi
API_BASE="http://localhost:${API_PORT:-8000}"
HEALTH_URL="${API_BASE}/health"
# Readiness probe (main.py:/health/ready) — 200 only once the boot lifespan has
# finished (migrations applied, trust gate, Phase-2 extensions, READY stage); 503
# before. This is what Railway healthchecks: /health/ready is true only when the
# local edition booted far enough to serve, /health whenever the process answers.
# Under uvicorn 0.24 the lifespan runs before the socket listens, so today both
# first answer at the same moment; they are timed separately so the gap shows
# once the boot moves work off the critical path.
READY_URL="${API_BASE}/health/ready"
BOOTSTRAP_URL="${API_BASE}/health/bootstrap"
FRONTEND_URL="http://localhost:${FRONTEND_PORT:-3000}/"
BACKEND_CONTAINER="automatos_backend"
FRONTEND_CONTAINER="automatos_frontend"
MAX_WAIT_SECONDS="${SMOKE_MAX_WAIT_SECONDS:-300}"
WARM_RESTARTS=2
OUT_DIR="${SMOKE_OUT_DIR:-${RUNNER_TEMP:-/tmp}/smoke-fresh-clone}"
BOOT_TOOL="scripts/ci/smoke_boot.py"
REPORT_TOOL="scripts/ci/smoke_report.py"
IMAGES_TOOL="scripts/ci/smoke_images.py"
RESTART_TOOL="scripts/ci/smoke_worker_restart.py"
CONTEXT_TOOL="scripts/ci/build_context.py"
DOCUMENT_TOOL="scripts/ci/smoke_document_presign.py"
BOOTS_FILE="${OUT_DIR}/boots.jsonl"
EVENTS_FILE="${OUT_DIR}/events.jsonl"
# smoke_images.py's records (it reads them back from OUT_DIR for the report).
PULL_FILE="${OUT_DIR}/pull.json"
# make up's output (SMOKE_IMAGES=pull), kept for the rate-limit search and the artifact,
# and its exit code, written when it ends.
MAKE_UP_LOG="${OUT_DIR}/make-up.log"
MAKE_UP_EXIT="${OUT_DIR}/make-up.exit"
PROCESSES_FILE="${OUT_DIR}/processes.json"
IMAGES_FILE="${OUT_DIR}/images.json"
# smoke_worker_restart.py's record (PRD-262 US-008).
WORKER_RESTART_FILE="${OUT_DIR}/worker-restart.json"
# The source build's log (PRD-262 US-017: build_context.py reads the context transfer off it).
BUILD_LOG="${OUT_DIR}/build.log"
# smoke_document_presign.py's record (PRD-262 US-021).
DOCUMENT_FILE="${OUT_DIR}/document-presign.json"
# Where compose publishes MinIO on this host: presigned links must point here
# (S3_PUBLIC_ENDPOINT_URL in envs/api.defaults).
STORAGE_URL="http://localhost:${MINIO_PORT:-9000}"
# Outside OUT_DIR, so the resolved compose file (with the stack's env) is not uploaded.
COMPOSE_CONFIG_FILE="${RUNNER_TEMP:-/tmp}/smoke-compose-config.json"
# PRD-262 US-035: the production backend's checks, and their records.
PRODUCTION_TOOL="scripts/ci/smoke_production_backend.py"
PRODUCTION_FILE="${OUT_DIR}/production-backend.json"
API_LOCAL_RECORD="${OUT_DIR}/api-local.json"
STOP_FILE="${OUT_DIR}/stop-backend.json"
# The SSE stream the browser keeps open (frontend/hooks/use-board-event-stream.ts). It
# sends its first event at once and then only heartbeats, so it never ends by itself.
SSE_URL="${API_BASE}/api/v1/tasks/stream"
STOP_MAX_SECONDS=8
# The night tooling's override file (docker-compose.yml's env_file), the key the smoke
# writes to it alone, and where a checkout's own copy waits until finish puts it back.
# Outside OUT_DIR, which the workflow uploads.
API_LOCAL="envs/api.local"
API_LOCAL_KEY="AUTOMATOS_SMOKE_API_LOCAL_PROBE"
API_LOCAL_SAVED="${RUNNER_TEMP:-/tmp}/smoke-api.local.saved"
# The backend's log up to its recreate: the new container's log starts empty, so the
# report reads the earlier boots' lines from this copy.
EARLIER_BACKEND_LOG="${RUNNER_TEMP:-/tmp}/smoke-backend-before-recreate.log"
# The production stage's CMD runs `uvicorn ... --workers 4` (orchestrator/Dockerfile).
BACKEND_WORKERS="${SMOKE_BACKEND_WORKERS:-4}"
# docker-compose.yml's container_name values and named volumes all start with this.
STACK_NAME_PATTERN='^automatos_'

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

outcome="fail"
failure="the smoke stopped before its first boot"
build_seconds=""
events_pid=""
up_started_ns=""
images_before=0
api_local_written=""
earlier_backend_id=""
make_up_pid=""

compose() {
  docker compose "${COMPOSE_ARGS[@]}" "$@"
}

fail() {
  failure="$1"
  echo "FAIL: $1" >&2
  exit 1
}

# The prebuilt mode needs the tag its images carry; `pull` lets make up pick one when
# none is given. A typo in the mode must not fall back to a build.
check_mode() {
  case "$SMOKE_IMAGES" in
    ""|pull) ;;
    prebuilt)
      [ -n "${AUTOMATOS_IMAGE_TAG:-}" ] || fail "SMOKE_IMAGES=prebuilt needs AUTOMATOS_IMAGE_TAG (the tag to boot)" ;;
    *) fail "SMOKE_IMAGES must be unset, pull or prebuilt, not '${SMOKE_IMAGES}'" ;;
  esac
}

# The tag the stack runs: the one exported, else the one make up wrote to .env.
image_tag() {
  if [ -n "${AUTOMATOS_IMAGE_TAG:-}" ]; then
    printf '%s' "$AUTOMATOS_IMAGE_TAG"
  else
    sed -n 's/^AUTOMATOS_IMAGE_TAG=//p' .env 2>/dev/null | tail -1
  fi
}

# Runs before the exit trap is set, like refuse_existing_stack, so a refusal deletes nothing.
refuse_emptying_a_kept_image_cache() {
  [ "$SMOKE_IMAGES" = "pull" ] || return 0
  [ "${CI:-}" = "true" ] || [ "${SMOKE_EMPTY_IMAGE_CACHE:-}" = "true" ] \
    || fail "SMOKE_IMAGES=pull starts from an empty Docker image cache, so it deletes every image on this host. Run it on a throwaway host such as a CI runner, or set SMOKE_EMPTY_IMAGE_CACHE=true to let it delete them here"
}

# Runs before the exit trap is set, so a refusal tears nothing down.
refuse_existing_stack() {
  local found
  found="$( { docker ps -a --format '{{.Names}}'; docker volume ls --format '{{.Name}}'; } 2>/dev/null \
    | grep -E "$STACK_NAME_PATTERN" | sort -u | tr '\n' ' ')"
  [ -z "$found" ] || fail "this Docker host already has Automatos containers or volumes (${found% }). The smoke's teardown would delete them, with their data: run it on a clean Docker host, as CI does"
}

print_banner() {
  echo "========================================="
  echo "PRD-176 fresh-clone smoke test (PRD-262 US-002 boot timings)"
  echo "  images     : ${SMOKE_IMAGES:-built from this checkout} (AUTOMATOS_IMAGE_TAG=${AUTOMATOS_IMAGE_TAG:-make up picks it})"
  echo "  health url : ${HEALTH_URL} (liveness), ${READY_URL} (readiness)"
  echo "  frontend   : ${FRONTEND_URL}"
  echo "  max wait   : ${MAX_WAIT_SECONDS}s per boot, polled every second"
  echo "  warm runs  : ${WARM_RESTARTS} x docker compose restart backend"
  echo "  auth       : AUTH_EDITION=${AUTH_EDITION} REQUIRE_AUTH=${REQUIRE_AUTH}"
  echo "  workspace  : DEFAULT_WORKSPACE_ID=${DEFAULT_WORKSPACE_ID}"
  echo "  external creds: NONE (no Clerk / AWS / LLM keys)"
  echo "  records    : ${OUT_DIR}"
  echo "========================================="
}

# Docker replays only a short buffer of past events, and every health probe adds
# exec events to it, so the start and health_status events are streamed to a file
# for the whole run instead of being read back at the end. --since replays the
# moments between this call and the stream attaching, so no early start is missed.
start_event_log() {
  docker events --since "$(date +%s)" \
    --filter type=container --filter event=start --filter event=health_status \
    --format '{{json .}}' > "$EVENTS_FILE" &
  events_pid=$!
}

stop_event_log() {
  if [ -n "$events_pid" ]; then
    kill "$events_pid" 2>/dev/null || true
    wait "$events_pid" 2>/dev/null || true
    events_pid=""
  fi
}

# The build is timed for the record only: CI build time has no bar (PRD-262 §7).
# Piped, so BuildKit prints its plain progress, `transferring context` lines included.
build_images() {
  echo "Building the backend (production stage), workspace-worker and frontend images..."
  local started=$SECONDS
  compose build backend workspace-worker frontend 2>&1 | tee "$BUILD_LOG" || fail "docker compose build failed"
  build_seconds=$((SECONDS - started))
  echo "Images built in ${build_seconds}s"
  check_build_context
}

# PRD-262 US-017: the backend's build sent only code. The production stage copies the
# context, and orchestrator/.dockerignore keeps the checkout's uploads and caches out of it.
check_build_context() {
  python3 "$CONTEXT_TOOL" check --service backend "$BUILD_LOG" \
    || fail "the backend's build context is over its budget, or the build log does not show it"
}

# The published path starts from an empty Docker cache, as a new self-hoster's does.
empty_docker_cache() {
  images_before="$(docker image ls -q | wc -l | tr -d ' ')"
  docker image prune -af > /dev/null || fail "could not empty the Docker image cache"
  [ -z "$(docker image ls -q)" ] || fail "the Docker image cache is not empty after the prune"
  echo "Docker image cache emptied (${images_before} image(s) were there)"
}

# `make up` on this clean checkout (PRD-262 US-036): it picks the app images' tag (unless
# AUTOMATOS_IMAGE_TAG pins one), pulls them and the datastores, and starts the whole
# stack with --no-build. smoke_images.py times it, counts the bytes the runner received
# and keeps its output in MAKE_UP_LOG, where it looks for registry rate limits. It runs
# in the background while the boots are timed; finish_make_up waits for it.
pull_images() {
  empty_docker_cache
  rm -f "$MAKE_UP_EXIT"
  echo "make up from an empty Docker cache (AUTOMATOS_IMAGE_TAG=${AUTOMATOS_IMAGE_TAG:-not set: make up picks it})..."
  {
    python3 "$IMAGES_TOOL" pull --out "$PULL_FILE" --images-before "$images_before" --log "$MAKE_UP_LOG" -- \
      make up
    echo "$?" > "$MAKE_UP_EXIT"
  } &
  make_up_pid=$!
}

# Waits for the background make up (SMOKE_IMAGES=pull), which must exit 0.
finish_make_up() {
  [ -n "$make_up_pid" ] || return 0
  wait "$make_up_pid"
  make_up_pid=""
  [ "$(cat "$MAKE_UP_EXIT" 2>/dev/null)" = "0" ] || fail "make up failed"
}

prepare_images() {
  case "$SMOKE_IMAGES" in
    "") build_images ;;
    pull) pull_images ;;
    prebuilt) echo "Booting the images this job built (${AUTOMATOS_IMAGE_TAG}): no build, no pull" ;;
  esac
}

# $1 = the boot's label (fresh, warm-1, ...); further arguments go to smoke_boot.py. Times
# /health and /health/ready from the backend's StartedAt, then fetches /health/bootstrap.
wait_backend() {
  local label="$1"
  shift
  python3 "$BOOT_TOOL" --label "$label" --container "$BACKEND_CONTAINER" \
    --url "health=${HEALTH_URL}" --url "ready=${READY_URL}" \
    --fetch "bootstrap=${BOOTSTRAP_URL}" \
    --timeout "$MAX_WAIT_SECONDS" --out "$BOOTS_FILE" "$@"
}

# SMOKE_IMAGES=pull: make up starts the stack, so the fresh boot is timed from the backend's
# own start as soon as make up has started it. A failed make up is the failure to report.
boot_fresh_from_make_up() {
  echo "Timing the fresh boot make up starts: liveness (/health), then readiness (/health/ready)..."
  wait_backend fresh --wait-start --stop-file "$MAKE_UP_EXIT" && return 0
  finish_make_up
  fail "fresh boot: readiness not reached within ${MAX_WAIT_SECONDS}s (/health and /health/ready must both answer 200)"
}

boot_fresh() {
  if [ "$SMOKE_IMAGES" = "pull" ]; then
    boot_fresh_from_make_up
    return
  fi
  echo "Starting compose stack (postgres, redis, minio, workspace-worker, backend) on an empty database..."
  up_started_ns="$(date +%s%N)"
  compose up -d --no-build postgres redis minio backend || fail "docker compose up failed"
  echo "Timing the fresh boot: liveness (/health), then readiness (/health/ready)..."
  wait_backend fresh \
    || fail "fresh boot: readiness not reached within ${MAX_WAIT_SECONDS}s (/health and /health/ready must both answer 200)"
}

# SMOKE_IMAGES=pull: make up starts the frontend once the backend is healthy; it is timed
# from its own start, and Docker says where it published :3000.
boot_frontend_from_make_up() {
  echo "Timing the frontend make up starts once the backend is healthy..."
  python3 "$BOOT_TOOL" --label frontend --container "$FRONTEND_CONTAINER" \
    --wait-start --stop-file "$MAKE_UP_EXIT" --published-port 3000 --url "ui=${FRONTEND_URL}" \
    --timeout "$MAX_WAIT_SECONDS" --out "$BOOTS_FILE" && return 0
  finish_make_up
  fail "the frontend never answered 200 on ${FRONTEND_URL} within ${MAX_WAIT_SECONDS}s"
}

boot_frontend() {
  if [ "$SMOKE_IMAGES" = "pull" ]; then
    boot_frontend_from_make_up
    return
  fi
  echo "Starting the frontend (compose holds it until the backend is healthy)..."
  compose up -d --no-build frontend || fail "docker compose up frontend failed"
  local address
  address="$(compose port frontend 3000)" || fail "docker compose port frontend 3000 failed"
  echo "Frontend published on ${address} (docker compose port frontend 3000)"
  python3 "$BOOT_TOOL" --label frontend --container "$FRONTEND_CONTAINER" \
    --url "ui=${FRONTEND_URL}" --meta "address=${address}" \
    --timeout "$MAX_WAIT_SECONDS" --out "$BOOTS_FILE" \
    || fail "the frontend never answered 200 on ${FRONTEND_URL} within ${MAX_WAIT_SECONDS}s"
}

# PRD-262 US-005: the frontend, and every other published port, listens on 127.0.0.1.
# The frontend is already running, so the check starts nothing; it only reads ports.
check_loopback_ports() {
  bash scripts/ci/smoke-loopback-ports.sh "$COMPOSE_FILE" \
    || fail "a published port listens beyond this machine (expected 127.0.0.1), or the ports could not be read"
}

# The first CI boot of the production stage (image modes): no --reload, the expected
# worker count. check_worker_restart runs right after it.
check_backend_processes() {
  [ -n "$SMOKE_IMAGES" ] || return 0
  python3 "$IMAGES_TOOL" workers --container "$BACKEND_CONTAINER" --expect "$BACKEND_WORKERS" \
    --out "$PROCESSES_FILE" \
    || fail "the backend does not run the production command (no --reload, ${BACKEND_WORKERS} workers)"
}

# PRD-262 US-008, smoke-images' pull-request variant only: SIGKILL one uvicorn worker and
# require uvicorn to replace it within 30 s, /health 200, and no container restart.
# The main runs (pull) are measurements, so they kill nothing.
check_worker_restart() {
  [ "$SMOKE_IMAGES" = "prebuilt" ] || return 0
  python3 "$RESTART_TOOL" --container "$BACKEND_CONTAINER" --health-url "$HEALTH_URL" \
    --out "$WORKER_RESTART_FILE" --summary "${GITHUB_STEP_SUMMARY:-/dev/stdout}" \
    || fail "uvicorn did not replace a killed worker within 30s with /health 200 and no container restart"
}

# Every mode: every app container runs the image compose names for it, with this
# commit's revision label. With `pull`, its RepoDigests prove it was pulled and nothing
# was built; a source build must carry a local-* tag (PRD-262 US-036).
check_images() {
  compose config --format json > "$COMPOSE_CONFIG_FILE" || fail "docker compose config failed"
  python3 "$IMAGES_TOOL" images --mode "$IMAGES_MODE" --config "$COMPOSE_CONFIG_FILE" \
    --expect-revision "${SMOKE_EXPECT_REVISION:-}" --out "$IMAGES_FILE" \
    || fail "an app container does not run the image this smoke expects: a wrong tag or revision label, or an image built instead of pulled"
}

# PRD-262 US-021: a document uploaded to the backend lands in MinIO through the pruned
# botocore, and the host fetches it back by its presigned URL.
check_document_storage() {
  python3 "$DOCUMENT_TOOL" --api "$API_BASE" --storage "$STORAGE_URL" \
    --out "$DOCUMENT_FILE" --summary "${GITHUB_STEP_SUMMARY:-/dev/stdout}" \
    || fail "a document uploaded to the backend did not come back from object storage by its presigned URL"
}

warm_restarts() {
  local i
  for i in $(seq 1 "$WARM_RESTARTS"); do
    echo "Warm restart ${i}/${WARM_RESTARTS}: docker compose restart backend (database kept)..."
    compose restart backend || fail "docker compose restart backend failed (warm restart ${i})"
    wait_backend "warm-${i}" \
      || fail "warm restart ${i}: readiness not reached within ${MAX_WAIT_SECONDS}s"
  done
}

# PRD-262 US-035: the backend runs the production stage. No bind mount lays the checkout
# over /app, nothing runs --reload, and SERVER_PID, the process the entrypoint's watchdog
# and `docker stop` signal, is uvicorn itself (docker top), with no sh in between.
check_production_backend() {
  python3 "$PRODUCTION_TOOL" process --container "$BACKEND_CONTAINER" \
    --out "$PRODUCTION_FILE" --summary "${GITHUB_STEP_SUMMARY:-/dev/stdout}" \
    || fail "the backend does not run as the production stage should: a bind mount over /app, --reload, or a shell between the entrypoint and uvicorn"
}

# $1 = the value. Appends the key to envs/api.local, after any lines a checkout's own copy
# holds; that copy waits in API_LOCAL_SAVED until finish puts it back.
write_api_local_probe() {
  rm -f "$API_LOCAL_SAVED"
  if [ -e "$API_LOCAL" ]; then
    cp -p "$API_LOCAL" "$API_LOCAL_SAVED" || fail "could not set this checkout's envs/api.local aside"
  fi
  api_local_written="true"
  printf '\n%s=%s\n' "$API_LOCAL_KEY" "$1" >> "$API_LOCAL" || fail "could not write envs/api.local"
}

restore_api_local() {
  [ -n "$api_local_written" ] || return 0
  if [ -e "$API_LOCAL_SAVED" ]; then
    mv -f "$API_LOCAL_SAVED" "$API_LOCAL"
  else
    rm -f "$API_LOCAL"
  fi
}

# PRD-262 US-035 (TESTER): the night tooling edits envs/api.local, then recreates the
# backend with `docker compose up -d --no-deps backend`. A key only that file sets must
# reach the new container. The old container's log is copied first, for the report.
check_api_local_reaches_backend() {
  local value="us035-$$-${RANDOM}"
  write_api_local_probe "$value"
  earlier_backend_id="$(docker inspect -f '{{.Id}}' "$BACKEND_CONTAINER")" \
    || fail "could not read the backend container's id before recreating it"
  docker logs -t "$BACKEND_CONTAINER" > "$EARLIER_BACKEND_LOG" 2>&1 \
    || echo "::warning title=smoke-fresh-clone::could not copy the backend's log before its recreate; the report misses the earlier boots' alembic runs"
  echo "Recreating the backend with a key only envs/api.local sets (docker compose up -d --no-deps backend)..."
  compose up -d --no-build --no-deps backend || fail "docker compose up -d --no-deps backend failed"
  wait_backend recreate \
    || fail "the recreated backend: readiness not reached within ${MAX_WAIT_SECONDS}s"
  python3 "$PRODUCTION_TOOL" env --container "$BACKEND_CONTAINER" --key "$API_LOCAL_KEY" --value "$value" \
    --out "$API_LOCAL_RECORD" --summary "${GITHUB_STEP_SUMMARY:-/dev/stdout}" \
    || fail "a key set only in envs/api.local did not reach the backend recreated by docker compose up -d --no-deps backend"
}

# PRD-262 US-035: with an SSE client connected, `docker compose stop backend` finishes in
# under STOP_MAX_SECONDS: uvicorn gives up on the open stream after
# UVICORN_TIMEOUT_GRACEFUL_SHUTDOWN (envs/api.defaults) instead of holding the stop until
# Docker's 10 s kill. It stops the backend, so it runs last.
check_stop_with_sse_client() {
  echo "Stopping the backend with an SSE client connected (${SSE_URL})..."
  python3 "$PRODUCTION_TOOL" stop --url "$SSE_URL" --max-seconds "$STOP_MAX_SECONDS" \
    --out "$STOP_FILE" --summary "${GITHUB_STEP_SUMMARY:-/dev/stdout}" \
    -- docker compose "${COMPOSE_ARGS[@]}" stop backend \
    || fail "docker compose stop backend did not finish in under ${STOP_MAX_SECONDS}s with an SSE client connected"
}

dump_failure_logs() {
  echo "----- postgres logs (tail) — initdb/schema errors surface here -----" >&2
  compose logs --tail 200 postgres >&2 || true
  echo "----- backend logs (tail) -----" >&2
  compose logs --tail 150 backend >&2 || true
  echo "----- frontend logs (tail) -----" >&2
  compose logs --tail 80 frontend >&2 || true
}

# The log of the backend container check_api_local_reaches_backend replaced, if it did.
replaced_backend_log() {
  [ -n "$earlier_backend_id" ] || return 0
  [ "$(docker inspect -f '{{.Id}}' "$BACKEND_CONTAINER" 2>/dev/null)" != "$earlier_backend_id" ] || return 0
  cat "$EARLIER_BACKEND_LOG" 2>/dev/null || true
}

# The step summary is written on every exit, pass or fail.
write_report() {
  { replaced_backend_log; docker logs -t "$BACKEND_CONTAINER" 2>&1; } > "${OUT_DIR}/backend.log" || true
  compose logs -t --no-color > "${OUT_DIR}/compose.log" 2>&1 || true
  stop_event_log
  python3 "$REPORT_TOOL" \
    --results "$BOOTS_FILE" --events "$EVENTS_FILE" --backend-log "${OUT_DIR}/backend.log" \
    --backend-container "$BACKEND_CONTAINER" --frontend-container "$FRONTEND_CONTAINER" \
    --build-seconds "$build_seconds" --outcome "$outcome" --failure "$failure" \
    --title "$(report_title)" \
    --json "${OUT_DIR}/smoke-report.json" --summary "${GITHUB_STEP_SUMMARY:-/dev/stdout}" || return 1
  python3 "$IMAGES_TOOL" report --mode "$IMAGES_MODE" --tag "$(image_tag)" \
    --dir "$OUT_DIR" --results "$BOOTS_FILE" --up-started-ns "$up_started_ns" \
    --json "${OUT_DIR}/smoke-images-report.json" --summary "${GITHUB_STEP_SUMMARY:-/dev/stdout}"
}

report_title() {
  case "$SMOKE_IMAGES" in
    pull) echo "smoke-images: make up pulls the published images and boots them (PRD-262 US-003, US-036)" ;;
    prebuilt) echo "smoke-images: this pull request's images, built in the job and booted (PRD-262 US-003)" ;;
    *) echo "Fresh-clone smoke: boot timings (PRD-262 US-002)" ;;
  esac
}

finish() {
  local code=$?
  # A make up still bringing the stack up ends before the teardown.
  [ -z "$make_up_pid" ] || wait "$make_up_pid" 2>/dev/null
  if [ "$outcome" != "pass" ]; then
    dump_failure_logs
  fi
  if ! write_report; then
    echo "FAIL: the boot report could not be written" >&2
    code=1
  fi
  compose down -v || true
  restore_api_local
  exit "$code"
}

main() {
  mkdir -p "$OUT_DIR" || exit 1
  : > "$BOOTS_FILE"
  check_mode
  print_banner
  refuse_existing_stack
  refuse_emptying_a_kept_image_cache
  trap finish EXIT
  # Before the images: make up (SMOKE_IMAGES=pull) starts the stack itself.
  start_event_log
  prepare_images
  boot_fresh
  check_backend_processes
  check_production_backend
  check_worker_restart
  boot_frontend
  finish_make_up
  check_loopback_ports
  check_images
  check_document_storage
  warm_restarts
  check_api_local_reaches_backend
  check_stop_with_sse_client
  outcome="pass"
  failure=""
  echo "OK: /health and /health/ready (readiness) answered 200 on the fresh boot and" \
    "${WARM_RESTARTS} warm restarts, :3000 answered 200, a document came back from" \
    "object storage and the backend ran as the production stage — the local edition" \
    "serves with NO external credentials"
}

main "$@"
