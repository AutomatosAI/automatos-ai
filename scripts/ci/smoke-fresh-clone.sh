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
#   (unset)   build them from this checkout (the fresh-clone lane, above);
#   pull      the published path: docker-compose.images.yml with AUTOMATOS_IMAGE_TAG.
#             The Docker image cache is emptied first, then `docker compose pull` is
#             timed and the bytes it brought in are counted;
#   prebuilt  the pull-request variant: the workflow built the three amd64 images and
#             tagged them with the GHCR names and AUTOMATOS_IMAGE_TAG, so compose
#             neither pulls nor builds them.
# Either way the stack starts with `up --no-build`, and scripts/ci/smoke_images.py
# asserts what booted: the backend runs the production command (no --reload, and the
# worker count compose's WEB_CONCURRENCY sets), and each app container runs its
# ghcr.io/automatosai/automatos-* image carrying the commit's revision label
# (SMOKE_EXPECT_REVISION). With `pull`, each image's RepoDigests must also hold the
# index digest its tag resolves to on GHCR, so nothing was built locally; no image may
# carry a local-* tag or compose's own <project>-<service> name. The step summary
# adds the pull, the first run from its start to :3000 200, and those checks.
#
# PRD-262 US-008, in the pull-request variant (`prebuilt`) only: once the worker count
# checks out, scripts/ci/smoke_worker_restart.py kills one uvicorn worker of the
# production-stage backend and needs uvicorn to start a replacement within 30 s, with
# /health 200 and no container restart. uvicorn 0.24 never replaced a dead worker.
# envs/api.defaults sets WEB_CONCURRENCY=1 (PRD-262 US-029), and a single uvicorn
# process has no manager to restart a worker, so this variant's backend runs
# RESTART_WORKERS through a compose override the script writes outside the checkout.
#
# PRD-262 US-029, in every mode: the entrypoint is the only owner of migrations. After
# the warm restarts, the database is stamped one revision behind head
# (scripts/ci/smoke_alembic_behind.py, run inside the backend) and the backend restarts
# once more. scripts/ci/smoke_alembic.py then reads the backend's log of every boot: the
# entrypoint's banner is there, no alembic line follows it (the CMD runs none), no boot
# runs alembic twice, and the behind-head boot runs it exactly once.
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
# Exit 0 = every boot answered 200 with no external creds, every published port is on
# loopback, the document came back, and every boot ran alembic at most once. Exit 1 = a
# boot failed, a port listens beyond this machine, the backend's build context is over
# its budget, the document did not come back from object storage, or a boot ran alembic
# twice (or the CMD ran it).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

COMPOSE_FILE="docker-compose.yml"
# The published images in place of the source builds (PRD-262 US-003).
IMAGES_COMPOSE_FILE="docker-compose.images.yml"
SMOKE_IMAGES="${SMOKE_IMAGES:-}"
COMPOSE_ARGS=(-f "$COMPOSE_FILE")
if [ -n "$SMOKE_IMAGES" ]; then
  COMPOSE_ARGS+=(-f "$IMAGES_COMPOSE_FILE")
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
# PRD-262 US-029: the alembic runs per boot, and the in-container stamp behind head.
ALEMBIC_TOOL="scripts/ci/smoke_alembic.py"
BEHIND_TOOL="scripts/ci/smoke_alembic_behind.py"
BOOTS_FILE="${OUT_DIR}/boots.jsonl"
EVENTS_FILE="${OUT_DIR}/events.jsonl"
# smoke_images.py's records (it reads them back from OUT_DIR for the report).
PULL_FILE="${OUT_DIR}/pull.json"
PROCESSES_FILE="${OUT_DIR}/processes.json"
IMAGES_FILE="${OUT_DIR}/images.json"
# smoke_worker_restart.py's record (PRD-262 US-008).
WORKER_RESTART_FILE="${OUT_DIR}/worker-restart.json"
# The source build's log (PRD-262 US-017: build_context.py reads the context transfer off it).
BUILD_LOG="${OUT_DIR}/build.log"
# smoke_document_presign.py's record (PRD-262 US-021).
DOCUMENT_FILE="${OUT_DIR}/document-presign.json"
# smoke_alembic.py's record (PRD-262 US-029).
ALEMBIC_FILE="${OUT_DIR}/alembic-runs.json"
BEHIND_HEAD_LABEL="behind-head"
# Where compose publishes MinIO on this host: presigned links must point here
# (S3_PUBLIC_ENDPOINT_URL in envs/api.defaults).
STORAGE_URL="http://localhost:${MINIO_PORT:-9000}"
# Outside OUT_DIR, so the resolved compose file (with the stack's env) is not uploaded.
COMPOSE_CONFIG_FILE="${RUNNER_TEMP:-/tmp}/smoke-compose-config.json"
# The production stage's worker count under compose: envs/api.defaults'
# WEB_CONCURRENCY (PRD-262 US-029; the image's own 4 is Railway's).
BACKEND_WORKERS="${SMOKE_BACKEND_WORKERS:-1}"
# The pull-request variant's count (US-008 kills one of them), set through an override
# written outside the checkout.
RESTART_WORKERS=2
WORKERS_OVERRIDE_FILE="${RUNNER_TEMP:-/tmp}/smoke-backend-workers.yml"
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

compose() {
  docker compose "${COMPOSE_ARGS[@]}" "$@"
}

fail() {
  failure="$1"
  echo "FAIL: $1" >&2
  exit 1
}

# Image modes need the tag to boot; a typo in the mode must not fall back to a build.
check_mode() {
  case "$SMOKE_IMAGES" in
    "") ;;
    pull|prebuilt)
      [ -n "${AUTOMATOS_IMAGE_TAG:-}" ] || fail "SMOKE_IMAGES=${SMOKE_IMAGES} needs AUTOMATOS_IMAGE_TAG (the tag to boot)" ;;
    *) fail "SMOKE_IMAGES must be unset, pull or prebuilt, not '${SMOKE_IMAGES}'" ;;
  esac
}

# PRD-262 US-029 and US-008: the pull-request variant's backend runs RESTART_WORKERS
# uvicorn workers, so check_worker_restart has a manager to watch. The main runs keep
# compose's count, the one a self-hoster gets.
use_restart_workers() {
  [ "$SMOKE_IMAGES" = "prebuilt" ] || return 0
  printf 'services:\n  backend:\n    environment:\n      WEB_CONCURRENCY: "%s"\n' "$RESTART_WORKERS" \
    > "$WORKERS_OVERRIDE_FILE" || fail "could not write the backend's worker-count override"
  COMPOSE_ARGS+=(-f "$WORKERS_OVERRIDE_FILE")
  BACKEND_WORKERS="$RESTART_WORKERS"
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
  echo "  images     : ${SMOKE_IMAGES:-built from this checkout}${SMOKE_IMAGES:+ (AUTOMATOS_IMAGE_TAG=${AUTOMATOS_IMAGE_TAG:-})}"
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
  echo "Building the backend (development stage), workspace-worker and frontend images..."
  local started=$SECONDS
  compose build backend workspace-worker frontend 2>&1 | tee "$BUILD_LOG" || fail "docker compose build failed"
  build_seconds=$((SECONDS - started))
  echo "Images built in ${build_seconds}s"
  check_build_context
}

# PRD-262 US-017: the backend's build sent only code. The development stage copies none
# (compose mounts the source), and orchestrator/.dockerignore keeps the rest out.
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

# Pulls every image of the default stack: the app images under AUTOMATOS_IMAGE_TAG and
# the datastores. smoke_images.py times it and counts the bytes the runner received.
pull_images() {
  empty_docker_cache
  echo "Pulling the stack's images (app images at ${AUTOMATOS_IMAGE_TAG})..."
  python3 "$IMAGES_TOOL" pull --out "$PULL_FILE" --images-before "$images_before" -- \
    docker compose "${COMPOSE_ARGS[@]}" pull --quiet || fail "docker compose pull failed"
}

prepare_images() {
  case "$SMOKE_IMAGES" in
    "") build_images ;;
    pull) pull_images ;;
    prebuilt) echo "Booting the images this job built (${AUTOMATOS_IMAGE_TAG}): no build, no pull" ;;
  esac
}

# $1 = the boot's label (fresh, warm-1, ...). Times /health and /health/ready from
# the backend's StartedAt, then fetches /health/bootstrap.
wait_backend() {
  python3 "$BOOT_TOOL" --label "$1" --container "$BACKEND_CONTAINER" \
    --url "health=${HEALTH_URL}" --url "ready=${READY_URL}" \
    --fetch "bootstrap=${BOOTSTRAP_URL}" \
    --timeout "$MAX_WAIT_SECONDS" --out "$BOOTS_FILE"
}

boot_fresh() {
  echo "Starting compose stack (postgres, redis, minio, workspace-worker, backend) on an empty database..."
  up_started_ns="$(date +%s%N)"
  compose up -d --no-build postgres redis minio backend || fail "docker compose up failed"
  echo "Timing the fresh boot: liveness (/health), then readiness (/health/ready)..."
  wait_backend fresh \
    || fail "fresh boot: readiness not reached within ${MAX_WAIT_SECONDS}s (/health and /health/ready must both answer 200)"
}

boot_frontend() {
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

# Image modes: every app container runs the image compose names for it, with this
# commit's revision label; with `pull`, its RepoDigests prove it was pulled.
check_images() {
  [ -n "$SMOKE_IMAGES" ] || return 0
  compose config --format json > "$COMPOSE_CONFIG_FILE" || fail "docker compose config failed"
  python3 "$IMAGES_TOOL" images --mode "$SMOKE_IMAGES" --config "$COMPOSE_CONFIG_FILE" \
    --expect-revision "${SMOKE_EXPECT_REVISION:-}" --out "$IMAGES_FILE" \
    || fail "an app container does not run the image this smoke expects, or an image was built locally"
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

# PRD-262 US-029: a database one revision behind head, then one more backend boot,
# whose entrypoint must run alembic exactly once.
boot_behind_head() {
  echo "Behind-head boot: stamping the database one revision behind head, then restarting the backend..."
  compose exec -T backend python - < "$BEHIND_TOOL" \
    || fail "could not stamp the database one revision behind head"
  compose restart backend || fail "docker compose restart backend failed (behind-head boot)"
  wait_backend "$BEHIND_HEAD_LABEL" \
    || fail "behind-head boot: readiness not reached within ${MAX_WAIT_SECONDS}s"
}

# PRD-262 US-029: every boot shows the entrypoint's banner and no alembic after it, none
# runs alembic twice, and the behind-head boot runs it exactly once.
check_alembic_runs() {
  docker logs -t "$BACKEND_CONTAINER" > "${OUT_DIR}/backend.log" 2>&1 || fail "could not read the backend's log"
  python3 "$ALEMBIC_TOOL" --results "$BOOTS_FILE" --backend-log "${OUT_DIR}/backend.log" \
    --backend-container "$BACKEND_CONTAINER" --expect "${BEHIND_HEAD_LABEL}=1" \
    --out "$ALEMBIC_FILE" --summary "${GITHUB_STEP_SUMMARY:-/dev/stdout}" \
    || fail "a backend boot ran alembic more than once, the CMD ran it, or the behind-head boot did not run it once"
}

dump_failure_logs() {
  echo "----- postgres logs (tail) — initdb/schema errors surface here -----" >&2
  compose logs --tail 200 postgres >&2 || true
  echo "----- backend logs (tail) -----" >&2
  compose logs --tail 150 backend >&2 || true
  echo "----- frontend logs (tail) -----" >&2
  compose logs --tail 80 frontend >&2 || true
}

# The step summary is written on every exit, pass or fail.
write_report() {
  docker logs -t "$BACKEND_CONTAINER" > "${OUT_DIR}/backend.log" 2>&1 || true
  compose logs -t --no-color > "${OUT_DIR}/compose.log" 2>&1 || true
  stop_event_log
  python3 "$REPORT_TOOL" \
    --results "$BOOTS_FILE" --events "$EVENTS_FILE" --backend-log "${OUT_DIR}/backend.log" \
    --backend-container "$BACKEND_CONTAINER" --frontend-container "$FRONTEND_CONTAINER" \
    --build-seconds "$build_seconds" --outcome "$outcome" --failure "$failure" \
    --title "$(report_title)" \
    --json "${OUT_DIR}/smoke-report.json" --summary "${GITHUB_STEP_SUMMARY:-/dev/stdout}" || return 1
  [ -n "$SMOKE_IMAGES" ] || return 0
  python3 "$IMAGES_TOOL" report --mode "$SMOKE_IMAGES" --tag "${AUTOMATOS_IMAGE_TAG:-}" \
    --dir "$OUT_DIR" --results "$BOOTS_FILE" --up-started-ns "$up_started_ns" \
    --json "${OUT_DIR}/smoke-images-report.json" --summary "${GITHUB_STEP_SUMMARY:-/dev/stdout}"
}

report_title() {
  case "$SMOKE_IMAGES" in
    pull) echo "smoke-images: the published images, pulled and booted (PRD-262 US-003)" ;;
    prebuilt) echo "smoke-images: this pull request's images, built in the job and booted (PRD-262 US-003)" ;;
    *) echo "Fresh-clone smoke: boot timings (PRD-262 US-002)" ;;
  esac
}

finish() {
  local code=$?
  if [ "$outcome" != "pass" ]; then
    dump_failure_logs
  fi
  if ! write_report; then
    echo "FAIL: the boot report could not be written" >&2
    code=1
  fi
  compose down -v || true
  exit "$code"
}

main() {
  mkdir -p "$OUT_DIR" || exit 1
  : > "$BOOTS_FILE"
  check_mode
  use_restart_workers
  print_banner
  refuse_existing_stack
  refuse_emptying_a_kept_image_cache
  trap finish EXIT
  prepare_images
  start_event_log
  boot_fresh
  check_backend_processes
  check_worker_restart
  boot_frontend
  check_loopback_ports
  check_images
  check_document_storage
  warm_restarts
  boot_behind_head
  check_alembic_runs
  outcome="pass"
  failure=""
  echo "OK: /health and /health/ready (readiness) answered 200 on the fresh boot," \
    "${WARM_RESTARTS} warm restarts and a behind-head boot, each ran alembic at most once," \
    ":3000 answered 200 and a document came back from object storage — the local" \
    "edition serves with NO external credentials"
}

main "$@"
