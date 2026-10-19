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
# land in $SMOKE_OUT_DIR, which the workflow uploads. No backend or frontend time is a bar yet: the
# lane fails when a boot or the frontend never answers 200 (PRD-262 US-004 records the baselines).
# The one time bar is the workspace worker's 20 s to healthy (PRD-262 US-045, below).
#
# PRD-262 US-005: once the frontend answers, scripts/ci/smoke-loopback-ports.sh asserts
# that it, and every other published port, listens on 127.0.0.1 (`docker compose port
# frontend 3000` prints 127.0.0.1:3000).
#
# PRD-262 US-003 runs the same boot steps on images instead of a source build, for
# .github/workflows/smoke-images.yml. SMOKE_IMAGES picks where the app images come from:
#   (unset)   `make build` on this checkout, as a contributor runs it (PRD-262 US-039):
#             tagged local-<sha7> and labelled with HEAD (AUTOMATOS_REVISION), both
#             exported for it. make build starts postgres, redis and minio, builds the
#             three images and starts the stack. It runs in the background, like make up
#             below, so the boots are timed from each container's own start; it must
#             then exit 0, and its output is the build log;
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
# The stack starts with `up --no-build` (make build's or make up's own; the smoke's in
# `prebuilt`), and scripts/ci/smoke_images.py asserts what booted: in the image modes the
# backend runs the production command (no --reload, and the worker count compose's
# WEB_CONCURRENCY sets), and in every mode each app container runs its
# ghcr.io/automatosai/automatos-* image carrying the commit's revision label
# (SMOKE_EXPECT_REVISION; HEAD for a source build, whose tag must be local-*). With
# `pull`, each image's RepoDigests must also hold the index digest its tag resolves to on
# GHCR, so nothing was built locally, and the app images exist only under that tag; no
# image may carry a local-* tag or compose's own <project>-<service> name. The step
# summary adds make up's run (or make build's start), the first run from its start to
# :3000 200, and those checks.
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
# PRD-262 US-030: a boot whose alembic_version already holds the code's head runs no
# alembic at all. The fresh boot (init_fresh_db stamps the head) and the warm restarts
# must each log `schema at head — alembic skipped` and run none: no `Context impl` line
# after `init_fresh_db: done`, and none at all on a warm restart.
#
# PRD-262 US-031: a warm restart redoes no seed work. The fresh boot runs the seed loader
# and stores each seed stage's fingerprint; each warm restart must log `seeds unchanged
# (fingerprint …) — skipped` for both stages (the entrypoint's loader and the lifespan
# leader's seeds) and start no seed loader, and credential_types (its rows' updated_at,
# read in the postgres container) must be the same after the warm restarts as before
# them (scripts/ci/smoke_seeds.py).
#
# PRD-262 US-034: the API answers once its core is up and boots the rest in the
# background, off the event loop. After the last boot has run its first 60 s,
# scripts/ci/smoke_ready.py reads every backend boot: the stages of its
# /health/bootstrap report up to and including trust_gate (what the server waits for
# before it listens) sum to 3 s or less, and its log holds no APScheduler `Run time of
# job … was missed` warning in its first 60 s. /health 200 and /health/ready 200 are
# timed separately, as above.
#
# PRD-262 US-017, when it builds from this checkout: make build's output is the build
# log, and once make build has exited, scripts/ci/build_context.py reads what the
# backend's build sent as its context off it. The workflow first seeds dummy uploads and
# caches where a self-hoster's checkout collects them, each as large as the budget, so
# one that reaches the context fails it.
#
# PRD-262 US-021, in every mode: the API image keeps only the botocore models of the
# services we call. Once the frontend and the images check out, and before the warm
# restarts, scripts/ci/smoke_document_presign.py uploads a document through the backend,
# which puts it in MinIO through that botocore, and the host fetches it back by the
# presigned URL the Documents page's download call answers with.
#
# PRD-262 US-037, SMOKE_IMAGES=pull: no secret is exported, as on a self-hoster's clean
# checkout. make up's first step, make env, creates .env from .env.example and writes
# POSTGRES_PASSWORD, REDIS_PASSWORD and API_KEY to it (openssl rand -hex 24); once make
# up has exited, the smoke finds three 48-hex values there, and every later compose
# command here reads them from .env. The other modes export throwaway ones (below).
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
# PRD-262 US-038, in every mode, once the frontend answers: compose probes each service
# every second while it starts (start_interval), and scripts/ci/smoke_health_gates.py
# reads the gates off Docker's start and health_status events. redis and minio must turn
# healthy within 2 s of their start; postgres is reported (on this fresh volume its TCP
# probe passes only after initdb's restart). Where make up or make build brings the stack
# up, compose starts the frontend the moment the backend turns healthy, and it must
# start within 3 s of that; `prebuilt` starts the frontend here, after its own checks, so
# its gap is only reported.
#
# PRD-262 US-039, make build only: postgres, redis and minio start before the images
# build, so postgres must turn healthy before the backend's image is built. Docker's
# events say both: postgres's health_status and the backend image's tag event, which
# BuildKit sends the moment that build has finished (the stream takes image tag events
# too). scripts/ci/smoke_build_overlap.py checks it once make build has exited, and
# reports the image build time from make build's start (report only). Compose's own log
# cannot show the order: it prints `postgres Healthy` only while a later up waits on it.
#
# PRD-262 US-041, in every mode, before the document check: MinIO runs as its image's own
# user, uid 65532, never root, and the one-shot minio-volume-owner that runs before it
# exited 0 (scripts/ci/smoke_minio.py server). Docker gives this fresh volume the owner of
# an image's /data as it creates the containers that mount it, which need not be 65532, so
# the one-shot may have re-owned it; either way the document below is stored by a MinIO
# that is not root. scripts/ci/smoke-minio-upgrade.sh covers a volume root wrote before.
#
# PRD-262 US-045, in every variant: scripts/ci/smoke_worker_health.py needs the workspace
# worker healthy within 20 s of its start (from the same Docker events) and its /metrics
# answering 200.
#
# Exit 0 = every boot answered 200 with no external creds, every published port is on
# loopback, the workspace worker turned healthy within 20 s and answered /metrics, MinIO
# ran as uid 65532, the document came back, the backend ran as the production stage
# should, the health gates held their bars, every boot ran alembic at most once (and only
# the behind-head boot ran it at all), and the warm restarts redid no seed work.
# Exit 1 = a boot failed, a port listens beyond this machine, the workspace worker was not
# healthy within 20 s of its start or its /metrics did not answer 200 (US-045), the
# backend's build context is over its budget, MinIO ran as root or minio-volume-owner
# failed (US-041), the document did not come back from object storage, a US-035 check
# failed, make up wrote no generated secrets to .env (US-037), a health gate
# missed its bar (US-038), make build or make up failed, or postgres was not healthy
# before make build had built the backend's image (US-039); or a boot ran alembic twice
# (or the CMD ran it), or ran it at head, or the behind-head boot did not, or a warm
# restart ran a seed stage, started the seed loader or rewrote credential_types, or a
# boot's core stages took over 3 s or APScheduler missed a job in a boot's first 60 s.
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
# Readiness probe (main.py:/health/ready), PRD-262 US-034. uvicorn listens, and
# /health answers, once the lifespan's core is up (tables, the leader's seeds, the
# trust gate). The rest of the boot (the starters, the Composio catalogue, Phase-2
# extensions, board dispatch, the READY stage) runs in the background; /health/ready
# answers 503 until it has finished, then 200. The two are timed separately, so the
# report shows the gap.
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
# make up's output (SMOKE_IMAGES=pull), kept for the rate-limit search and the artifact.
MAKE_UP_LOG="${OUT_DIR}/make-up.log"
# The make target that brings the stack up, which runs in the background: make build for
# a source build (PRD-262 US-039), make up for SMOKE_IMAGES=pull (US-036); none in
# `prebuilt`, which starts the stack itself. Its exit code is written here when it ends.
case "$SMOKE_IMAGES" in
  "") MAKE_TARGET="build" ;;
  pull) MAKE_TARGET="up" ;;
  *) MAKE_TARGET="" ;;
esac
MAKE_EXIT="${OUT_DIR}/make.exit"
# make build builds the three app images before it starts the backend, so the fresh
# boot waits this long for that start (make up's pull fits in MAX_WAIT_SECONDS).
BUILD_MAX_WAIT_SECONDS="${SMOKE_BUILD_MAX_WAIT_SECONDS:-1200}"
PROCESSES_FILE="${OUT_DIR}/processes.json"
IMAGES_FILE="${OUT_DIR}/images.json"
# smoke_worker_restart.py's record (PRD-262 US-008).
WORKER_RESTART_FILE="${OUT_DIR}/worker-restart.json"
# The source build's log, make build's output (PRD-262 US-017: build_context.py reads the
# context transfer off it).
BUILD_LOG="${OUT_DIR}/build.log"
# PRD-262 US-039: postgres healthy before the backend image was built, read off
# EVENTS_FILE; its record, and the image build's whole seconds for the report.
OVERLAP_TOOL="scripts/ci/smoke_build_overlap.py"
OVERLAP_FILE="${OUT_DIR}/build-overlap.json"
BUILD_SECONDS_FILE="${OUT_DIR}/build-seconds.txt"
# smoke_document_presign.py's record (PRD-262 US-021).
DOCUMENT_FILE="${OUT_DIR}/document-presign.json"
# smoke_alembic.py's record (PRD-262 US-029).
ALEMBIC_FILE="${OUT_DIR}/alembic-runs.json"
BEHIND_HEAD_LABEL="behind-head"
# PRD-262 US-031: smoke_seeds.py's record, and credential_types before and after the
# warm restarts: the row count and a digest of every row's updated_at.
SEEDS_TOOL="scripts/ci/smoke_seeds.py"
SEEDS_FILE="${OUT_DIR}/seed-stages.json"
CREDENTIAL_TYPES_BEFORE="${OUT_DIR}/credential-types-before.txt"
CREDENTIAL_TYPES_AFTER="${OUT_DIR}/credential-types-after.txt"
# PRD-262 US-034: smoke_ready.py's record.
READY_TOOL="scripts/ci/smoke_ready.py"
READY_FILE="${OUT_DIR}/ready-after-core.json"
CREDENTIAL_TYPES_SQL="SELECT count(*) || ' ' || coalesce(md5(string_agg(name || '=' || coalesce(updated_at::text, 'null'), ',' ORDER BY name)), '') FROM credential_types"
# PRD-262 US-041: MinIO's user and the one-shot that hands it its volume, and their record.
MINIO_TOOL="scripts/ci/smoke_minio.py"
MINIO_CONTAINER="automatos_minio"
MINIO_OWNER_CONTAINER="automatos_minio_volume_owner"
MINIO_FILE="${OUT_DIR}/minio-user.json"
# Where compose publishes MinIO on this host: presigned links must point here
# (S3_PUBLIC_ENDPOINT_URL in envs/api.defaults).
STORAGE_URL="http://localhost:${MINIO_PORT:-9000}"
# PRD-262 US-045: the workspace worker, its health bar and smoke_worker_health.py's record.
WORKER_CONTAINER="automatos_workspace_worker"
WORKER_HEALTH_TOOL="scripts/ci/smoke_worker_health.py"
WORKER_HEALTHY_MAX_SECONDS=20
WORKER_HEALTH_FILE="${OUT_DIR}/worker-health.json"
# Outside OUT_DIR, so the resolved compose file (with the stack's env) is not uploaded.
COMPOSE_CONFIG_FILE="${RUNNER_TEMP:-/tmp}/smoke-compose-config.json"
# PRD-262 US-037: the .env make up writes, and a secret line as make env writes it.
ENV_FILE=".env"
GENERATED_SECRET='^(POSTGRES_PASSWORD|REDIS_PASSWORD|API_KEY)=[0-9a-f]{48}$'
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
# PRD-262 US-038: the health gates, read off EVENTS_FILE, and their record.
GATES_TOOL="scripts/ci/smoke_health_gates.py"
GATES_FILE="${OUT_DIR}/health-gates.json"
# The backend's log up to its recreate: the new container's log starts empty, so the
# report reads the earlier boots' lines from this copy.
EARLIER_BACKEND_LOG="${RUNNER_TEMP:-/tmp}/smoke-backend-before-recreate.log"
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
# SMOKE_IMAGES=pull exports none of the three: make up generates them (PRD-262 US-037).
# ---------------------------------------------------------------------------
if [ "$SMOKE_IMAGES" = "pull" ]; then
  unset POSTGRES_PASSWORD REDIS_PASSWORD API_KEY
else
  export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-smoke_pg_pw}"
  export REDIS_PASSWORD="${REDIS_PASSWORD:-smoke_redis_pw}"
  export API_KEY="${API_KEY:-smoke_api_key}"
fi
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
make_pid=""
env_existed=""

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
  echo "  images     : ${SMOKE_IMAGES:-make build from this checkout} (AUTOMATOS_IMAGE_TAG=${AUTOMATOS_IMAGE_TAG:-make up picks it})"
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
# Image tag events come too: BuildKit tags an image the moment its build has finished,
# which is when make build's backend image was built (PRD-262 US-039). The other tools
# read only the compose services' container events.
start_event_log() {
  docker events --since "$(date +%s)" \
    --filter type=container --filter type=image \
    --filter event=start --filter event=health_status --filter event=tag \
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

# `make build` on this checkout (PRD-262 US-039): make env, postgres, redis and minio
# started first, the production stages of the backend, the frontend and the workspace
# worker built under the exported local-<sha7>, then the whole stack started with
# --no-build. It runs in the background while the boots are timed; finish_make waits for
# it. Piped, so BuildKit prints its plain progress, `transferring context` lines
# included. The image build is timed off Docker's events, for the record only: CI build
# time has no bar (PRD-262 §7).
build_images() {
  rm -f "$MAKE_EXIT"
  echo "make build: postgres, redis and minio first, then the backend (production stage), workspace-worker and frontend images from this checkout (AUTOMATOS_IMAGE_TAG=${AUTOMATOS_IMAGE_TAG})..."
  up_started_ns="$(date +%s%N)"
  {
    make build 2>&1 | tee "$BUILD_LOG"
    echo "${PIPESTATUS[0]}" > "$MAKE_EXIT"
  } &
  make_pid=$!
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
# in the background while the boots are timed; finish_make waits for it.
pull_images() {
  empty_docker_cache
  rm -f "$MAKE_EXIT"
  echo "make up from an empty Docker cache (AUTOMATOS_IMAGE_TAG=${AUTOMATOS_IMAGE_TAG:-not set: make up picks it})..."
  {
    python3 "$IMAGES_TOOL" pull --out "$PULL_FILE" --images-before "$images_before" --log "$MAKE_UP_LOG" -- \
      make up
    echo "$?" > "$MAKE_EXIT"
  } &
  make_pid=$!
}

# Waits for the background make build or make up, which must exit 0. make build's log is
# then whole, so the backend's build context is checked here (PRD-262 US-017), even when
# a boot failed first.
finish_make() {
  [ -n "$make_pid" ] || return 0
  wait "$make_pid"
  make_pid=""
  [ "$(cat "$MAKE_EXIT" 2>/dev/null)" = "0" ] || fail "make ${MAKE_TARGET} failed"
  [ "$MAKE_TARGET" != "build" ] || check_build_context
}

# PRD-262 US-037, SMOKE_IMAGES=pull: make up ran with no secret exported on a checkout
# with no .env, so make env created .env and wrote the three secrets to it.
check_generated_secrets() {
  [ "$SMOKE_IMAGES" = "pull" ] || return 0
  if [ -n "$env_existed" ]; then
    echo "::warning title=smoke-fresh-clone::.env existed before make up, so the check that make env generated the three secrets was skipped"
    return 0
  fi
  local found
  found="$(grep -cE "$GENERATED_SECRET" "$ENV_FILE" 2>/dev/null)"
  [ "${found:-0}" = "3" ] \
    || fail "make up did not write POSTGRES_PASSWORD, REDIS_PASSWORD and API_KEY to .env as 48 hex characters each (make env): ${found:-0} of 3"
  echo "make env wrote POSTGRES_PASSWORD, REDIS_PASSWORD and API_KEY to .env (48 hex characters each)"
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

# make build or make up starts the stack, so the fresh boot is timed from the backend's
# own start as soon as make has started it. make build first builds the images, so the
# wait for that start is BUILD_MAX_WAIT_SECONDS. A failed make is the failure to report.
boot_fresh_from_make() {
  local start_wait="$MAX_WAIT_SECONDS"
  [ "$MAKE_TARGET" != "build" ] || start_wait="$BUILD_MAX_WAIT_SECONDS"
  echo "Timing the fresh boot make ${MAKE_TARGET} starts: liveness (/health), then readiness (/health/ready)..."
  wait_backend fresh --wait-start --start-timeout "$start_wait" --stop-file "$MAKE_EXIT" && return 0
  finish_make
  fail "fresh boot: readiness not reached within ${MAX_WAIT_SECONDS}s (/health and /health/ready must both answer 200)"
}

boot_fresh() {
  if [ -n "$MAKE_TARGET" ]; then
    boot_fresh_from_make
    return
  fi
  echo "Starting compose stack (postgres, redis, minio, workspace-worker, backend) on an empty database..."
  up_started_ns="$(date +%s%N)"
  compose up -d --no-build postgres redis minio backend || fail "docker compose up failed"
  echo "Timing the fresh boot: liveness (/health), then readiness (/health/ready)..."
  wait_backend fresh \
    || fail "fresh boot: readiness not reached within ${MAX_WAIT_SECONDS}s (/health and /health/ready must both answer 200)"
}

# make build or make up starts the frontend once the backend is healthy; it is timed
# from its own start, and Docker says where it published :3000.
boot_frontend_from_make() {
  echo "Timing the frontend make ${MAKE_TARGET} starts once the backend is healthy..."
  python3 "$BOOT_TOOL" --label frontend --container "$FRONTEND_CONTAINER" \
    --wait-start --stop-file "$MAKE_EXIT" --published-port 3000 --url "ui=${FRONTEND_URL}" \
    --timeout "$MAX_WAIT_SECONDS" --out "$BOOTS_FILE" && return 0
  finish_make
  fail "the frontend never answered 200 on ${FRONTEND_URL} within ${MAX_WAIT_SECONDS}s"
}

boot_frontend() {
  if [ -n "$MAKE_TARGET" ]; then
    boot_frontend_from_make
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

# PRD-262 US-038: redis and minio healthy within 2 s of their start, postgres reported,
# from Docker's events (EVENTS_FILE, streamed since before the stack started). Where make
# build or make up brought the stack up, their compose started the frontend once the
# backend was healthy, so that gap is gated too; in `prebuilt` the smoke starts it
# itself, so it is reported.
check_health_gates() {
  local frontend="report"
  if [ -n "$MAKE_TARGET" ]; then
    frontend="gate"
  fi
  python3 "$GATES_TOOL" --events "$EVENTS_FILE" --frontend "$frontend" \
    --out "$GATES_FILE" --summary "${GITHUB_STEP_SUMMARY:-/dev/stdout}" \
    || fail "a health gate missed its bar: redis or minio turned healthy more than 2s after it started, or the frontend started more than 3s after the backend turned healthy (PRD-262 US-038)"
}

# PRD-262 US-039, make build only: postgres turned healthy before the backend's image was
# built, from Docker's events; also the image build's seconds from make build's start,
# which the report shows (no bar).
check_build_overlap() {
  [ "$MAKE_TARGET" = "build" ] || return 0
  local passed="true"
  python3 "$OVERLAP_TOOL" --events "$EVENTS_FILE" --tag "$(image_tag)" --started-ns "$up_started_ns" \
    --out "$OVERLAP_FILE" --seconds-out "$BUILD_SECONDS_FILE" --summary "${GITHUB_STEP_SUMMARY:-/dev/stdout}" \
    || passed=""
  build_seconds="$(cat "$BUILD_SECONDS_FILE" 2>/dev/null)"
  [ -n "$passed" ] \
    || fail "postgres did not turn healthy before the backend image was built: make build must start postgres, redis and minio before it builds (PRD-262 US-039)"
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

# PRD-262 US-045: the workspace worker, started with the backend, reported healthy within
# WORKER_HEALTHY_MAX_SECONDS of its start (its HEALTHCHECK probes every 2 s while it
# starts), and its /metrics answers 200.
check_workspace_worker() {
  python3 "$WORKER_HEALTH_TOOL" --events "$EVENTS_FILE" --service workspace-worker \
    --container "$WORKER_CONTAINER" --max-seconds "$WORKER_HEALTHY_MAX_SECONDS" \
    --out "$WORKER_HEALTH_FILE" --summary "${GITHUB_STEP_SUMMARY:-/dev/stdout}" \
    || fail "the workspace worker was not healthy within ${WORKER_HEALTHY_MAX_SECONDS}s of its start, or its /metrics did not answer 200"
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

# PRD-262 US-041: MinIO is healthy as its image's user, uid 65532, on the fresh volume, and
# minio-volume-owner exited 0. Before the one-shot runs, a fresh volume has the owner of an
# image's /data (Docker copies it in), not necessarily 65532, so whether it re-owned this
# one is recorded, not required.
check_minio_user() {
  python3 "$MINIO_TOOL" server --container "$MINIO_CONTAINER" --owner-container "$MINIO_OWNER_CONTAINER" \
    --reowned any --out "$MINIO_FILE" --summary "${GITHUB_STEP_SUMMARY:-/dev/stdout}" \
    || fail "MinIO is not healthy as uid 65532, or minio-volume-owner did not exit 0 (PRD-262 US-041)"
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
# runs alembic twice, and the behind-head boot runs it exactly once. US-030: the fresh
# boot and the warm restarts, at head, run none and log `schema at head — alembic skipped`.
check_alembic_runs() {
  local i
  # The labels boot_fresh and warm_restarts give wait_backend.
  local expects=(--expect "fresh=0" --expect "${BEHIND_HEAD_LABEL}=1")
  for i in $(seq 1 "$WARM_RESTARTS"); do
    expects+=(--expect "warm-${i}=0")
  done
  docker logs -t "$BACKEND_CONTAINER" > "${OUT_DIR}/backend.log" 2>&1 || fail "could not read the backend's log"
  python3 "$ALEMBIC_TOOL" --results "$BOOTS_FILE" --backend-log "${OUT_DIR}/backend.log" \
    --backend-container "$BACKEND_CONTAINER" "${expects[@]}" \
    --out "$ALEMBIC_FILE" --summary "${GITHUB_STEP_SUMMARY:-/dev/stdout}" \
    || fail "a backend boot ran alembic more than once or at head, the CMD ran it, or the behind-head boot did not run it once"
}

# PRD-262 US-031: credential_types' rows, read in the postgres container (its local socket
# needs no password). $1 = the file the snapshot goes to.
snapshot_credential_types() {
  compose exec -T postgres psql -U "${POSTGRES_USER:-postgres}" -d "${POSTGRES_DB:-orchestrator_db}" \
    -tAc "$CREDENTIAL_TYPES_SQL" > "$1" || fail "could not read credential_types into ${1##*/}"
}

# PRD-262 US-031: the fresh boot runs the seed loader; each warm restart skips both seed
# stages, starts no seed loader and leaves credential_types as it was.
check_seed_stages() {
  local i
  # The labels boot_fresh and warm_restarts give wait_backend.
  local expects=(--expect-loaded "fresh")
  for i in $(seq 1 "$WARM_RESTARTS"); do
    expects+=(--expect-skipped "warm-${i}")
  done
  docker logs -t "$BACKEND_CONTAINER" > "${OUT_DIR}/backend.log" 2>&1 || fail "could not read the backend's log"
  python3 "$SEEDS_TOOL" --results "$BOOTS_FILE" --backend-log "${OUT_DIR}/backend.log" \
    --backend-container "$BACKEND_CONTAINER" "${expects[@]}" \
    --credential-types "$CREDENTIAL_TYPES_BEFORE" "$CREDENTIAL_TYPES_AFTER" \
    --out "$SEEDS_FILE" --summary "${GITHUB_STEP_SUMMARY:-/dev/stdout}" \
    || fail "a warm restart ran a seed stage again, started the seed loader or rewrote credential_types"
}

# PRD-262 US-034: every backend boot's core stages (to trust_gate) took 3 s or less, and
# APScheduler missed no job in its first 60 s. The last boot's 60 s must be in the log
# before it is read, so this waits for them first.
check_ready_after_core() {
  local left
  left="$(python3 "$READY_TOOL" --results "$BOOTS_FILE" --backend-container "$BACKEND_CONTAINER" --window-left)" \
    || fail "could not work out when the last backend boot's first 60 s end"
  if [ "$left" -gt 0 ]; then
    echo "Waiting ${left}s for the last backend boot's first 60 s to be in its log..."
    sleep "$left"
  fi
  docker logs -t "$BACKEND_CONTAINER" > "${OUT_DIR}/backend.log" 2>&1 || fail "could not read the backend's log"
  python3 "$READY_TOOL" --results "$BOOTS_FILE" --backend-log "${OUT_DIR}/backend.log" \
    --backend-container "$BACKEND_CONTAINER" \
    --out "$READY_FILE" --summary "${GITHUB_STEP_SUMMARY:-/dev/stdout}" \
    || fail "a backend boot's core stages (to trust_gate) took over 3s, or APScheduler missed a job in a boot's first 60s"
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
  echo "----- workspace-worker logs (tail) -----" >&2
  compose logs --tail 80 workspace-worker >&2 || true
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
  # A make build or make up still bringing the stack up ends before the teardown.
  [ -z "$make_pid" ] || wait "$make_pid" 2>/dev/null
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
  use_restart_workers
  print_banner
  refuse_existing_stack
  refuse_emptying_a_kept_image_cache
  trap finish EXIT
  [ ! -e "$ENV_FILE" ] || env_existed="true"
  # Before the images: make build and make up start the stack themselves, and make build's
  # image tag events are read too.
  start_event_log
  prepare_images
  boot_fresh
  check_backend_processes
  check_production_backend
  check_worker_restart
  boot_frontend
  finish_make
  check_generated_secrets
  check_health_gates
  check_build_overlap
  check_loopback_ports
  check_images
  check_workspace_worker
  check_minio_user
  check_document_storage
  snapshot_credential_types "$CREDENTIAL_TYPES_BEFORE"
  warm_restarts
  snapshot_credential_types "$CREDENTIAL_TYPES_AFTER"
  boot_behind_head
  check_alembic_runs
  check_seed_stages
  check_ready_after_core
  # Last: they replace the backend container, then stop it (PRD-262 US-035). The checks
  # above read every earlier boot from the container's own log.
  check_api_local_reaches_backend
  check_stop_with_sse_client
  outcome="pass"
  failure=""
  echo "OK: /health and /health/ready (readiness) answered 200 on the fresh boot," \
    "${WARM_RESTARTS} warm restarts and a behind-head boot; only the behind-head boot ran alembic, once;" \
    "the warm restarts redid no seed work; every boot's core took 3s or less and no job was missed;" \
    ":3000 answered 200, MinIO ran as uid 65532, a document came back from object storage," \
    "the backend ran as the production stage and the health gates held" \
    "— the local edition serves with NO external credentials"
}

main "$@"
