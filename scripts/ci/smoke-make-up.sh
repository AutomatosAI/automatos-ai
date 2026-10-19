#!/usr/bin/env bash
# PRD-262 US-036, on pull requests: `make up` on the pull request's checkout runs the
# images CI published for the merge base, and builds nothing.
#
# smoke-images.yml's make-up-pr job pins AUTOMATOS_IMAGE_TAG to sha-<7> of the merge base
# (or of its nearest ancestor CI published, which it picks with `smoke_images.py base`)
# and runs this script. Those images predate the change, so only what make up promises
# is checked here; the main runs (smoke-images, after images.yml publishes) check the rest
# on the images of their own commit:
#   - make up exits 0;
#   - nothing was built: each app container runs its ghcr.io/automatosai/automatos-*
#     image at the pinned tag, whose RepoDigests holds the index GHCR serves for it, the
#     app images exist only under that tag, and no local-* tag or compose-built image is
#     present (scripts/ci/smoke_images.py images --mode pull), and each carries the merge
#     base's revision label (SMOKE_EXPECT_REVISION);
#   - /health/ready answers 200 and :3000 answers 200 (scripts/ci/smoke_boot.py). No time
#     is recorded as a measurement here: make up has returned before they are polled.
#
# Built for a throwaway Docker host, such as a CI runner: its teardown removes the stack
# with its volumes, so it refuses to start on a host that has Automatos containers or
# volumes. The records and make up's output land in $SMOKE_OUT_DIR, which the workflow
# uploads; the secrets are the fresh-clone lane's throwaway ones, from the environment.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_DIR="${SMOKE_OUT_DIR:-${RUNNER_TEMP:-/tmp}/smoke-make-up}"
API_BASE="http://localhost:${API_PORT:-8000}"
FRONTEND_URL="http://localhost:${FRONTEND_PORT:-3000}/"
MAX_WAIT_SECONDS="${SMOKE_MAX_WAIT_SECONDS:-300}"
BOOT_TOOL="scripts/ci/smoke_boot.py"
IMAGES_TOOL="scripts/ci/smoke_images.py"
BOOTS_FILE="${OUT_DIR}/boots.jsonl"
IMAGES_FILE="${OUT_DIR}/images.json"
MAKE_UP_LOG="${OUT_DIR}/make-up.log"
# Outside OUT_DIR, so the resolved compose file (with the stack's env) is not uploaded.
COMPOSE_CONFIG_FILE="${RUNNER_TEMP:-/tmp}/smoke-make-up-config.json"
# docker-compose.yml's container_name values and named volumes all start with this.
STACK_NAME_PATTERN='^automatos_'

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
}

# $1 = the label, $2 = the container, $3 = name=url; polled from the container's own start.
wait_for() {
  python3 "$BOOT_TOOL" --label "$1" --container "$2" --url "$3" \
    --timeout "$MAX_WAIT_SECONDS" --out "$BOOTS_FILE"
}

check_nothing_was_built() {
  docker compose config --format json > "$COMPOSE_CONFIG_FILE" || fail "docker compose config failed"
  python3 "$IMAGES_TOOL" images --mode pull --config "$COMPOSE_CONFIG_FILE" \
    --expect-revision "${SMOKE_EXPECT_REVISION:-}" --out "$IMAGES_FILE" \
    || fail "make up did not run the published ${AUTOMATOS_IMAGE_TAG} images alone: something was built or tagged here"
}

finish() {
  local code=$?
  if [ "$outcome" != "pass" ]; then
    docker compose logs --tail 150 backend frontend >&2 || true
  fi
  docker compose logs -t --no-color > "${OUT_DIR}/compose.log" 2>&1 || true
  docker compose down -v || true
  exit "$code"
}

main() {
  mkdir -p "$OUT_DIR" || exit 1
  : > "$BOOTS_FILE"
  [ -n "${AUTOMATOS_IMAGE_TAG:-}" ] || fail "AUTOMATOS_IMAGE_TAG must name the published tag make up runs"
  refuse_existing_stack
  trap finish EXIT
  echo "make up with AUTOMATOS_IMAGE_TAG=${AUTOMATOS_IMAGE_TAG} on this pull request's checkout..."
  make up 2>&1 | tee "$MAKE_UP_LOG" || fail "make up failed"
  wait_for fresh automatos_backend "ready=${API_BASE}/health/ready" \
    || fail "/health/ready did not answer 200 within ${MAX_WAIT_SECONDS}s"
  wait_for frontend automatos_frontend "ui=${FRONTEND_URL}" \
    || fail "the frontend did not answer 200 on ${FRONTEND_URL} within ${MAX_WAIT_SECONDS}s"
  check_nothing_was_built
  outcome="pass"
  echo "OK: make up ran the published ${AUTOMATOS_IMAGE_TAG} images, built nothing, and the stack answered" \
    "/health/ready 200 and :3000 200"
}

main "$@"
