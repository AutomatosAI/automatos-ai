#!/usr/bin/env bash
# PRD-262 US-041: MinIO runs as its own user, and an old volume root wrote still serves.
#
# docker-compose.yml runs MinIO as its image's user, uid 65532 (no `user: "0:0"` any
# more), and the one-shot minio-volume-owner re-owns the minio_data volume to
# 65532:65532 before MinIO starts whenever root owns it. smoke-images.yml's
# smoke-images-pr job runs this after the core-stack smoke, on the images that job built
# and tagged (AUTOMATOS_IMAGE_TAG), so neither make up nor compose pulls or builds an
# app image. Two upgrades, one after the other, each from a volume pre-populated as root:
#   1. through `make up`, on a checkout with no .env (make env writes it);
#   2. through plain `docker compose up -d`, on the .env the first one left.
# Each starts by seeding the volume the way an install from before US-041 left it: this
# stack's MinIO, run as root (`docker compose run --user 0:0`, as `user: "0:0"` did),
# creates a bucket and stores an object. Then everything in the volume goes to root and
# /data to mode 0755, as the official minio/minio image wrote it (Chainguard's image
# ships /data world-writable, with .minio.sys its own user's, which an old volume never
# had), and scripts/ci/smoke_minio.py checks that root owns every path.
# After each upgrade:
#   - /health/ready answers 200: the backend, which waits for minio: service_healthy,
#     started, so MinIO came up healthy on the old volume;
#   - MinIO is healthy and runs as uid 65532 (its Config.User and docker top), and
#     minio-volume-owner exited 0 and logged that it re-owned the volume;
#   - every path in the volume is 65532:65532;
#   - the object root wrote reads back byte for byte, and its bucket takes a new object,
#     which root's 0755 directories would refuse uid 65532.
#
# Built for a throwaway Docker host, such as a CI runner: its teardown removes the stack
# with its volumes, so it refuses to start on a host that has Automatos containers or
# volumes, or on a checkout with a .env. The deliverables folder is a temporary one
# (AUTOMATOS_WORKSPACE_DIR), so nothing in the checkout changes owner. Records land in
# $SMOKE_OUT_DIR.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_DIR="${SMOKE_OUT_DIR:-${RUNNER_TEMP:-/tmp}/smoke-minio-upgrade}"
API_BASE="http://localhost:${API_PORT:-8000}"
MAX_WAIT_SECONDS="${SMOKE_MAX_WAIT_SECONDS:-300}"
BOOT_TOOL="scripts/ci/smoke_boot.py"
MINIO_TOOL="scripts/ci/smoke_minio.py"
BOOTS_FILE="${OUT_DIR}/boots.jsonl"
ENV_FILE=".env"
# docker-compose.yml's container_name values and named volumes all start with this.
STACK_NAME_PATTERN='^automatos_'
MINIO_CONTAINER="automatos_minio"
OWNER_CONTAINER="automatos_minio_volume_owner"
# The root MinIO that seeds the volume: a one-off of compose's minio service.
SEED_CONTAINER="automatos_minio_seed"
VOLUME="automatos_minio_data"
OWNER_SERVICE="minio-volume-owner"
ROOT_OWNER="0:0"
MINIO_OWNER="65532:65532"
BUCKET="us041-upgrade"
OLD_KEY="written-as-root.txt"
NEW_KEY="written-as-65532.txt"
SEED_HEALTH_SECONDS=60
# Compose interpolates the three secrets even for a command that never uses them; these
# stand in while there is no .env yet. Only the seed, the config read and the teardown
# use them.
UNUSED_SECRETS=(POSTGRES_PASSWORD=unused REDIS_PASSWORD=unused API_KEY=unused)
WORKSPACE_DIR="${RUNNER_TEMP:-/tmp}/smoke-minio-deliverables"

outcome="fail"
owner_image=""

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

# docker compose on docker-compose.yml with stand-in secrets, for the commands that use none.
compose_any() {
  env "${UNUSED_SECRETS[@]}" docker compose -f docker-compose.yml "$@"
}

# The stack, its volumes and the seed go; the .env stays, for the next upgrade.
teardown() {
  docker rm -f "$SEED_CONTAINER" > /dev/null 2>&1 || true
  compose_any logs -t --no-color >> "${OUT_DIR}/compose.log" 2>&1 || true
  compose_any down -v --remove-orphans > /dev/null 2>&1 || true
}

finish() {
  local code=$?
  if [ "$outcome" != "pass" ]; then
    compose_any ps -a >&2 || true
    compose_any logs --tail 80 "$OWNER_SERVICE" minio backend >&2 || true
  fi
  teardown
  rm -f "$ENV_FILE"
  exit "$code"
}

# $1 = the step's name, then smoke_minio.py's subcommand and arguments; its record goes
# to OUT_DIR/<name>.json and its line to the step summary.
minio_check() {
  local name="$1"
  shift
  python3 "$MINIO_TOOL" "$@" --out "${OUT_DIR}/${name}.json" --summary "${GITHUB_STEP_SUMMARY:-/dev/stdout}"
}

# The image minio-volume-owner runs (the stack's redis), which has find, stat and chown.
read_owner_image() {
  owner_image="$(compose_any config --format json | jq -r --arg s "$OWNER_SERVICE" '.services[$s].image // empty')"
  [ -n "$owner_image" ] || fail "docker-compose.yml has no ${OWNER_SERVICE} service with an image"
}

wait_seed_healthy() {
  local i status=""
  for i in $(seq 1 "$SEED_HEALTH_SECONDS"); do
    status="$(docker inspect -f '{{if .State.Health}}{{.State.Health.Status}}{{end}}' "$SEED_CONTAINER" 2>/dev/null)"
    [ "$status" != "healthy" ] || return 0
    sleep 1
  done
  docker logs --tail 40 "$SEED_CONTAINER" >&2 || true
  fail "MinIO run as root did not turn healthy within ${SEED_HEALTH_SECONDS}s (last status: ${status:-none})"
}

# $1 = the label. The volume as an install from before US-041 left it: MinIO, as root,
# stores an object; then root owns every path, and /data is 0755.
seed_root_volume() {
  local label="$1"
  echo "${label}: MinIO, run as root as before US-041, stores ${BUCKET}/${OLD_KEY} in a fresh ${VOLUME}..."
  compose_any run -d --no-deps --user 0:0 --name "$SEED_CONTAINER" minio > /dev/null \
    || fail "${label}: could not start MinIO as root on a fresh ${VOLUME}"
  wait_seed_healthy
  minio_check "${label}-seed-put" put --container "$SEED_CONTAINER" --bucket "$BUCKET" --key "$OLD_KEY" \
    --text "$old_text" || fail "${label}: MinIO run as root could not store the object"
  docker stop -t 30 "$SEED_CONTAINER" > /dev/null && docker rm "$SEED_CONTAINER" > /dev/null \
    || fail "${label}: could not stop the root MinIO"
  docker run --rm --network none --user 0:0 -v "${VOLUME}:/data" --entrypoint /bin/sh "$owner_image" \
    -c 'chown -R 0:0 /data && chmod 755 /data' || fail "${label}: could not hand ${VOLUME} to root"
  minio_check "${label}-seed-volume" volume --volume "$VOLUME" --image "$owner_image" --owner "$ROOT_OWNER" \
    || fail "${label}: the seeded volume is not root's, so this would not test an upgrade"
}

# $1 = the label: what both upgrades must leave behind.
check_upgraded() {
  local label="$1"
  python3 "$BOOT_TOOL" --label "$label" --container automatos_backend --url "ready=${API_BASE}/health/ready" \
    --timeout "$MAX_WAIT_SECONDS" --out "$BOOTS_FILE" \
    || fail "${label}: /health/ready did not answer 200 within ${MAX_WAIT_SECONDS}s"
  minio_check "${label}-server" server --container "$MINIO_CONTAINER" --owner-container "$OWNER_CONTAINER" \
    --reowned yes || fail "${label}: MinIO is not healthy as uid 65532, or minio-volume-owner did not re-own the root volume and exit 0"
  minio_check "${label}-volume" volume --volume "$VOLUME" --image "$owner_image" --owner "$MINIO_OWNER" \
    || fail "${label}: paths in ${VOLUME} are not ${MINIO_OWNER} after the upgrade"
  minio_check "${label}-get-old" get --container "$MINIO_CONTAINER" --bucket "$BUCKET" --key "$OLD_KEY" \
    --text "$old_text" || fail "${label}: the object root wrote did not read back"
  minio_check "${label}-put-new" put --container "$MINIO_CONTAINER" --bucket "$BUCKET" --key "$NEW_KEY" \
    --text "$new_text" || fail "${label}: the old bucket refused a new object from uid 65532"
  minio_check "${label}-get-new" get --container "$MINIO_CONTAINER" --bucket "$BUCKET" --key "$NEW_KEY" \
    --text "$new_text" || fail "${label}: the new object did not read back"
  echo "${label}: MinIO runs as uid 65532 on the old volume, which minio-volume-owner re-owned"
}

# $1 = the label. The texts each upgrade writes, unique to this run.
set_texts() {
  old_text="PRD-262 US-041 ${1}: written by MinIO as root before the upgrade ($$-${RANDOM})"
  new_text="PRD-262 US-041 ${1}: written by MinIO as uid 65532 after the upgrade ($$-${RANDOM})"
}

# 1. make up on a checkout with no .env: make env writes it, then the stack starts.
upgrade_with_make_up() {
  set_texts make-up
  seed_root_volume make-up
  echo "make up on a volume root owns..."
  make up > "${OUT_DIR}/make-up.log" 2>&1 \
    || { cat "${OUT_DIR}/make-up.log" >&2; fail "make up failed on a volume MinIO wrote as root"; }
  check_upgraded make-up
}

# 2. Plain docker compose up, on the .env make up left (its secrets and image tag).
upgrade_with_compose_up() {
  set_texts compose-up
  seed_root_volume compose-up
  echo "docker compose up -d on a volume root owns..."
  docker compose up -d > "${OUT_DIR}/compose-up.log" 2>&1 \
    || { cat "${OUT_DIR}/compose-up.log" >&2; fail "docker compose up -d failed on a volume MinIO wrote as root"; }
  check_upgraded compose-up
}

main() {
  mkdir -p "$OUT_DIR" || exit 1
  : > "$BOOTS_FILE"
  [ -n "${AUTOMATOS_IMAGE_TAG:-}" ] || fail "AUTOMATOS_IMAGE_TAG must name the images to run (smoke-images-pr's tag)"
  export DEFAULT_WORKSPACE_ID="${DEFAULT_WORKSPACE_ID:-00000000-0000-0000-0000-0000000000c1}"
  # The .env holds the secrets: none from the shell. The deliverables folder is a temporary one.
  unset POSTGRES_PASSWORD REDIS_PASSWORD API_KEY COMPOSE_FILE COMPOSE_PROFILES
  export AUTOMATOS_WORKSPACE_DIR="$WORKSPACE_DIR" LOCAL_PROJECTS_DIR="${WORKSPACE_DIR}/projects"
  mkdir -p "$LOCAL_PROJECTS_DIR" || fail "could not create ${LOCAL_PROJECTS_DIR}"
  refuse_existing_stack
  trap finish EXIT
  read_owner_image
  upgrade_with_make_up
  teardown
  upgrade_with_compose_up
  outcome="pass"
  echo "OK: a volume MinIO wrote as root was re-owned once by minio-volume-owner, through make up and" \
    "through plain docker compose up; MinIO ran healthy as uid 65532, the old object read back and its" \
    "bucket took a new one"
}

main "$@"
