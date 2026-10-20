#!/usr/bin/env bash
# PRD-262 US-037: on an install that has started, `make env` changes nothing and stops.
#
# Postgres keeps the password it was created with, so once the automatos_postgres_data
# volume exists make env generates no secret. An .env that still holds .env.example's
# CHANGE_ME placeholders is left exactly as it is, and make env exits non-zero with how
# to rotate them: orchestrator/docker-secrets-guard.sh's message, which the backend's
# entrypoint prints too.
#
# The check, with the real Docker and make, and no secret exported:
#   1. create the automatos_postgres_data volume;
#   2. write .env.example to .env;
#   3. run `make env`: it must exit non-zero, leave .env byte for byte as it was, name
#      the volume and say how to rotate (ALTER USER for Postgres).
# Pass or fail, the volume it created is removed and a checkout's own .env is put back.
#
# Built for a throwaway Docker host such as a CI runner (smoke-images.yml's make-up-pr
# job, after the make up smoke's teardown): it refuses to start when the volume already
# exists, because that volume is a database.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

VOLUME="automatos_postgres_data"
ENV_FILE=".env"
WORK_DIR="${RUNNER_TEMP:-/tmp}/smoke-make-env"
SAVED_ENV="${WORK_DIR}/env.saved"
BEFORE="${WORK_DIR}/env.before"
OUTPUT="${WORK_DIR}/make-env.out"

volume_created=""
env_set_aside=""
env_written=""

fail() {
  echo "FAIL: $1" >&2
  exit 1
}

# Runs before the exit trap is set, so a refusal removes nothing.
refuse_an_existing_volume() {
  if docker volume inspect "$VOLUME" > /dev/null 2>&1; then
    fail "this Docker host already has the $VOLUME volume, a database. Run this on a throwaway host, as CI does"
  fi
}

# The checkout's own .env waits in WORK_DIR until finish puts it back.
set_env_aside() {
  [ -e "$ENV_FILE" ] || return 0
  mv -f "$ENV_FILE" "$SAVED_ENV" || fail "could not set this checkout's .env aside"
  env_set_aside="true"
}

finish() {
  local code=$?
  if [ -n "$volume_created" ]; then
    docker volume rm "$VOLUME" > /dev/null || code=1
  fi
  [ -z "$env_written" ] || rm -f "$ENV_FILE"
  [ -z "$env_set_aside" ] || mv -f "$SAVED_ENV" "$ENV_FILE" || code=1
  exit "$code"
}

check_make_env_changed_nothing() {
  local code=0
  make env > "$OUTPUT" 2>&1 || code=$?
  cat "$OUTPUT"
  [ "$code" -ne 0 ] || fail "make env exited 0 on an install that started with CHANGE_ME placeholders"
  cmp -s "$BEFORE" "$ENV_FILE" || fail "make env changed .env on an install that has started"
  grep -q "the $VOLUME volume exists, so this install has started" "$OUTPUT" \
    || fail "make env did not say that the $VOLUME volume means this install has started"
  grep -q "ALTER USER" "$OUTPUT" || fail "make env did not say how to rotate the Postgres password"
}

main() {
  mkdir -p "$WORK_DIR" || exit 1
  refuse_an_existing_volume
  unset POSTGRES_PASSWORD REDIS_PASSWORD API_KEY
  trap finish EXIT
  set_env_aside
  docker volume create "$VOLUME" > /dev/null || fail "docker volume create $VOLUME failed"
  volume_created="true"
  env_written="true"
  cp .env.example "$ENV_FILE" && cp "$ENV_FILE" "$BEFORE" || fail "could not write .env from .env.example"
  check_make_env_changed_nothing
  echo "OK: with the $VOLUME volume present, make env left the placeholder .env as it was, exited non-zero" \
    "and said how to rotate"
}

main "$@"
