#!/usr/bin/env bash
# PRD-262 US-046: boots a built frontend image the way a host runs it, and checks what
# its standalone server carries and loads.
#
#   check-standalone-image.sh <image> <page path> [docker run options...]
#
# - None of the paths next.config.js keeps out of the standalone trace is in /app.
# - Started with no network at all (--network none), the server answers <page path>
#   with 200 (a signed-out request needs no network) and /_next/image with 404 (image
#   optimisation is off, so nothing asks for sharp).
# - The server's log never mentions sharp.
#
# A container with no network publishes no port, so each request is made from inside
# it, with the image's own node (fetch, redirects not followed). test.yml's
# prod-image-parity job runs this against the SaaS and the local build (amd64), and
# images.yml against the frontend image each build job pushed (amd64 and arm64); it needs
# only Docker. frontend/lib/__tests__/standalone-diet.test.ts holds TRACE_EXCLUDED equal to
# next.config.js's STANDALONE_TRACE_EXCLUDES.
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <image> <page path> [docker run options...]" >&2
  exit 2
fi
IMAGE=$1
PAGE=$2
shift 2

TRACE_EXCLUDED=(
  node_modules/@img
  node_modules/sharp
  node_modules/typescript
  node_modules/next/dist/compiled/amphtml-validator
)
READY_TIMEOUT_S=60
# A static file: the middleware's matcher skips paths with an extension.
READY_PATH=/favicon.ico
OPTIMISER_PATH='/_next/image?url=%2Fbrand%2Fautomatos-mark.png&w=96&q=75'
STATUS_JS='fetch(`http://127.0.0.1:${process.env.PORT || 3000}${process.argv[1]}`, { redirect: "manual" }).then((r) => console.log(r.status), () => console.log(0))'
CONTAINER="standalone-check-$$"

bad=0
fail() {
  echo "::error::${IMAGE}: $*"
  bad=1
}

# The HTTP status the server gives a path, or 0 when nothing answers.
status_of() {
  docker exec "$CONTAINER" node -e "$STATUS_JS" "$1" 2> /dev/null || echo 0
}

running() {
  [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2> /dev/null)" = true ]
}

check_trace() {
  local present
  # shellcheck disable=SC2016 # $@ expands inside the container's sh
  present=$(docker run --rm "$IMAGE" sh -c 'for p in "$@"; do [ ! -e "$p" ] || echo "$p"; done' sh "${TRACE_EXCLUDED[@]}")
  for path in $present; do fail "the standalone build carries ${path}"; done
  if [ -z "$present" ]; then echo "absent from /app: ${TRACE_EXCLUDED[*]}"; fi
}

wait_until_ready() {
  local waited=0
  until [ "$(status_of "$READY_PATH")" = 200 ]; do
    if ! running || [ "$waited" -ge "$READY_TIMEOUT_S" ]; then
      docker logs "$CONTAINER" 2>&1 | tail -n 80
      echo "::error::${IMAGE}: the server never answered ${READY_PATH} with 200 (${waited} s)"
      exit 1
    fi
    sleep 1
    waited=$((waited + 1))
  done
  echo "the server answered ${READY_PATH} after ${waited} s"
}

expect_status() {
  local path=$1 want=$2 why=$3 got
  got=$(status_of "$path")
  echo "GET ${path}: ${got}"
  if [ "$got" != "$want" ]; then fail "GET ${path} answered ${got}, not ${want}: ${why}"; fi
}

check_trace
docker run -d --name "$CONTAINER" --network none "$@" "$IMAGE" > /dev/null
trap 'docker rm -f "$CONTAINER" > /dev/null 2>&1 || true' EXIT
wait_until_ready
expect_status "$PAGE" 200 "the page must render with no network"
expect_status "$OPTIMISER_PATH" 404 "image optimisation must be off"

logs=$(docker logs "$CONTAINER" 2>&1)
echo "----- server log -----"
echo "$logs"
if grep -qi sharp <<< "$logs"; then fail "the server log mentions sharp"; fi
exit "$bad"
