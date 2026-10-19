#!/usr/bin/env bash
# PRD-262 US-046: boots a built frontend image the way a host runs it, and checks what
# its standalone server carries and loads.
#
#   check-standalone-image.sh <image> <page path> [docker run options...]
#
# - None of the globs next.config.js keeps out of the standalone trace matches a file in
#   /app (the image's sh expands each one).
# - /app/.next holds none of the build's trace files (*.nft.json): the build deletes them,
#   and the server never reads them (PRD-262 #47).
# - /app/.next/server holds no code of the packages only the browser runs, which
#   next.config.js resolves to an empty module on the server (PRD-262 #47): a string literal
#   from each package's own code marks it.
# - node runs, and its binary has no symbol table or debug section (strip --strip-all in
#   the Dockerfile's node-strip stage, PRD-262 #24). The image has no readelf, so its own
#   node reads its section headers.
# - Started with no network at all (--network none), the server answers <page path>
#   with 200 (a signed-out request needs no network) and /_next/image with 404 (image
#   optimisation is off, so nothing asks for sharp).
# - It answers every page the build's app-path-routes-manifest.json lists without a
#   dynamic segment, and none with a 5xx or with nothing. A redirect or a 404 passes: the
#   SaaS build sends a signed-out request to sign-in.
# - The server's log never mentions sharp, MODULE_NOT_FOUND or "Cannot find module", the
#   error a request gets when the trace left out a file it loads (PRD-262 #29).
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
  'node_modules/@img/**'
  'node_modules/sharp/**'
  'node_modules/typescript/**'
  'node_modules/next/dist/compiled/amphtml-validator/**'
  'node_modules/next/dist/server/capsize-font-metrics.json'
  'node_modules/next/dist/server/font-utils.js'
  'node_modules/next/dist/compiled/@next/font/**'
  'node_modules/next/dist/compiled/babel/bundle.js'
  'node_modules/next/dist/compiled/babel-packages/**'
  'node_modules/next/dist/compiled/browserslist/**'
  'node_modules/caniuse-lite/**'
  'node_modules/next/dist/compiled/postcss-preset-env/**'
  'node_modules/next/dist/compiled/cssnano-simple/**'
  'node_modules/next/dist/compiled/crypto-browserify/**'
  'node_modules/next/dist/compiled/@vercel/nft/**'
  'node_modules/next/dist/compiled/next-devtools/**'
  'node_modules/next/dist/compiled/next-server/*-experimental.runtime.prod.js'
  'node_modules/next/dist/compiled/next-server/*-turbo.runtime.prod.js'
  'node_modules/react/cjs/*.development.js'
  'node_modules/react-dom/cjs/*.development.js'
  'node_modules/scheduler/cjs/*.development.js'
  'node_modules/next/dist/compiled/react-is/cjs/*.development.js'
  'node_modules/next/dist/compiled/react-refresh/**'
)
# <package>:<a string literal of that package's code, in none of ours>.
CLIENT_ONLY_MARKERS=(
  'retell-client-js-sdk:retell-ai-4ihahnq7.livekit.cloud'
  'livekit-client:livekit-room'
  'xlsx:SheetJS Table Export'
)
READY_TIMEOUT_S=60
# A static file: the middleware's matcher skips paths with an extension.
READY_PATH=/favicon.ico
OPTIMISER_PATH='/_next/image?url=%2Fbrand%2Fautomatos-mark.png&w=96&q=75'
STATUS_JS='fetch(`http://127.0.0.1:${process.env.PORT || 3000}${process.argv[1]}`, { redirect: "manual" }).then((r) => console.log(r.status), () => console.log(0))'
# Every page the build lists whose path has no dynamic segment, each with the status the
# server gives it: "<status> <path>" per line, 0 when nothing answers.
# shellcheck disable=SC2016 # JavaScript, not shell: nothing here expands
PAGES_JS='const manifest = require("/app/.next/app-path-routes-manifest.json");
const pages = [...new Set(Object.entries(manifest).filter(([entry, path]) => entry.endsWith("/page") && !path.includes("[")).map(([, path]) => path))].sort();
const origin = `http://127.0.0.1:${process.env.PORT || 3000}`;
(async () => {
  for (const path of pages) {
    const status = await fetch(origin + path, { redirect: "manual", signal: AbortSignal.timeout(20000) }).then((r) => r.status, () => 0);
    console.log(`${status} ${path}`);
  }
})();'
CONTAINER="standalone-check-$$"
# The names of the ELF sections strip --strip-all removes that the image's node binary still
# has: .symtab, .strtab and .debug_*. Prints nothing for a stripped binary.
# shellcheck disable=SC2016 # JavaScript, not shell: nothing here expands
ELF_SECTIONS_JS='const fs = require("fs"); const file = process.argv[1] || process.execPath; const fd = fs.openSync(file, "r");
const read = (at, length) => { const b = Buffer.alloc(length); fs.readSync(fd, b, 0, length, at); return b };
const head = read(0, 64);
if (head.readUInt32BE(0) !== 0x7f454c46 || head[4] !== 2 || head[5] !== 1) throw new Error(`${file}: not a 64-bit little-endian ELF file`);
const size = head.readUInt16LE(58); const count = head.readUInt16LE(60);
const table = read(Number(head.readBigUInt64LE(40)), size * count);
const field = (i, offset) => Number(table.readBigUInt64LE(i * size + offset));
const shstrndx = head.readUInt16LE(62); const names = read(field(shstrndx, 24), field(shstrndx, 32));
for (let i = 0; i < count; i++) {
  const start = table.readUInt32LE(i * size); const name = names.toString("latin1", start, names.indexOf(0, start));
  if (/^\.(symtab|strtab|debug_)/.test(name)) console.log(name);
}'

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

# Each glob is expanded by the image's sh (where ** works as *); one that still matches a
# file is printed once.
check_trace() {
  local present glob
  # shellcheck disable=SC2016 # $glob expands inside the container's sh
  present=$(docker run --rm "$IMAGE" sh -c 'for glob in "$@"; do for p in $glob; do if [ -e "$p" ]; then echo "$glob"; break; fi; done; done' sh "${TRACE_EXCLUDED[@]}")
  if [ -z "$present" ]; then
    echo "absent from /app: ${TRACE_EXCLUDED[*]}"
    return
  fi
  while IFS= read -r glob; do fail "the standalone build carries ${glob}"; done <<< "$present"
}

check_no_trace_files() {
  local found
  found=$(docker run --rm "$IMAGE" sh -c 'find .next -name "*.nft.json"')
  if [ -n "$found" ]; then
    fail "/app/.next holds trace files (*.nft.json): $(wc -l <<< "$found" | tr -d ' '), such as $(head -n 1 <<< "$found")"
  else
    echo "no *.nft.json in /app/.next"
  fi
}

check_client_only_packages() {
  local found package
  # shellcheck disable=SC2016 # $entry expands inside the container's sh
  found=$(docker run --rm "$IMAGE" sh -c 'for entry in "$@"; do if grep -rqF -- "${entry#*:}" .next/server; then echo "${entry%%:*}"; fi; done' sh "${CLIENT_ONLY_MARKERS[@]}")
  if [ -z "$found" ]; then
    echo "no client-only package in /app/.next/server: ${CLIENT_ONLY_MARKERS[*]%%:*}"
    return
  fi
  while IFS= read -r package; do fail "the server output carries ${package}, which only the browser runs"; done <<< "$found"
}

check_node() {
  local version sections
  if ! version=$(docker run --rm "$IMAGE" node -v); then fail "node does not run"; return; fi
  if ! sections=$(docker run --rm "$IMAGE" node -e "$ELF_SECTIONS_JS"); then
    fail "node could not read its own section headers"
    return
  fi
  if [ -n "$sections" ]; then
    fail "node ${version} keeps sections strip --strip-all removes: ${sections//$'\n'/ }"
  else
    echo "node ${version}: no symbol table or debug section"
  fi
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

check_every_page() {
  local results failing
  if ! results=$(docker exec "$CONTAINER" node -e "$PAGES_JS"); then fail "could not request the pages"; return; fi
  if [ -z "$results" ]; then fail "app-path-routes-manifest.json lists no page"; return; fi
  echo "$results"
  failing=$(awk '$1 == 0 || $1 >= 500' <<< "$results")
  if [ -n "$failing" ]; then
    fail "pages that answered 5xx or nothing: ${failing//$'\n'/, }"
  else
    echo "$(wc -l <<< "$results" | tr -d ' ') pages answered"
  fi
}

expect_status() {
  local path=$1 want=$2 why=$3 got
  got=$(status_of "$path")
  echo "GET ${path}: ${got}"
  if [ "$got" != "$want" ]; then fail "GET ${path} answered ${got}, not ${want}: ${why}"; fi
}

check_trace
check_no_trace_files
check_client_only_packages
check_node
docker run -d --name "$CONTAINER" --network none "$@" "$IMAGE" > /dev/null
trap 'docker rm -f "$CONTAINER" > /dev/null 2>&1 || true' EXIT
wait_until_ready
expect_status "$PAGE" 200 "the page must render with no network"
expect_status "$OPTIMISER_PATH" 404 "image optimisation must be off"
check_every_page

logs=$(docker logs "$CONTAINER" 2>&1)
echo "----- server log -----"
echo "$logs"
if grep -qi sharp <<< "$logs"; then fail "the server log mentions sharp"; fi
if grep -qE "MODULE_NOT_FOUND|Cannot find module" <<< "$logs"; then
  fail "the server log reports a missing module: a file the trace left out is still loaded"
fi
exit "$bad"
