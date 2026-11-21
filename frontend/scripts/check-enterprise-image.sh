#!/usr/bin/env bash
# PRD-277 US-505, PRD-279 US-402: starts a built enterprise web image the way an install starts
# it, and checks what only a real start can show.
#
#   check-enterprise-image.sh <image>
#
# The enterprise image fixes no address and no media origin when it is built: the server reads
# BACKEND_INTERNAL_URL (required) and MEDIA_ORIGINS (optional) when it starts
# (lib/enterprise/start-up.ts) and stops, naming the setting and never its value, when one is
# missing or bad. Each numbered check prints a line starting "proved:" when it passes:
#
# 1. Started with no BACKEND_INTERNAL_URL, the server stops with a non-zero exit and its log
#    names BACKEND_INTERNAL_URL.
# 2. Started with a valid BACKEND_INTERNAL_URL and a bad MEDIA_ORIGINS (an http origin that is
#    not a local address), it stops the same way, and its log names MEDIA_ORIGINS but not the
#    value it was given.
# 3. Started with both set as an install sets them, it passes check-standalone-image.sh, the
#    checks the local image gets: with no network, /sign-in answers 200, no page answers a 5xx
#    or nothing, and the image carries nothing the trimmed build leaves out.
# 4. Started again with those settings, the middleware's answer to a page (`/`, sent to sign in)
#    and the sign-in page itself carry a content security policy whose img-src and media-src
#    name the origin MEDIA_ORIGINS gave when the container started, and no other directive names
#    it, while a static file, which keeps the policy fixed at build time, carries one that names
#    it in no directive: the origin came from the running server, not from the build, and the
#    server put it in img-src and media-src only.
# 5. (PRD-277 D-12) Started with a malformed ENTERPRISE_CALLS_CLOUD_SYNC, it stops the same way.
#    Started with it on, the sign-in page's img-src names Automatos Cloud's files path while the
#    static file's build-time policy does not, and neither names any other Automatos, Clerk or
#    Cloudflare host.
#
# Not proved here: that the web app reaches an API (nothing listens at the address given, and
# the container has no network) or that anyone can sign in (no identity provider). chart.yml's
# install lane starts the image as the chart renders it (deploy/kind/strict.sh enterprise).
#
# Needs only Docker. Each request is made from inside the container with the image's own node
# (fetch, redirects not followed), as check-standalone-image.sh makes its requests.
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: $0 <image>" >&2
  exit 2
fi
IMAGE=$1
HERE="$(cd "$(dirname "$0")" && pwd)"

# Obviously made-up values of the right shape: the API's in-cluster origin, and a storage host.
BACKEND="http://automatos-api:8000"
MEDIA_ORIGIN="https://media.example.com"
BAD_MEDIA_ORIGIN="http://media.example.com"
# Cloud sync's switch (lib/enterprise/cloud-sync-setting.ts), a value that is not one, and the one
# Automatos source it adds to img-src (csp-policy.mjs cloudImageSources).
BAD_CLOUD_SYNC="sometimes"
CLOUD_IMAGES="https://api.automatos.app/api/marketplace/catalog/files/"
STOP_TIMEOUT_S=60
READY_TIMEOUT_S=60
READY_PATH=/favicon.ico
CONTAINER="enterprise-check-$$"
# "<status> <Content-Security-Policy>" of one path, or "0" when nothing answers.
ANSWER_JS='fetch(`http://127.0.0.1:${process.env.PORT || 3000}${process.argv[1]}`, { redirect: "manual" }).then((r) => console.log(`${r.status} ${r.headers.get("content-security-policy") || ""}`), () => console.log("0"))'

bad=0
fail() {
  echo "::error::${IMAGE}: $*"
  bad=1
}

remove_container() {
  docker rm -f "$CONTAINER" > /dev/null 2>&1 || true
}
trap remove_container EXIT

running() {
  [ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER" 2> /dev/null)" = true ]
}

answer_of() {
  docker exec "$CONTAINER" node -e "$ANSWER_JS" "$1" 2> /dev/null || echo 0
}

# Starts the image with no network and the given docker run options, under $CONTAINER.
start() {
  remove_container
  docker run -d --name "$CONTAINER" --network none "$@" "$IMAGE" > /dev/null
}

# Waits for the container to stop, asking it for a page each second in case the server only
# runs its start-up check on a first request. 0 once it has stopped, 1 if it is still running.
wait_until_stopped() {
  local waited=0
  while running; do
    if [ "$waited" -ge "$STOP_TIMEOUT_S" ]; then return 1; fi
    answer_of "$READY_PATH" > /dev/null
    sleep 1
    waited=$((waited + 1))
  done
}

# A start the server must refuse: SETTING is the name its log must give, HIDDEN a value it must
# not print (or ""), then the docker run options.
expect_refused() {
  local setting=$1 hidden=$2 logs code
  shift 2
  start "$@"
  if ! wait_until_stopped; then
    fail "the server kept running ${STOP_TIMEOUT_S} s with a bad or missing ${setting}"
    return
  fi
  code=$(docker inspect -f '{{.State.ExitCode}}' "$CONTAINER")
  logs=$(docker logs "$CONTAINER" 2>&1)
  echo "----- log of the start without a usable ${setting} (exit ${code}) -----"
  echo "$logs"
  if [ "$code" = 0 ]; then fail "the server stopped with exit 0 without a usable ${setting}"; return; fi
  if ! grep -qF "The web app cannot start: ${setting}" <<< "$logs"; then
    fail "the server stopped, but its log does not say ${setting} is why"
    return
  fi
  if [ -n "$hidden" ] && grep -qF -- "$hidden" <<< "$logs"; then
    fail "the refusal prints the value of ${setting}"
    return
  fi
  echo "proved: without a usable ${setting} the server stops at start-up (exit ${code}) and its log names the setting"
}

wait_until_ready() {
  local waited=0
  until [ "$(answer_of "$READY_PATH" | cut -d' ' -f1)" = 200 ]; do
    if ! running || [ "$waited" -ge "$READY_TIMEOUT_S" ]; then
      docker logs "$CONTAINER" 2>&1 | tail -n 80
      fail "the server never answered ${READY_PATH} with 200 (${waited} s)"
      return 1
    fi
    sleep 1
    waited=$((waited + 1))
  done
  echo "the server answered ${READY_PATH} after ${waited} s"
}

# The value of one directive of a policy: DIRECTIVE POLICY.
directive() {
  tr ';' '\n' <<< "$2" | sed 's/^ *//' | grep "^$1 " || true
}

# The policy of PATH, answered with STATUS, must name (WANT=yes) or must not name (WANT=no)
# the media origin in img-src and media-src; either way, no other directive may name it.
expect_policy() {
  local path=$1 status=$2 want=$3 answer got policy name others
  answer=$(answer_of "$path")
  got=${answer%% *}
  policy=${answer#* }
  echo "GET ${path}: ${got}"
  if [ "$got" != "$status" ]; then fail "GET ${path} answered ${got}, not ${status}"; return; fi
  if [ "$policy" = "$answer" ] || [ -z "$policy" ]; then fail "GET ${path} carries no content security policy"; return; fi
  for name in img-src media-src; do
    if grep -qF -- " ${MEDIA_ORIGIN}" <<< "$(directive "$name" "$policy")"; then
      [ "$want" = yes ] || fail "GET ${path}: ${name} names ${MEDIA_ORIGIN}, which only the running server was given"
    else
      [ "$want" = no ] || fail "GET ${path}: ${name} does not name ${MEDIA_ORIGIN}: $(directive "$name" "$policy")"
    fi
  done
  others=$(other_directives_naming "$MEDIA_ORIGIN" "$policy")
  [ -z "$others" ] || fail "GET ${path}: ${MEDIA_ORIGIN} is also in ${others}: only img-src and media-src may name it"
}

# The names of the directives of POLICY, other than img-src and media-src, that name ORIGIN:
# ORIGIN POLICY. Empty when none does.
other_directives_naming() {
  tr ';' '\n' <<< "$2" | sed 's/^ *//' | grep -v -e '^img-src ' -e '^media-src ' \
    | grep -F -- "$1" | cut -d' ' -f1 | tr '\n' ' ' || true
}

# PRD-277 D-12: the policy of PATH names no Automatos origin and no sign-in or challenge host,
# except Automatos Cloud's files path in img-src, which it must name (WANT=yes) or must not (WANT=no).
expect_own_hosts() {
  local path=$1 want=$2 answer policy rest
  answer=$(answer_of "$path")
  policy=${answer#* }
  if grep -qF -- " ${CLOUD_IMAGES}" <<< "$(directive img-src "$policy")"; then
    [ "$want" = yes ] || fail "GET ${path}: img-src names ${CLOUD_IMAGES}, which only Cloud sync allows"
  else
    [ "$want" = no ] || fail "GET ${path}: img-src does not name ${CLOUD_IMAGES} with Cloud sync on"
  fi
  rest=${policy//" ${CLOUD_IMAGES}"/}
  if grep -qiE 'automatos\.app|clerk|cloudflare' <<< "$rest"; then
    fail "GET ${path}: the policy names an Automatos, sign-in or challenge host: ${rest}"
  fi
}

check_running_policy() {
  start -e "BACKEND_INTERNAL_URL=${BACKEND}" -e "MEDIA_ORIGINS=${MEDIA_ORIGIN}" -e "ENTERPRISE_CALLS_CLOUD_SYNC=true"
  wait_until_ready || return
  local before=$bad
  expect_policy / 307 yes
  expect_policy /sign-in 200 yes
  expect_policy "$READY_PATH" 200 no
  expect_own_hosts /sign-in yes
  expect_own_hosts "$READY_PATH" no
  if [ "$bad" = "$before" ]; then
    echo "proved: the policy of a page names the media origin given at start-up in img-src and media-src only, and the build-time policy names it nowhere; with Cloud sync switched on at start-up a page's img-src names Automatos Cloud's files path, the build-time policy does not, and neither names another Automatos, sign-in or challenge host"
  fi
}

expect_refused BACKEND_INTERNAL_URL "" -e "MEDIA_ORIGINS=${MEDIA_ORIGIN}"
expect_refused MEDIA_ORIGINS "$BAD_MEDIA_ORIGIN" -e "BACKEND_INTERNAL_URL=${BACKEND}" -e "MEDIA_ORIGINS=${BAD_MEDIA_ORIGIN}"
expect_refused ENTERPRISE_CALLS_CLOUD_SYNC "$BAD_CLOUD_SYNC" -e "BACKEND_INTERNAL_URL=${BACKEND}" -e "ENTERPRISE_CALLS_CLOUD_SYNC=${BAD_CLOUD_SYNC}"
if bash "${HERE}/check-standalone-image.sh" "$IMAGE" /sign-in \
    -e "BACKEND_INTERNAL_URL=${BACKEND}" -e "MEDIA_ORIGINS=${MEDIA_ORIGIN}"; then
  echo "proved: started with both settings, it serves /sign-in and every page with no network, and carries nothing the trimmed build leaves out"
else
  fail "check-standalone-image.sh failed on the enterprise image (above)"
fi
check_running_policy
exit "$bad"
