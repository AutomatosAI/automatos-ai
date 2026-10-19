#!/usr/bin/env bash
# PRD-262 US-033 (FR-7): the API boots with no internet egress.
#
# A self-hoster on a locked-down network runs the stack with no route to the internet.
# Before US-033, `import main` fetched tiktoken's encoding from
# openaipublic.blob.core.windows.net through requests.get with no timeout. Refused
# egress killed the import, and egress that drops packets hung it for about two minutes
# per attempt. This smoke boots the backend with its egress cut in one of two ways
# (SMOKE_EGRESS):
#
#   refused     every HTTP(S) client in the backend goes through a proxy at
#               127.0.0.1:9, where nothing listens, so each connect is refused at once.
#               NO_PROXY keeps the stack's own services direct. The compose override
#               that sets this is written outside the checkout and applies to every
#               backend container the script starts.
#   blackholed  the backend stays on the stack's normal compose network. Once
#               `docker compose up` of the datastores has created that network, the
#               runner drops every TCP packet from its subnet to anywhere outside it:
#               `iptables -I DOCKER-USER -s <subnet> ! -d <subnet> -p tcp -j DROP`.
#               DNS still answers and the host still reaches the published port, but a
#               connect to the internet is silently dropped, which is the case where a
#               timeout-less requests.get hangs. An internal-only Docker network would
#               not test it: connects there fail at once, and the runner could not
#               reach the published port.
#
# Before the backend starts, the script proves the cut is the kind it says. From a
# one-off container on the backend's network, with the backend's environment, a connect
# to the encoding's host fails at once (refused: curl exit 7) or runs into curl's
# timeout while DNS still resolves the host (blackholed: curl exit 28).
#
# Then, with egress cut throughout:
#   1. the first boot, on an empty database: the entrypoint builds the schema and loads
#      the seeds, then the server starts. /health must answer 200 within the lane's
#      usual wait (SMOKE_MAX_WAIT_SECONDS, 300 s, as in smoke-fresh-clone.sh);
#   2. a new backend container, as a redeploy creates (`up --force-recreate`): /health
#      must answer 200 within PRD-262 §8's fresh-boot bar, 30 s from the container's
#      start, so a hang shows up as a timeout;
#   3. inside that container, core.context_guard.get_tokenizer() loads its encoding from
#      the image's TIKTOKEN_CACHE_DIR within 60 s.
# The first boot is held to the lane's wait, not to the bar. An empty-database boot took
# 45-50 s in CI with internet on 19 Oct 2026, about 28 s of it init_fresh_db's migration
# replay, so the bar would fail it on speed rather than egress. Both boots run the code a
# hang hides in (import main, the entrypoint, the lifespan), and the second one is held
# to the bar.
#
# The step summary shows each boot (smoke_report.py) and how the egress was cut; the
# records and logs land in $SMOKE_OUT_DIR, which the workflow uploads.
#
# Like smoke-fresh-clone.sh, this script is for a throwaway Docker host such as a CI
# runner. Its teardown removes the stack together with its volumes, so it refuses to
# start on a host that already has Automatos containers or volumes. The blackholed mode
# changes the host's firewall (the rule is deleted again on exit), so it refuses to run
# outside CI (CI=true).
#
# Exit 0 = with the egress cut, both boots answered /health 200 (the second within the
# bar) and the tokenizer loaded. Exit 1 = the cut was not the kind it should be, a boot
# never answered or the second one answered late, or the tokenizer did not load.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

COMPOSE_ARGS=(-f docker-compose.yml)
SMOKE_EGRESS="${SMOKE_EGRESS:-}"
API_BASE="http://localhost:${API_PORT:-8000}"
HEALTH_URL="${API_BASE}/health"
BOOTSTRAP_URL="${API_BASE}/health/bootstrap"
BACKEND_CONTAINER="automatos_backend"
# A datastore's container: the network it is on is the one the backend joins.
NETWORK_PROBE_CONTAINER="automatos_postgres"
DATASTORES=(postgres redis minio)
# PRD-262 §8: a backend's fresh boot, container start to /health 200.
FRESH_BOOT_BAR_SECONDS=30
# How long past the bar the new container is still polled, so a slow boot shows its time.
BAR_GRACE_SECONDS=30
MAX_WAIT_SECONDS="${SMOKE_MAX_WAIT_SECONDS:-300}"
TOKENIZER_TIMEOUT_SECONDS=60
# Nothing listens there, so every proxied connect is refused at once.
REFUSING_PROXY="http://127.0.0.1:9"
# The stack's own services, reached directly, never through the proxy.
DIRECT_HOSTS="localhost,127.0.0.1,::1,backend,postgres,redis,minio,workspace-worker,media-render,mem0-server,qdrant,searxng"
# Where tiktoken downloads its encoding from: what a timeout-less fetch hangs on.
ENCODING_HOST="openaipublic.blob.core.windows.net"
ENCODING_URL="https://${ENCODING_HOST}/encodings/cl100k_base.tiktoken"
CONNECT_CHECK_SECONDS=10
CURL_COULDNT_CONNECT=7
CURL_TIMED_OUT=28
OUT_DIR="${SMOKE_OUT_DIR:-${RUNNER_TEMP:-/tmp}/smoke-egress}"
BOOT_TOOL="scripts/ci/smoke_boot.py"
REPORT_TOOL="scripts/ci/smoke_report.py"
BOOTS_FILE="${OUT_DIR}/boots.jsonl"
EVENTS_FILE="${OUT_DIR}/events.jsonl"
BACKEND_LOG="${OUT_DIR}/backend.log"
# The first container's log, kept before `--force-recreate` removes it.
FIRST_CONTAINER_LOG="${OUT_DIR}/backend-first-container.log"
# Outside OUT_DIR and the checkout: the refused mode's compose override.
PROXY_OVERRIDE_FILE="${RUNNER_TEMP:-/tmp}/smoke-egress-proxy.yml"
FIRST_LABEL="offline-fresh"
NEW_CONTAINER_LABEL="offline-new-container"
# docker-compose.yml's container_name values and named volumes all start with this.
STACK_NAME_PATTERN='^automatos_'
# Run inside the backend container: the tokenizer must load from the image's cache.
TOKENIZER_CHECK='import sys
from core.context_guard import TOKENIZER_ENCODING, get_tokenizer
encoding = get_tokenizer()
if encoding is None or encoding.name != TOKENIZER_ENCODING:
    sys.exit(f"the tokenizer did not load: {encoding!r}")
print("tokenizer", encoding.name, "loaded with no egress")'

# Throwaway values for an ephemeral local stack, as in smoke-fresh-clone.sh. No external
# SaaS credential is set.
export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-smoke_pg_pw}"
export REDIS_PASSWORD="${REDIS_PASSWORD:-smoke_redis_pw}"
export API_KEY="${API_KEY:-smoke_api_key}"
export AUTH_EDITION="${AUTH_EDITION:-local}"
export REQUIRE_AUTH="${REQUIRE_AUTH:-false}"
export DEFAULT_WORKSPACE_ID="${DEFAULT_WORKSPACE_ID:-00000000-0000-0000-0000-0000000000c1}"
# A build from this checkout gets a local-<sha7> tag and its commit as the revision
# label, never an official tag (PRD-262 US-036, FR-13), wherever compose names them.
AUTOMATOS_REVISION="${AUTOMATOS_REVISION:-$(git rev-parse HEAD 2>/dev/null)}"
export AUTOMATOS_REVISION
export AUTOMATOS_IMAGE_TAG="${AUTOMATOS_IMAGE_TAG:-local-${AUTOMATOS_REVISION:0:7}}"

outcome="fail"
failure="the smoke stopped before its first boot"
build_seconds=""
events_pid=""
cut_note="egress not cut yet"
tokenizer_note="not checked"
# "<iptables|ip6tables> <subnet>" for each DROP rule added, so the exit removes it.
drop_rules=()

compose() {
  docker compose "${COMPOSE_ARGS[@]}" "$@"
}

fail() {
  failure="$1"
  echo "FAIL: $1" >&2
  exit 1
}

# Runs before the exit trap is set, so a refusal changes nothing on the host.
check_mode() {
  case "$SMOKE_EGRESS" in
    refused) ;;
    blackholed)
      [ "${CI:-}" = "true" ] \
        || fail "SMOKE_EGRESS=blackholed adds a rule to this host's firewall (sudo iptables). Run it on a throwaway host such as a CI runner (CI=true)" ;;
    *) fail "SMOKE_EGRESS must be refused or blackholed, not '${SMOKE_EGRESS}'" ;;
  esac
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
  echo "PRD-262 US-033 egress smoke: the backend boots with no internet"
  echo "  egress     : ${SMOKE_EGRESS}"
  echo "  health url : ${HEALTH_URL}"
  echo "  first boot : empty database, up to ${MAX_WAIT_SECONDS}s"
  echo "  new boot   : a new container, within the ${FRESH_BOOT_BAR_SECONDS}s fresh-boot bar (PRD-262 §8)"
  echo "  records    : ${OUT_DIR}"
  echo "========================================="
}

# Docker's start and health_status events, streamed for the whole run (smoke_report.py's
# Started → Healthy table), as smoke-fresh-clone.sh does.
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

# The backend image compose runs, built from this checkout. Timed for the record only.
build_backend() {
  echo "Building the backend image..."
  local started=$SECONDS
  compose build backend || fail "docker compose build backend failed"
  build_seconds=$((SECONDS - started))
}

# This also creates the stack's network, which the blackholed mode cuts off.
start_datastores() {
  echo "Starting the datastores (${DATASTORES[*]})..."
  compose up -d --wait "${DATASTORES[@]}" || fail "the datastores did not start healthy"
}

# The refused mode: every backend container this script starts reads the override.
refuse_egress() {
  local name
  {
    printf 'services:\n  backend:\n    environment:\n'
    for name in HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy; do
      printf '      %s: "%s"\n' "$name" "$REFUSING_PROXY"
    done
    for name in NO_PROXY no_proxy; do
      printf '      %s: "%s"\n' "$name" "$DIRECT_HOSTS"
    done
  } > "$PROXY_OVERRIDE_FILE" || fail "could not write the backend's proxy override"
  COMPOSE_ARGS+=(-f "$PROXY_OVERRIDE_FILE")
  cut_note="refused: the backend's HTTP(S) clients go through ${REFUSING_PROXY}, where nothing listens (NO_PROXY: ${DIRECT_HOSTS})"
}

# The network the datastores are on, which the backend joins.
stack_network() {
  local networks
  networks="$(docker inspect -f '{{range $name, $settings := .NetworkSettings.Networks}}{{$name}} {{end}}' \
    "$NETWORK_PROBE_CONTAINER")" || return 1
  networks="${networks%% *}"
  [ -n "$networks" ] || return 1
  echo "$networks"
}

# The blackholed mode: drop TCP from the stack's subnet(s) to anywhere outside them.
blackhole_egress() {
  local network subnets subnet tool
  network="$(stack_network)" || fail "could not read the network ${NETWORK_PROBE_CONTAINER} is on"
  subnets="$(docker network inspect -f '{{range .IPAM.Config}}{{.Subnet}} {{end}}' "$network")" \
    || fail "could not read the subnets of ${network}"
  [ -n "${subnets// /}" ] || fail "${network} has no subnet"
  for subnet in $subnets; do
    tool=iptables
    case "$subnet" in *:*) tool=ip6tables ;; esac
    sudo "$tool" -I DOCKER-USER -s "$subnet" ! -d "$subnet" -p tcp -j DROP \
      || fail "could not add the DROP rule for ${subnet} to DOCKER-USER"
    drop_rules+=("${tool} ${subnet}")
  done
  cut_note="blackholed: DOCKER-USER drops TCP from ${network} (${subnets% }) to anywhere outside it"
}

restore_egress() {
  local rule
  for rule in ${drop_rules[@]+"${drop_rules[@]}"}; do
    sudo "${rule%% *}" -D DOCKER-USER -s "${rule#* }" ! -d "${rule#* }" -p tcp -j DROP \
      || echo "::warning title=Egress smoke::could not delete the DROP rule for ${rule#* }"
  done
}

# A one-off container on the backend's network, with the backend's environment.
# $1 = the program to run, the rest its arguments.
egress_probe() {
  compose run --rm -T --no-deps --entrypoint "$1" backend "${@:2}"
}

# The cut is the kind the mode says: refused at once, or dropped while DNS answers.
check_egress_cut() {
  local expected code
  expected="$CURL_COULDNT_CONNECT"
  if [ "$SMOKE_EGRESS" = "blackholed" ]; then
    expected="$CURL_TIMED_OUT"
    egress_probe getent ahosts "$ENCODING_HOST" > /dev/null \
      || fail "DNS did not resolve ${ENCODING_HOST} on the backend's network: the blackholed case needs DNS to answer"
  fi
  egress_probe curl -sS -o /dev/null --max-time "$CONNECT_CHECK_SECONDS" "$ENCODING_URL"
  code=$?
  [ "$code" -eq "$expected" ] \
    || fail "a connect to ${ENCODING_URL} on the backend's network ended with curl exit ${code}, not ${expected}: egress is not ${SMOKE_EGRESS}"
  echo "Egress ${SMOKE_EGRESS}: a connect to ${ENCODING_HOST} ends with curl exit ${code}, as it should"
}

cut_egress() {
  case "$SMOKE_EGRESS" in
    refused) refuse_egress ;;
    blackholed) blackhole_egress ;;
  esac
  echo "Egress ${cut_note}"
  check_egress_cut
}

# $1 = the boot's label; the rest go to smoke_boot.py (its wait and any bar).
wait_backend() {
  python3 "$BOOT_TOOL" --label "$1" --container "$BACKEND_CONTAINER" \
    --url "health=${HEALTH_URL}" --fetch "bootstrap=${BOOTSTRAP_URL}" --out "$BOOTS_FILE" "${@:2}"
}

boot_first() {
  echo "First boot, on an empty database, with egress ${SMOKE_EGRESS}..."
  compose up -d --no-build --no-deps backend || fail "docker compose up backend failed"
  wait_backend "$FIRST_LABEL" --timeout "$MAX_WAIT_SECONDS" \
    || fail "first boot with egress ${SMOKE_EGRESS}: /health never answered 200 within ${MAX_WAIT_SECONDS}s"
}

boot_new_container() {
  docker logs -t "$BACKEND_CONTAINER" > "$FIRST_CONTAINER_LOG" 2>&1 \
    || fail "could not keep the first backend container's log"
  echo "A new backend container, as a redeploy creates, with egress ${SMOKE_EGRESS}..."
  compose up -d --no-build --no-deps --force-recreate backend \
    || fail "docker compose up --force-recreate backend failed"
  wait_backend "$NEW_CONTAINER_LABEL" --timeout "$((FRESH_BOOT_BAR_SECONDS + BAR_GRACE_SECONDS))" \
    --within "$FRESH_BOOT_BAR_SECONDS" \
    || fail "a new backend container with egress ${SMOKE_EGRESS}: /health did not answer 200 within the ${FRESH_BOOT_BAR_SECONDS}s fresh-boot bar (PRD-262 §8)"
}

check_tokenizer() {
  docker exec "$BACKEND_CONTAINER" timeout "$TOKENIZER_TIMEOUT_SECONDS" python -c "$TOKENIZER_CHECK" \
    || fail "the tokenizer did not load from the image's TIKTOKEN_CACHE_DIR with egress ${SMOKE_EGRESS} (orchestrator/Dockerfile)"
  tokenizer_note="core.context_guard.get_tokenizer() loaded its encoding inside the new container"
}

dump_failure_logs() {
  echo "----- backend logs (tail) -----" >&2
  compose logs --tail 150 backend >&2 || true
}

# Both containers' logs, in order: the first one's is gone once it was recreated.
collect_backend_log() {
  {
    [ ! -f "$FIRST_CONTAINER_LOG" ] || cat "$FIRST_CONTAINER_LOG"
    docker logs -t "$BACKEND_CONTAINER" 2>&1
  } > "$BACKEND_LOG" || true
}

# The step summary is written on every exit, pass or fail.
write_report() {
  local summary="${GITHUB_STEP_SUMMARY:-/dev/stdout}"
  collect_backend_log
  compose logs -t --no-color > "${OUT_DIR}/compose.log" 2>&1 || true
  stop_event_log
  python3 "$REPORT_TOOL" \
    --results "$BOOTS_FILE" --events "$EVENTS_FILE" --backend-log "$BACKEND_LOG" \
    --backend-container "$BACKEND_CONTAINER" \
    --build-seconds "$build_seconds" --outcome "$outcome" --failure "$failure" \
    --title "Egress ${SMOKE_EGRESS}: the backend boots with no internet (PRD-262 US-033)" \
    --json "${OUT_DIR}/smoke-report.json" --summary "$summary" || return 1
  {
    echo "- Egress ${cut_note}."
    echo "- \`${FIRST_LABEL}\`: the first boot, on an empty database; /health 200 within ${MAX_WAIT_SECONDS} s."
    echo "- \`${NEW_CONTAINER_LABEL}\`: a new container; /health 200 within the ${FRESH_BOOT_BAR_SECONDS} s fresh-boot bar (PRD-262 §8)."
    echo "- Tokenizer: ${tokenizer_note}."
    echo ""
  } >> "$summary"
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
  restore_egress
  compose down -v || true
  exit "$code"
}

main() {
  mkdir -p "$OUT_DIR" || exit 1
  : > "$BOOTS_FILE"
  check_mode
  print_banner
  refuse_existing_stack
  trap finish EXIT
  build_backend
  start_event_log
  start_datastores
  cut_egress
  boot_first
  boot_new_container
  check_tokenizer
  outcome="pass"
  failure=""
  echo "OK: with egress ${SMOKE_EGRESS}, the first boot and a new container answered /health 200" \
    "(the new one within ${FRESH_BOOT_BAR_SECONDS}s) and the tokenizer loaded from the image"
}

main "$@"
