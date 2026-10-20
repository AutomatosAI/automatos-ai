#!/usr/bin/env bash
# PRD-262 US-039: does `git pull && make build` reuse the dependency installs?
#
# make up, make build and make dev end with make clean. Before US-039 it ran a plain
# `docker builder prune -f`, which deletes every build cache record no image holds, and
# the two dependency installs are in no image: the frontend's npm install and the
# backend's pip install run in build stages whose output the production images only
# copy. make clean now runs `docker builder prune -f --filter until=168h`, which keeps
# the cache the last week's builds used. This experiment shows both, on the layer cache
# alone (no Railway-built Dockerfile has a BuildKit cache mount, PRD-262 §10):
#   1. it builds the backend and the frontend from this checkout, as make build does
#      (the production stages, under a local-* tag);
#   2. it runs `docker builder prune -f`, changes one line of each app's source as a git
#      pull would, rebuilds, and records whether npm install and pip install were CACHED;
#   3. it runs `make clean` itself, changes another line, rebuilds, and requires both
#      installs CACHED.
# scripts/ci/build_cache.py reads each rebuild's log (BuildKit's plain progress, which a
# piped `docker compose build` prints) and writes the step summary. The logs and the
# records land in $BUILD_CACHE_OUT_DIR, which the workflow uploads. Each build's time is
# reported, not gated: CI build time has no bar (PRD-262 §7).
#
# Both prunes delete this Docker host's unused build cache, whoever's it is, so outside
# CI (CI=true) it refuses unless BUILD_CACHE_PRUNE_OK=true says the cache may go. The two
# source files are put back on exit from git, so it also refuses while either has
# uncommitted changes, which that would discard. Nothing is started.
#
# Exit 0 = after make clean, a rebuild reused both installs. Exit 1 = a build or a prune
# failed, or an install ran again after make clean.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT_DIR="${BUILD_CACHE_OUT_DIR:-${RUNNER_TEMP:-/tmp}/build-cache}"
TOOL="scripts/ci/build_cache.py"
SUMMARY="${GITHUB_STEP_SUMMARY:-/dev/stdout}"
# The two services whose builds hold the installs. The worker installs in its only
# stage, which its image keeps, so a plain prune cannot cost it anything.
SERVICES=(backend frontend)
# One line of each app's source changes before each rebuild, as a git pull would.
BACKEND_SOURCE="orchestrator/main.py"
FRONTEND_SOURCE="frontend/app/layout.tsx"
# What make clean ran before PRD-262 US-039, as the summary names it.
PLAIN_PRUNE="docker builder prune -f"

# Throwaway values: docker-compose.yml refuses to load with these unset (${…:?}), even
# for a build. Nothing starts, so nothing reads them.
export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-build_cache_pg_pw}"
export REDIS_PASSWORD="${REDIS_PASSWORD:-build_cache_redis_pw}"
export API_KEY="${API_KEY:-build_cache_api_key}"
# A source build's tag and revision label, as make build sets them (PRD-262 US-036).
AUTOMATOS_REVISION="${AUTOMATOS_REVISION:-$(git rev-parse HEAD 2>/dev/null)}"
export AUTOMATOS_REVISION
export AUTOMATOS_IMAGE_TAG="${AUTOMATOS_IMAGE_TAG:-local-build-cache}"

fail() {
  echo "FAIL: $1" >&2
  printf '\n**FAIL**: %s\n' "$1" >> "$SUMMARY"
  exit 1
}

# Runs before anything is pruned or changed.
refuse_pruning_a_kept_build_cache() {
  [ "${CI:-}" = "true" ] || [ "${BUILD_CACHE_PRUNE_OK:-}" = "true" ] \
    || fail "this experiment prunes the Docker host's unused build cache, whoever's it is. Run it on a throwaway host such as a CI runner, or set BUILD_CACHE_PRUNE_OK=true to let it prune here"
}

# Runs before anything is changed: restore_sources puts back HEAD's copy.
refuse_changing_uncommitted_sources() {
  git diff --quiet HEAD -- "$BACKEND_SOURCE" "$FRONTEND_SOURCE" \
    || fail "${BACKEND_SOURCE} or ${FRONTEND_SOURCE} has uncommitted changes, which putting the sources back on exit would discard. Commit or stash them first"
}

restore_sources() {
  git checkout -- "$BACKEND_SOURCE" "$FRONTEND_SOURCE" 2>/dev/null || true
}

# $1 = the build's name, also its log's. Piped, so BuildKit prints its plain progress.
build() {
  local started=$SECONDS
  echo "docker compose build ${SERVICES[*]} (${1})..."
  docker compose build "${SERVICES[@]}" 2>&1 | tee "${OUT_DIR}/${1}.log" \
    || fail "docker compose build ${SERVICES[*]} failed (${1})"
  printf -- '- `%s`: built in %s s (report only).\n' "$1" "$((SECONDS - started))" >> "$SUMMARY"
}

# $1 = what ran before the rebuild. One line more in each app's source.
change_a_line() {
  printf '\n# PRD-262 US-039 build-cache experiment: a line changed after %s\n' "$1" >> "$BACKEND_SOURCE"
  printf '\n// PRD-262 US-039 build-cache experiment: a line changed after %s\n' "$1" >> "$FRONTEND_SOURCE"
}

# $1 = the rebuild's name, $2 = what ran before it; further arguments go to build_cache.py.
read_installs() {
  local name="$1" arm="$2"
  shift 2
  python3 "$TOOL" "${OUT_DIR}/${name}.log" --arm "$arm" --out "${OUT_DIR}/${name}.json" \
    --summary "$SUMMARY" "$@"
}

main() {
  mkdir -p "$OUT_DIR" || exit 1
  refuse_pruning_a_kept_build_cache
  refuse_changing_uncommitted_sources
  trap restore_sources EXIT
  printf '## Source build cache (PRD-262 US-039)\n\n' >> "$SUMMARY"
  build first-build
  docker builder prune -f || fail "${PLAIN_PRUNE} failed"
  change_a_line "$PLAIN_PRUNE"
  build after-plain-prune
  read_installs after-plain-prune "$PLAIN_PRUNE" || fail "the rebuild after ${PLAIN_PRUNE} could not be read"
  make clean || fail "make clean failed"
  change_a_line "make clean"
  build after-make-clean
  read_installs after-make-clean "make clean" --expect-cached \
    || fail "after make clean, a rebuild ran a dependency install again: the next make build after a git pull would reinstall it (PRD-262 US-039)"
  echo "OK: after make clean, the rebuild reused the npm and pip installs (CACHED)"
}

main "$@"
