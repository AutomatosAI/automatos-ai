#!/usr/bin/env bash
# PRD-262 US-048: a frontend update downloads only the layers whose files changed.
#
#   check-layer-reuse.sh <first image> <second image>
#
# The two images are builds of frontend/Dockerfile's production stage from the same lock and
# the same public/ files, with different bundles: test.yml's prod-image-parity job passes its
# SaaS and local builds, which differ only in their NEXT_PUBLIC_* build args. The stage ends
# with four filesystem layers, the one that changes least often first: node_modules, public,
# .next/static and the server (orchestrator/tests/test_prd262_frontend_layers.py holds the
# Dockerfile to that). BuildKit keys a COPY on its files' content, not their mtimes, so:
# - the second build reuses every layer up to and including public (the same layer ids);
# - it makes its own server layer, which shows the pair really built two bundles.
#
# One case cannot show the reuse: when the two builds traced different dependencies, their
# node_modules layers differ, and public, above it, is rebuilt too. The script then lists the
# traced files that differ and warns. When the traced files are the same and the layer still
# differs, BuildKit did not reuse it, and the script fails.
#
# It needs only Docker: each image's own node lists its traced files, with no network.
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: $0 <first image> <second image>" >&2
  exit 2
fi
FIRST=$1
SECOND=$2

APP_LAYERS=(node_modules public static server)
DIFF_LINES=40
# Every entry under /app/node_modules, sorted: a directory's or file's mode (and a file's
# sha256), or a symlink's target. Those are what BuildKit's content key covers.
TRACED_FILES_JS='
const fs = require("fs"), path = require("path"), crypto = require("crypto");
const rows = [];
const sha256 = (file) => crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const walk = (dir) => fs.readdirSync(dir, { withFileTypes: true }).forEach((entry) => {
  const file = path.join(dir, entry.name);
  const mode = fs.lstatSync(file).mode.toString(8);
  if (entry.isSymbolicLink()) rows.push(`${file} -> ${fs.readlinkSync(file)}`);
  else if (entry.isDirectory()) { rows.push(`${file}/ ${mode}`); walk(file); }
  else rows.push(`${file} ${mode} ${sha256(file)}`);
});
walk("/app/node_modules");
console.log(rows.sort().join("\n"));
'

bad=0
fail() {
  echo "::error::$*"
  bad=1
}

# The image's layer ids (its RootFS diff ids), lowest first, on one line.
layer_ids() {
  docker image inspect -f '{{range .RootFS.Layers}}{{println .}}{{end}}' "$1" | tr '\n' ' '
}

traced_files() {
  docker run --rm --network none --entrypoint node "$1" -e "$TRACED_FILES_JS"
}

first_ids=$(layer_ids "$FIRST")
second_ids=$(layer_ids "$SECOND")
read -r -a first <<< "$first_ids"
read -r -a second <<< "$second_ids"
count=${#APP_LAYERS[@]}
if [ "${#first[@]}" -lt "$count" ] || [ "${#first[@]}" -ne "${#second[@]}" ]; then
  echo "::error::${FIRST} has ${#first[@]} layers and ${SECOND} ${#second[@]}: expected the same number, at least ${count}"
  exit 1
fi
# The position of node_modules, the first app layer; the base, Node and the user sit below it.
BASE=$((${#first[@]} - count))

shared() {
  [ "${first[$1]}" = "${second[$1]}" ]
}

state_of() {
  if shared "$1"; then echo reused; else echo rebuilt; fi
}

report() {
  local i line="" state
  for i in "${!APP_LAYERS[@]}"; do
    state=$(state_of $((BASE + i)))
    echo "${APP_LAYERS[$i]}: ${state} (${second[$((BASE + i))]:0:19})"
    line="${line}${line:+ · }${APP_LAYERS[$i]} ${state}"
  done
  { echo "### Frontend layer reuse (PRD-262 US-048)"
    echo "${SECOND} against ${FIRST}: ${line}"
  } >> "${GITHUB_STEP_SUMMARY:-/dev/null}"
}

check_the_base() {
  local i
  for ((i = 0; i < BASE; i++)); do
    shared "$i" || fail "layer $((i + 1)) differs, below node_modules: the two images do not share a base"
  done
}

check_the_server() {
  if shared $((BASE + count - 1)); then
    fail "${SECOND} reused ${FIRST}'s server layer: the two builds made the same bundle, so this pair shows nothing"
  fi
}

compare_traced_files() {
  local listing_first listing_second
  listing_first=$(traced_files "$FIRST")
  listing_second=$(traced_files "$SECOND")
  if [ "$listing_first" = "$listing_second" ]; then
    fail "node_modules was rebuilt although both images trace the same files: BuildKit did not reuse it"
    return 0
  fi
  echo "::warning::the two builds traced different dependencies, so node_modules and public above it were rebuilt: this pair cannot show their reuse"
  diff <(echo "$listing_first") <(echo "$listing_second") | head -n "$DIFF_LINES" || true
}

check_node_modules_and_public() {
  if ! shared "$BASE"; then
    compare_traced_files
    return 0
  fi
  if ! shared $((BASE + 1)); then
    fail "public/ was rebuilt although node_modules below it was reused"
  fi
}

report
check_the_base
check_the_server
check_node_modules_and_public
exit "$bad"
