#!/bin/bash
# Acceptance gate — PRD-258 Auto Live, back on track, Wave 1: measure the voice turn (US-001..US-004).
#   bash scripts/ralph/acceptance-prd258w1.sh                  run the gate (the wave's ONE CI run must be green on HEAD)
#   bash scripts/ralph/acceptance-prd258w1.sh --print-base     print the diff base
#   bash scripts/ralph/acceptance-prd258w1.sh --no-ci          every check but CI (the loop's local smoke, never its gate)
#   bash scripts/ralph/acceptance-prd258w1.sh --negative-only  only the scope/convention checks (smoke them on the base itself)
#   GATE_BASE=<ref> overrides the diff base (smoke runs only).
# Nothing runs on the owner's machine (owner rule: CI is the only gate): no pytest, no npm, no docker.
# The checks are greps, git and stdlib python; the tests are proven by test.yml on the wave's draft PR, once.
# No `pipefail` on purpose (a `! producer | grep -q X` can PASS under pipefail when X matched). Negative greps
# read only the wave's ADDED code lines (tests excluded), so a pattern that exists on the base never trips them.
set -u
exec </dev/null
cd "$(dirname "$0")/../.." || exit 1

export DATABASE_URL="postgresql://ralph:ralph@127.0.0.1:1/ralph_no_db"
export POSTGRES_HOST=127.0.0.1 POSTGRES_PORT=1 POSTGRES_USER=ralph POSTGRES_PASSWORD=ralph POSTGRES_DB=ralph_no_db
export REDIS_URL="redis://127.0.0.1:1/0" REDIS_HOST=127.0.0.1 REDIS_PORT=1
export ENVIRONMENT=development

if [ -n "${GATE_BASE:-}" ]; then
  BASE=$(git rev-parse --verify -q "$GATE_BASE^{commit}") || { echo "GATE_BASE $GATE_BASE does not resolve"; exit 1; }
else
  BASE=$(git merge-base HEAD origin/main) || { echo "cannot resolve merge-base with origin/main"; exit 1; }
fi
MODE=full
case "${1:-}" in
  --print-base) echo "$BASE"; exit 0 ;;
  --no-ci) MODE=no-ci ;;
  --negative-only) MODE=negative ;;
  "") ;;
  *) echo "unknown argument: $1"; exit 2 ;;
esac
echo "base: $(git log -1 --format='%h %s' "$BASE") · HEAD: $(git log -1 --format='%h %s' HEAD) · mode: $MODE"
FAIL=0
check() {
  local name="$1" body="$2"
  echo ""; echo "── $name"
  if ( eval "$body" ); then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi
}
warn() { local name="$1" body="$2"; echo ""; echo "── $name"; if ( eval "$body" ); then echo "   ✅ ok: $name"; else echo "   ⚠️  WARN (not a failure): $name"; fi; }

KIT=scripts/ralph/prd-258w1.json
VOICE=orchestrator/modules/voice
RETELL=orchestrator/api/voice_retell.py
TESTS=orchestrator/tests
EVAL=orchestrator/scripts/eval/voice_latency
VERSIONS=orchestrator/alembic/versions

# Added code lines of the wave under the given paths, tests, the kit and the eval CLI excluded (negative greps).
added_code() { git diff "$BASE"..HEAD -- "$@" ':!orchestrator/tests' ':!scripts/ralph' | grep '^+' | grep -v '^+++'; }
env_read_added() { added_code orchestrator ':!orchestrator/config.py' | grep -E 'os\.(getenv|environ)' | grep -q .; }
print_added()    { added_code orchestrator ':!orchestrator/scripts' | grep -E '^\+\s*print\(' | grep -q .; }
added_files()    { git diff --name-only --diff-filter=A "$BASE"..HEAD -- "$@"; }
lines_at()       { git show "$1:$2" 2>/dev/null | wc -l | tr -d ' '; }
every_commit_signed() { [ -z "$(git log --format='%H %(trailers:key=Signed-off-by,valueonly)' "$BASE"..HEAD | awk 'NF<2{print $1}')" ]; }
file_sizes_ok() { local f n bad=0; for f in $(added_files orchestrator scripts/voice); do [ -f "$f" ] || continue; n=$(wc -l < "$f" | tr -d ' '); [ "$n" -le 800 ] || { echo "   $f: $n lines"; bad=1; }; done; [ $bad -eq 0 ]; }
new_revisions()  { added_files "$VERSIONS" | grep '\.py$'; }
one_revision()   { local n; n=$(new_revisions | grep -c .); echo "   new revisions: $n"; [ "$n" -eq 1 ]; }
revision_ok() {
  local f rev; f=$(new_revisions | head -1); [ -n "$f" ] && [ -f "$f" ] || { echo "   no new revision file"; return 1; }
  grep -q 'voice_turns' "$f" && grep -q 'first_frame_ms' "$f" && grep -q 'vendor_latency' "$f" || { echo "   $f lacks a D2 column"; return 1; }
  grep -qi 'IF NOT EXISTS' "$f" || { echo "   $f: no IF NOT EXISTS"; return 1; }
  rev=$(grep -E "^revision *(: *str)? *= *['\"]" "$f" | head -1 | sed -E "s/.*['\"]([^'\"]+)['\"].*/\1/")
  [ -n "$rev" ] || { echo "   cannot read the revision id"; return 1; }
  grep -q "EXPECTED_HEAD = \"$rev\"" $TESTS/test_prd209_alembic_single_head.py || { echo "   test_prd209 EXPECTED_HEAD is not $rev"; return 1; }
  grep -q "$rev" $TESTS/test_prd236_w1_routes.py || { echo "   test_prd236_w1_routes.py does not name $rev"; return 1; }
  echo "   revision $rev in $f; both pins moved"
}
retell_not_grown() { local b h; b=$(lines_at "$BASE" $RETELL); h=$(wc -l < $RETELL | tr -d ' '); echo "   voice_retell.py: base $b, HEAD $h"; [ "$h" -le "$b" ]; }
public_host_once() { local n; n=$(grep -cE '^\s+PUBLIC_API_HOST\s*(:[^=]*)?=' orchestrator/config.py); echo "   PUBLIC_API_HOST assignments: $n"; [ "$n" -eq 1 ]; }
eval_ignores_data() { test -f $EVAL/.gitignore && grep -q 'results.md' $EVAL/.gitignore && grep -q 'asks' $EVAL/.gitignore; }
analyst_touched() { git diff "$BASE"..HEAD -- . ':!scripts/ralph' ':!docs/PRDS' ':!scripts/voice/README.md' ':!'"$EVAL"'/README.md' | grep '^+' | grep -q 'automatos-analyst'; }
data_added()      { git diff --name-only --diff-filter=A "$BASE"..HEAD | grep -qE '(^|/)(results\.md|asks[^/]*\.json)$'; }
all_acs_done() { python3 - "$KIT" <<'PY'
import json, re, sys
d = json.load(open(sys.argv[1])); bad = []
strip = lambda x: re.sub(r"`[^`]*`", "", x)   # the marks are plain text; the instructions quote them in backticks
for s in d["userStories"]:
    for raw in s["acceptanceCriteria"]:
        ac = strip(raw)
        if ac.startswith("→ OWNER:") or "→ DONE" in ac or "SKIPPED" in ac: continue
        bad.append(f"{s['id']}: {raw[:80]}")
print("\n".join(bad)); sys.exit(1 if bad else 0)
PY
}
ci_commit_for_head() {
  local sha; sha=$(git rev-parse HEAD)
  while git log -1 --format=%s "$sha" | grep -q 'skip ci' && [ -z "$(git show --format= --name-only "$sha" | grep -v '^scripts/ralph/')" ]; do
    sha=$(git rev-parse "$sha^")
  done
  echo "$sha"
}
ci_green_on_head() {
  local br sha waited=0 run id jobs job hits
  br=$(git rev-parse --abbrev-ref HEAD); sha=$(ci_commit_for_head)
  [ "$sha" = "$(git rev-parse HEAD)" ] || echo "   HEAD is kit-only [skip ci] commits over $(git log -1 --format=%h "$sha"): reading that commit's run"
  command -v gh >/dev/null || { echo "   gh is required"; return 1; }
  [ -n "$(gh pr list --head "$br" --state open --json number --jq '.[0].number' 2>/dev/null)" ] || { echo "   no open PR for $br: test.yml runs on pull_request only (#993)"; return 1; }
  while :; do
    run=$(gh run list --branch "$br" --workflow test.yml --limit 20 --json databaseId,headSha,status --jq ".[] | select(.headSha == \"$sha\") | \"\(.databaseId) \(.status)\"" 2>/dev/null | head -1)
    id="${run%% *}"
    [ -n "$run" ] && [ "${run##* }" = "completed" ] && break
    [ "$waited" -ge 3600 ] && { echo "   no completed test.yml run for $sha after 60 min (${run:-no run})"; return 1; }
    sleep 60; waited=$((waited + 60))
  done
  jobs=$(gh run view "$id" --json jobs --jq '.jobs[] | "\(.conclusion)\t\(.name)"')
  # media-render left test.yml for the on-demand media-render.yml (7 Oct 2026): not a job of this run.
  local REQUIRED=("orchestrator-tests" "Alembic from-zero" "Schema-drift check" "Frontend CI" "Prod images built" "Code standards on changed lines")
  local PATH_GATED=("Prod images built" "Frontend CI")
  for job in "${REQUIRED[@]}"; do
    hits=$(echo "$jobs" | grep -F "$job" | cut -f1)
    [ -n "$hits" ] || { echo "   missing job: $job"; return 1; }
    if printf '%s\n' "${PATH_GATED[@]}" | grep -qxF "$job"; then echo "$hits" | grep -qvE '^(success|skipped)$' && { echo "   red: $job"; return 1; }
    else echo "$hits" | grep -qvE '^success$' && { echo "   red: $job"; return 1; }; fi
  done
  echo "   test.yml run $id is green on $sha for all ${#REQUIRED[@]} required jobs (path-gated lanes: green or skipped)"
}

if [ "$MODE" != "negative" ]; then
  echo ""; echo "=== Wave 1 stories"
  check "US-001 the lifecycle-event helper moved to modules/voice/call_events.py" "test -f $VOICE/call_events.py && grep -qE 'def _?apply_call_event' $VOICE/call_events.py && ! grep -qE 'def _apply_call_event' $RETELL"
  check "US-001 the binding helpers left voice_retell.py" "! grep -qE 'def _(resolve_caller_int_id|named_call_agent)' $RETELL"
  check "US-001 one new Alembic revision with the D2 columns; both head pins moved" "one_revision && revision_ok"
  check "US-001 the ORM models carry the columns" "grep -q 'first_frame_ms' orchestrator/core/models/voice_turns.py && grep -q 'tools_before_speech' orchestrator/core/models/voice_turns.py && grep -q 'vendor_latency' orchestrator/core/models/voice_calls.py"
  check "US-001 turn_timing.py and the telemetry fields" "test -f $VOICE/turn_timing.py && grep -q 'first_auto_frame_ms' $VOICE/telemetry.py"
  check "US-001 tests" "test -f $TESTS/test_prd258_turn_timings.py"
  check "US-002 vendor latency subset + the get-call URL constant" "grep -q 'vendor_latency' $VOICE/call_events.py && grep -q 'get-call' $VOICE/retell_api.py"
  check "US-002 tests" "test -f $TESTS/test_prd258_vendor_latency.py"
  check "US-003 PUBLIC_API_HOST is assigned once in config.py" "public_host_once"
  check "US-003 the re-arm re-points the agent (the arming test asserts the PATCH carries response_engine and webhook_url)" "test -f $TESTS/test_prd258_local_arming.py && grep -q 'response_engine' $TESTS/test_prd258_local_arming.py && grep -q 'webhook_url' $TESTS/test_prd258_local_arming.py"
  check "US-003 the local agent name exists" "grep -rq 'Auto Live (local)' orchestrator/config.py $VOICE/"
  check "US-003 tunnel README + script, bash -n" "test -f scripts/voice/README.md && test -f scripts/voice/tunnel.sh && bash -n scripts/voice/tunnel.sh && grep -q 'ngrok' scripts/voice/tunnel.sh"
  check "US-003 tests" "test -f $TESTS/test_prd258_local_arming.py"
  check "US-004 the runner package" "for f in __init__.py README.md protocol.py report.py run.py; do test -f $EVAL/\$f || { echo \"   missing $EVAL/\$f\"; exit 1; }; done"
  check "US-004 its .gitignore keeps the owner's data out" "eval_ignores_data"
  check "US-004 tests" "test -f $TESTS/test_prd258_voice_latency_runner.py"
  check "every acceptance criterion in $KIT is DONE" "all_acs_done"
fi

echo ""; echo "=== Scope and conventions"
check "N1 at most one Alembic revision (D2)" "[ \$(new_revisions | grep -c .) -le 1 ]"
check "N2 no env reads added outside config.py" "! env_read_added"
check "N3 no print() added in orchestrator code outside orchestrator/scripts" "! print_added"
check "N4 no new Python dependency" "git diff --quiet $BASE..HEAD -- orchestrator/requirements.txt"
check "N5 no new npm dependency" "git diff --quiet $BASE..HEAD -- frontend/package.json"
check "N6 the route manifest is unchanged (no new route)" "git diff --quiet $BASE..HEAD -- orchestrator/reports/route-manifest.json"
check "N7 service.py untouched (PRD-256 is rewriting it)" "git diff --quiet $BASE..HEAD -- orchestrator/consumers/chatbot/service.py"
check "N8 voice_retell.py no longer than on the base (D6)" "retell_not_grown"
check "N9 the Composio deny list and the generated Auto skill seed are untouched" "git diff --quiet $BASE..HEAD -- orchestrator/core/composio/deny_list.py orchestrator/core/seeds/platform-management-skill.md"
check "N10 no node_modules in the diff" "! git diff --name-only $BASE..HEAD | grep -q node_modules"
check "N11 every commit is DCO-signed" "every_commit_signed"
check "N12 a new file is at most 800 lines" "file_sizes_ok"
check "N13 the hierarchy gate passes (stdlib ast)" "python3 orchestrator/scripts/check_hierarchy_gate.py >/dev/null"
check "N14 no eval data or run claimed: the analyst folder untouched, no results or ask files added (D4)" "! analyst_touched && ! data_added"
check "N15 the Retell tuning values are unchanged (measure, don't change)" "git diff $BASE..HEAD -- orchestrator/config.py | grep -E '^[-+].*VOICE_LIVE_(INTERRUPTION|RESPONSIVENESS|VOICE_|STT_MODE|DENOISING|LANGUAGE|FIRST_FRAME_ACK|SKIP_COMPOSIO|SPEECH_)' | grep -q . && exit 1 || exit 0"
warn "N16 story commits carry [skip ci] (all but the final CI commit)" "[ -z \"\$(git log --format='%s' $BASE..HEAD^ | grep -v 'skip ci' | grep -v '^chore(prd-258): seed')\" ]"

if [ "$MODE" = "full" ]; then
  echo ""; echo "=== CI (once per wave)"
  check "CI: the required jobs present and green on HEAD" "ci_green_on_head"
fi

echo ""
if [ "$FAIL" -eq 0 ]; then echo "✅ PRD-258 Wave 1 acceptance ($MODE): PASS"; else echo "❌ PRD-258 Wave 1 acceptance ($MODE): FAIL"; fi
exit "$FAIL"
