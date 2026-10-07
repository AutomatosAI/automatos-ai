#!/bin/bash
# Acceptance gate — PRD-256 Auto, receipts not narration, Wave 1: receipts, honesty, gates, tool contracts, the instrument (US-001..US-007).
#   bash scripts/ralph/acceptance-prd256w1.sh                  run the gate (the wave's ONE CI run must be green on HEAD)
#   bash scripts/ralph/acceptance-prd256w1.sh --print-base     print the diff base
#   bash scripts/ralph/acceptance-prd256w1.sh --no-ci          every check but CI (the loop's local smoke, never its gate)
#   bash scripts/ralph/acceptance-prd256w1.sh --negative-only  only the scope/convention checks (smoke them on the base itself)
#   GATE_BASE=<ref> overrides the diff base (smoke runs only).
# Nothing runs on the owner's machine (owner rule: CI is the only gate): no pytest, no npm, no docker.
# The checks are greps, git and stdlib python; the tests are proven by test.yml on the pushed HEAD, once per wave.
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

KIT=scripts/ralph/prd-256w1.json
CHAT=orchestrator/consumers/chatbot
EXEC=orchestrator/modules/tools/execution
DISC=orchestrator/modules/tools/discovery
TESTS=orchestrator/tests

# Added code lines of the wave, tests and fixtures excluded (for the negative greps).
# Added code lines (no tests) under the given paths; `:!` pathspecs exclude.
added_code() { git diff "$BASE"..HEAD -- "$@" ':!orchestrator/tests' ':!*__tests__*' ':!*.test.ts' ':!*.test.tsx' ':!scripts/ralph' | grep '^+' | grep -v '^+++'; }
env_read_added()     { added_code orchestrator ':!orchestrator/config.py' | grep -E 'os\.(getenv|environ)' | grep -q .; }
print_added()        { added_code orchestrator | grep -E '^\+\s*print\(' | grep -q .; }
raw_fetch_added()    { added_code frontend | grep -E "fetch\(['\"]/api" | grep -q .; }
receipts_read_logs() { test -f $CHAT/receipts.py && grep -qE 'tool_execution_logs|ToolExecutionLog' $CHAT/receipts.py; }
added_files() { git diff --name-only --diff-filter=A "$BASE"..HEAD -- "$@"; }
count_at() { git show "$1:$2" 2>/dev/null | grep -c "$3"; }
lines_at() { git show "$1:$2" 2>/dev/null | wc -l | tr -d ' '; }
every_commit_signed() { [ -z "$(git log --format='%H %(trailers:key=Signed-off-by,valueonly)' "$BASE"..HEAD | awk 'NF<2{print $1}')" ]; }
file_sizes_ok() { local f n bad=0; for f in $(added_files orchestrator frontend); do [ -f "$f" ] || continue; n=$(wc -l < "$f" | tr -d ' '); [ "$n" -le 800 ] || { echo "   $f: $n lines"; bad=1; }; done; [ $bad -eq 0 ]; }
all_acs_done() { python3 - "$KIT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); bad = []
strip = lambda x: __import__("re").sub(r"`[^`]*`", "", x)   # the marks are plain text; the instructions quote them in backticks
for s in d["userStories"]:
    for raw in s["acceptanceCriteria"]:
        ac = strip(raw)
        if ac.startswith("→ OWNER:") or "→ DONE" in ac or "SKIPPED" in ac: continue
        bad.append(f"{s['id']}: {raw[:80]}")
print("\n".join(bad)); sys.exit(1 if bad else 0)
PY
}
ci_green_on_head() {
  local br sha waited=0 run id jobs job hits
  br=$(git rev-parse --abbrev-ref HEAD); sha=$(git rev-parse HEAD)
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
  local REQUIRED=("orchestrator-tests" "Alembic from-zero" "Schema-drift check" "Frontend CI" "Prod images built" "media-render" "Code standards on changed lines")
  local PATH_GATED=("media-render" "Prod images built")
  for job in "${REQUIRED[@]}"; do
    hits=$(echo "$jobs" | grep -F "$job" | cut -f1)
    [ -n "$hits" ] || { echo "   missing job: $job"; return 1; }
    if printf '%s\n' "${PATH_GATED[@]}" | grep -qxF "$job"; then
      echo "$hits" | grep -qvE '^(success|skipped)$' && { echo "   red: $job"; return 1; }
    else
      echo "$hits" | grep -qvE '^success$' && { echo "   red: $job"; return 1; }
    fi
  done
  echo "   test.yml run $id is green on $sha for all ${#REQUIRED[@]} required jobs (path-gated lanes: green or skipped)"
}
service_not_grown() { local b h; b=$(lines_at "$BASE" $CHAT/service.py); h=$(wc -l < $CHAT/service.py | tr -d ' '); echo "   service.py: base $b, HEAD $h"; [ "$h" -le $((b + 40)) ]; }
families_not_grown() { local f ok=1 b h; for f in $EXEC/action_claims.py $EXEC/document_claims.py $EXEC/shop_and_team_claims.py; do b=$(count_at "$BASE" "$f" "_Family(\|family("); h=$(grep -c "_Family(\|family(" "$f"); echo "   $f: base $b, HEAD $h"; [ "$h" -le "$b" ] || ok=0; done; [ $ok -eq 1 ]; }
no_new_lane_module() { local f bad=0; for f in $(added_files $CHAT); do case "$(basename "$f")" in receipts.py|__init__.py) ;; *) echo "   new module: $f"; bad=1 ;; esac; done; [ $bad -eq 0 ]; }
promoted_named() { local a bad=0; for a in platform_create_task platform_update_task platform_update_task_status platform_assign_task platform_create_mission platform_execute_playbook platform_schedule_playbook platform_create_social_post platform_store_memory platform_get_task platform_query_data; do git diff "$BASE"..HEAD -- orchestrator/modules/tools orchestrator/config.py | grep '^+' | grep -q "$a" || { echo "   not promoted in the diff: $a"; bad=1; }; done; [ $bad -eq 0 ]; }

if [ "$MODE" != "negative" ]; then
  echo ""; echo "=== Wave 1 stories"
  check "US-001 receipts module exists" "test -f $CHAT/receipts.py"
  check "US-001 the receipts part type reaches reply_parts and the frame" "grep -q '\"receipts\"' $CHAT/narration.py $CHAT/receipts.py"
  check "US-001 the frontend MessagePart union has receipts" "grep -q \"'receipts'\" frontend/types/chat.ts"
  check "US-001 the hook absorbs the receipts frame and the message renders it" "grep -q 'receipts' frontend/lib/chat/hooks.ts && grep -q -i 'receipt' frontend/components/chatbot/message.tsx"
  check "US-001 tests: backend + vitest" "test -f $TESTS/test_prd256_receipts_part.py && ls frontend/components/chatbot/__tests__/ | grep -qi receipt"
  check "US-002 honesty rule test with the 25 cases and the frozen counts" "test -f $TESTS/test_prd256_honesty_rule.py && grep -q 'families_frozen' $TESTS/test_prd256_honesty_rule.py"
  check "US-003 memory takes receipts (store signature + test)" "grep -q 'receipts' $CHAT/integration.py && test -f $TESTS/test_prd256_memory_takes_receipts.py"
  check "US-004 owner_only module with OWNER_ONLY_ACTIONS, wired into the executor" "grep -q 'OWNER_ONLY_ACTIONS' $DISC/owner_only.py && grep -q 'owner_only' $DISC/platform_executor.py"
  check "US-004 follows_the_owner no longer applies APPROVE/CANCEL/GO_AHEAD" "! grep -qE '\b(APPROVE|CANCEL|GO_AHEAD)\b' $DISC/follows_the_owner.py"
  check "US-004 test: the ask, the grant, the signer" "test -f $TESTS/test_prd256_owner_actions_wait_for_the_click.py"
  check "US-005 done needs an artifact (handler + test)" "grep -qi 'artifact' $DISC/handlers_board_task_done.py && test -f $TESTS/test_prd256_done_needs_artifact.py"
  check "US-006 the eleven write actions appear in the promotion diff" "promoted_named"
  check "US-006 the refused-write rule in the nudges" "grep -qi 'refused' $EXEC/nudges.py"
  check "US-006 tests" "test -f $TESTS/test_prd256_tool_contracts.py"
  check "US-007 the instrument's docs page" "test -f docs/testing/AUTO-EVAL.md"
  check "every acceptance criterion in $KIT is DONE (the owner's are left to the owner)" "all_acs_done"
fi

echo ""; echo "=== Scope and conventions"
check "N1 no Alembic revision (no migration in this PRD)" "[ -z \"\$(added_files orchestrator/alembic/versions)\" ]"
check "N2 no env reads added outside config.py" "! env_read_added"
check "N3 no print() added in orchestrator code" "! print_added"
check "N4 no raw fetch('/api…') added in the frontend" "! raw_fetch_added"
check "N5 no new Python dependency" "git diff --quiet $BASE..HEAD -- orchestrator/requirements.txt"
check "N6 no new npm dependency" "git diff --quiet $BASE..HEAD -- frontend/package.json"
check "N7 the Composio deny list is untouched" "git diff --quiet $BASE..HEAD -- orchestrator/core/composio/deny_list.py"
check "N8 the generated Auto skill seed is untouched" "git diff --quiet $BASE..HEAD -- orchestrator/core/seeds/platform-management-skill.md"
check "N9 no node_modules in the diff" "! git diff --name-only $BASE..HEAD | grep -q node_modules"
check "N10 every commit is DCO-signed" "every_commit_signed"
check "N11 a new file is at most 800 lines" "file_sizes_ok"
check "N12 service.py did not grow beyond the wiring (≤ base + 40 lines)" "service_not_grown"
check "N13 the claim families did not grow (D10)" "families_not_grown"
check "N14 no new lane module under consumers/chatbot other than receipts.py" "no_new_lane_module"
check "N15 the hierarchy gate passes (stdlib ast)" "python3 orchestrator/scripts/check_hierarchy_gate.py >/dev/null"
check "N16 receipts never read tool_execution_logs (D8)" "! receipts_read_logs"
check "N17 no eval run claimed: the loop never touched the analyst folder" "! git diff $BASE..HEAD | grep -q 'automatos-analyst'"
warn "N18 story commits carry [skip ci] (all but the final CI commit)" "[ -z \"\$(git log --format='%s' $BASE..HEAD^ | grep -v 'skip ci' | grep -v '^chore(prd-256): seed')\" ]"

if [ "$MODE" = "full" ]; then
  echo ""; echo "=== CI (once per wave)"
  check "CI: the required jobs present and green on HEAD" "ci_green_on_head"
fi

echo ""
if [ "$FAIL" -eq 0 ]; then echo "✅ PRD-256 Wave 1 acceptance ($MODE): PASS"; else echo "❌ PRD-256 Wave 1 acceptance ($MODE): FAIL"; fi
exit "$FAIL"
