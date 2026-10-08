#!/bin/bash
# Acceptance gate — PRD-256 Auto, the night-12 fix wave (FX-001..FX-017).
#   bash scripts/ralph/acceptance-prd256fix.sh | --print-base | --no-ci | --negative-only ; GATE_BASE=<ref> (smoke runs only).
# Base: stacked on Wave 2 (feat/prd-256-w2-model-lanes, which carries Wave 1) until Wave 2 is on main, then main;
# the NEWER fork point wins. Nothing runs on the owner's machine; greps, git and stdlib python only; CI once per wave
# on the pushed HEAD. No pipefail.
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
  for PREV_REF in origin/feat/prd-256-w1-receipts-gates origin/feat/prd-256-w2-model-lanes; do
    if git rev-parse -q --verify "$PREV_REF" >/dev/null; then
      PREV_BASE=$(git merge-base HEAD "$PREV_REF" 2>/dev/null || true)
      if [ -n "$PREV_BASE" ] && [ "$PREV_BASE" != "$BASE" ] && git merge-base --is-ancestor "$BASE" "$PREV_BASE"; then BASE="$PREV_BASE"; fi
    fi
  done
fi
MODE=full
case "${1:-}" in --print-base) echo "$BASE"; exit 0 ;; --no-ci) MODE=no-ci ;; --negative-only) MODE=negative ;; "") ;; *) echo "unknown argument: $1"; exit 2 ;; esac
echo "base: $(git log -1 --format='%h %s' "$BASE") · HEAD: $(git log -1 --format='%h %s' HEAD) · mode: $MODE"
FAIL=0
check() { local name="$1" body="$2"; echo ""; echo "── $name"; if ( eval "$body" ); then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi; }
warn() { local name="$1" body="$2"; echo ""; echo "── $name"; if ( eval "$body" ); then echo "   ✅ ok: $name"; else echo "   ⚠️  WARN (not a failure): $name"; fi; }

KIT=scripts/ralph/prd-256fix.json
CHAT=orchestrator/consumers/chatbot
EXEC=orchestrator/modules/tools/execution
DISC=orchestrator/modules/tools/discovery
TESTS=orchestrator/tests
SIM=tests/sim

added_code() { git diff "$BASE"..HEAD -- "$@" ':!orchestrator/tests' ':!tests/sim/tests' ':!*__tests__*' ':!*.test.ts' ':!*.test.tsx' ':!scripts/ralph' | grep '^+' | grep -v '^+++'; }
env_read_added()      { added_code orchestrator ':!orchestrator/config.py' | grep -E 'os\.(getenv|environ)' | grep -q .; }
print_added()         { added_code orchestrator | grep -E '^\+\s*print\(' | grep -q .; }
raw_fetch_added()     { added_code frontend | grep -E "fetch\(['\"]/api" | grep -q .; }
failover_default_set() { added_code orchestrator | grep -E "LLM_FAILOVER_MODEL[^=]*=\s*[\"'][A-Za-z]" | grep -q .; }
owner_words_regex_added() { added_code $DISC/owner_only.py $DISC/follows_the_owner.py | grep -E 're\.(compile|search|match)\(.*(approve|cancel|go ahead|yes)' | grep -q .; }
added_files() { git diff --name-only --diff-filter=A "$BASE"..HEAD -- "$@"; }
lines_at() { git show "$1:$2" 2>/dev/null | wc -l | tr -d ' '; }
every_commit_signed() { [ -z "$(git log --format='%H %(trailers:key=Signed-off-by,valueonly)' "$BASE"..HEAD | awk 'NF<2{print $1}')" ]; }
file_sizes_ok() { local f n bad=0; for f in $(added_files orchestrator frontend tests); do [ -f "$f" ] || continue; n=$(wc -l < "$f" | tr -d ' '); [ "$n" -le 800 ] || { echo "   $f: $n lines"; bad=1; }; done; [ $bad -eq 0 ]; }
giants_not_grown() { local f b h bad=0; for f in $CHAT/service.py $CHAT/auto.py orchestrator/modules/tools/tool_router.py $CHAT/smart_memory.py; do b=$(lines_at "$BASE" "$f"); h=$(wc -l < "$f" | tr -d ' '); [ "$h" -le "$b" ] || { echo "   $f grew: base $b, HEAD $h"; bad=1; }; done; [ $bad -eq 0 ]; }
all_acs_done() { python3 - "$KIT" <<'PY'
import json, sys, re
d = json.load(open(sys.argv[1])); bad = []
strip = lambda x: re.sub(r"`[^`]*`", "", x)   # the marks are plain text; the instructions quote them in backticks
for s in d["userStories"]:
    for raw in s["acceptanceCriteria"]:
        ac = strip(raw)
        if ac.startswith("→ OWNER:") or "→ DONE" in ac: continue
        bad.append(f"{s['id']}: {raw[:80]}")
print("\n".join(bad)); sys.exit(1 if bad else 0)
PY
}
tests_not_weakened() { local d a; d=$(git diff "$BASE"..HEAD -- $TESTS $SIM/tests | grep -cE '^-\s*(async )?def test_'); a=$(git diff "$BASE"..HEAD -- $TESTS $SIM/tests | grep -cE '^\+\s*(async )?def test_'); echo "   test functions removed $d, added $a"; [ "$a" -ge "$d" ]; }
service_shorter() { local b h; b=$(lines_at "$BASE" $CHAT/service.py); h=$(wc -l < $CHAT/service.py | tr -d ' '); echo "   service.py: base $b, HEAD $h"; [ "$h" -lt "$b" ]; }
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
    if printf '%s\n' "${PATH_GATED[@]}" | grep -qxF "$job"; then echo "$hits" | grep -qvE '^(success|skipped)$' && { echo "   red: $job"; return 1; }
    else echo "$hits" | grep -qvE '^success$' && { echo "   red: $job"; return 1; }; fi
  done
  echo "   test.yml run $id is green on $sha for all ${#REQUIRED[@]} required jobs"
}

if [ "$MODE" != "negative" ]; then
  echo ""; echo "=== Fix stories"
  check "FX-001 the US-004 test fixture drives the executor with the chat's user" "grep -q 'driving_user_id' $TESTS/test_prd256_owner_actions_wait_for_the_click.py"
  check "FX-001 the prd232 ranker tests assert the promoted (first-class) truth" "grep -qE 'first_class' $TESTS/test_prd232_us006_corpus_embeddings.py"
  check "FX-002 the sim client keeps the receipts frame, the above lines and applies retractions" "grep -q 'receipts' $SIM/sse.py && grep -q 'retracted' $SIM/sse.py && grep -q 'text_raw' $SIM/customer.py"
  check "FX-002 sim tests cover the frames" "grep -rlq 'receipts' $SIM/tests/"
  check "FX-003 the signature rule yields to the owner-only gate" "grep -q 'is_owner_only' $DISC/follows_the_owner.py && test -f $TESTS/test_prd256_fix_card_before_signature.py"
  check "FX-004 the waiting status exists on both sides" "grep -q '\"waiting\"' $CHAT/receipts.py && grep -q 'waiting' frontend/lib/chat/receipts.ts && test -f $TESTS/test_prd256_fix_waiting_receipt.py"
  check "FX-005 the tool-end flag reads the result; one producer for the nothing-done line" "grep -q 'the_results_flag(event)' $EXEC/tool_loop.py && grep -q 'result.get(\"success\"' $EXEC/card_raised.py && test -f $TESTS/test_prd256_fix_stream_truth.py"
  check "FX-006 honesty per claim tests" "test -f $TESTS/test_prd256_fix_honesty_per_claim.py"
  check "FX-007 the families are deleted" "! test -f $EXEC/action_claims.py && ! test -f $EXEC/document_claims.py && ! test -f $EXEC/shop_and_team_claims.py && ! test -f $EXEC/social_post_claims.py && ! test -f $CHAT/figure_disputes.py && ! test -f $CHAT/shop_figures.py && ! test -f $CHAT/team_corrections.py"
  check "FX-007 service.py is shorter than the base and under 3200 lines" "service_shorter && [ \$(wc -l < $CHAT/service.py | tr -d ' ') -le 3200 ]"
  check "FX-007 tests and the W2 kit's US-012 marks" "test -f $TESTS/test_prd256_families_deleted.py && grep -q 'built in the fix wave as FX-007' scripts/ralph/prd-256w2.json"
  check "FX-008 cards say what they approve: tests" "test -f $TESTS/test_prd256_fix_cards_say_what_they_approve.py"
  check "FX-009 mission ids, click results, the two pins" "grep -q 'platform_approve_mission' orchestrator/config.py && grep -q 'platform_cancel_mission' orchestrator/config.py && test -f $TESTS/test_prd256_fix_mission_ids_and_click_results.py"
  check "FX-010 the owner-only list grew" "grep -q 'platform_configure_agent_heartbeat' $DISC/owner_only.py && grep -q 'platform_delete_agent' $DISC/owner_only.py && test -f $TESTS/test_prd256_fix_owner_only_list.py"
  check "FX-011 an agent's send on an Auto-written ticket goes through the click" "grep -qE 'is_composio_send|asks_before_a_send|send_ask' orchestrator/services/session_tools.py && test -f $TESTS/test_prd256_fix_agent_sends_wait_for_the_click.py"
  check "FX-012 task tools take agent_id" "grep -q '\"agent_id\"' $DISC/actions_board_tasks.py && test -f $TESTS/test_prd256_fix_agent_ids.py"
  check "FX-013 schema aliases tests" "test -f $TESTS/test_prd256_fix_schema_aliases.py"
  check "FX-014 the verdict parser and the lane" "! grep -q 'falling back to ATOM' $CHAT/auto.py && test -f $TESTS/test_prd256_fix_verdict_parser.py && grep -qi 'get ops' $TESTS/fixtures/prd256_routing_golden.json"
  check "FX-015 standing rules and memory ownership tests" "test -f $TESTS/test_prd256_fix_standing_rules.py"
  check "FX-016 the runtime parameter and its default" "grep -q '\"runtime\"' $DISC/actions_agents.py && grep -q 'DEFAULT_AGENT_RUNTIME' orchestrator/config.py && grep -q 'DEFAULT_AGENT_RUNTIME' orchestrator/reports/config-surface.json && test -f $TESTS/test_prd256_fix_agent_runtime.py"
  check "FX-017 one answer on screen tests" "test -f $TESTS/test_prd256_fix_one_answer_on_screen.py"
  check "every acceptance criterion in $KIT is DONE (the owner's are left)" "all_acs_done"
fi

echo ""; echo "=== Scope and conventions"
check "N1 no Alembic revision" "[ -z \"\$(added_files orchestrator/alembic/versions)\" ]"
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
check "N12 the four giants did not grow (service.py, auto.py, tool_router.py, smart_memory.py)" "giants_not_grown"
check "N13 no Jev live: the decision engine is untouched (D4)" "git diff --quiet $BASE..HEAD -- orchestrator/core/llm/decisions"
check "N14 no failover model set by default (D5)" "! failover_default_set"
check "N15 the hierarchy gate passes (stdlib ast)" "python3 orchestrator/scripts/check_hierarchy_gate.py >/dev/null"
check "N16 no eval run claimed: no added line outside tests/sim names the analyst or sim folders" "! git diff $BASE..HEAD -- . ':!scripts/ralph' ':!tests/sim' | grep '^+' | grep -v '^+++' | grep -qE 'automatos-analyst|\.automatos-sim'"
check "N17 no regex on the owner's words added to the gates (D1)" "! owner_words_regex_added"
check "N18 no test weakened: added test functions ≥ removed" "tests_not_weakened"
warn "N19 story commits carry [skip ci] (all but the final CI commit)" "[ -z \"\$(git log --format='%s' $BASE..HEAD^ | grep -v 'skip ci' | grep -v '^chore(prd-256): seed')\" ]"

if [ "$MODE" = "full" ]; then
  echo ""; echo "=== CI (once per wave)"
  check "CI: the required jobs present and green on HEAD" "ci_green_on_head"
fi
echo ""
if [ "$FAIL" -eq 0 ]; then echo "✅ PRD-256 night-12 fix wave acceptance ($MODE): PASS"; else echo "❌ PRD-256 night-12 fix wave acceptance ($MODE): FAIL"; fi
exit "$FAIL"
