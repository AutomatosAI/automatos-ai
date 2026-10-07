#!/bin/bash
# Acceptance gate — PRD-256 Auto, receipts not narration, Wave 2: the model, Auto always answers, one hand-off table, the families deleted (US-008..US-013).
#   bash scripts/ralph/acceptance-prd256w2.sh | --print-base | --no-ci | --negative-only ; GATE_BASE=<ref> (smoke runs only).
# Base: stacked on Wave 1 (feat/prd-256-w1-receipts-gates) until Wave 1 is on main, then main; the NEWER fork point wins.
# Nothing runs on the owner's machine; greps, git and stdlib python only; CI once per wave on the pushed HEAD. No pipefail.
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
  PREV_REF=origin/feat/prd-256-w1-receipts-gates
  if git rev-parse -q --verify "$PREV_REF" >/dev/null; then
    PREV_BASE=$(git merge-base HEAD "$PREV_REF" 2>/dev/null || true)
    if [ -n "$PREV_BASE" ] && [ "$PREV_BASE" != "$BASE" ] && git merge-base --is-ancestor "$BASE" "$PREV_BASE"; then BASE="$PREV_BASE"; fi
  fi
fi
MODE=full
case "${1:-}" in --print-base) echo "$BASE"; exit 0 ;; --no-ci) MODE=no-ci ;; --negative-only) MODE=negative ;; "") ;; *) echo "unknown argument: $1"; exit 2 ;; esac
echo "base: $(git log -1 --format='%h %s' "$BASE") · HEAD: $(git log -1 --format='%h %s' HEAD) · mode: $MODE"
FAIL=0
check() { local name="$1" body="$2"; echo ""; echo "── $name"; if ( eval "$body" ); then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi; }
warn() { local name="$1" body="$2"; echo ""; echo "── $name"; if ( eval "$body" ); then echo "   ✅ ok: $name"; else echo "   ⚠️  WARN (not a failure): $name"; fi; }

KIT=scripts/ralph/prd-256w2.json
CHAT=orchestrator/consumers/chatbot
EXEC=orchestrator/modules/tools/execution
TESTS=orchestrator/tests
LLM=orchestrator/core/llm

added_code() { git diff "$BASE"..HEAD -- "$@" ':!orchestrator/tests' ':!*__tests__*' ':!*.test.ts' ':!*.test.tsx' ':!scripts/ralph' | grep '^+' | grep -v '^+++'; }
env_read_added()      { added_code orchestrator ':!orchestrator/config.py' | grep -E 'os\.(getenv|environ)' | grep -q .; }
print_added()         { added_code orchestrator | grep -E '^\+\s*print\(' | grep -q .; }
raw_fetch_added()     { added_code frontend | grep -E "fetch\(['\"]/api" | grep -q .; }
failover_default_set() { added_code orchestrator | grep -E "LLM_FAILOVER_MODEL[^=]*=\s*[\"'][A-Za-z]" | grep -q .; }
seed_reads_defaults() { test -f $LLM/defaults.py && grep -qE 'core\.llm\.defaults|from core\.llm import defaults' orchestrator/core/seeds/seed_auto_agent.py; }
added_files() { git diff --name-only --diff-filter=A "$BASE"..HEAD -- "$@"; }
lines_at() { git show "$1:$2" 2>/dev/null | wc -l | tr -d ' '; }
every_commit_signed() { [ -z "$(git log --format='%H %(trailers:key=Signed-off-by,valueonly)' "$BASE"..HEAD | awk 'NF<2{print $1}')" ]; }
file_sizes_ok() { local f n bad=0; for f in $(added_files orchestrator frontend); do [ -f "$f" ] || continue; n=$(wc -l < "$f" | tr -d ' '); [ "$n" -le 800 ] || { echo "   $f: $n lines"; bad=1; }; done; [ $bad -eq 0 ]; }
story_state() { python3 - "$KIT" "$1" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); sid = sys.argv[2]
for s in d["userStories"]:
    if s["id"].endswith(sid):
        import re
        acs = [re.sub(r"`[^`]*`", "", a) for a in s["acceptanceCriteria"] if not a.startswith("→ OWNER:")]
        print("SKIPPED" if all("SKIPPED —" in a for a in acs) else ("BLOCKED" if all("→ BLOCKED" in a for a in acs) else "BUILT")); break
PY
}
all_acs_done() { python3 - "$KIT" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); bad = []
strip = lambda x: __import__("re").sub(r"`[^`]*`", "", x)   # the marks are plain text; the instructions quote them in backticks
for s in d["userStories"]:
    for raw in s["acceptanceCriteria"]:
        ac = strip(raw)
        if ac.startswith("→ OWNER:") or "→ DONE" in ac or "SKIPPED —" in ac or "→ BLOCKED" in ac: continue
        bad.append(f"{s['id']}: {raw[:80]}")
print("\n".join(bad)); sys.exit(1 if bad else 0)
PY
}
tests_not_weakened() { local d a; d=$(git diff "$BASE"..HEAD -- $TESTS | grep -cE '^-\s*(async )?def test_'); a=$(git diff "$BASE"..HEAD -- $TESTS | grep -cE '^\+\s*(async )?def test_'); echo "   test functions removed $d, added $a"; [ "$a" -ge "$d" ]; }
service_shorter() { local b h; b=$(lines_at "$BASE" $CHAT/service.py); h=$(wc -l < $CHAT/service.py | tr -d ' '); echo "   service.py: base $b, HEAD $h"; [ "$h" -lt "$b" ]; }
no_new_lane_module() { local f bad=0; for f in $(added_files $CHAT); do case "$(basename "$f")" in handoffs.py|receipts.py|__init__.py) ;; *) echo "   new module: $f"; bad=1 ;; esac; done; [ $bad -eq 0 ]; }
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
  echo ""; echo "=== Wave 2 stories"
  check "US-008 the price table prices the Claude 5 ids" "grep -q 'claude-sonnet-5' $LLM/manager.py && grep -q 'claude-haiku-4-5' $LLM/manager.py"
  check "US-008 LLM_FAILOVER_MODEL in config.py and config-surface.json, empty by default" "grep -q 'LLM_FAILOVER_MODEL' orchestrator/config.py && grep -q 'LLM_FAILOVER_MODEL' orchestrator/reports/config-surface.json"
  check "US-008 the per-model sampling rule changed in base.py (one place)" "! git diff --quiet $BASE..HEAD -- $LLM/clients/base.py"
  check "US-008 tests" "test -f $TESTS/test_prd256_llm_manager_claude5_params.py"
  check "US-009 the pin test: the seeded Auto model equals core/llm/defaults.py (the constant already exists on main)" "seed_reads_defaults && test -f $TESTS/test_prd256_default_model_pin.py"
  check "US-010 UniversalRouter is gone from the chat dispatch" "! grep -q 'UniversalRouter' orchestrator/api/chat.py"
  check "US-010 the rubric's delegate-by-default line is gone" "! grep -q 'Most molecule/cell/organ work' $CHAT/auto.py"
  check "US-010 tests" "test -f $TESTS/test_prd256_auto_answers_unless_named.py"
  check "US-011 one hand-off table, four lanes deleted" "test -f $CHAT/handoffs.py && ! test -f $CHAT/brand_assign_lane.py && ! test -f $CHAT/paperwork_to_the_team.py"
  check "US-011 the routing golden file and its test" "test -f $TESTS/fixtures/prd256_routing_golden.json && test -f $TESTS/test_prd256_handoffs.py"
  case "$(story_state US-012)" in
    BUILT)
      check "US-012 the families are deleted" "! test -f $EXEC/action_claims.py && ! test -f $EXEC/document_claims.py && ! test -f $EXEC/shop_and_team_claims.py && ! test -f $CHAT/figure_disputes.py && ! test -f $CHAT/shop_figures.py && ! test -f $CHAT/team_corrections.py"
      check "US-012 no test weakened: added test functions ≥ removed" "tests_not_weakened"
      check "US-012 service.py is shorter than the base and under 3200 lines" "service_shorter && [ \$(wc -l < $CHAT/service.py | tr -d ' ') -le 3200 ]"
      check "US-012 tests" "test -f $TESTS/test_prd256_families_deleted.py" ;;
    *) echo ""; echo "── US-012: $(story_state US-012) per the owner's note (decision rule) — its checks are skipped" ;;
  esac
  case "$(story_state US-013)" in
    BUILT) check "US-013 context assembly test" "test -f $TESTS/test_prd256_context_assembly.py" ;;
    *) echo ""; echo "── US-013: $(story_state US-013) per Decision D3 — its checks are skipped" ;;
  esac
  check "every acceptance criterion in $KIT is DONE, SKIPPED or BLOCKED per its rule (the owner's are left)" "all_acs_done"
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
check "N12 no new lane module under consumers/chatbot other than handoffs.py" "no_new_lane_module"
check "N13 no Jev live: the decision engine is untouched (D4)" "git diff --quiet $BASE..HEAD -- orchestrator/core/llm/decisions"
check "N14 no failover model set by default (D5)" "! failover_default_set"
check "N15 the hierarchy gate passes (stdlib ast)" "python3 orchestrator/scripts/check_hierarchy_gate.py >/dev/null"
check "N16 no eval run claimed: the loop never touched the analyst folder" "! git diff $BASE..HEAD | grep -q 'automatos-analyst'"
warn "N17 story commits carry [skip ci] (all but the final CI commit)" "[ -z \"\$(git log --format='%s' $BASE..HEAD^ | grep -v 'skip ci' | grep -v '^chore(prd-256): seed')\" ]"

if [ "$MODE" = "full" ]; then
  echo ""; echo "=== CI (once per wave)"
  check "CI: the required jobs present and green on HEAD" "ci_green_on_head"
fi
echo ""
if [ "$FAIL" -eq 0 ]; then echo "✅ PRD-256 Wave 2 acceptance ($MODE): PASS"; else echo "❌ PRD-256 Wave 2 acceptance ($MODE): FAIL"; fi
exit "$FAIL"
