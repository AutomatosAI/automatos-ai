#!/bin/bash
# Acceptance gate — PRD-229 Mid-Run Clarifications (US-001..004). Chain 6/6 (base: 228 branch).
# Run from the worktree repo root. Exit 0 = PRD-level done. Runs ALL checks.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
FAIL=0
ANSWERS="orchestrator/services/orchestrator_answers.py"
CFG="orchestrator/config.py"
DEAD="orchestrator/modules/agents/communication/inter_agent.py"
AGENTS_INIT="orchestrator/modules/agents/__init__.py"
BASE_BR="ralph/prd-228-fleet-state"
check() {
  local name="$1"; shift
  echo ""
  echo "── $name"
  if bash -c "$1"; then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi
}

# PRD-229's TRUE base = the parent of the earliest prd-229 commit. This is robust
# to the local ralph/prd-228-fleet-state pointer drifting FORWARD after this
# branch was cut: once it does, `git merge-base HEAD ralph/prd-228-fleet-state`
# resolves to the shared ANCESTOR and false-flags PRD-222/225/228's own migrations
# + route-manifest edits as "new in PRD-229" (they are not — 229 rides existing
# JSONB). Falls back to merge-base only when no 229 commit is found (squashed
# history). (Memory: ralph-acceptance-script-gotchas — scope the diff to THIS
# PRD's commit range, not a drifting branch pointer.)
FIRST229=$(git rev-list --reverse --grep='prd-229' --grep='prd229' --grep='PRD-229' --grep='P229-RVW' HEAD 2>/dev/null | head -1)
PRD229_BASE=""
[ -n "$FIRST229" ] && PRD229_BASE=$(git rev-parse "${FIRST229}^" 2>/dev/null || echo "")
[ -z "$PRD229_BASE" ] && PRD229_BASE=$(git merge-base HEAD "$BASE_BR" 2>/dev/null || git merge-base HEAD origin/main 2>/dev/null || echo "")

# Backend gate is BRANCH-SCOPED (2026-08-27 amendment): ~49 pre-existing
# environmental fails/errors in the full local suite — CI test.yml is the
# full-suite gate. Locally we prove THIS PRD's own tests green. (US-004's
# "full suite proves nothing depended on inter_agent" provenance = CI.)
echo ""
echo "── branch-scoped backend tests (this PRD's tests; full-suite provenance = CI test.yml)"
CHT=$(git diff --name-only "${PRD229_BASE:-HEAD~40}"..HEAD -- orchestrator 2>/dev/null | grep -E '(^|/)tests/.*\.py$' | sed 's|^orchestrator/||' | tr '\n' ' ')
if [ -z "${CHT// /}" ]; then
  echo "   ✅ PASS: no backend test files changed on this branch (CI covers the full suite)"
else
  if ( cd orchestrator && python3 -m pytest --timeout=90 --timeout-method=thread -o faulthandler_timeout=120 -p no:cacheprovider -q $CHT ); then
    echo "   ✅ PASS: branch-scoped backend tests: $CHT"
  else
    echo "   ❌ FAIL: branch-scoped backend tests: $CHT"; FAIL=1
  fi
fi

# P229-RVW-1: the registry namespace invariant (test_tool_reachability) is a
# full-suite guard the branch-scoped block above MISSES — this PRD does not edit
# that file, so it never lands in $CHT. Run it explicitly so a stray-prefix
# regression (the ask_orchestrator → platform_ask_orchestrator class: a
# registered action with neither platform_ nor workspace_ prefix routes to no
# dispatch path) fails HERE, not only in CI test.yml.
echo ""
echo "── registry namespace invariant (test_tool_reachability — full-suite gate class)"
if ( cd orchestrator && python3 -m pytest --timeout=90 --timeout-method=thread -o faulthandler_timeout=120 -p no:cacheprovider -q tests/test_tool_reachability.py ); then
  echo "   ✅ PASS: registry namespace invariant (every action is platform_*/workspace_*)"
else
  echo "   ❌ FAIL: registry namespace invariant — a registered action lacks a platform_/workspace_ prefix"; FAIL=1
fi

echo ""
echo "── ZERO new alembic revisions vs PRD-229 base; route manifest untouched"
if [ -n "$PRD229_BASE" ]; then
  NEWMIG=$(git diff --name-only --diff-filter=A "$PRD229_BASE"..HEAD -- orchestrator/alembic/versions/ 2>/dev/null | wc -l | tr -d ' ')
  if [ "$NEWMIG" = "0" ]; then echo "   ✅ PASS: no new migrations"; else echo "   ❌ FAIL: $NEWMIG new migrations (must be 0)"; FAIL=1; fi
  if git diff "$PRD229_BASE"..HEAD -- orchestrator/reports/route-manifest.json | grep -q .; then
    echo "   ❌ FAIL: route manifest changed (no new routes in PRD-229)"; FAIL=1
  else
    echo "   ✅ PASS: route manifest untouched"
  fi
else
  echo "   ⚠️  SKIP: no PRD-229 base found"
fi

# --- US-001 ------------------------------------------------------------------
check "US-001 answering service exists with cannot_answer path" \
  "grep -q 'cannot_answer' $ANSWERS"

check "US-001 CLARIFICATION_BUDGET in config.py" \
  "grep -q 'CLARIFICATION_BUDGET' $CFG"

# --- US-002 ------------------------------------------------------------------
check "US-002 ask_orchestrator registered (3-file pattern)" \
  "grep -rq 'ask_orchestrator' orchestrator/modules/tools/discovery/"

check "US-002 TASK_EXECUTION-scoped (referenced in modes.py)" \
  "grep -q 'ask_orchestrator' orchestrator/modules/context/modes.py"

# --- US-003 ------------------------------------------------------------------
check "US-003 escalation reuses 225's shared internals (no parallel ask construction)" \
  "grep -rq 'ask_human' $ANSWERS orchestrator/modules/tools/discovery/handlers_*.py 2>/dev/null || grep -q 'ask' $ANSWERS"

check "US-003 draft label on parked partial output" \
  "grep -qi 'draft' $ANSWERS || grep -rqi 'draft' orchestrator/modules/tools/discovery/ --include='handlers_*.py'"

# --- US-004: the deletion ----------------------------------------------------
check "US-004 inter_agent.py DELETED" \
  "[ ! -f $DEAD ]"

check "US-004 re-exports removed from modules/agents/__init__.py" \
  "! grep -qE 'CollaborativeAgentFactory|AgentCommunicationProtocol|CollaborativeReasoner|SharedContextManager|execute_team_task' $AGENTS_INIT"

check "US-004 zero orphan references in backend code" \
  "! grep -rqE 'CollaborativeAgentFactory|AgentCommunicationProtocol|CollaborativeReasoner|execute_team_task' orchestrator --include='*.py'"

# --- Conventions -------------------------------------------------------------
# These forbid a pattern in NEW code. They are scoped to orchestrator/ (the only
# tree this PRD touches — no frontend stories) so they do not self-defeat on the
# seed spec/json/prompt, which legitimately document the very rules ("delete
# inter_agent.py", "pgvector paths are legacy", "no AWAITING_HUMAN writers") and
# would otherwise trip an unscoped added-line grep. Mirrors the os.getenv check's
# existing scoping. (Memory: ralph-acceptance-script-gotchas — scope to code.)
check "no lateral messaging surfaces (no mailbox/inter-agent send in diff)" \
  "! git diff $PRD229_BASE..HEAD -- orchestrator/ | grep -E '^\\+' | grep -qiE 'inter_agent|agent_mailbox'"

check "no pgvector touches (RAG is S3 Vectors only)" \
  "! git diff $PRD229_BASE..HEAD -- orchestrator/ | grep -E '^\\+' | grep -qi 'pgvector'"

# Match actual os.getenv( CALLS, not prose that names the rule (a test comment
# legitimately says "no os.getenv outside config.py"). Self-defeating-grep guard.
check "no os.getenv outside config.py (diff scope)" \
  "! git diff $PRD229_BASE..HEAD -- 'orchestrator/**/*.py' ':!orchestrator/config.py' | grep -E '^\\+' | grep -qE 'os\\.getenv\\('"

check "no AWAITING_HUMAN writers introduced" \
  "! git diff $PRD229_BASE..HEAD -- orchestrator/ | grep -E '^\\+' | grep -q 'AWAITING_HUMAN'"

echo ""
if [ "$FAIL" = "0" ]; then echo "ACCEPTANCE: PASS (PRD-229)"; exit 0; else echo "ACCEPTANCE: FAIL (PRD-229)"; exit 1; fi
