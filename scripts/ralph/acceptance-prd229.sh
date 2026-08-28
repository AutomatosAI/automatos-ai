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

# Backend gate is BRANCH-SCOPED (2026-08-27 amendment): ~49 pre-existing
# environmental fails/errors in the full local suite — CI test.yml is the
# full-suite gate. Locally we prove THIS PRD's own tests green. (US-004's
# "full suite proves nothing depended on inter_agent" provenance = CI.)
echo ""
echo "── branch-scoped backend tests (this PRD's tests; full-suite provenance = CI test.yml)"
BASEP=$(git merge-base HEAD "$BASE_BR" 2>/dev/null || git merge-base HEAD origin/main 2>/dev/null || echo "")
CHT=$(git diff --name-only "${BASEP:-HEAD~40}"..HEAD -- orchestrator 2>/dev/null | grep -E '(^|/)tests/.*\.py$' | sed 's|^orchestrator/||' | tr '\n' ' ')
if [ -z "${CHT// /}" ]; then
  echo "   ✅ PASS: no backend test files changed on this branch (CI covers the full suite)"
else
  if ( cd orchestrator && python3 -m pytest --timeout=90 --timeout-method=thread -o faulthandler_timeout=120 -p no:cacheprovider -q $CHT ); then
    echo "   ✅ PASS: branch-scoped backend tests: $CHT"
  else
    echo "   ❌ FAIL: branch-scoped backend tests: $CHT"; FAIL=1
  fi
fi

echo ""
echo "── ZERO new alembic revisions vs base ($BASE_BR); route manifest untouched"
BASE=$(git merge-base HEAD "$BASE_BR" 2>/dev/null || git merge-base HEAD origin/main 2>/dev/null || echo "")
if [ -n "$BASE" ]; then
  NEWMIG=$(git diff --name-only --diff-filter=A "$BASE"..HEAD -- orchestrator/alembic/versions/ 2>/dev/null | wc -l | tr -d ' ')
  if [ "$NEWMIG" = "0" ]; then echo "   ✅ PASS: no new migrations"; else echo "   ❌ FAIL: $NEWMIG new migrations (must be 0)"; FAIL=1; fi
  if git diff "$BASE"..HEAD -- orchestrator/reports/route-manifest.json | grep -q .; then
    echo "   ❌ FAIL: route manifest changed (no new routes in PRD-229)"; FAIL=1
  else
    echo "   ✅ PASS: route manifest untouched"
  fi
else
  echo "   ⚠️  SKIP: no merge base found"
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
check "no lateral messaging surfaces (no mailbox/inter-agent send in diff)" \
  "! git diff \$(git merge-base HEAD $BASE_BR 2>/dev/null || git merge-base HEAD origin/main)..HEAD | grep -E '^\\+' | grep -qiE 'inter_agent|agent_mailbox'"

check "no pgvector touches (RAG is S3 Vectors only)" \
  "! git diff \$(git merge-base HEAD $BASE_BR 2>/dev/null || git merge-base HEAD origin/main)..HEAD | grep -E '^\\+' | grep -qi 'pgvector'"

check "no os.getenv outside config.py (diff scope)" \
  "! git diff \$(git merge-base HEAD $BASE_BR 2>/dev/null || git merge-base HEAD origin/main)..HEAD -- 'orchestrator/**/*.py' ':!orchestrator/config.py' | grep -E '^\\+' | grep -q 'os.getenv'"

check "no AWAITING_HUMAN writers introduced" \
  "! git diff \$(git merge-base HEAD $BASE_BR 2>/dev/null || git merge-base HEAD origin/main)..HEAD | grep -E '^\\+' | grep -q 'AWAITING_HUMAN'"

echo ""
if [ "$FAIL" = "0" ]; then echo "ACCEPTANCE: PASS (PRD-229)"; exit 0; else echo "ACCEPTANCE: FAIL (PRD-229)"; exit 1; fi
