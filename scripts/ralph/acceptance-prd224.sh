#!/bin/bash
# Acceptance gate — PRD-224 The Ticket Lane (US-001..005). Chain 2/6 (base: 227 branch).
# Run from the worktree repo root. Exit 0 = PRD-level done. Runs ALL checks.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
FAIL=0
HANDLERS="orchestrator/modules/tools/discovery/handlers_board_tasks.py"
AUTO="orchestrator/consumers/chatbot/auto.py"
CHAT="orchestrator/api/chat.py"
WENUMS="orchestrator/core/models/watch_enums.py"
WACTIONS="orchestrator/modules/tools/discovery/actions_watches.py"
CFG="orchestrator/config.py"
BASE_BR="ralph/prd-227-board-light-up"
check() {
  local name="$1"; shift
  echo ""
  echo "── $name"
  if bash -c "$1"; then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi
}

# Backend gate is BRANCH-SCOPED (2026-08-27 amendment): ~49 pre-existing
# environmental fails/errors in the full local suite — CI test.yml is the
# full-suite gate. Locally we prove THIS PRD's own tests green.
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

# --- ZERO new migrations / routes vs the CHAIN base ---------------------------
echo ""
echo "── ZERO new alembic revisions vs base ($BASE_BR)"
BASE=$(git merge-base HEAD "$BASE_BR" 2>/dev/null || git merge-base HEAD origin/main 2>/dev/null || echo "")
if [ -n "$BASE" ]; then
  NEWMIG=$(git diff --name-only --diff-filter=A "$BASE"..HEAD -- orchestrator/alembic/versions/ 2>/dev/null | wc -l | tr -d ' ')
  if [ "$NEWMIG" = "0" ]; then echo "   ✅ PASS: no new migrations"; else echo "   ❌ FAIL: $NEWMIG new migrations (must be 0 — target_type is a plain string)"; FAIL=1; fi
  if git diff "$BASE"..HEAD -- orchestrator/reports/route-manifest.json | grep -q .; then
    echo "   ❌ FAIL: route manifest changed (no new routes in PRD-224)"; FAIL=1
  else
    echo "   ✅ PASS: route manifest untouched"
  fi
else
  echo "   ⚠️  SKIP: no merge base found"
fi

# --- US-001 -------------------------------------------------------------------
check "US-001 chat handlers call notify_task_available" \
  "grep -q 'notify_task_available' $HANDLERS"

# --- US-002 -------------------------------------------------------------------
check "US-002 WatchTargetType gains board_task" \
  "grep -qi 'board_task' $WENUMS"

check "US-002 create_watch schema offers board_task" \
  "grep -q 'board_task' $WACTIONS"

check "US-002 decider handles board tasks" \
  "grep -qi 'board_task' orchestrator/services/watch_decider.py"

# --- US-003 -------------------------------------------------------------------
check "US-003 watch_actions handles board_task correctives" \
  "grep -qi 'board_task' orchestrator/services/watch_actions.py"

# --- US-004 -------------------------------------------------------------------
check "US-004 Action.ASSIGN exists (WORKFLOW untouched)" \
  "grep -q 'ASSIGN' $AUTO && grep -q 'WORKFLOW' $AUTO"

check "US-004 chat.py dispatches the ASSIGN lane" \
  "grep -q 'ASSIGN' $CHAT"

# --- US-005 -------------------------------------------------------------------
check "US-005 AUTO_TICKET_WATCH in config.py" \
  "grep -q 'AUTO_TICKET_WATCH' $CFG"

check "US-005 handler-level auto-attach (not prompt-level)" \
  "grep -qi 'AUTO_TICKET_WATCH' $HANDLERS"

check "no os.getenv outside config.py (diff scope)" \
  "! git diff \$(git merge-base HEAD $BASE_BR 2>/dev/null || git merge-base HEAD origin/main)..HEAD -- 'orchestrator/**/*.py' ':!orchestrator/config.py' | grep -E '^\\+' | grep -q 'os.getenv'"

# --- Dead vocabulary stays dead ----------------------------------------------
# Scoped to orchestrator source (like the os.getenv check above): a "writer"
# lives in code. The unscoped diff also matches this PRD's own scaffolding — the
# spec, the Ralph prompts, prd-224.json, and this check's own string all NAME the
# token to forbid it — which is documentation of the rule, not a violation.
check "no AWAITING_HUMAN writers introduced" \
  "! git diff \$(git merge-base HEAD $BASE_BR 2>/dev/null || git merge-base HEAD origin/main)..HEAD -- 'orchestrator/**/*.py' | grep -E '^\\+' | grep -q 'AWAITING_HUMAN'"

echo ""
if [ "$FAIL" = "0" ]; then echo "ACCEPTANCE: PASS (PRD-224)"; exit 0; else echo "ACCEPTANCE: FAIL (PRD-224)"; exit 1; fi
