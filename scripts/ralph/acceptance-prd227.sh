#!/bin/bash
# Acceptance gate — PRD-227 Board Light-Up (US-001..003). Chain 1/6.
# Run from the worktree repo root. Exit 0 = PRD-level done. Runs ALL checks.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
FAIL=0
HANDLERS="orchestrator/modules/tools/discovery/handlers_board_tasks.py"
COORD="orchestrator/services/coordinator_service.py"
CFG="orchestrator/config.py"
BELL="frontend/components/notifications/notification-bell.tsx"
check() {
  local name="$1"; shift
  echo ""
  echo "── $name"
  if bash -c "$1"; then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi
}

# --- Primary gates: full suites green ---------------------------------------
check "orchestrator-full-suite (pure; @integration skips with no DB)" \
  'cd orchestrator && python3 -m pytest --timeout=90 --timeout-method=thread -o faulthandler_timeout=120 -p no:cacheprovider -q'

check "frontend-vitest-suite" \
  'cd frontend && npm run -s test'

# --- ZERO new migrations / routes on this branch ------------------------------
echo ""
echo "── ZERO new alembic revisions vs base (main)"
BASE=$(git merge-base HEAD origin/main 2>/dev/null || git merge-base HEAD main 2>/dev/null || echo "")
if [ -n "$BASE" ]; then
  NEWMIG=$(git diff --name-only --diff-filter=A "$BASE"..HEAD -- orchestrator/alembic/versions/ 2>/dev/null | wc -l | tr -d ' ')
  if [ "$NEWMIG" = "0" ]; then echo "   ✅ PASS: no new migrations"; else echo "   ❌ FAIL: $NEWMIG new migrations (must be 0)"; FAIL=1; fi
  if git diff "$BASE"..HEAD -- orchestrator/reports/route-manifest.json | grep -q .; then
    echo "   ❌ FAIL: route manifest changed (no new routes in PRD-227)"; FAIL=1
  else
    echo "   ✅ PASS: route manifest untouched"
  fi
else
  echo "   ⚠️  SKIP: no merge base found"
fi

# --- US-001: agent-side SSE + status parity ----------------------------------
check "US-001 agent handlers call notify_board_event" \
  "grep -q 'notify_board_event' $HANDLERS"

check "US-001 agent handler accepts blocked + failed" \
  "grep -q 'blocked' $HANDLERS && grep -q 'failed' $HANDLERS"

check "US-001 fail-soft test exists (NOTIFY failure does not fail the tool call)" \
  "grep -rlE 'notify_board_event' orchestrator/tests | xargs grep -lE 'monkeypatch|patch' | grep -q ."

# --- US-002: mission narration ------------------------------------------------
check "US-002 coordinator narrates via deliver_background_message" \
  "grep -q 'deliver_background_message' $COORD"

check "US-002 MISSION_NARRATION_TASK_CAP in config.py" \
  "grep -q 'MISSION_NARRATION_TASK_CAP' $CFG"

check "US-002 no os.getenv outside config.py (diff scope)" \
  "! git diff \$(git merge-base HEAD origin/main)..HEAD -- 'orchestrator/**/*.py' ':!orchestrator/config.py' | grep -E '^\\+' | grep -q 'os.getenv'"

# --- US-003: bell cases + drift guard ----------------------------------------
check "US-003 linkFor handles approval_grant" \
  "grep -q \"approval_grant\" $BELL"

check "US-003 linkFor handles watch" \
  "grep -qE \"case 'watch'|'watch':\" $BELL"

check "US-003 drift-guard test exists" \
  "grep -rlq 'approval_grant' frontend/components/notifications/__tests__ 2>/dev/null || grep -rl 'linkFor' frontend --include='*.test.*' | grep -q ."

echo ""
if [ "$FAIL" = "0" ]; then echo "ACCEPTANCE: PASS (PRD-227)"; exit 0; else echo "ACCEPTANCE: FAIL (PRD-227)"; exit 1; fi
