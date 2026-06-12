#!/bin/bash
# Acceptance gate — PRD-164 Planning Intelligence & Integration Seams (WS-9).
# Run from the worktree repo root. Exit 0 = local DB-free gates pass.
#
# THIS MACHINE HAS NO LOCAL DATABASE / NO QDRANT / NO CONTAINERS. The orchestrator
# pytest suite blocks on test_82c_wiring's real Postgres connect, so it is NOT run
# here — it is validated by CI (test.yml, real Postgres) on push, and the overnight
# runner records that CI result in the night report. This gate is DB-FREE only:
# py_compile on touched backend + frontend tsc/vitest/lint + grep/deletion gates.
# The golden planning suite, matcher matrix, flywheel e2e and heartbeat-memory tests
# need a DB/Qdrant and are CI-owned. Browser ACs (deliverables tab, chat widgets) are
# confirmed by their vitest/reachability proxies and flagged for a morning human check.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
FAIL=0
check() {
  local name="$1"; shift
  echo ""
  echo "── $name"
  if bash -c "$1"; then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi
}

# --- Backend: compile-only (no DB). Full suite runs on CI. ---
check "backend compiles (py_compile — DB-free; full suite is on CI)" \
  'cd orchestrator && python3 -m compileall -q modules/coordination/planner.py modules/coordination/agent_matcher.py api/board_tasks.py consumers/chatbot/auto.py modules/rag/ingestion/pipeline.py modules/context/adapters/vector_field.py'

# --- S2: matcher EXTENDED, not forked — exactly one AgentMatcher in coordination ---
check "S2 single AgentMatcher (no parallel matcher module)" \
  '[ "$(grep -rl "class AgentMatcher" orchestrator/modules/ | wc -l | tr -d " ")" = "1" ]'

# --- S5: dead TOOL_WIDGET_MAP deleted (delete-what-you-replace, Q62) ---
check "S5 TOOL_WIDGET_MAP removed from widgets/router.ts (Q62)" \
  '! grep -q "TOOL_WIDGET_MAP" frontend/components/widgets/router.ts'

# NOTE: S1 pack convergence (one assembler, three consumers), the golden plan-delta,
# S3 flywheel + opt-out, S4 ≥60% dispatch shrink, and S5 heartbeat-memory recall are
# enforced by their pytest suites on CI (DB/Qdrant) — a grep here would false-arbitrate.

check "frontend-typecheck (no DB)" \
  'cd frontend && npx tsc --noEmit'

check "frontend-units (vitest — widget routing + reachability proxies for the deferred browser ACs)" \
  'cd frontend && npm run test'

check "frontend-lint (no DB)" \
  'cd frontend && npm run lint'

echo ""
if [ $FAIL -eq 0 ]; then
  echo "ACCEPTANCE: PRD-164 local gates PASS"
  echo "NOTE: backend golden/matrix/flywheel/heartbeat suites are validated by CI (test.yml, real Postgres + Qdrant) — confirm the branch CI run is green before merge."
  echo "NOTE: browser ACs (mission deliverables tab, chat tool-widgets) are DEFERRED — eyeball in the morning per prd-164.json DEFERRED notes."
  echo "NOTE: if S3 added a source_type alembic revision, the Railway redeploy must apply it before the flywheel is exercised in prod."
else
  echo "ACCEPTANCE: PRD-164 local gates FAIL"
fi
exit $FAIL
