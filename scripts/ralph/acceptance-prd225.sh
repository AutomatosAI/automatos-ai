#!/bin/bash
# Acceptance gate — PRD-225 Agent Questions: ASK ME + Telegram (US-001..006). Chain 4/6 (base: 226 branch).
# Run from the worktree repo root. Exit 0 = PRD-level done. Runs ALL checks.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
FAIL=0
GRANTS_MODEL="orchestrator/core/models/approval_grants.py"
GRANTS_API="orchestrator/api/approval_grants.py"
DISPATCH="orchestrator/core/services/notification_dispatcher.py"
WEBHOOKS="orchestrator/api/webhooks.py"
SHELL_TSX="frontend/components/command-center/command-center-shell.tsx"
MANIFEST="orchestrator/reports/route-manifest.json"
BASE_BR="ralph/prd-226-manager-doctrine"
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

check "frontend-vitest-suite" \
  'cd frontend && npm run -s test'

# --- EXACTLY ONE new migration + single head ---------------------------------
echo ""
echo "── EXACTLY ONE new alembic revision vs base ($BASE_BR) + single head"
BASE=$(git merge-base HEAD "$BASE_BR" 2>/dev/null || git merge-base HEAD origin/main 2>/dev/null || echo "")
if [ -n "$BASE" ]; then
  NEWMIG=$(git diff --name-only --diff-filter=A "$BASE"..HEAD -- orchestrator/alembic/versions/ 2>/dev/null | wc -l | tr -d ' ')
  if [ "$NEWMIG" = "1" ]; then echo "   ✅ PASS: exactly one new migration"; else echo "   ❌ FAIL: $NEWMIG new migrations (must be exactly 1)"; FAIL=1; fi
else
  echo "   ⚠️  SKIP: no merge base found"
fi
HEADS=$(cd orchestrator && python3 -c "
from alembic.config import Config
from alembic.script import ScriptDirectory
print(len(ScriptDirectory.from_config(Config('alembic.ini')).get_heads()))
" 2>/dev/null || echo "err")
if [ "$HEADS" = "1" ]; then echo "   ✅ PASS: alembic heads == 1"; else echo "   ❌ FAIL: alembic heads == $HEADS (must be 1)"; FAIL=1; fi

# --- US-001 model ------------------------------------------------------------
check "US-001 asks columns on the model (kind, question_md, answer_text, channel_refs)" \
  "grep -q 'question_md' $GRANTS_MODEL && grep -q 'answer_text' $GRANTS_MODEL && grep -q 'channel_refs' $GRANTS_MODEL && grep -q 'kind' $GRANTS_MODEL"

echo ""
echo "── US-001 existing approval test files not MODIFIED (new files fine)"
if [ -n "$BASE" ]; then
  TOUCHED=$(git diff --name-only --diff-filter=MD "$BASE"..HEAD -- orchestrator/tests/ 2>/dev/null | grep -ci 'approval' || true)
  if [ "$TOUCHED" = "0" ]; then echo "   ✅ PASS: approval regression tests untouched"; else echo "   ❌ FAIL: $TOUCHED existing approval test file(s) modified/deleted"; FAIL=1; fi
else
  echo "   ⚠️  SKIP: no merge base found"
fi

# --- US-002 tool -------------------------------------------------------------
check "US-002 platform_ask_human registered (3-file pattern)" \
  "grep -rq 'platform_ask_human' orchestrator/modules/tools/discovery/"

check "US-002 question_pending in VALID_EVENT_TYPES" \
  "grep -q 'question_pending' $DISPATCH"

# --- US-003 answer route + manifest ------------------------------------------
check "US-003 answer endpoint on the existing grants router" \
  "grep -q '/answer' $GRANTS_API"

check "US-003 resume reuse (grep _requeue_subject call from the answer path)" \
  "grep -q '_requeue_subject' $GRANTS_API"

check "US-003 route manifest carries the answer route" \
  "grep -q 'answer' $MANIFEST"

check "US-003 route-manifest test green" \
  'cd orchestrator && python3 -m pytest -q tests/test_route_manifest.py'

check "US-003 frontend route-contract green" \
  'node frontend/scripts/check-route-contract.js'

# --- US-004 tab --------------------------------------------------------------
check "US-004 questions tab wired into the shell (TabKey + TABS + render)" \
  "grep -q 'questions' $SHELL_TSX"

check "US-004 questions tab component exists" \
  "ls frontend/components/command-center/ | grep -qi 'question'"

check "US-004 duplicate grant hooks consolidated (one module remains)" \
  "[ \$(ls frontend/hooks/ | grep -c 'use-approval-grants') -eq 1 ]"

# --- US-005 telegram ---------------------------------------------------------
check "US-005 reply correlation in webhooks (reply_to_message_id)" \
  "grep -q 'reply_to_message_id' $WEBHOOKS"

check "US-005 /answer command fallback" \
  "grep -q '/answer' $WEBHOOKS"

# --- US-006 trust gate -------------------------------------------------------
check "US-006 trigger_mode gate in webhooks" \
  "grep -q 'trigger_mode' $WEBHOOKS"

check "US-006 strict is the default (grep default literal)" \
  "grep -rqE \"strict\" $WEBHOOKS orchestrator/core/models/channels.py"

# --- Conventions -------------------------------------------------------------
check "no os.getenv outside config.py (diff scope)" \
  "! git diff \$(git merge-base HEAD $BASE_BR 2>/dev/null || git merge-base HEAD origin/main)..HEAD -- 'orchestrator/**/*.py' ':!orchestrator/config.py' | grep -E '^\\+' | grep -q 'os.getenv'"

check "no AWAITING_HUMAN writers introduced" \
  "! git diff \$(git merge-base HEAD $BASE_BR 2>/dev/null || git merge-base HEAD origin/main)..HEAD | grep -E '^\\+' | grep -q 'AWAITING_HUMAN'"

echo ""
if [ "$FAIL" = "0" ]; then echo "ACCEPTANCE: PASS (PRD-225)"; exit 0; else echo "ACCEPTANCE: FAIL (PRD-225)"; exit 1; fi
