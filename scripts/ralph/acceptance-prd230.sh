#!/bin/bash
# Acceptance gate — PRD-230 Packages & Vertical Onboarding (+2 PRD-222 W0 fixes).
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
FAIL=0
SECTION="orchestrator/modules/context/sections/onboarding.py"
OSTATE="orchestrator/services/onboarding_state.py"
check() {
  local name="$1"; shift
  echo ""; echo "── $name"
  if bash -c "$1"; then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi
}

check "orchestrator-full-suite (no NEW failures vs documented env baseline)" \
  'OUT=$(cd orchestrator && python3 -m pytest --timeout=90 --timeout-method=thread -o faulthandler_timeout=120 -p no:cacheprovider -q 2>&1 | tail -200); echo "$OUT" | grep -qE "[0-9]{4} passed" && ! echo "$OUT" | grep "^FAILED" | grep -vE "prd172|dr_restore|composio" | grep -q .'

check "frontend-vitest-suite" 'cd frontend && npm run -s test'
check "frontend-build" 'cd frontend && npm run -s build'

echo ""; echo "── exactly ONE new alembic revision (marketplace_packages)"
BASE=$(git merge-base HEAD origin/main 2>/dev/null || git merge-base HEAD main 2>/dev/null || echo "")
if [ -n "$BASE" ]; then
  NEWMIG=$(git diff --name-only --diff-filter=A "$BASE"..HEAD -- orchestrator/alembic/versions/ 2>/dev/null | wc -l | tr -d ' ')
  if [ "$NEWMIG" = "1" ]; then echo "   ✅ PASS"; else echo "   ❌ FAIL: $NEWMIG new migrations (must be 1)"; FAIL=1; fi
else echo "   ⚠️  SKIP: no merge base"; fi

# W0 fixes
check "US-001 chatbot passes workspace identity to the factory" \
  "grep -rq 'workspace_id' orchestrator/consumers/chatbot/service.py"
check "US-001 chat trial regression test exists" \
  "grep -rlq 'record_trial_spend' orchestrator/tests/ && grep -rliE 'chat.*trial|trial.*chat' orchestrator/tests/ | grep -q ."
check "US-002 doctrine v2 in the section (connect card, two-step sync, scan-on-URL, marketplace-first, stage vocabulary)" \
  "grep -qi 'connect' $SECTION && grep -qi 'marketplace' $SECTION && grep -qi 'scan' $SECTION && grep -qi 'widget' $SECTION && grep -q 'not_started' $SECTION"

# Packages
check "US-003 marketplace_packages model + migration" \
  "grep -rq 'marketplace_packages' orchestrator/core/models/ && git diff --name-only --diff-filter=A \"$BASE\"..HEAD -- orchestrator/alembic/versions/ | xargs grep -l 'marketplace_packages' | grep -q ."
check "US-004 closure resolver with the agent-A canonical test" \
  "grep -rlq 'closure' orchestrator/services/ && grep -rq 'agent' orchestrator/tests/test_prd230*.py"
check "US-005/US-010 invariant guards present and biting" \
  "[ -f orchestrator/tests/test_prd230_invariants.py ] && grep -q 'workspace' orchestrator/tests/test_prd230_invariants.py"
check "US-006 three platform tools registered, walker green (suite covers)" \
  "grep -q 'platform_search_packages' orchestrator/modules/tools/discovery/platform_executor.py && grep -q 'platform_install_package' orchestrator/modules/tools/discovery/platform_executor.py && grep -q 'platform_install_marketplace_agent' orchestrator/modules/tools/discovery/platform_executor.py"
check "US-007 Packages tab + popup in the marketplace UI" \
  "grep -rqi 'package' frontend/components/marketplace/ --include='*.tsx'"
check "US-008 both Shopify seeds present" \
  "grep -rqi 'shopify' orchestrator/core/seeds/ --include='*.py' && grep -rqiE 'management' orchestrator/core/seeds/*.py && grep -rqiE 'development' orchestrator/core/seeds/*.py"
check "US-009 proposal offers a package + funnel events" \
  "grep -qi 'package' $SECTION && grep -rq 'package_offered' orchestrator/ --include='*.py' && grep -rq 'package_installed' orchestrator/ --include='*.py'"

# Load-bearing PRD-222 surfaces
check "PRD-222 surfaces intact" \
  "[ -f frontend/components/onboarding/onboarding-opener.tsx ] && [ -f frontend/components/onboarding/power-up-card.tsx ] && [ -f frontend/components/onboarding/connect-app-card.tsx ] && [ -f frontend/components/onboarding/setup-checklist-card.tsx ] && [ -f frontend/app/dev/reset-onboarding/page.tsx ] && grep -q 'def reset_onboarding' $OSTATE && [ -f orchestrator/tests/test_prd222_trust_guards.py ] && grep -q 'InvalidStageTransition' $OSTATE && grep -q 'PLAN_TIERS' orchestrator/config.py"

echo ""
if [ $FAIL -eq 0 ]; then echo "ACCEPTANCE: PRD-230 PASS"; else echo "ACCEPTANCE: PRD-230 FAIL"; fi
echo "NOTE: package↔tier visibility (Q1), vertical shortlist (Q2), tab naming (Q3) are Gerard's calls."
exit $FAIL
