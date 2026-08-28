#!/bin/bash
# Acceptance gate — PRD-222 Wave 2b (US-023..US-025): tier config v1, exposure profiles,
# plan recommendation. Exit 0 = wave done. Numbers contract = the approved strawman.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
FAIL=0
CFG="orchestrator/config.py"
WSAPI="orchestrator/api/workspaces.py"
SECTION="orchestrator/modules/context/sections/onboarding.py"
OSTATE="orchestrator/services/onboarding_state.py"
check() {
  local name="$1"; shift
  echo ""
  echo "── $name"
  if bash -c "$1"; then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi
}

# --- Primary gates ------------------------------------------------------------------
check "orchestrator-full-suite (no NEW failures vs documented env baseline)" \
  'OUT=$(cd orchestrator && python3 -m pytest --timeout=90 --timeout-method=thread -o faulthandler_timeout=120 -p no:cacheprovider -q 2>&1 | tail -200); echo "$OUT" | grep -qE "[0-9]{4} passed" && ! echo "$OUT" | grep "^FAILED" | grep -vE "prd172|dr_restore|composio" | grep -q .'

check "frontend-vitest-suite" \
  'cd frontend && npm run -s test'

check "frontend-build" \
  'cd frontend && npm run -s build'

# --- ONE migration (US-023 rename + backfill) ----------------------------------------
echo ""
echo "── exactly ONE new alembic revision vs base (plan default rename + backfill)"
BASE=$(git merge-base HEAD origin/main 2>/dev/null || git merge-base HEAD main 2>/dev/null || echo "")
if [ -n "$BASE" ]; then
  NEWMIG=$(git diff --name-only --diff-filter=A "$BASE"..HEAD -- orchestrator/alembic/versions/ 2>/dev/null | wc -l | tr -d ' ')
  if [ "$NEWMIG" = "1" ]; then echo "   ✅ PASS: one new migration"; else echo "   ❌ FAIL: $NEWMIG new migrations (must be exactly 1)"; FAIL=1; fi
else
  echo "   ⚠️  SKIP: no merge base found"
fi

# --- US-023: tier config v1 -----------------------------------------------------------
check "US-023 PLAN_TIERS in config.py with the approved names + display prices" \
  "grep -q 'PLAN_TIERS' $CFG && grep -q 'basic' $CFG && grep -q 'pro' $CFG && grep -q 'business' $CFG && grep -q '19' $CFG && grep -q '49' $CFG && grep -q '99' $CFG"

check "US-023 enterprise is coming-soon display only (not assignable)" \
  "grep -qi 'coming_soon\|coming soon' $CFG"

check "US-023 plan default renamed to basic (model)" \
  "grep -q '\"basic\"' orchestrator/core/models/workspaces.py"

# --- US-024: exposure -----------------------------------------------------------------
check "US-024 exposure block on the current-workspace surface" \
  "grep -q 'exposure' $WSAPI"

check "US-024 marketplace plan-label chips present" \
  "grep -qiE 'plan|tier' frontend/components/marketplace/marketplace-grid.tsx"

check "US-024 tool-surface family filter reads config (not hardcoded families in the filter)" \
  "grep -rq 'PLAN_TIERS\|tool_famil' orchestrator/modules/tools/discovery/ --include='*.py'"

# --- US-025: recommendation -----------------------------------------------------------
check "US-025 proposal block recommends a plan (section carries tier copy)" \
  "grep -qiE 'basic|pro|business' $SECTION"

check "US-025 platform_update_onboarding accepts a validated plan field" \
  "grep -q 'plan' orchestrator/modules/tools/discovery/actions_onboarding.py && grep -q 'plan' orchestrator/modules/tools/discovery/handlers_onboarding.py"

check "US-025 funnel events exist" \
  "grep -rq 'plan_recommended' orchestrator/ --include='*.py' && grep -rq 'plan_accepted' orchestrator/ --include='*.py'"

# --- Scope guards ---------------------------------------------------------------------
echo ""
echo "── scope guard: no commerce code (Q5 — display pricing only)"
if [ -n "$BASE" ] && git diff "$BASE"..HEAD -- orchestrator/ frontend/ 2>/dev/null | grep -iE '^\+.*(stripe|checkout)' | grep -vq 'coming'; then
  echo "   ❌ FAIL: commerce code in the diff — Q5 keeps billing out of Wave 2"; FAIL=1
else
  echo "   ✅ PASS: no commerce code"
fi

# --- Load-bearing surfaces intact -----------------------------------------------------
check "wave-1/2a surfaces intact" \
  "[ -f frontend/components/onboarding/onboarding-opener.tsx ] && [ -f frontend/components/onboarding/power-up-card.tsx ] && [ -f frontend/components/onboarding/trial-balance-pill.tsx ] && [ -f frontend/components/onboarding/intake-progress-card.tsx ] && [ -f frontend/components/onboarding/connect-app-card.tsx ] && [ -f frontend/components/onboarding/setup-checklist-card.tsx ] && [ -f frontend/app/dev/reset-onboarding/page.tsx ]"

check "reset + section + trust guards + validator intact" \
  "grep -q 'def reset_onboarding' $OSTATE && grep -q 'powerup' $SECTION && [ -f orchestrator/tests/test_prd222_trust_guards.py ] && grep -q 'InvalidStageTransition' $OSTATE"

echo ""
if [ $FAIL -eq 0 ]; then echo "ACCEPTANCE: PRD-222 Wave 2b PASS"; else echo "ACCEPTANCE: PRD-222 Wave 2b FAIL"; fi
echo "NOTE: tiers/prices are config — Gerard tunes them live while testing. Commerce (Q5) and Wave 3 remain separate decisions."
exit $FAIL
