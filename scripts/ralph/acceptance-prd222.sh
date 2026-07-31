#!/bin/bash
# Acceptance gate — PRD-222 Auto-Led Onboarding (Mission Zero v2), Wave 1 (US-001..US-015).
# Run from the worktree repo root. Exit 0 = PRD-level done. Runs ALL checks.
# Full backend + frontend suites are the regression gate; the asserts prove the state
# machine + trial ledger + tools + section v2 + frontend surfaces exist by their pinned
# names, exactly ONE migration shipped, the trust guards hold, and W2 planes are untouched.
# Setting TRIAL_* values in Railway + the Q1 tier decision are Gerard's, not gated here.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
FAIL=0
OSTATE="orchestrator/services/onboarding_state.py"
TRIAL="orchestrator/services/trial_ledger.py"
SECTION="orchestrator/modules/context/sections/onboarding.py"
CFG="orchestrator/config.py"
PROVIDERS="frontend/components/providers.tsx"
ACT_REPORTS="orchestrator/modules/tools/discovery/actions_reports.py"
ACT_MISSIONS="orchestrator/modules/tools/discovery/actions_missions.py"
EXECUTOR="orchestrator/modules/tools/discovery/platform_executor.py"
check() {
  local name="$1"; shift
  echo ""
  echo "── $name"
  if bash -c "$1"; then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi
}

# --- Primary gates: full suites green -------------------------------------------
check "orchestrator-full-suite (pure PRD-222 tests + no regression; @integration skips with no DB)" \
  'cd orchestrator && python3 -m pytest --timeout=90 --timeout-method=thread -o faulthandler_timeout=120 -p no:cacheprovider -q'

check "frontend-vitest-suite" \
  'cd frontend && npm run -s test'

# --- US-001: state machine + the ONE migration ------------------------------------
check "US-001 onboarding_state service with pinned entrypoints" \
  "grep -qE 'def get_onboarding' $OSTATE && grep -qE 'def advance_onboarding_stage' $OSTATE"

check "US-001 stage vocabulary present (powerup + skipped + completed)" \
  "grep -q 'powerup' $OSTATE && grep -q 'skipped' $OSTATE && grep -q 'completed' $OSTATE"

echo ""
echo "── US-001 exactly one new alembic revision vs base"
BASE=$(git merge-base HEAD origin/main 2>/dev/null || git merge-base HEAD main 2>/dev/null || echo "")
if [ -n "$BASE" ]; then
  NEWMIG=$(git diff --name-only --diff-filter=A "$BASE"..HEAD -- orchestrator/alembic/versions/ 2>/dev/null | wc -l | tr -d ' ')
  if [ "$NEWMIG" = "1" ]; then echo "   ✅ PASS: one new migration"; else echo "   ❌ FAIL: $NEWMIG new migrations (must be exactly 1)"; FAIL=1; fi
else
  echo "   ⚠️  SKIP: no merge base found"
fi

# --- US-002: workspace surface -----------------------------------------------------
check "US-002 current-workspace response carries onboarding (backend)" \
  "grep -qE 'onboarding' orchestrator/api/workspaces.py"

check "US-002 workspace-provider exposes onboarding (frontend)" \
  "grep -qE 'onboarding' frontend/components/workspace-provider.tsx"

# --- US-003/US-008: the three new platform tools registered ------------------------
check "US-003 platform_update_onboarding registered" \
  "grep -q 'platform_update_onboarding' $EXECUTOR"

check "US-008 intake tools registered" \
  "grep -q 'platform_scan_business_site' $EXECUTOR && grep -q 'platform_get_intake_status' $EXECUTOR"

# --- US-004: trial config keys through config.py only -------------------------------
check "US-004 TRIAL_* keys in config.py" \
  "grep -q 'TRIAL_CREDIT_USD' $CFG && grep -q 'TRIAL_GLOBAL_DAILY_USD' $CFG && grep -q 'TRIAL_MODEL_ALLOWLIST' $CFG && grep -q 'TRIAL_ENABLED' $CFG"

# --- US-005: trial ledger at the choke point ---------------------------------------
check "US-005 trial ledger with pinned entrypoint" \
  "grep -qE 'def resolve_trial_routing' $TRIAL"

check "US-005 typed exhaustion error exists" \
  "grep -rq 'trial_exhausted' orchestrator/ --include='*.py'"

# --- US-007: capability report ------------------------------------------------------
check "US-007 onboarding_capabilities callable exists" \
  "grep -rqE 'def onboarding_capabilities' orchestrator/ --include='*.py'"

# --- US-009: section v2 -------------------------------------------------------------
check "US-009 section is stage-aware (powerup variant present)" \
  "grep -q 'powerup' $SECTION"

check "US-009 section carries the OpenRouter recommendation" \
  "grep -qi 'openrouter' $SECTION"

# --- US-010: trust guards -----------------------------------------------------------
check "US-010 no skip_verification/auto_approve enabled in onboarding-owned code" \
  "! grep -nE '(skip_verification|auto_approve).{0,6}(True|true)' $SECTION $OSTATE $TRIAL 2>/dev/null | grep -q ."

check "US-010 trust guard test exists" \
  "grep -rlEq 'def test_.*(no_skip_verification|trust_guard|awaiting_approval)' orchestrator/tests"

# --- US-011: schema truth pass ------------------------------------------------------
check "US-011 no required[] in actions_reports.py still lists report_type (flatten + extract)" \
  "! (tr '\\n' ' ' < $ACT_REPORTS | grep -oE 'required[^]]*\\]' | grep -q report_type)"

check "US-011 walker guard test exists" \
  "grep -rlEq 'def test_.*(schema.*handler|required.*truth|tool_schema_truth)' orchestrator/tests"

check "US-011 platform_create_mission schema NOT given source" \
  "! grep -qE '\\\"source\\\"' $ACT_MISSIONS"

# --- US-012..US-015: frontend surfaces ----------------------------------------------
check "US-012 FirstLoginGuard unmounted from providers" \
  "! grep -q 'FirstLoginGuard' $PROVIDERS"

check "US-012 shepherd + modal FILES survive (deletion is W2)" \
  "[ -d frontend/lib/shepherd ] && [ -f frontend/components/onboarding/welcome-modal.tsx ] && [ -f frontend/components/onboarding/first-login-guard.tsx ]"

check "US-012 onboarding opener exists and is imported" \
  "[ -f frontend/components/onboarding/onboarding-opener.tsx ] && grep -rq 'onboarding-opener' frontend/components frontend/app --include='*.tsx' --exclude='onboarding-opener.tsx'"

check "US-013 power-up card exists and is imported" \
  "[ -f frontend/components/onboarding/power-up-card.tsx ] && grep -rq 'power-up-card' frontend/components --include='*.tsx' --exclude='power-up-card.tsx'"

check "US-014 trial pill + exhausted banner exist" \
  "[ -f frontend/components/onboarding/trial-balance-pill.tsx ] && [ -f frontend/components/onboarding/trial-exhausted-banner.tsx ]"

check "US-015 intake progress card exists" \
  "[ -f frontend/components/onboarding/intake-progress-card.tsx ]"

# --- Convention guards ---------------------------------------------------------------
check "no os.getenv in the new onboarding/trial services" \
  "! grep -nE 'os\\.getenv' $OSTATE $TRIAL 2>/dev/null | grep -q ."

# --- Scope guards: W2 retirement targets untouched -----------------------------------
echo ""
echo "── scope guard: W2 planes untouched (shepherd lib, wizard UI, onboarding-agent seeds)"
if [ -n "$BASE" ] && git diff --name-only "$BASE"..HEAD 2>/dev/null | grep -qE 'frontend/lib/shepherd/|frontend/components/wizard/|orchestrator/core/seeds/seed_onboarding_agents\.py'; then
  echo "   ❌ FAIL: this run touched a W2 retirement plane — out of scope for Wave 1"; FAIL=1
else
  echo "   ✅ PASS: W2 planes untouched"
fi

echo ""
echo "── scope guard: no deletions of onboarding modal/guard components"
if [ -n "$BASE" ] && git diff --name-only --diff-filter=D "$BASE"..HEAD 2>/dev/null | grep -qE 'frontend/components/onboarding/(welcome-modal|first-login-guard)'; then
  echo "   ❌ FAIL: modal/guard component deleted — that is W2"; FAIL=1
else
  echo "   ✅ PASS: no premature deletions"
fi

echo ""
if [ $FAIL -eq 0 ]; then echo "ACCEPTANCE: PRD-222 PASS"; else echo "ACCEPTANCE: PRD-222 FAIL"; fi
echo "NOTE: TRIAL_* Railway values, the Q1 tier decision (gates W2), and waitlist opening are Gerard's calls; not gated here."
exit $FAIL
