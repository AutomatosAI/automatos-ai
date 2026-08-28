#!/bin/bash
# Acceptance gate — PRD-222 Wave 2a (US-019..US-022): connect card, checklist card,
# retirement of Shepherd/wizard/4-agent machinery, is_new_workspace migration.
# Run from the worktree repo root. Exit 0 = wave done. Runs ALL checks.
# The inverse (deletion) checks are the heart: gone means GONE (files, imports,
# deps, attrs), while every Wave-1 surface provably survives.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
FAIL=0
OSTATE="orchestrator/services/onboarding_state.py"
COORD="orchestrator/services/coordinator_service.py"
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

check "frontend-build (the only import-graph link-check browser ESM gets)" \
  'cd frontend && npm run -s build'

# --- EXACTLY ONE migration (US-021 seeded-agent cleanup) ------------------------------
echo ""
echo "── exactly ONE new alembic revision vs base (the seeded-agent cleanup)"
BASE=$(git merge-base HEAD origin/main 2>/dev/null || git merge-base HEAD main 2>/dev/null || echo "")
if [ -n "$BASE" ]; then
  NEWMIG=$(git diff --name-only --diff-filter=A "$BASE"..HEAD -- orchestrator/alembic/versions/ 2>/dev/null | wc -l | tr -d ' ')
  if [ "$NEWMIG" = "1" ]; then echo "   ✅ PASS: one new migration"; else echo "   ❌ FAIL: $NEWMIG new migrations (must be exactly 1)"; FAIL=1; fi
else
  echo "   ⚠️  SKIP: no merge base found"
fi

# --- US-019: connect card -------------------------------------------------------------
check "US-019 connect card exists and is wired into chat" \
  "[ -f frontend/components/onboarding/connect-app-card.tsx ] && grep -rq 'connect-app-card' frontend/components --include='*.tsx' --exclude='connect-app-card.tsx'"

check "US-019 no new OAuth/redirect endpoint (reuse proven)" \
  "! git diff --name-only --diff-filter=A \"$BASE\"..HEAD -- orchestrator/api/ 2>/dev/null | grep -q ."

check "US-019 first-integration funnel event exists" \
  "grep -rq 'first_integration_connected' orchestrator/ --include='*.py'"

# --- US-020: checklist ----------------------------------------------------------------
check "US-020 checklist entrypoint in onboarding_state.py" \
  "grep -q 'checklist' $OSTATE"

check "US-020 checklist card exists (chat + Command Center wiring)" \
  "[ -f frontend/components/onboarding/setup-checklist-card.tsx ] && grep -rq 'setup-checklist-card' frontend/components --include='*.tsx' --exclude='setup-checklist-card.tsx'"

check "US-020 no localStorage in the checklist card (D8)" \
  "! grep -q 'localStorage' frontend/components/onboarding/setup-checklist-card.tsx"

# --- US-021: retirement — gone means GONE ---------------------------------------------
check "US-021 shepherd surfaces deleted" \
  "[ ! -d frontend/lib/shepherd ] && [ ! -f frontend/components/onboarding/welcome-modal.tsx ] && [ ! -f frontend/components/onboarding/first-login-guard.tsx ] && [ ! -f frontend/hooks/use-auto-tour.ts ] && [ ! -f frontend/styles/shepherd-custom.css ]"

check "US-021 shepherd deps out of package.json" \
  "! grep -qE 'shepherd\\.js|react-shepherd' frontend/package.json"

check "US-021 zero data-tour attributes" \
  "[ \$(grep -rl 'data-tour' frontend --include='*.tsx' 2>/dev/null | grep -v node_modules | wc -l | tr -d ' ') = '0' ]"

check "US-021 wizard UI deleted; intake card re-homed to use-intake-progress" \
  "[ ! -d frontend/components/wizard ] && [ ! -d frontend/app/onboarding/wizard ] && [ ! -f frontend/hooks/use-wizard-progress.ts ] && [ ! -f frontend/hooks/use-wizard-api.ts ] && [ -f frontend/hooks/use-intake-progress.ts ] && grep -q 'use-intake-progress' frontend/components/onboarding/intake-progress-card.tsx"

check "US-021 backend intake pipeline SURVIVES (tool substrate)" \
  "[ -d orchestrator/modules/intake ] && [ -f orchestrator/api/wizard.py ]"

check "US-021 4-agent machinery gone from coordinator + boot + reseed" \
  "! grep -qE '_clone_onboarding_agents|_cleanup_ephemeral_agents|mission_zero' $COORD && ! grep -q 'seed_onboarding' orchestrator/main.py && ! grep -q '_ensure_onboarding_agents' orchestrator/api/wizard.py && [ ! -f orchestrator/core/seeds/seed_onboarding_agents.py ]"

check "US-021 admin surface gone (tab + router + mounts)" \
  "[ ! -f orchestrator/api/onboarding_agents.py ] && [ ! -f frontend/components/settings/OnboardingAgentsTab.tsx ] && ! grep -q 'onboarding_agents' orchestrator/main.py orchestrator/router_manifest.py && ! grep -q 'OnboardingAgentsTab' frontend/components/settings/SystemSettingsTab.tsx"

check "US-021 org-chart empty state re-pointed (no Mission Zero CTA)" \
  "! grep -qi 'mission zero' frontend/components/agents/org-chart-tab.tsx"

check "US-021 repo grep-clean in code dirs" \
  "! grep -rqiE 'shepherd|use-auto-tour|OnboardingAgentsTab|seed_onboarding_agents' orchestrator/api orchestrator/services orchestrator/core orchestrator/modules frontend/components frontend/app frontend/hooks frontend/lib 2>/dev/null"

# --- US-022: is_new_workspace gone ----------------------------------------------------
check "US-022 is_new_workspace fully migrated out" \
  "! grep -rqE 'is_new_workspace|isNewWorkspace' orchestrator/api orchestrator/services frontend/components frontend/app frontend/hooks 2>/dev/null"

# --- Wave-1 surfaces MUST survive the retirement --------------------------------------
check "wave-1 chat surfaces intact" \
  "[ -f frontend/components/onboarding/onboarding-opener.tsx ] && [ -f frontend/components/onboarding/power-up-card.tsx ] && [ -f frontend/components/onboarding/trial-balance-pill.tsx ] && [ -f frontend/components/onboarding/trial-exhausted-banner.tsx ] && [ -f frontend/components/onboarding/intake-progress-card.tsx ]"

check "wave-1 reset + section + trust guards intact" \
  "grep -q 'def reset_onboarding' $OSTATE && [ -f frontend/app/dev/reset-onboarding/page.tsx ] && grep -q 'powerup' orchestrator/modules/context/sections/onboarding.py && [ -f orchestrator/tests/test_prd222_trust_guards.py ]"

check "advance_onboarding_stage validator untouched" \
  "grep -q 'InvalidStageTransition' $OSTATE"

# --- Scope guard: no W2·S1/S2 (Q1-gated) ----------------------------------------------
echo ""
echo "── scope guard: no exposure-profile / plan-recommendation code (Q1-gated)"
if [ -n "$BASE" ] && git diff "$BASE"..HEAD 2>/dev/null | grep -qiE '^\+.*(exposure_profile|plan_recommendation)'; then
  echo "   ❌ FAIL: W2·S1/S2 territory touched — that kit waits on Q1"; FAIL=1
else
  echo "   ✅ PASS: Q1-gated planes untouched"
fi

echo ""
if [ $FAIL -eq 0 ]; then echo "ACCEPTANCE: PRD-222 Wave 2a PASS"; else echo "ACCEPTANCE: PRD-222 Wave 2a FAIL"; fi
echo "NOTE: the cleanup migration runs on deploy (seeded onboarding agents removed from live DBs). Q1 tiers still gate W2·S1/S2. TRIAL_* Railway values remain Gerard's pre-pilot step."
exit $FAIL
