#!/bin/bash
# Acceptance gate — PRD-222 Auto-Led Onboarding, Wave 1b (US-011..US-016).
# Run from the worktree repo root. Exit 0 = PRD-level done. Runs ALL checks.
# Wave 1's backend spine (US-001..010) is already on main — the full suites are the
# regression gate for it; the asserts below prove the Wave-1b surfaces exist by their
# pinned names, ZERO new migrations shipped, the reset is gated + survivor-safe by
# construction, and the W2 planes are untouched.
# Setting ONBOARDING_RESET_ENABLED / TRIAL_* in Railway + the Q1 tier decision are
# Gerard's, not gated here.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
FAIL=0
OSTATE="orchestrator/services/onboarding_state.py"
PURGE="orchestrator/services/workspace_purge.py"
WSAPI="orchestrator/api/workspaces.py"
CFG="orchestrator/config.py"
PROVIDERS="frontend/components/providers.tsx"
ACT_REPORTS="orchestrator/modules/tools/discovery/actions_reports.py"
ACT_MISSIONS="orchestrator/modules/tools/discovery/actions_missions.py"
MANIFEST="orchestrator/reports/route-manifest.json"
DEVPAGE="frontend/app/dev/reset-onboarding/page.tsx"
check() {
  local name="$1"; shift
  echo ""
  echo "── $name"
  if bash -c "$1"; then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi
}

# --- Primary gates: full suites green -------------------------------------------
check "orchestrator-full-suite (pure tests + no regression; @integration skips with no DB)" \
  'cd orchestrator && python3 -m pytest --timeout=90 --timeout-method=thread -o faulthandler_timeout=120 -p no:cacheprovider -q'

check "frontend-vitest-suite" \
  'cd frontend && npm run -s test'

# --- ZERO new migrations (US-001 shipped the one; a new file here is a hard fail) ----
echo ""
echo "── ZERO new alembic revisions vs base"
BASE=$(git merge-base HEAD origin/main 2>/dev/null || git merge-base HEAD main 2>/dev/null || echo "")
if [ -n "$BASE" ]; then
  NEWMIG=$(git diff --name-only --diff-filter=A "$BASE"..HEAD -- orchestrator/alembic/versions/ 2>/dev/null | wc -l | tr -d ' ')
  if [ "$NEWMIG" = "0" ]; then echo "   ✅ PASS: no new migrations"; else echo "   ❌ FAIL: $NEWMIG new migrations (must be 0 — the JSONB carries all Wave-1b state)"; FAIL=1; fi
else
  echo "   ⚠️  SKIP: no merge base found"
fi

# --- US-011: schema truth pass ------------------------------------------------------
check "US-011 required[] in actions_reports.py no longer lists report_type (flatten + extract)" \
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

# --- US-016: dev reset ---------------------------------------------------------------
check "US-016 reset_onboarding entrypoint with the three flags" \
  "grep -qE 'def reset_onboarding' $OSTATE && grep -q 'reset_trial' $OSTATE && grep -q 'wipe_built' $OSTATE && grep -q 'wipe_credentials' $OSTATE"

check "US-016 ONBOARDING_RESET_ENABLED in config.py only (default off)" \
  "grep -q 'ONBOARDING_RESET_ENABLED' $CFG && ! grep -rn 'ONBOARDING_RESET_ENABLED' orchestrator/ --include='*.py' | grep -v 'config.py' | grep -v 'config\\.' | grep -v tests | grep -q ."

check "US-016 endpoint on the existing workspaces router" \
  "grep -q 'onboarding/reset' $WSAPI"

check "US-016 committed route manifest carries the reset route" \
  "grep -q 'onboarding/reset' $MANIFEST"

check "US-016 wipe reuses workspace_purge internals (call-site outside the purge module itself)" \
  "git grep -l 'workspace_purge' -- 'orchestrator/*.py' 'orchestrator/**/*.py' | grep -v 'services/workspace_purge.py' | grep -v tests | grep -q ."

check "US-016 dev page exists" \
  "[ -f $DEVPAGE ]"

check "US-016 dev page is unlinked (no nav/sidebar reference)" \
  "! grep -rq 'dev/reset-onboarding' frontend/components --include='*.tsx'"

check "US-016 .env.example documents the temporary flag" \
  "grep -q 'ONBOARDING_RESET_ENABLED' .env.example"

check "US-016 reset gating + survivor tests exist" \
  "grep -rlq 'reset_onboarding' orchestrator/tests"

# --- Trust + convention guards -------------------------------------------------------
check "trust-guard test still present (never weakened away)" \
  "ls orchestrator/tests/test_prd222_trust_guards.py >/dev/null 2>&1"

check "no os.getenv in the onboarding-owned services touched this wave" \
  "! grep -nE 'os\\.getenv' $OSTATE $PURGE 2>/dev/null | grep -q ."

check "advance_onboarding_stage validator untouched (reset must not loosen the spine)" \
  "grep -q 'InvalidStageTransition' $OSTATE"

# --- Scope guards: W2 retirement targets untouched -----------------------------------
echo ""
echo "── scope guard: W2 planes untouched (shepherd lib, wizard UI, onboarding-agent seeds)"
if [ -n "$BASE" ] && git diff --name-only "$BASE"..HEAD 2>/dev/null | grep -qE 'frontend/lib/shepherd/|frontend/components/wizard/|orchestrator/core/seeds/seed_onboarding_agents\.py'; then
  echo "   ❌ FAIL: this run touched a W2 retirement plane — out of scope for Wave 1b (importing from tour-storage is fine; MODIFYING it is not)"; FAIL=1
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
echo "── scope guard: no new router file"
if [ -n "$BASE" ] && git diff --name-only --diff-filter=A "$BASE"..HEAD 2>/dev/null | grep -qE 'orchestrator/api/.*\.py'; then
  echo "   ❌ FAIL: new file under orchestrator/api/ — the reset endpoint belongs on the existing workspaces router"; FAIL=1
else
  echo "   ✅ PASS: no new router file"
fi

echo ""
if [ $FAIL -eq 0 ]; then echo "ACCEPTANCE: PRD-222 Wave 1b PASS"; else echo "ACCEPTANCE: PRD-222 Wave 1b FAIL"; fi
echo "NOTE: ONBOARDING_RESET_ENABLED (test window only) + TRIAL_* Railway values, the Q1 tier decision (gates W2), and waitlist opening are Gerard's calls; not gated here."
exit $FAIL
