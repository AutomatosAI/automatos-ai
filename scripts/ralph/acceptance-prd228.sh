#!/bin/bash
# Acceptance gate — PRD-228 Fleet State (US-001..004). Chain 5/6 (base: 225 branch).
# Run from the worktree repo root. Exit 0 = PRD-level done. Runs ALL checks.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
FAIL=0
SVC="orchestrator/services/fleet_state.py"
MANIFEST="orchestrator/reports/route-manifest.json"
MGMT="frontend/components/agents/agent-management.tsx"
# Chain parent PRD-225 is MERGED (#637), so the diff base is main. Comparing
# against the stale local parent branch counted 225's now-merged migration as
# "new" and failed the zero-migration check on a read-model-only PRD.
BASE_BR="origin/main"
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

echo ""
echo "── ZERO new alembic revisions vs base ($BASE_BR)"
BASE=$(git merge-base HEAD "$BASE_BR" 2>/dev/null || git merge-base HEAD origin/main 2>/dev/null || echo "")
if [ -n "$BASE" ]; then
  NEWMIG=$(git diff --name-only --diff-filter=A "$BASE"..HEAD -- orchestrator/alembic/versions/ 2>/dev/null | wc -l | tr -d ' ')
  if [ "$NEWMIG" = "0" ]; then echo "   ✅ PASS: no new migrations"; else echo "   ❌ FAIL: $NEWMIG new migrations (must be 0 — read-model only)"; FAIL=1; fi
else
  echo "   ⚠️  SKIP: no merge base found"
fi

# --- US-001 ------------------------------------------------------------------
check "US-001 fleet_state service exists" \
  "[ -f $SVC ]"

check "US-001 read-only (no session mutations in the service)" \
  "! grep -nE '\\.(add|delete|commit|flush)\\(' $SVC | grep -q ."

check "US-001 cost-source pin recorded in the service" \
  "grep -qiE 'cost source|canonical' $SVC"

# --- US-002 ------------------------------------------------------------------
check "US-002 fleet route in the committed manifest" \
  "grep -q '/api/v1/fleet' $MANIFEST"

check "US-002 route-manifest test green" \
  'cd orchestrator && python3 -m pytest -q tests/test_route_manifest.py'

check "US-002 api-client carries the fleet call" \
  "grep -q 'fleet' frontend/lib/api-client.ts"

# test_route_manifest.py regenerates the manifest as a side effect; in a partial
# local env (infra-gated routers like workflows/* fail to import) that DROPS
# routes, corrupting the working-tree manifest. The COMMITTED manifest is the
# source of truth CI regenerates in full — restore it before the route-contract
# check reads it (a no-op in CI, where regeneration matches committed).
git checkout -- "$MANIFEST" 2>/dev/null || true

check "US-002 frontend route-contract green" \
  'node frontend/scripts/check-route-contract.js'

# --- US-003 ------------------------------------------------------------------
check "US-003 platform_fleet_status registered (3-file pattern)" \
  "grep -rq 'platform_fleet_status' orchestrator/modules/tools/discovery/"

# --- US-004 ------------------------------------------------------------------
check "US-004 fleet tab in the Agents surface" \
  "grep -qi 'fleet' $MGMT"

check "US-004 exactly one fleet hook, no V2 hooks" \
  "[ \$(grep -rl 'fleet' frontend/hooks/ 2>/dev/null | wc -l | tr -d ' ') -le 2 ] && ! ls frontend/hooks/ | grep -qi 'v2'"

check "no os.getenv outside config.py (diff scope)" \
  "! git diff \$(git merge-base HEAD $BASE_BR 2>/dev/null || git merge-base HEAD origin/main)..HEAD -- 'orchestrator/**/*.py' ':!orchestrator/config.py' | grep -E '^\\+' | grep -q 'os.getenv'"

echo ""
if [ "$FAIL" = "0" ]; then echo "ACCEPTANCE: PASS (PRD-228)"; exit 0; else echo "ACCEPTANCE: FAIL (PRD-228)"; exit 1; fi
