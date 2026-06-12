#!/bin/bash
# Acceptance gate — PRD-150 Auth Decoupling (Open Core).
# Run from the worktree repo root. Exit 0 = PRD-level done. Runs ALL checks.
# Verifier fixes baked in: clerk_user_id allowlist (data-continuity keeps),
# ClerkAuth grep excludes tests/, OSS build neutralizes frontend/.env.local.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
FAIL=0
check() {
  local name="$1"; shift
  echo ""
  echo "── $name"
  if bash -c "$1"; then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi
}

check "orchestrator-full-suite (CI-parity, AUTH_PROVIDER=local default)" \
  'cd orchestrator && python3 -m pytest --timeout=60 --timeout-method=thread -o faulthandler_timeout=90 -p no:cacheprovider -q'

check "saas-package-suite (ClerkAuthProvider contract + parity)" \
  'python3 -m pip install -e ./automatos-saas --quiet && cd automatos-saas && python3 -m pytest tests -q'

check "new-seam-targeted (+ untouched 27-test PRD-09 board-SDK suite)" \
  'cd orchestrator && python3 -m pytest tests/test_auth_providers.py tests/test_auth_registry.py tests/test_local_seed.py tests/test_hybrid_dispatch.py tests/test_identity_resolution.py tests/test_auth_provider_parity.py tests/test_no_saas_imports.py tests/test_board_sdk_auth.py -q -p no:cacheprovider'

check "import-direction (core/api/services/modules never import automatos_saas)" \
  'cd orchestrator && python3 -m pip install --quiet import-linter && lint-imports && [ -z "$(grep -rn "automatos_saas" core api services modules --include="*.py" 2>/dev/null)" ]'

check "clerk-gone-from-core (clerk.py deleted; zero ClerkAuth refs outside tests/)" \
  'test ! -f orchestrator/core/auth/clerk.py && [ -z "$(grep -rEn "from core\.auth\.clerk|import core\.auth\.clerk|ClerkAuth\b" orchestrator --include="*.py" --exclude-dir=tests 2>/dev/null)" ]'

check "leakage-layer-deleted (zero clerk_user_id under api/+services/ outside the 4 allowlisted data-continuity keeps)" \
  '[ -z "$(grep -rn "clerk_user_id" orchestrator/api orchestrator/services --include="*.py" 2>/dev/null | grep -vE "api/board_tasks\.py|api/team\.py|services/coordinator_service\.py|services/workspace_purge\.py")" ]'

check "frontend-typecheck-and-units" \
  'cd frontend && npx tsc --noEmit && npm run test'

# OSS build must prove the no-Clerk property: .env.local (which contains Clerk
# keys and is auto-loaded by next build) is moved aside for the duration.
check "frontend-oss-build-no-clerk (.env.local neutralized, Clerk env unset)" \
  'cd frontend && moved=0; trap "[ \$moved -eq 1 ] && mv .env.local.gate-bak .env.local" EXIT; if [ -f .env.local ]; then mv .env.local .env.local.gate-bak; moved=1; fi; env -u NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY -u CLERK_SECRET_KEY NEXT_PUBLIC_EDITION=oss npm run build'

echo ""
if [ $FAIL -eq 0 ]; then echo "ACCEPTANCE: PRD-150 PASS"; else echo "ACCEPTANCE: PRD-150 FAIL"; fi
exit $FAIL
