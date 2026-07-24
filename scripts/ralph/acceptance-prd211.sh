#!/bin/bash
# Acceptance gate — PRD-211 In-repo topology discipline (US-001..US-003).
# Run from the worktree repo root. Exit 0 = done + safe.
# Gates: (1) full orchestrator suite green, (2) the import-linter contract exists,
# parses, and (if installed) is green on the branch tip, (3) the 7 mem0-residue files
# are gone + guarded + the un-split is locked, (4) the /api/tasks guard passes,
# (5) SCOPE GUARD — this PRD adds a contract, it does NOT refactor feature modules.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
FAIL=0
check() {
  local name="$1"; shift
  echo ""; echo "── $name"
  if bash -c "$1"; then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi
}

# --- Primary gate: full suite green ----------------------------------------------
check "orchestrator-full-suite green (@integration skips with no DB)" \
  'cd orchestrator && python3 -m pytest --timeout=90 --timeout-method=thread -o faulthandler_timeout=120 -p no:cacheprovider -q'

# --- US-001: the import-linter contract ------------------------------------------
check "US-001 orchestrator/.importlinter exists" "[ -f orchestrator/.importlinter ]"
check "US-001 .importlinter parses + declares an independence contract" \
  "python3 -c \"import configparser,sys; c=configparser.ConfigParser(); c.read('orchestrator/.importlinter'); sys.exit(0 if any('independence' in (c[s].get('type','')) for s in c.sections()) else 1)\""
check "US-001 import-linter pinned in requirements" \
  "grep -qiE 'import-linter|import_linter' orchestrator/requirements.txt"
check "US-001 dedicated CI lane present" "[ -f .github/workflows/import-linter.yml ]"
check "US-001 contract-present test added" \
  "[ -n \"\$(grep -rlE 'def test_import_contract_present' orchestrator/tests 2>/dev/null)\" ]"
# lint-imports is the real gate in CI; run it here only if it is installed.
echo ""; echo "── US-001 lint-imports green on the branch tip (best-effort — CI lane is the real gate)"
if command -v lint-imports >/dev/null 2>&1; then
  if (cd orchestrator && lint-imports >/dev/null 2>&1); then echo "   ✅ PASS: lint-imports exits 0"; else echo "   ❌ FAIL: lint-imports non-zero on tip (contract not ratcheted green)"; FAIL=1; fi
else
  echo "   ⚠️  lint-imports not installed in this shell — deferred to the CI lane (config parses, above)"
fi

# --- US-002: mem0 residue gone + un-split locked ---------------------------------
check "US-002 the 7 mem0 residue files deleted" \
  "[ ! -e orchestrator/mem0_openapi.json ] && [ ! -e orchestrator/scripts/probe_mem0_endpoints.py ] && [ ! -e orchestrator/scripts/seed_mem0_user.py ] && [ ! -e scripts/test_mem0_railway.py ] && [ ! -e docs/PRDS/39-MEM0-MIGRATION-PRD.md ] && [ ! -e docs/PRDS/PRD-152-MEM0-INTERNAL-SERVICES-DECOUPLING.md ] && [ ! -e docs/memory-system/phase1-mem0-async-rollback.md ]"
check "US-002 guard test present" \
  "[ -n \"\$(grep -rlE 'def test_no_mem0_residue' orchestrator/tests 2>/dev/null)\" ]"
check "US-002 un-split locked: no HTTP mem0 client under modules/memory" \
  "! git grep -nE 'MEM0_API_URL|mem0_client|httpx' -- orchestrator/modules/memory 2>/dev/null | grep -q ."

# --- US-003: /api/tasks stays gone (existing guard passes) ------------------------
check "US-003 tasks-lane-deleted guard passes" \
  "cd orchestrator && python3 -m pytest -q tests/test_p2w2_tasks_lane_deleted.py"

# --- SCOPE GUARD: a contract, not a refactor -------------------------------------
echo ""; echo "── scope guard: no feature-module code refactor, no node_modules"
BASE=$(git merge-base HEAD origin/main 2>/dev/null || git merge-base HEAD main 2>/dev/null || echo "")
if [ -n "$BASE" ]; then
  OVERREACH=$(git diff --name-only "$BASE"..HEAD 2>/dev/null | grep -nE 'node_modules|alembic/versions' || true)
  if [ -n "$OVERREACH" ]; then
    echo "   ❌ FAIL: touched node_modules or a migration (out of scope for a contract-only PRD):"; echo "$OVERREACH"; FAIL=1
  else
    echo "   ✅ PASS: no node_modules / no migration in the diff"
  fi
else
  echo "   ⚠️  could not compute base — skipping scope guard"
fi
check "no os.getenv added outside config.py" \
  "[ -z \"\$(git diff \$BASE..HEAD -- orchestrator ':!orchestrator/config.py' 2>/dev/null | grep -E '^\\+' | grep -E 'os\\.getenv')\" ]"

echo ""
if [ $FAIL -eq 0 ]; then echo "ACCEPTANCE: PRD-211 PASS"; else echo "ACCEPTANCE: PRD-211 FAIL"; fi
echo "NOTE: modules.learning/evaluation are in the contract list on this branch; they drop out when PRD-184 merges first (rebase). The contract ratchets — flipping the lane to required in branch protection is PRD-210's call."
exit $FAIL
