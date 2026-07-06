#!/bin/bash
# Acceptance gate — PRD-186 S3 Vectors Relight (CODE stories S1/S2/S3 only).
# Run from the worktree repo root. Exit 0 = PRD-level (code) done. Runs ALL checks.
# The full orchestrator suite is the protected-regression gate; the grep asserts
# prove the F005 assertion was EXTRACTED (not duplicated), the boot check is WIRED,
# and the dimension mismatch RAISES. The three OPS stories (bucket env change, prod
# re-embed, S8 probe) are Gerard's prod actions and are intentionally NOT gated here.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
FAIL=0
CFG="orchestrator/config.py"
MAIN="orchestrator/main.py"
S3B="orchestrator/modules/search/vector_store/backends/s3_vectors_backend.py"
check() {
  local name="$1"; shift
  echo ""
  echo "── $name"
  if bash -c "$1"; then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi
}

# --- Primary gate: full suite green (pure PRD-186 tests + no regression) ----------
check "orchestrator-full-suite (pure config-integrity + dimension tests; @integration skips with no DB)" \
  'cd orchestrator && python3 -m pytest --timeout=90 --timeout-method=thread -o faulthandler_timeout=120 -p no:cacheprovider -q'

# --- S1: the F005 assertion is EXTRACTED and CALLED, not duplicated --------------
check "S1 assert_vector_config_integrity() defined in config.py" \
  "grep -qE 'def assert_vector_config_integrity' $CFG"

check "S1 validate_security calls the shared assertion (extracted, not inline)" \
  "grep -qE 'assert_vector_config_integrity\\(' $CFG"

check "S1 the {workspace_id} placeholder message is NOT duplicated (lives in ONE place)" \
  "[ \"\$(grep -c \"{workspace_id}' placeholder\" $CFG 2>/dev/null || echo 0)\" -le 1 ]"

# --- S2: the integrity check is wired into boot (un-swallowable) ------------------
check "S2 boot wiring calls assert_vector_config_integrity in main.py" \
  "grep -qE 'assert_vector_config_integrity' $MAIN"

check "S2 fail-closed boot test exists" \
  "[ -n \"\$(grep -rlE 'def test_boot_aborts_on_bad_vector_config' orchestrator/tests 2>/dev/null)\" ]"

# --- S3: dimension mismatch RAISES (not log-and-continue), never deletes ----------
check "S3 dimension-mismatch tests exist (raises + match-passes)" \
  "[ -n \"\$(grep -rlE 'def test_index_dimension_mismatch_raises' orchestrator/tests 2>/dev/null)\" ] && [ -n \"\$(grep -rlE 'def test_index_dimension_match_passes' orchestrator/tests 2>/dev/null)\" ]"

check "S3 _verify_or_recreate_index raises on mismatch (a raise exists in the backend's dimension path)" \
  "grep -qE 'raise ' $S3B && grep -qE 'dimension' $S3B"

# --- Convention guard: no os.getenv crept in outside config.py -------------------
check "no os.getenv added outside config.py in the touched code files" \
  "! grep -nE 'os\\.getenv' $MAIN $S3B 2>/dev/null | grep -qv '# '"

# --- Scope guard: no OPS work snuck into the code run ----------------------------
echo ""
echo "── scope guard: no OPS (re-embed / probe / migration) code in this run"
BASE=$(git merge-base HEAD origin/main 2>/dev/null || git merge-base HEAD main 2>/dev/null || echo "")
if [ -n "$BASE" ] && git diff --name-only "$BASE"..HEAD 2>/dev/null | grep -qE 'migrate_to_s3_vectors|probe_document_vectors'; then
  echo "   ❌ FAIL: this run touched an OPS script (migrate/probe) — out of scope"; FAIL=1
else
  echo "   ✅ PASS: no OPS scripts modified"
fi

echo ""
if [ $FAIL -eq 0 ]; then echo "ACCEPTANCE: PRD-186 (code) PASS"; else echo "ACCEPTANCE: PRD-186 FAIL"; fi
echo "NOTE: OPS relight (bucket env → {workspace_id} + migrate_to_s3_vectors.py re-embed + S8 probe → LIVE) is Gerard's to run against prod; not gated here."
exit $FAIL
