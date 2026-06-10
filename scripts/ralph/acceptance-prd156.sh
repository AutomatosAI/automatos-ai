#!/bin/bash
# Acceptance gate — PRD-156 Security & Tenancy Hardening.
# Run from the worktree repo root. Exit 0 = PRD-level done. Runs ALL checks.
# The deterministic suite (tenancy matrix / IDOR / SSTI / auth tests the stories
# author) is the gate; the adversarial security-reviewer pass runs in the review
# cycle. S2 lives in the sibling ../automatos-mem0 repo (cross-repo).
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
FAIL=0
check() {
  local name="$1"; shift
  echo ""
  echo "── $name"
  if bash -c "$1"; then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi
}

check "orchestrator-full-suite (tenancy matrix / IDOR / SSTI / auth tests; @integration skips with no DB)" \
  'cd orchestrator && python3 -m pytest --timeout=60 --timeout-method=thread -o faulthandler_timeout=90 -p no:cacheprovider -q'

check "S5 mock memory surface deleted (AdvancedMemoryManager gone from the tree)" \
  '[ -z "$(grep -rn "AdvancedMemoryManager" orchestrator --include="*.py" 2>/dev/null)" ]'

check "S3 NL2SQL unreachable from chat (query_main_database not wired into chat tool routing)" \
  '[ -z "$(grep -rn "query_main_database" orchestrator/consumers/chatbot 2>/dev/null)" ]'

check "S5 route-contract suite still green (proves deleted routes have zero surviving callers)" \
  'cd frontend && npm run test:contract'

check "frontend-typecheck" \
  'cd frontend && npx tsc --noEmit'

# S2 — mem0 token auth lives in the sibling repo; its test harness may not be
# auto-discoverable from here. Attempt it; if absent, flag for the review cycle
# (which inspects the ../automatos-mem0 diff) rather than silently passing.
echo ""
echo "── S2 mem0 token-auth (cross-repo: ../automatos-mem0)"
MEM0="../automatos-mem0"
if [ -d "$MEM0/openmemory/api/tests" ] || ls "$MEM0"/openmemory/api/conftest.py >/dev/null 2>&1; then
  if (cd "$MEM0/openmemory/api" && python3 -m pytest -q); then
    echo "   ✅ PASS: mem0 fork auth tests"
  else
    echo "   ❌ FAIL: mem0 fork auth tests"; FAIL=1
  fi
else
  echo "   ⚠️  MANUAL VERIFY: mem0 fork test harness not auto-discovered here."
  echo "      → review cycle inspects ../automatos-mem0 diff; set MEM0_API_KEY on Railway BEFORE deploy."
fi

echo ""
if [ $FAIL -eq 0 ]; then echo "ACCEPTANCE: PRD-156 PASS"; else echo "ACCEPTANCE: PRD-156 FAIL"; fi
exit $FAIL
