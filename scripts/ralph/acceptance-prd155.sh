#!/bin/bash
# Acceptance gate — PRD-155 Route Contract & Mount Honesty.
# Run from the worktree repo root. Exit 0 = PRD-level done. Runs ALL checks.
# The net must be HONEST: zero suppressions, manifest from the real app.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
FAIL=0
check() {
  local name="$1"; shift
  echo ""
  echo "── $name"
  if bash -c "$1"; then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi
}

check "S1 route-manifest deterministic (two runs, no DB, identical output)" \
  'cd orchestrator && python3 -m scripts.dump_routes && cp reports/route-manifest.json /tmp/rm1.json && python3 -m scripts.dump_routes && diff -q /tmp/rm1.json reports/route-manifest.json'

check "S3 mount honesty (zero try/except ImportError around router mounts in main.py)" \
  '! grep -nE "except[[:space:]]+ImportError" orchestrator/main.py'

check "S4 tool reachability suite" \
  'cd orchestrator && python3 -m pytest tests/test_tool_reachability.py -q -p no:cacheprovider'

check "S2 frontend route-contract suite (extracted paths ⊆ manifest, zero suppressions)" \
  'cd frontend && npm run test:contract'

check "S5 contract + reachability jobs wired into CI" \
  'grep -qiE "contract|reachability" .github/workflows/test.yml'

check "orchestrator-full-suite (CI-parity; @integration skips cleanly with no DB)" \
  'cd orchestrator && python3 -m pytest --timeout=60 --timeout-method=thread -o faulthandler_timeout=90 -p no:cacheprovider -q'

check "frontend-typecheck" \
  'cd frontend && npx tsc --noEmit'

echo ""
if [ $FAIL -eq 0 ]; then echo "ACCEPTANCE: PRD-155 PASS"; else echo "ACCEPTANCE: PRD-155 FAIL"; fi
exit $FAIL
