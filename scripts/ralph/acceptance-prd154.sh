#!/bin/bash
# Acceptance gate — PRD-154 Wave-0 Quick Wins.
# Run from the worktree repo root. Exit 0 = PRD-level done. Runs ALL checks.
# DETERMINISTIC ONLY: browser-verify ACs (S6/S7/S9/S10/S11/S12) are confirmed
# by their paired vitest/grep proxies here and flagged for a morning human
# browser check — they never gate a headless run.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
FAIL=0
check() {
  local name="$1"; shift
  echo ""
  echo "── $name"
  if bash -c "$1"; then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi
}

check "orchestrator-full-suite (CI-parity; @integration skips cleanly with no DB)" \
  'cd orchestrator && python3 -m pytest --timeout=60 --timeout-method=thread -o faulthandler_timeout=90 -p no:cacheprovider -q'

check "frontend-typecheck" \
  'cd frontend && npx tsc --noEmit'

check "frontend-units (vitest — includes S6/S7/S11 ErrorBoundary+behavior proxies)" \
  'cd frontend && npm run test'

check "frontend-lint (S9 sonner codemod + S10 raw-fetch('\''/api'\'') ban rule)" \
  'cd frontend && npm run lint'

check "S7 R3F stack deleted (@react-three/fiber + drei gone from package.json)" \
  '! grep -qE "\"@react-three/fiber\"|\"@react-three/drei\"" frontend/package.json'

check "S9 losing toast systems gone (react-hot-toast + use-toast: no deps, no call sites)" \
  '! grep -q "\"react-hot-toast\"" frontend/package.json \
   && [ -z "$(grep -rn "react-hot-toast" frontend --include="*.ts" --include="*.tsx" --exclude-dir=node_modules 2>/dev/null)" ] \
   && [ -z "$(grep -rn "use-toast" frontend --include="*.ts" --include="*.tsx" --exclude-dir=node_modules 2>/dev/null)" ]'

check "S10 fake agent-performance literal removed from api/agents.py" \
  '! grep -q "85.5" orchestrator/api/agents.py'

echo ""
if [ $FAIL -eq 0 ]; then
  echo "ACCEPTANCE: PRD-154 PASS"
  echo "NOTE: browser ACs (field graph render, toast render, global search, Databases tabs, calendar retry) are DEFERRED — eyeball in the morning per prd-154.json DEFERRED notes."
else
  echo "ACCEPTANCE: PRD-154 FAIL"
fi
exit $FAIL
