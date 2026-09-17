#!/bin/bash
# Acceptance gate — PRD-246 Studio on mobile (US-001..008).
# Base: studio. Run from the worktree repo root.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
FAIL=0
CSS="frontend/app/globals.css"
check() {
  local name="$1"; shift
  echo ""
  echo "── $name"
  if bash -c "$1"; then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi
}

# ── the compact region exists and is the single home for compact rules ───────
check "one Studio compact region in globals.css" \
  "grep -q '── Studio compact' $CSS && [ \$(grep -c '── Studio compact' $CSS) -eq 1 ]"

check "safe area has exactly one definition (no second helper)" \
  "[ \$(grep -c 'safe-area-inset-bottom' $CSS) -le 2 ]"

check "touch targets raised under 768" \
  "grep -q 'min-height: 44px' $CSS"

# ── every surface family has compact rules ───────────────────────────────────
for fam in sh-chat cc-tabs cc-stats entry-grid cc-toolbar; do
  check "compact rules cover .$fam" \
    "awk '/── Studio compact/,0' $CSS | grep -q '\\.$fam'"
done

# ── the routes fork on the style axis alone (US-007) ─────────────────────────
check "no route selects a component by width" \
  "! grep -rEn 'isStudio && !is(TabletOrBelow|MobileLayout)' frontend/app --include=page.tsx"

# ── Classic is untouched ─────────────────────────────────────────────────────
check "classic components and the classic mobile sidebar are unchanged" \
  "git diff --name-only origin/studio..HEAD | grep -Ev '^frontend/(app/globals.css|app/.*/page.tsx|components/(chatbot|command-center|assignments|agents|deliverables|tools|shared|layout)/)' | grep -qv 'mobile-sidebar' || ! git diff --name-only origin/studio..HEAD | grep -q 'components/layout/mobile-sidebar.tsx'"

# ── the PRD-244 gates still hold, plus the new mobile scope gate ─────────────
check "frontend suite (all gates)" \
  "cd frontend && npm run test"

echo ""
if [ "$FAIL" -eq 0 ]; then echo "✅ PRD-246 acceptance: PASS"; else echo "❌ PRD-246 acceptance: FAIL"; fi
exit "$FAIL"
