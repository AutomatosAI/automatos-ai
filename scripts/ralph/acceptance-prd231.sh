#!/bin/bash
# Acceptance gate — PRD-231 Auto's Context Diet (US-002..006).
# Base: feat/auto-skill-seed-sync (#640). Run from the worktree repo root.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
FAIL=0
CHARTER_SEED="orchestrator/core/seeds/platform-management-skill.md"
OPS_SEED="orchestrator/core/seeds/platform-operations-skill.md"
SOUL="orchestrator/core/seeds/auto-cto-custom-soul.txt"
SEEDER="orchestrator/core/seeds/seed_auto_agent.py"
LOADER="orchestrator/modules/agents/services/skill_loader.py"
SECTION="orchestrator/modules/context/sections/skills.py"
BASE_BR="feat/auto-skill-seed-sync"
check() {
  local name="$1"; shift
  echo ""
  echo "── $name"
  if bash -c "$1"; then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi
}

# ── branch-scoped backend tests (full-suite provenance = CI test.yml) ────────
echo ""
echo "── branch-scoped backend tests"
BASEP=$(git merge-base HEAD "$BASE_BR" 2>/dev/null || git merge-base HEAD origin/main 2>/dev/null || echo "")
CHT=$(git diff --name-only "${BASEP:-HEAD~40}"..HEAD -- orchestrator 2>/dev/null | grep -E '(^|/)tests/.*\.py$' | sed 's|^orchestrator/||' | tr '\n' ' ')
if [ -z "${CHT// /}" ]; then
  echo "   ❌ FAIL: no test files changed — this PRD requires tests"; FAIL=1
else
  if ( cd orchestrator && python3 -m pytest --timeout=90 --timeout-method=thread -o faulthandler_timeout=120 -p no:cacheprovider -q $CHT ); then
    echo "   ✅ PASS: branch-scoped tests: $CHT"
  else
    echo "   ❌ FAIL: branch-scoped tests: $CHT"; FAIL=1
  fi
fi

# ── 226 regression: doctrine/contract/backfill suites UNTOUCHED and green ────
echo ""
echo "── test_prd226_* files untouched + green"
if [ -n "$BASEP" ] && git diff --name-only "$BASEP"..HEAD -- orchestrator/tests/ | grep -q "test_prd226"; then
  echo "   ❌ FAIL: a test_prd226_* file was modified (the _default_persona hypothesis must hold, or the build should have BLOCKED)"; FAIL=1
else
  if ( cd orchestrator && python3 -m pytest -q -p no:cacheprovider --timeout=90 tests/test_prd226_doctrine.py tests/test_prd226_contract.py tests/test_prd226_backfill_mode.py tests/test_prd226_prof_tech_backfill.py >/dev/null 2>&1 ); then
    echo "   ✅ PASS: 226 suites green, files untouched"
  else
    echo "   ❌ FAIL: a 226 suite went red"; FAIL=1
  fi
fi

# ── migrations / manifest ────────────────────────────────────────────────────
echo ""
echo "── ZERO new alembic revisions; manifest untouched"
if [ -n "$BASEP" ]; then
  NEWMIG=$(git diff --name-only --diff-filter=A "$BASEP"..HEAD -- orchestrator/alembic/versions/ 2>/dev/null | wc -l | tr -d ' ')
  [ "$NEWMIG" = "0" ] && echo "   ✅ PASS: no new migrations" || { echo "   ❌ FAIL: $NEWMIG new migrations"; FAIL=1; }
  git diff "$BASEP"..HEAD -- orchestrator/reports/route-manifest.json | grep -q . && { echo "   ❌ FAIL: route manifest changed"; FAIL=1; } || echo "   ✅ PASS: route manifest untouched"
fi

# ── US-002: both seeds generated, correct shape ──────────────────────────────
check "US-002 charter seed exists WITHOUT the cookbook" \
  "[ -f $CHARTER_SEED ] && ! grep -q '# Platform Operations Reference' $CHARTER_SEED"
check "US-002 ops seed exists WITH the cookbook" \
  "[ -f $OPS_SEED ] && grep -q 'platform-operations' $OPS_SEED && grep -q '## 19\.' $OPS_SEED"
check "US-002 banners carry recorded body sha" \
  "grep -q 'sha' $CHARTER_SEED && grep -q 'sha' $OPS_SEED"
check "US-002 sync --check passes on fresh tree" \
  "python3 scripts/sync-auto-skill.py --check"

# ── US-003: seeding + non-core discipline ────────────────────────────────────
check "US-003 seeder upserts+assigns platform-operations" \
  "grep -q 'platform-operations' $SEEDER"
check "US-003 _BUILTIN_PATHS has both entries" \
  "[ \$(grep -c 'core/seeds/platform-' $LOADER) -ge 2 ]"
check "US-003 platform-operations NOT in core-always-on (config + section default)" \
  "! grep -rn 'platform-operations' orchestrator/config.py $SECTION | grep -iv 'not\|never\|#' | grep -q 'ALWAYS_ON'"

# ── US-004: soul surgery ─────────────────────────────────────────────────────
check "US-004 five rulebook sections REMOVED from soul" \
  "! grep -qE '\*\*My Role:\*\*|\*\*My Authority:\*\*|\*\*How I Think:\*\*|\*\*My Operating Rhythm:\*\*|\*\*My Routing Rules:\*\*' $SOUL"
check "US-004 personality kept (spot anchors)" \
  "grep -q 'Sacred Ground' $SOUL && grep -q 'My Promise' $SOUL && grep -qi 'one-armed plasterer' $SOUL"
check "US-004 cross-reference line present" \
  "grep -q 'single source' $SOUL"
check "US-004 pre-231 hash frozen into known hashes" \
  "grep -qi 'pre-231\|fat default' $SEEDER"

# ── US-005/US-006 ────────────────────────────────────────────────────────────
check "US-005 drift-guard pytest exists (sha banner recompute)" \
  "grep -rlq 'sha' orchestrator/tests --include='*prd231*' "
check "US-006 activation log carries size counts" \
  "grep -q 'core_tokens' $SECTION"

echo ""
if [ "$FAIL" = "0" ]; then echo "ACCEPTANCE: PASS (PRD-231)"; exit 0; else echo "ACCEPTANCE: FAIL (PRD-231)"; exit 1; fi
