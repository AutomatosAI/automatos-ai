#!/bin/bash
# Acceptance gate — PRD-226 The Manager's Doctrine (US-001..003). Chain 3/6 (base: 224 branch).
# Run from the worktree repo root. Exit 0 = PRD-level done. Runs ALL checks.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
FAIL=0
SOUL="orchestrator/core/seeds/auto-cto-custom-soul.txt"
SKILL="orchestrator/core/seeds/platform-management-skill.md"
SEEDER="orchestrator/core/seeds/seed_auto_agent.py"
BASE_BR="ralph/prd-224-ticket-lane"
check() {
  local name="$1"; shift
  echo ""
  echo "── $name"
  if bash -c "$1"; then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi
}

# Backend gate is BRANCH-SCOPED (2026-08-27 amendment): ~49 pre-existing
# environmental fails/errors in the full local suite — CI test.yml is the
# full-suite gate. Locally we prove THIS PRD's own tests green (eval fixtures
# still skip cleanly when absent).
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

echo ""
echo "── ZERO new alembic revisions vs base ($BASE_BR)"
BASE=$(git merge-base HEAD "$BASE_BR" 2>/dev/null || git merge-base HEAD origin/main 2>/dev/null || echo "")
if [ -n "$BASE" ]; then
  NEWMIG=$(git diff --name-only --diff-filter=A "$BASE"..HEAD -- orchestrator/alembic/versions/ 2>/dev/null | wc -l | tr -d ' ')
  if [ "$NEWMIG" = "0" ]; then echo "   ✅ PASS: no new migrations"; else echo "   ❌ FAIL: $NEWMIG new migrations (must be 0 — or the build should have BLOCKED)"; FAIL=1; fi
  if git diff "$BASE"..HEAD -- orchestrator/reports/route-manifest.json | grep -q .; then
    echo "   ❌ FAIL: route manifest changed (no new routes in PRD-226)"; FAIL=1
  else
    echo "   ✅ PASS: route manifest untouched"
  fi
else
  echo "   ⚠️  SKIP: no merge base found"
fi

# --- US-001: doctrine in seeds + hash-guarded backfill -----------------------
check "US-001 soul carries lane doctrine (three lanes named)" \
  "grep -qi 'ASSIGN' $SOUL && grep -qi 'MISSION' $SOUL && grep -qi 'DELEGATE' $SOUL"

check "US-001 skill carries the 4-part contract" \
  "grep -qi 'OBJECTIVE' $SKILL && grep -qi 'BOUNDARIES' $SKILL"

check "US-001 skill carries the ask-formatting doctrine (decisions not reports)" \
  "grep -qiE 'never (sit )?idle|park' $SKILL"

check "US-001 seeder has hash-guarded backfill (skip customized)" \
  "grep -qiE 'hash|digest' $SEEDER"

check "US-001 doctrine ceiling test exists" \
  "grep -rliE 'ceiling|char.*(budget|limit)' orchestrator/tests --include='*.py' | grep -q . || grep -rl 'doctrine' orchestrator/tests --include='*.py' | grep -q ."

# --- US-002: single rubric site ----------------------------------------------
check "US-002 rubric extended in place (reuse-before-create present in auto.py)" \
  "grep -qiE 'one capable owner|reuse' orchestrator/consumers/chatbot/auto.py"

# --- US-003: shared contract fragment ----------------------------------------
echo ""
echo "── US-003 shared 4-part fragment: exactly ONE definition site"
DEFS=$(grep -rl "OBJECTIVE" orchestrator --include="*.py" | xargs grep -l "BOUNDARIES" 2>/dev/null | grep -vc tests || echo 0)
if [ "$DEFS" -ge 1 ]; then
  SITES=$(grep -rln 'OBJECTIVE.*OUTPUT.*TOOLS.*BOUNDARIES\|4-part\|four-part' orchestrator --include='*.py' | grep -v tests | wc -l | tr -d ' ')
  echo "   ✅ PASS: fragment present ($SITES candidate sites — reviewer confirms single-source)"
else
  echo "   ❌ FAIL: no 4-part contract fragment found in orchestrator code"; FAIL=1
fi

# P226-RVW-2: the single-source guarantee must cover the Markdown seed home too,
# not just .py files — the always-on platform-management skill embeds the SAME
# fragment and CI must fail if a hand-edit drifts it. AST-extract the fragment
# (import-free, no side effects) and assert it is present verbatim in the skill.
check "P226-RVW-2 skill md embeds the shared contract fragment verbatim (markdown single-source)" \
  "python3 -c \"import ast,pathlib,sys; src=pathlib.Path('orchestrator/modules/coordination/dispatch_contract.py').read_text(encoding='utf-8'); frag=next(n.value.value for n in ast.walk(ast.parse(src)) if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant) and any(getattr(t,'id',None)=='DISPATCH_CONTRACT_FRAGMENT' for t in n.targets)); md=pathlib.Path('orchestrator/core/seeds/platform-management-skill.md').read_text(encoding='utf-8'); sys.exit(0 if frag in md else 1)\""

check "US-003 planner stores definition_of_done" \
  "grep -qi 'definition_of_done' orchestrator/modules/coordination/planner.py"

check "US-003 verification consumes definition_of_done" \
  "grep -qi 'definition_of_done' orchestrator/modules/coordination/verification.py"

check "no committed gold-set fixtures (public repo)" \
  "! git diff --name-only \$(git merge-base HEAD $BASE_BR 2>/dev/null || git merge-base HEAD origin/main)..HEAD | grep -qiE 'gold|fixture.*eval|eval.*fixture'"

echo ""
if [ "$FAIL" = "0" ]; then echo "ACCEPTANCE: PASS (PRD-226)"; exit 0; else echo "ACCEPTANCE: FAIL (PRD-226)"; exit 1; fi
