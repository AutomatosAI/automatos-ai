#!/bin/bash
# Acceptance gate — PRD-209 Open-Core Phase 0 (fresh-clone boot + honest CI).
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
FAIL=0
check() {
  local name="$1"; shift
  echo ""; echo "── $name"
  if bash -c "$1"; then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi
}

check "orchestrator-full-suite (no NEW failures vs documented env baseline)" \
  'OUT=$(cd orchestrator && python3 -m pytest --timeout=90 --timeout-method=thread -o faulthandler_timeout=120 -p no:cacheprovider -q 2>&1 | tail -200); echo "$OUT" | grep -qE "[0-9]{4} passed" && ! echo "$OUT" | grep "^FAILED" | grep -vE "prd172|dr_restore|composio" | grep -q .'

check "prd209 guard tests green" \
  'cd orchestrator && python3 -m pytest -q -k "prd209" 2>&1 | tail -5 | grep -qE "passed" '

check "frontend-vitest-suite" 'cd frontend && npm run -s test'
check "frontend-build" 'cd frontend && npm run -s build'

# US-001 — entrypoint executable in the INDEX (not the filesystem)
check "US-001 docker-entrypoint.sh tracked 100755" \
  'git ls-files -s docker-entrypoint.sh | grep -q "^100755"'

# US-002 — lineage safety: NO deleted revision files (stranded-DB guard), stamp present
BASE=$(git merge-base HEAD origin/main 2>/dev/null || git merge-base HEAD main 2>/dev/null || echo "")
echo ""; echo "── US-002 no alembic revision files deleted (prod lineage)"
if [ -n "$BASE" ]; then
  DELMIG=$(git diff --name-only --diff-filter=D "$BASE"..HEAD -- orchestrator/alembic/versions/ 2>/dev/null | wc -l | tr -d ' ')
  if [ "$DELMIG" = "0" ]; then echo "   ✅ PASS"; else echo "   ❌ FAIL: $DELMIG revision files deleted"; FAIL=1; fi
else echo "   ⚠️  SKIP: no merge base"; fi
check "US-002 fresh path: init_fresh_db exists, stamps heads, entrypoint wired, stale SQL gone" \
  'grep -q "build_schema(engine)" orchestrator/scripts/init_fresh_db.py && grep -q "command.upgrade(cfg, rev)" orchestrator/scripts/generate_schema_baseline.py && grep -q "init_fresh_if_empty" docker-entrypoint.sh && [ ! -f orchestrator/core/database/init_complete_schema.sql ] && ! grep -q "init_complete_schema" docker-compose.yml'

# US-003 — compose local defaults
check "US-003 compose consumes envs/api.defaults via env_file" \
  'grep -q "env_file" docker-compose.yml && grep -q "envs/api.defaults" docker-compose.yml'
check "US-003 api.defaults speaks shipped vocab (AUTH_EDITION=local + DEFAULT_WORKSPACE_ID; dead keys gone)" \
  'grep -q "^AUTH_EDITION=local" envs/api.defaults && grep -q "^DEFAULT_WORKSPACE_ID=" envs/api.defaults && ! grep -qE "^(EDITION|AUTH_PROVIDER)=" envs/api.defaults'
check "US-003 the three secrets stay required" \
  'grep -q "POSTGRES_PASSWORD:?" docker-compose.yml && grep -q "REDIS_PASSWORD:?" docker-compose.yml && grep -q "API_KEY:?" docker-compose.yml'

# US-004 — de-masked lanes + smoke script fix
check "US-004 no continue-on-error in smoke-fresh-clone.yml" \
  '! grep -q "continue-on-error" .github/workflows/smoke-fresh-clone.yml'
check "US-004 smoke script exports DEFAULT_WORKSPACE_ID" \
  'grep -q "DEFAULT_WORKSPACE_ID" scripts/ci/smoke-fresh-clone.sh'

# US-005 — readiness beyond /health
check "US-005 smoke asserts readiness past bare /health" \
  'grep -qE "readiness|document_backend|rag" scripts/ci/smoke-fresh-clone.sh'

# US-006 — drift check exists + wired
check "US-006 schema drift check present + wired into CI" \
  '[ -f scripts/ci/schema_drift_check.py ] && grep -rq "schema_drift_check" .github/workflows/'

# US-007 — one lockfile
echo ""; echo "── US-007 exactly one frontend lockfile tracked"
LOCKS=$(git ls-files frontend/ | grep -cE "(package-lock\.json|yarn\.lock|pnpm-lock\.yaml)$" | tr -d ' ')
if [ "$LOCKS" = "1" ]; then echo "   ✅ PASS"; else echo "   ❌ FAIL: $LOCKS lockfiles"; FAIL=1; fi

# US-008 — one canonical compose
echo ""; echo "── US-008 exactly one tracked docker-compose*.yml repo-wide"
COMPOSES=$(git ls-files | grep -cE "docker-compose[^/]*\.yml$" | tr -d ' ')
if [ "$COMPOSES" = "1" ]; then echo "   ✅ PASS"; else echo "   ❌ FAIL: $COMPOSES compose files"; FAIL=1; fi

# US-009 — honest QUICKSTART
check "US-009 QUICKSTART names the three required secrets" \
  'grep -q "POSTGRES_PASSWORD" QUICKSTART.md && grep -q "REDIS_PASSWORD" QUICKSTART.md && grep -q "API_KEY" QUICKSTART.md'
check "US-009 no stale no-env claim" \
  '! grep -rqi "no \.env file needed" QUICKSTART.md README.md'

# Off-limits surfaces untouched
echo ""; echo "── required CI lanes untouched (orchestrator-tests, ioc-scan definitions)"
if [ -n "$BASE" ]; then
  if git diff "$BASE"..HEAD -- .github/workflows/ | grep -E "^[-+].*(orchestrator-tests|ioc-scan)" | grep -vE "^[-+]{3}" | grep -q .; then
    echo "   ❌ FAIL: required-lane definitions modified"; FAIL=1
  else echo "   ✅ PASS"; fi
else echo "   ⚠️  SKIP: no merge base"; fi

echo ""
if [ $FAIL -eq 0 ]; then echo "ACCEPTANCE: PRD-209 PASS"; else echo "ACCEPTANCE: PRD-209 FAIL"; fi
echo "NOTE: Q2 (flip smoke/from-zero/drift lanes to REQUIRED in branch protection) is Gerard's repo-admin action post-merge."
exit $FAIL
