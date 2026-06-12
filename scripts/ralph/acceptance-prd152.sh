#!/bin/bash
# Acceptance gate — PRD-152 mem0 / internal-services decoupling.
# Run from the worktree repo root. Exit 0 = PRD-level done. Runs ALL checks.
# Verifier fixes baked in: compose freeze checked against the BRANCH BASE
# (catches committed drift, not just working-tree drift); python3 everywhere.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
FAIL=0
check() {
  local name="$1"; shift
  echo ""
  echo "── $name"
  if bash -c "$1"; then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi
}

check "config-railway-sweep (zero railway.internal defaults in config.py)" \
  '! grep -n "railway.internal" orchestrator/config.py'

check "orchestrator-railway-sweep (zero railway.internal in any orchestrator python)" \
  '! grep -rn "railway.internal" orchestrator/ --include="*.py"'

check "services-railway-sweep (all three standalone automatos_logging.py copies clean)" \
  '! grep -rn "railway.internal" services/ --include="*.py"'

check "log-relay-default-off" \
  'grep -q "LOG_RELAY_ENABLED\", \"false\"" orchestrator/config.py'

check "mem0-local-default (compose DNS, matching envs/api.defaults)" \
  'grep -q "MEM0_API_URL\", \"http://mem0-server:8765\"" orchestrator/config.py'

check "config-test-suite (centralization suite green with new defaults)" \
  'cd orchestrator && python3 -m pytest tests/test_config_env_centralization.py -q --timeout=60'

check "mem0-degraded-lane (breaker, [] reads, L2 survival under new defaults)" \
  'cd orchestrator && python3 -m pytest tests/test_memory_degrades_when_mem0_down.py tests/test_mem0_circuit_breaker.py tests/test_mem0_async_client.py -q --timeout=60'

check "feature-off-guards (optional consumers no-op cleanly on empty URL)" \
  'cd orchestrator && python3 -m pytest tests/test_optional_services_feature_off.py -q --timeout=60'

check "compose-files-untouched (vs branch base AND working tree — PRD-153 owns compose)" \
  'BASE=$(git merge-base HEAD ralph/prd-151-storage-minio 2>/dev/null || git merge-base HEAD main); test -f infrastructure/docker-compose.memory.yml && git diff --quiet "$BASE" HEAD -- docker-compose.yml infrastructure/ && git diff --quiet HEAD -- docker-compose.yml infrastructure/'

check "mem0-arch-doc (key-wiring facts documented)" \
  'test -f docs/architecture/MEM0-INTERNAL-SERVICE.md && grep -q "OPENAI_API_KEY" docs/architecture/MEM0-INTERNAL-SERVICE.md'

echo ""
if [ $FAIL -eq 0 ]; then echo "ACCEPTANCE: PRD-152 PASS"; else echo "ACCEPTANCE: PRD-152 FAIL"; fi
exit $FAIL
