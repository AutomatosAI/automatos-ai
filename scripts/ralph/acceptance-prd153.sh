#!/bin/bash
# Acceptance gate — PRD-153 One-Command Local Run (integration capstone).
# Run from the worktree repo root. Exit 0 = PRD-level done. Runs ALL checks.
# Verifier fixes baked in: LLM keys sourced from orchestrator/.env (root .env
# keys are empty), smoke bounded by timeout, venv-or-python3 guard tests.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
FAIL=0
TIMEOUT_BIN=$(command -v timeout || command -v gtimeout || true)
check() {
  local name="$1"; shift
  echo ""
  echo "── $name"
  if bash -c "$1"; then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi
}

# Smoke chat check needs a real LLM key; root .env values are EMPTY — pull
# from orchestrator/.env (copied into the worktree) without printing values.
if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ] && [ -f orchestrator/.env ]; then
  set -a
  eval "$(grep -E '^(ANTHROPIC_API_KEY|OPENAI_API_KEY)=.+' orchestrator/.env || true)"
  set +a
fi

check "infra-compose-folded (only landing compose may remain under infrastructure/)" \
  '[ -z "$(ls infrastructure/docker-compose*.yml 2>/dev/null | grep -v landing)" ]'

check "single-schema-lifecycle (SQL-dump init + raw migrations gone; no initdb mount)" \
  '[ ! -f orchestrator/core/database/init_complete_schema.sql ] && [ ! -d orchestrator/core/database/migrations ] && ! grep -q init_complete_schema docker-compose.yml && ! grep -q docker-entrypoint-initdb docker-compose.yml'

check "no-plural-heads-in-boot-path (single-head alembic from the shared entrypoint)" \
  '! grep -rn "alembic upgrade heads" orchestrator/Dockerfile orchestrator/docker-entrypoint.sh docker-compose.yml 2>/dev/null | grep -q . && grep -q "alembic upgrade head" orchestrator/docker-entrypoint.sh'

check "compose-renders-all-profiles (config -q across profile combos, no container_name)" \
  'bash scripts/compose-smoke.sh --config-only'

if [ -n "$TIMEOUT_BIN" ]; then
  check "isolated-boot-golden-path (clean boot on offset ports in -p automatos-smoke: health, frontend, no-Clerk identity, task create, chat answer, single alembic head, seed sentinel, scoped teardown; 45m cap)" \
    "\"$TIMEOUT_BIN\" --kill-after=60s 45m bash scripts/compose-smoke.sh"
else
  check "isolated-boot-golden-path (UNBOUNDED — install coreutils for timeout)" \
    'bash scripts/compose-smoke.sh'
fi

check "orchestrator-guard-tests (single-head guard + config validation, no DB needed)" \
  'cd orchestrator && if [ -d ../.venv ]; then source ../.venv/bin/activate; fi; python3 -m pytest tests/test_alembic_lifecycle.py tests/test_config_validation.py -q -m "not integration"'

check "quickstart-honest (real one-command flow documented; stale claims gone)" \
  'grep -q COMPOSE_PROFILES QUICKSTART.md && grep -q compose-smoke.sh QUICKSTART.md && ! grep -q "No .env file needed" QUICKSTART.md && ! grep -q automatos_dev_pass QUICKSTART.md'

echo ""
if [ $FAIL -eq 0 ]; then echo "ACCEPTANCE: PRD-153 PASS"; else echo "ACCEPTANCE: PRD-153 FAIL"; fi
exit $FAIL
