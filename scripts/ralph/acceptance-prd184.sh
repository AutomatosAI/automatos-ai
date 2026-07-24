#!/bin/bash
# Acceptance gate — PRD-184 Kill-list (DELETE-NOW tier US-001..US-006 only).
# Run from the worktree repo root. Exit 0 = delete-now tier done + safe.
# Gates: (1) full orchestrator suite green (deletion broke no import), (2) each
# target is GONE + carries a source-grep guard, (3) SCOPE GUARD — the diff must
# NOT touch any held retire-tier file or another PRD's surface (that would be an
# over-reach on an unattended deletion run), (4) no node_modules / no os.getenv crept in.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
FAIL=0
check() {
  local name="$1"; shift
  echo ""; echo "── $name"
  if bash -c "$1"; then echo "   ✅ PASS: $name"; else echo "   ❌ FAIL: $name"; FAIL=1; fi
}

# --- Primary gate: full suite green (no deletion broke a live import) --------------
check "orchestrator-full-suite green (@integration skips with no DB)" \
  'cd orchestrator && python3 -m pytest --timeout=90 --timeout-method=thread -o faulthandler_timeout=120 -p no:cacheprovider -q'

# --- US-001: learning/evaluation gone + guarded ----------------------------------
check "US-001 modules/evaluation deleted" "[ ! -e orchestrator/modules/evaluation ]"
# learning/__init__.py + playbooks/ SURVIVE BY DESIGN — they are the live reachability
# chain for the held-S10 PlaybookMiner (`from modules.learning import PlaybookMiner` in the
# held, out-of-scope api/api_playbooks.py). The deletion gate forbids breaking that live
# caller, so US-001 razes only the DEAD theatre subpackages (feedback/ + patterns/: empty
# __init__, zero importers). The in-suite guard test_held_playbook_miner_chain_survives pins
# this boundary. (The earlier `[ ! -e learning/__init__.py ]` check demanded a deletion-gate
# violation and matched docstring-prose Usage examples — corrected to the true dead theatre.)
check "US-001 dead-theatre learning subpkgs deleted (feedback/ + patterns/; __init__/playbooks held for PlaybookMiner)" \
  "[ ! -e orchestrator/modules/learning/feedback ] && [ ! -e orchestrator/modules/learning/patterns ]"
check "US-001 guard test present" \
  "[ -n \"\$(grep -rlE 'def test_no_learning_evaluation_imports' orchestrator/tests 2>/dev/null)\" ]"
# Precise dotted-subpackage tokens (mirror the guard test's _GONE_TOKENS): a bare
# `modules.learning` would false-positive on the surviving held barrel + its docstring prose.
check "US-001 no live import of the DELETED packages (modules.evaluation / learning.feedback / learning.patterns)" \
  "! git grep -nE 'modules\\.evaluation|modules\\.learning\\.feedback|modules\\.learning\\.patterns' -- orchestrator ':!orchestrator/tests' 2>/dev/null | grep -q ."

# --- US-002: llm-core dead scaffolding gone + guarded ----------------------------
check "US-002 the 6 llm-core files deleted (incl anthropic_client.py)" \
  "[ ! -e orchestrator/core/llm/function_executor.py ] && [ ! -e orchestrator/core/llm/function_registry.py ] && [ ! -e orchestrator/core/llm/response_parser.py ] && [ ! -e orchestrator/core/llm/semantic_skill_matcher.py ] && [ ! -e orchestrator/core/global_function_registry.py ] && [ ! -e orchestrator/api/anthropic_client.py ]"
check "US-002 guard test present" \
  "[ -n \"\$(grep -rlE 'def test_llm_core_no_dead_scaffolding' orchestrator/tests 2>/dev/null)\" ]"

# --- US-003: exec_planning gone + de-routed --------------------------------------
check "US-003 exec_planning.py deleted" "[ ! -e orchestrator/modules/tools/execution/exec_planning.py ]"
check "US-003 unified_executor no longer dispatches exec_planning" \
  "! grep -qE 'exec_planning' orchestrator/modules/tools/execution/unified_executor.py"
check "US-003 guard test present" \
  "[ -n \"\$(grep -rlE 'def test_exec_planning_deleted_and_unrouted' orchestrator/tests 2>/dev/null)\" ]"

# --- US-004: concurrency helper gone + guarded (ToolService best-effort) ----------
check "US-004 concurrency.py deleted" "[ ! -e orchestrator/modules/tools/execution/concurrency.py ]"
check "US-004 guard test present" \
  "[ -n \"\$(grep -rlE 'def test_no_tools_concurrency_import' orchestrator/tests 2>/dev/null)\" ]"
check "US-004 composio_tool_router.py FILE still present (live — only a dead delegate may be excised)" \
  "[ -e orchestrator/modules/tools/execution/composio_tool_router.py ] || [ -e orchestrator/modules/tools/composio_tool_router.py ]"

# --- US-005: legacy channel adapters gone + guarded ------------------------------
check "US-005 the 7 legacy channel adapters deleted" \
  "! ls orchestrator/channels/{teams,google_chat,signal,imessage,irc,matrix,line}_adapter.py >/dev/null 2>&1"
check "US-005 _ping_platform_legacy removed" \
  "! grep -qE '_ping_platform_legacy' orchestrator/api/channels.py 2>/dev/null"
check "US-005 guard test present" \
  "[ -n \"\$(grep -rlE 'def test_no_legacy_channel_adapters' orchestrator/tests 2>/dev/null)\" ]"

# --- US-006: frontend placebo relics gone + guarded ------------------------------
check "US-006 /api-control and /styleguide deleted" \
  "[ ! -e frontend/app/api-control ] && [ ! -e frontend/app/styleguide ]"
check "US-006 workspaceMeta pill literal removed" \
  "! grep -rqE 'pilot · 11 op' frontend/components/layout/studio-sidebar.tsx 2>/dev/null"
check "US-006 guard test present" \
  "[ -n \"\$(grep -rlE 'test_no_placebo_routes' orchestrator/tests frontend 2>/dev/null)\" ]"

# --- SCOPE GUARD: no held retire-tier / other-PRD surface touched ----------------
echo ""; echo "── scope guard: the run touched ONLY delete-now surface"
BASE=$(git merge-base HEAD origin/main 2>/dev/null || git merge-base HEAD main 2>/dev/null || echo "")
if [ -n "$BASE" ]; then
  OVERREACH=$(git diff --name-only "$BASE"..HEAD 2>/dev/null | grep -nE \
    'modules/learning/playbooks/miner\.py|api/api_playbooks\.py|api/workflows\.py|api/workflow_templates\.py|app/chat/\[id\]|package-lock\.json|pnpm-lock\.yaml|yarn\.lock|mem0_openapi|probe_mem0_endpoints|seed_mem0_user|node_modules|alembic/versions' || true)
  if [ -n "$OVERREACH" ]; then
    echo "   ❌ FAIL: touched held/other-PRD surface (retire-tier / lockfiles / mem0 / migration / node_modules):"; echo "$OVERREACH"; FAIL=1
  else
    echo "   ✅ PASS: no held retire-tier, lockfile, mem0, migration, or node_modules path in the diff"
  fi
else
  echo "   ⚠️  could not compute base — skipping scope guard"
fi

# --- convention guard: no os.getenv crept in outside config.py -------------------
check "no os.getenv added outside config.py in the diff" \
  "[ -z \"\$(git diff \$BASE..HEAD -- orchestrator ':!orchestrator/config.py' 2>/dev/null | grep -E '^\\+' | grep -E 'os\\.getenv')\" ]"

echo ""
if [ $FAIL -eq 0 ]; then echo "ACCEPTANCE: PRD-184 delete-now PASS"; else echo "ACCEPTANCE: PRD-184 FAIL"; fi
echo "NOTE: retire tier (S9 workflow-engine, S10 miner+/chat/[id]), /execute (S7), KG (S8) and the 4 decide-then-cut items are Gerard's — intentionally NOT in this run."
exit $FAIL
