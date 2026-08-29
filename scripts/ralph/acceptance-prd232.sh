#!/bin/bash
# Acceptance gate — PRD-232 The Intent Graph (US-001..011, 014, 012A).
# Base: fix/main-ci-wave-drift (#643). Run from the worktree repo root.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
FAIL=0
STR="orchestrator/consumers/chatbot/smart_tool_router.py"
IDX="orchestrator/modules/tools/discovery/action_semantic_index.py"
GRP="orchestrator/modules/tools/discovery/graph_router.py"
REC="orchestrator/modules/tools/discovery/signal_recorder.py"
TRT="orchestrator/modules/tools/tool_router.py"
CFG="orchestrator/config.py"
UTT="orchestrator/core/seeds/utterances"
BASE_BR="fix/main-ci-wave-drift"
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
  if ( cd orchestrator && python3 -m pytest --timeout=120 --timeout-method=thread -o faulthandler_timeout=150 -p no:cacheprovider -q $CHT ); then
    echo "   ✅ PASS: branch-scoped tests: $CHT"
  else
    echo "   ❌ FAIL: branch-scoped tests: $CHT"; FAIL=1
  fi
fi

# ── RVW-5: regression-prone UNCHANGED-file suites ────────────────────────────
# The changed-file net above cannot see a regression in a test file THIS branch
# did not touch — the exact trap that hid the P232-RVW-1..3 required-CI reds
# (test_prd143_graph_seed is unchanged on this branch, so the diff-scoped net
# skips it while the required orchestrator-tests job is RED on it). Run the four
# promoted/enum/head-guard contract suites explicitly so an unchanged-file
# regression cannot pass acceptance green (RVW-5 AC option b, network-free).
echo ""
echo "── RVW-5 regression-prone suites (unchanged-file guard for the required-CI contract)"
RVW5_SUITES="tests/test_prd143_su_surface.py tests/test_prd143_selection_at_scale.py tests/test_prd143_graph_seed.py tests/test_prd225_asks_model.py"
if ( cd orchestrator && python3 -m pytest --timeout=120 --timeout-method=thread -o faulthandler_timeout=150 -p no:cacheprovider -q $RVW5_SUITES ); then
  echo "   ✅ PASS: regression-prone suites green: $RVW5_SUITES"
else
  echo "   ❌ FAIL: unchanged-file regression on the required-CI contract (RVW-2 §12 block still open): $RVW5_SUITES"; FAIL=1
fi

# ── #643 files untouched; migrations/manifest frozen ─────────────────────────
echo ""
echo "── #643 files untouched; zero migrations; manifest frozen"
if [ -n "$BASEP" ]; then
  git diff --name-only "$BASEP"..HEAD -- orchestrator/tests/test_prd222_w2s1_plan_tiers.py orchestrator/tests/authz_sweep_probe.py | grep -q . \
    && { echo "   ❌ FAIL: #643's files modified"; FAIL=1; } || echo "   ✅ PASS: #643 files untouched"
  NEWMIG_FILES=$(git diff --name-only --diff-filter=A "$BASEP"..HEAD -- orchestrator/alembic/versions/ 2>/dev/null)
  NEWMIG=$(echo "$NEWMIG_FILES" | grep -c . | tr -d ' ')
  if [ "$NEWMIG" = "0" ]; then echo "   ✅ PASS: no new migrations (US-007 revision optional-but-expected)";
  elif [ "$NEWMIG" = "1" ] && echo "$NEWMIG_FILES" | grep -q "prd232.*provenance"; then echo "   ✅ PASS: exactly the authorized US-007 provenance revision";
  else echo "   ❌ FAIL: unauthorized migrations: $NEWMIG_FILES"; FAIL=1; fi
  git diff "$BASEP"..HEAD -- orchestrator/reports/route-manifest.json | grep -q . \
    && { echo "   ❌ FAIL: route manifest changed"; FAIL=1; } || echo "   ✅ PASS: route manifest untouched"
fi

# ── US-001/002: dispatcher survival + flag split ─────────────────────────────
check "US-001 dispatcher-survival test exists and names platform_execute" \
  "grep -rln 'platform_execute' orchestrator/tests --include='*prd232*' | grep -q ."
check "US-002 graph call gated on TOOL_ROUTING_GRAPH in smart_tool_router" \
  "grep -n 'TOOL_ROUTING_GRAPH' $STR | grep -q ."
check "US-002 no graph call under bare SEMANTIC_TOOL_ROUTING gate" \
  "! grep -B2 'rank_chains' $STR | grep -q 'SEMANTIC_TOOL_ROUTING'"

# ── US-005/006: corpus + embeddings ──────────────────────────────────────────
check "US-005 corpus directory exists with YAML files" \
  "[ -d $UTT ] && ls $UTT/*.yaml >/dev/null 2>&1"
check "US-005 update_task_status carries close/ticket/blocked vocabulary" \
  "grep -rliq 'close' $UTT && grep -rliq 'ticket' $UTT && grep -rliq 'blocked' $UTT"
check "US-006 embedding text builder references utterances" \
  "grep -qi 'utterance' $IDX"
check "US-006 no DeterministicEmbeddingProvider outside fixtures (branch-scope; embedding_manager's PRE-EXISTING PRD-185 fallback sites are out of 232 scope)" \
  "! grep -rn 'DeterministicEmbeddingProvider' orchestrator --include='*.py' | grep -v tests | grep -v 'base.py' |  grep -v 'core/llm/embedding_manager.py' | grep -v 'core/llm/clients/__init__.py' | grep -q ."

# ── US-007/009/010/011: seeding + learning ───────────────────────────────────
check "US-007 seed script gained an utterances step (still --yes gated)" \
  "grep -qi 'utterance' orchestrator/scripts/seed_tool_routing_graph.py && grep -q -- '--yes' orchestrator/scripts/seed_tool_routing_graph.py"
check "US-009 recorder default ON" \
  "grep -n 'TOOL_SIGNAL_RECORDER_ENABLED' $CFG | grep -qi '\"true\"'"
check "US-010 affinity reads carry intent_cluster predicate" \
  "grep -q 'intent_cluster' $GRP"
check "US-011 gap signal exists (tool_gap marker)" \
  "grep -rqi 'tool_gap' orchestrator --include='*.py'"

# ── US-014: promotion-as-prior ───────────────────────────────────────────────
check "US-014 pins list lives in config" \
  "grep -qi 'PIN' $CFG"
check "US-014 first-class attach no longer unconditional-all-promoted" \
  "! grep -n 'to_first_class_schemas' $TRT | grep -q 'get_promoted()'"

# ── US-012A: eval prep ───────────────────────────────────────────────────────
check "US-012A abstain rows present in eval set" \
  "grep -qi 'abstain' orchestrator/scripts/eval/tool_routing/eval_set.jsonl 2>/dev/null || grep -rqi 'abstain' orchestrator/scripts/eval/tool_routing/"

# ── convention: flags Ralph must NOT flip ────────────────────────────────────
check "TOOL_ROUTING_GRAPH default still false (flip is human, post-eval)" \
  "grep -n 'TOOL_ROUTING_GRAPH:' $CFG | head -1 | grep -qi '\"false\"'"

echo ""
if [ "$FAIL" = "0" ]; then echo "ACCEPTANCE: PASS (PRD-232)"; exit 0; else echo "ACCEPTANCE: FAIL (PRD-232)"; exit 1; fi
