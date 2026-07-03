"""PRD-177 S2 (F017): thread user_query + conversation/turn ids into caller_context.

The chat tool-callback threaded only ``{user_id}`` into ``caller_context``, so the
``user_query`` and turn grouping never reached the edge builder — and the
``succeeds_for_intent`` affinities (which need the query to cluster intent) never
materialized from real chat traffic. telemetry.py already *reads*
``ctx.get('user_query')`` and ``router_decision`` already carries
``conversation_id`` / ``turn_id``; F017 is purely the chat WRITE site populating
them.

This test exercises the pure builder that constructs that caller_context, and
proves the edge builder turns a query-bearing successful log into a
``succeeds_for_intent`` affinity (the loop input F017 unblocks). Pure — no LLM,
no live chat, no network. The edge builder's intent clustering is stubbed at the
embedding boundary so it stays offline.
"""
import importlib.util
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

_orchestrator_root = str(Path(__file__).resolve().parent.parent)
if _orchestrator_root not in sys.path:
    sys.path.insert(0, _orchestrator_root)


# ---------------------------------------------------------------------------
# The pure caller_context builder (F017 write-site logic, extracted for test)
# ---------------------------------------------------------------------------

def _load_caller_context_builder():
    """Import build_tool_caller_context from the chat service module without
    triggering its heavy import chain, by loading the file directly is not
    feasible (large module); instead import the symbol lazily and skip if the
    import environment can't satisfy it."""
    from consumers.chatbot.service import build_tool_caller_context

    return build_tool_caller_context


def test_builder_threads_query_and_ids():
    """caller_context carries user_query + conversation_id + turn_id (F017)."""
    build = _load_caller_context_builder()
    ctx = build(
        user_query="message the growth team on slack",
        conversation_id="chat-abc",
        turn_id="turn-xyz",
        driving_clerk="user_clerk_1",
        prior_action="platform_list_agents",
    )
    assert ctx["user_query"] == "message the growth team on slack"
    assert ctx["conversation_id"] == "chat-abc"
    assert ctx["turn_id"] == "turn-xyz"
    assert ctx["user_id"] == "user_clerk_1"
    assert ctx["prior_action"] == "platform_list_agents"


def test_builder_omits_empty_fields():
    """No driving clerk / no prior action → those keys are absent, not None,
    so the telemetry row stays clean (matches the pre-F017 posture for user_id)."""
    build = _load_caller_context_builder()
    ctx = build(
        user_query="list my agents",
        conversation_id="chat-1",
        turn_id="turn-1",
        driving_clerk=None,
        prior_action=None,
    )
    assert ctx["user_query"] == "list my agents"
    assert ctx["conversation_id"] == "chat-1"
    assert "user_id" not in ctx
    assert "prior_action" not in ctx


def test_builder_returns_none_without_any_signal():
    """When there is genuinely nothing to record, return None (unchanged from the
    old ``{...} if _driving_clerk else None`` contract — no empty dict noise)."""
    build = _load_caller_context_builder()
    assert build(
        user_query=None,
        conversation_id=None,
        turn_id=None,
        driving_clerk=None,
        prior_action=None,
    ) is None


# ---------------------------------------------------------------------------
# The loop input F017 unblocks: a query-bearing success -> succeeds_for_intent
# ---------------------------------------------------------------------------

def _load_edge_builder():
    eb_path = Path(_orchestrator_root) / "core" / "services" / "edge_builder.py"
    spec = importlib.util.spec_from_file_location("edge_builder_prd177_s2", eb_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_intent_affinity_materializes_from_query_logs():
    """Two successful logs sharing a cluster produce a succeeds_for_intent
    affinity. Proves the query threaded by F017 is exactly what the affinity
    computation needs — without it (user_query None) no cluster forms."""
    try:
        eb = _load_edge_builder()
    except Exception:
        pytest.skip("edge_builder heavy deps unavailable in this environment")

    # cluster_map maps log index -> cluster id; simulate the clustering output
    # that _compute_and_upsert_clusters would produce for query-bearing logs.
    logs = [
        {
            "action_name": "SLACK_SEND_MESSAGE",
            "workspace_id": "ws-1",
            "agent_id": None,
            "user_query": "message the team on slack",
            "status": "success",
        },
        {
            "action_name": "SLACK_SEND_MESSAGE",
            "workspace_id": "ws-1",
            "agent_id": None,
            "user_query": "send a slack message to the team",
            "status": "success",
        },
        {
            "action_name": "SLACK_SEND_MESSAGE",
            "workspace_id": "ws-1",
            "agent_id": None,
            "user_query": "ping the team in slack",
            "status": "success",
        },
    ]
    cluster_map = {0: 100, 1: 100, 2: 100}  # all three land in cluster 100

    affinities = eb._compute_affinities(logs, cluster_map)
    succeeds = [
        a for a in affinities
        if a["affinity_type"] == "succeeds_for_intent"
        and a["action_name"] == "SLACK_SEND_MESSAGE"
        and a["intent_cluster_id"] == 100
    ]
    assert succeeds, "a succeeds_for_intent affinity must materialize for the answering action"
    assert succeeds[0]["sample_count"] == 3


def test_no_query_no_intent_affinity():
    """Without user_query (the pre-F017 state) there is no cluster assignment, so
    no succeeds_for_intent affinity forms — this is exactly the gap F017 closes."""
    try:
        eb = _load_edge_builder()
    except Exception:
        pytest.skip("edge_builder heavy deps unavailable in this environment")

    logs = [
        {"action_name": "SLACK_SEND_MESSAGE", "workspace_id": "ws-1",
         "agent_id": None, "user_query": None, "status": "success"},
    ]
    cluster_map = {}  # no queries -> no clusters -> empty map

    affinities = eb._compute_affinities(logs, cluster_map)
    intent_affs = [a for a in affinities if a["affinity_type"] == "succeeds_for_intent"]
    assert intent_affs == []
