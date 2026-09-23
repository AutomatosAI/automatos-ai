"""F121 (run 4, reopened) — the turn's narrowing ranks only actions that can run here.

F078 made the registry leave an unavailable action out of the surfaces it
builds. The narrowing does not start there: the semantic and lexical rankers
take their candidates from ActionSemanticIndex._eligible_actions(), which read
registry.get_all() unfiltered. On the local stack, with PROMETHEUS_URL unset, a
cost/health question still ranked platform_query_prometheus and
platform_query_loki_logs into the narrowed list. Each took a top-K slot, and
the enum then dropped it. Eligibility now honours is_available(). The embedding
build still embeds every action, so one configured later ranks without a
re-index. The other readers of the full list leave it out too:
platform_list_tools, the graph router's chains, both unknown-name errors and the
prompt catalog.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest

from config import config
from modules.tools.discovery import action_semantic_index as asi
from modules.tools.discovery.action_registry import ActionDefinition, ActionRegistry, get_action_registry

PROMETHEUS, LOKI = "platform_query_prometheus", "platform_query_loki_logs"
MONITORING = {PROMETHEUS, LOKI}
ALERTS = "platform_get_alerts"          # reads our own DB: always runs
QUESTION = "What does the platform cost to run, is it healthy, and what are the error rate and latency?"
LOG_QUESTION = "Show me the error logs from the last hour"


@pytest.fixture
def unconfigured(monkeypatch):
    for key in ("PROMETHEUS_CONFIGURED", "LOKI_CONFIGURED"):
        monkeypatch.setattr(config, key, False, raising=False)
    for key in ("GRAFANA_URL", "GRAFANA_SERVICE_ACCOUNT_TOKEN", "RAILWAY_API_TOKEN", "RAILWAY_PROJECT_ID"):
        monkeypatch.setattr(config, key, "", raising=False)
    # the ranking dials at their legacy zero, so the order below is the cosine's
    for key in ("TOOL_ROUTING_PROMOTION_BOOST", "SEMANTIC_TOOL_ROUTING_FLOOR", "SEMANTIC_TOOL_ROUTING_FLOOR_RATIO"):
        monkeypatch.setattr(config, key, 0, raising=False)
    # the narrowed-enum mode: the enum is the narrowed list itself (cache-stable
    # mode publishes the same list as the turn's late system line instead)
    monkeypatch.setattr(config, "TOOL_ENUM_CACHE_STABLE", False, raising=False)
    yield monkeypatch
    from modules.tools.turn_narrowing import clear_narrowed_actions

    clear_narrowed_actions()


def _index(*, embed_times_out=False):
    """The real index over the real registry with its embeddings replaced: the
    monitoring actions and the alerts sit exactly on the question, everything
    else is orthogonal to it — so an eligible monitoring action ranks first."""
    index = asi.ActionSemanticIndex.__new__(asi.ActionSemanticIndex)
    index._registry = get_action_registry()
    index._action_embeddings = {
        a.name: [1.0, 0.0] if a.name in MONITORING | {ALERTS} else [0.0, 1.0]
        for a in index._registry.get_all()
    }

    async def indexed(**_kw):
        return None

    async def embed(query, model_key=None, timeout_s=None):
        return (None, False, True) if embed_times_out else ([1.0, 0.0], False, False)

    index.ensure_indexed = indexed
    index._embed_query_bounded = embed
    index._cache_model_key = lambda: "f121"
    index._embed_timeout_s = lambda: None
    return index


def _narrowed(monkeypatch, index, top_k=5, question=QUESTION):
    """The dispatcher's narrowing for a super-admin caller — the one path that
    ever sees these su-only actions."""
    from modules.tools import tool_router

    monkeypatch.setattr(asi, "get_action_semantic_index", lambda: index)
    return asyncio.run(tool_router._rank_actions_for_dispatcher_async(
        question, top_k=top_k, exclude_admin=False, exclude_promoted=False, include_super_admin=True))


def _enum(names):
    schema = get_action_registry().to_dispatcher_schema(
        exclude_admin=False, exclude_promoted=False, allowed_names=names, include_super_admin=True)
    return schema["function"]["parameters"]["properties"]["action"].get("enum", [])


# ── the narrowing: semantic, then the lexical shortlist ─────────────────────

def test_a_health_question_narrows_to_what_can_run(unconfigured):
    names = _narrowed(unconfigured, _index())
    assert names and names[0] == ALERTS
    assert not set(names) & MONITORING
    assert sorted(_enum(names)) == sorted(names)        # every narrowed slot is callable


def test_with_prometheus_configured_it_ranks_first_again(unconfigured):
    unconfigured.setattr(config, "PROMETHEUS_CONFIGURED", True)
    names = _narrowed(unconfigured, _index())
    assert set(names[:2]) == {PROMETHEUS, ALERTS} and LOKI not in names
    assert PROMETHEUS in _enum(names)


def test_the_lexical_shortlist_leaves_them_out_too(unconfigured):
    """The embed timed out: the shortlist is plain token overlap."""
    index = _index(embed_times_out=True)
    for question in (QUESTION, LOG_QUESTION):
        names = _narrowed(unconfigured, index, top_k=15, question=question)
        assert names and not set(names) & MONITORING, question
    unconfigured.setattr(config, "PROMETHEUS_CONFIGURED", True)
    assert PROMETHEUS in _narrowed(unconfigured, index, top_k=15)
    unconfigured.setattr(config, "LOKI_CONFIGURED", True)
    assert LOKI in _narrowed(unconfigured, index, top_k=15, question=LOG_QUESTION)


def test_the_embedding_build_still_covers_every_action(unconfigured):
    index = _index()
    ranked = {a.name for a in index._eligible_actions(False, False, include_super_admin=True)}
    embedded = {a.name for a in index._eligible_actions(False, False, include_super_admin=True,
                                                        available_only=False)}
    assert not ranked & MONITORING and MONITORING <= embedded


# ── the other readers of the full list ──────────────────────────────────────

def test_platform_list_tools_lists_only_what_can_run(unconfigured):
    from modules.tools.discovery.handlers_tools_llms import list_tools

    def listed():
        # The monitoring tier is super_admin_only, so the listing is a super
        # admin's: the flag the executor injects for one (F122).
        params = {"category": "platform", "_caller_is_super_admin": True}
        result = asyncio.run(list_tools(None, uuid4(), params))
        return {t["name"] for t in result["tools"]}

    assert ALERTS in listed() and not listed() & MONITORING
    unconfigured.setattr(config, "PROMETHEUS_CONFIGURED", True)
    assert PROMETHEUS in listed()


def test_the_graph_router_drops_a_chain_through_what_cannot_run(unconfigured):
    from modules.tools.discovery.graph_router import GraphRouter

    router = GraphRouter.__new__(GraphRouter)
    router._semantic_index = NS(_registry=get_action_registry())
    chains = [("health", 1.0, [ALERTS, PROMETHEUS]), ("alerts", 0.9, [ALERTS])]
    kept = router._drop_ineligible_chains(chains, exclude_admin=False, include_super_admin=True)
    assert [c[0] for c in kept] == ["alerts"]
    unconfigured.setattr(config, "PROMETHEUS_CONFIGURED", True)
    kept = router._drop_ineligible_chains(chains, exclude_admin=False, include_super_admin=True)
    assert [c[0] for c in kept] == ["health", "alerts"]


def _action(name, available=None):
    return ActionDefinition(name=name, description=name, category="t",
                            parameters={"type": "object", "properties": {}}, available=available)


def test_the_unknown_action_error_suggests_only_what_can_run():
    from modules.tools.execution.unified_executor import unknown_action_error

    registry = ActionRegistry()
    registry._initialized = True
    for action in (_action("platform_down", available=lambda: False), _action("platform_up")):
        registry.register(action)
    error = unknown_action_error("platform_nope", registry)
    assert "platform_up" in error and "platform_down" not in error


def test_the_unknown_tool_error_never_names_an_action_that_cannot_run(unconfigured):
    from modules.tools.execution.unified_executor import UnifiedToolExecutor

    executor = UnifiedToolExecutor.__new__(UnifiedToolExecutor)
    executor.tool_routes = {}
    executor._tool_registry = NS(get_all_tools=lambda: [])
    assert PROMETHEUS not in executor._unknown_tool_error("query_prometheus")
    unconfigured.setattr(config, "PROMETHEUS_CONFIGURED", True)
    assert PROMETHEUS in executor._unknown_tool_error("query_prometheus")


def test_the_prompt_catalog_never_describes_what_cannot_run():
    registry = ActionRegistry()
    registry._initialized = True
    for action in (_action("platform_down", available=lambda: False), _action("platform_up")):
        registry.register(action)
    for catalog in (registry.build_prompt_summary(),
                    registry.build_filtered_prompt_summary(["platform_down", "platform_up"])):
        assert "platform_up" in catalog and "platform_down" not in catalog
