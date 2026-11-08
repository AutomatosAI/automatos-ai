"""#1430 Wave 0 (Automatos for products): a chat turn's long-term recall reads the workspace's facts.

Distillation files an exchange's durable facts in the WORKSPACE namespace (``mem:{ws}``):
PRD-159 S5 retired the double write to the workspace and the agent namespaces, keeping one
row per fact, filed globally unless the owner names the agent ("for this agent"). The
Context Router's long-term fetch still asked for the turn's agent namespace only
(``mem:{ws}:agent:{id}``): every chat turn has an agent (Auto included), so a chat turn
never recalled a distilled fact. Found when a new-session "catch me up" injected ~0 memory
tokens with 46 facts stored.

Now the router's fetch reads the workspace's facts and the agent's own; nothing else
changes: a search that does not ask for the workspace (the widget lane, F155) stays on
the agent's namespace, and another agent's or another workspace's facts never come back.
The real DurableMemoryStore over an in-memory Qdrant with a fixed score.
"""
from __future__ import annotations

import asyncio
import uuid
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock

import pytest

WS, OTHER_WS = str(uuid.UUID(int=1)), str(uuid.UUID(int=2))
AUTO, OTHER_AGENT = 21, 22

WORKSPACE_FACT = "Old Weir sold the chapel's Vigil Bell to Corvus Marle to mend the roof."
AGENT_FACT = "Auto answers the owner in short paragraphs."
OTHER_AGENT_FACT = "The brand designer prefers teal."
OTHER_WS_FACT = "Another business closes on Fridays."


def _matches(payload, condition):
    value, match = payload.get(condition.key), condition.match
    return value in match.any if hasattr(match, "any") else value == match.value


class _Qdrant:
    """query_points over the payload filter (MatchValue and MatchAny), every hit scoring 0.9."""

    def __init__(self) -> None:
        self.points: list = []
        self.filters: list = []

    async def query_points(self, collection_name, query, query_filter, limit, **_kwargs):
        self.filters.append(query_filter)
        hits = [p for p in self.points if all(_matches(p.payload, c) for c in query_filter.must or [])]
        return NS(points=[NS(id=p.id, payload=p.payload, score=0.9) for p in hits[:limit]])

    def seed(self, content, namespace, workspace=WS):
        self.points.append(NS(id=str(uuid.uuid4()), payload={
            "namespace": namespace, "workspace_id": workspace, "content": content,
            "metadata": {"type": "decision", "importance": 0.8},
            "created_at": "2026-11-08T20:00:00+00:00",
        }))


@pytest.fixture
def memory(monkeypatch):
    import modules.memory.unified_memory_service as ums
    from config import config
    from modules.memory.durable_store import DurableMemoryStore

    client = _Qdrant()
    store = DurableMemoryStore.__new__(DurableMemoryStore)
    store._enabled, store._bootstrap_done, store._collection = True, True, "durable_memory"
    store._client = client
    store._embedder = NS(generate_embedding=AsyncMock(return_value=[0.0]))
    service = ums.UnifiedMemoryService.__new__(ums.UnifiedMemoryService)
    service._durable = store
    service._redis_client_getter = lambda: None
    # The router's other layers have nothing to say here: this is about long-term recall.
    service.search_short_term = AsyncMock(return_value=[])
    service.search_short_term_semantic = AsyncMock(return_value=[])
    service.get_all_daily_logs = AsyncMock(return_value=[])
    monkeypatch.setattr(ums, "get_unified_memory_service", lambda: service)
    monkeypatch.setattr(config, "QDRANT_URL", "http://qdrant.test")

    client.seed(WORKSPACE_FACT, f"mem:{WS}")
    client.seed(AGENT_FACT, f"mem:{WS}:agent:{AUTO}")
    client.seed(OTHER_AGENT_FACT, f"mem:{WS}:agent:{OTHER_AGENT}")
    client.seed(OTHER_WS_FACT, f"mem:{OTHER_WS}", workspace=OTHER_WS)
    return service, client


def _texts(rows):
    return {r.get("memory") for r in rows}


# ── the service ──────────────────────────────────────────────────────────────────


def test_an_agents_search_that_asks_for_the_workspace_reads_both_namespaces(memory):
    service, _ = memory
    rows = asyncio.run(service.search_long_term(WS, "the chapel bell", agent_id=AUTO, include_workspace=True))
    assert _texts(rows) == {WORKSPACE_FACT, AGENT_FACT}


def test_an_agents_search_that_does_not_ask_stays_on_its_own_namespace(memory):
    """The widget lane (F155) and every other caller keep today's agent-only read."""
    service, _ = memory
    rows = asyncio.run(service.search_long_term(WS, "the chapel bell", agent_id=AUTO))
    assert _texts(rows) == {AGENT_FACT}


def test_a_search_with_no_agent_reads_the_workspace_namespace_only(memory):
    service, _ = memory
    rows = asyncio.run(service.search_long_term(WS, "the chapel bell", include_workspace=True))
    assert _texts(rows) == {WORKSPACE_FACT}


def test_one_query_one_embedding(memory):
    service, client = memory
    asyncio.run(service.search_long_term(WS, "the chapel bell", agent_id=AUTO, include_workspace=True))
    assert len(client.filters) == 1 and service._durable._embedder.generate_embedding.await_count == 1


# ── the chat turn: the Context Router ─────────────────────────────────────────────


@pytest.mark.parametrize("query", [
    "Who has the chapel bell now?",
    "We are starting a new session after last time. Catch me up on the story so far.",   # read as temporal
])
def test_a_chat_turn_recalls_the_workspaces_distilled_facts(memory, query):
    from modules.memory.context_router import ContextRouter

    bundle = asyncio.run(ContextRouter().retrieve_context(workspace_id=WS, agent_id=AUTO, query=query))
    recalled = _texts(bundle.long_term_memories)
    assert WORKSPACE_FACT in recalled and AGENT_FACT in recalled
    assert not recalled & {OTHER_AGENT_FACT, OTHER_WS_FACT}
