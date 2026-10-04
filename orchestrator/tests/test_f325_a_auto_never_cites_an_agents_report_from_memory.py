"""F325 (night 9b): Auto never gives an agent's report as the source of the owner's facts
from its memory.

Chat 99a25490: "Where did that come from?" → platform_search_memory → "I found that
information in my memory, which was stored from a document titled
'2026-10-04_125617_e5c675_task-payment-terms-for-cafés.md'" (an agent's report, #1547).
Night 9's chat 9a128fe3 had cited that file and the exchange was kept as a memory. A
memory that names a document an agent wrote is now left out of the memory tools and of
the memories put into a prompt (modules/memory/agent_sources).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

OWNERS = "Quay Coffee House moves to 30-day payment terms from November."
OWNERS_PAPER = "The wholesale terms are in wholesale-terms-2026.md."
CITED = "The café payment terms came from '2026-10-04_125617_e5c675_task-payment-terms-for-cafés.md'."
BY_NAME = "Quay's terms are checked in quay-coffee-house-terms-check.md."
BY_TITLE = "The retail takings figure is from task-retail-takings-in-september."
MISSION = "Christmas boxes need 30 kg, says mission-output-christmas-boxes.md."


class _Memories:
    """The memory service: the workspace's memories, and each agent's."""

    is_durable_configured = True

    def __init__(self, by_agent):
        self.by_agent = by_agent

    async def search_long_term(self, workspace_id, query, agent_id=None, limit=10):
        return [{"memory": text, "score": 0.9, "created_at": "2026-10-04"} for text in self.by_agent.get(agent_id, [])]

    async def get_all_memories(self, workspace_id, agent_id=None, limit=20):
        return await self.search_long_term(workspace_id, "", agent_id, limit)


@pytest.fixture
def shop(monkeypatch, db_session, seed_workspace):
    """A workspace with one agent, the owner's paper and two documents agents wrote."""
    from core.models import Agent
    from core.models.core import Document
    from modules.memory import unified_memory_service

    ws = UUID(seed_workspace())
    analyst = Agent(name="Analyst", agent_type="custom", description="", status="active", configuration={},
                    workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws))
    db_session.add(analyst)
    for filename, source_type in (("wholesale-terms-2026.md", "upload"),
                                  ("quay-coffee-house-terms-check.md", "agent_output"),
                                  ("2026-10-04_130140_bf90d7_task-retail-takings-in-september.md", "agent_output")):
        db_session.add(Document(filename=filename, workspace_id=ws, status="completed", source_type=source_type))
    db_session.flush()
    memories = _Memories({None: [OWNERS, CITED, BY_NAME, OWNERS_PAPER, BY_TITLE], analyst.id: [MISSION]})
    monkeypatch.setattr(unified_memory_service, "get_unified_memory_service", lambda: memories)
    return NS(db=db_session, ws=ws)


def test_autos_memory_search_leaves_out_what_names_an_agents_report(shop):
    from modules.tools.discovery.handlers_search import search_memory

    found = asyncio.run(search_memory(shop.db, shop.ws, {"query": "Quay Coffee House payment terms"}))

    assert found["success"] is True
    assert [r["memory"] for r in found["results"]] == [OWNERS, OWNERS_PAPER]       # the owner's, as they were
    assert found["total"] == 2 and found["global_count"] == 2 and found["agent_count"] == 0
    assert "task-payment-terms" not in found["formatted"] and "mission-output" not in found["formatted"]
    assert found["left_out"] == 4
    assert "never the owner's facts" in found["left_out_note"] and found["left_out_note"] in found["formatted"]


def test_browsing_memories_leaves_them_out_too(shop):
    from modules.tools.discovery.handlers_search import browse_memories

    found = asyncio.run(browse_memories(shop.db, shop.ws, {"query": "terms"}))

    assert [m["content"] for m in found["memories"]] == [OWNERS, OWNERS_PAPER]
    assert found["total"] == 2 and found["left_out"] == 3


def test_a_memory_put_into_a_prompt_never_names_an_agents_report():
    from modules.memory.injection_filter import filter_injectable_memories

    recalled = [{"memory": CITED, "score": 0.9}, {"memory": OWNERS, "score": 0.9},
                {"content": MISSION, "score": 0.9}, {"memory": OWNERS_PAPER, "score": 0.9}]

    kept = filter_injectable_memories(recalled, floor=0.3)

    assert [m.get("memory") for m in kept] == [OWNERS, OWNERS_PAPER]
    assert len(recalled) == 4                                                   # a new list, not changed in place


def test_without_a_database_session_only_the_filed_names_are_known():
    from modules.memory.agent_sources import agents_document_names, owners_memories

    assert agents_document_names(None, "ws") == frozenset()
    kept = owners_memories([{"memory": BY_NAME}, {"memory": CITED}, "not a memory"])
    assert kept == [{"memory": BY_NAME}]
