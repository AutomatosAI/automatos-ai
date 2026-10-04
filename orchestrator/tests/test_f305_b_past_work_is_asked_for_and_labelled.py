"""F305 (night 9): with agent outputs out of the owner's documents, an agent finds
earlier work only by asking for it (``scope: "past_work"``), and every result says
who wrote it and when, never the owner's facts. The default search returns none of it.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

from services import past_work

ANSWER = "Cafés pay on 30-day terms; Quay Coffee House pays in 14 days."


@pytest.fixture
def board(db_session, seed_workspace):
    from core.models import Agent
    from core.models.core import BoardTask

    ws = UUID(seed_workspace())
    writer = Agent(name="Ledger Clerk", agent_type="custom", description="", status="active", configuration={},
                   workspace_id=ws, created_by="test", owner_type="workspace", owner_id=str(ws))
    db_session.add(writer)
    db_session.flush()

    def card(title, status, result=ANSWER, workspace=ws):
        made = BoardTask(workspace_id=workspace, title=title, status=status, priority="low", result=result,
                         assigned_agent_id=writer.id, completed_at=datetime(2026, 10, 4, 12, 56, tzinfo=timezone.utc))
        db_session.add(made)
        db_session.flush()
        return made

    return NS(db=db_session, ws=ws, writer=writer, card=card)


def test_past_work_comes_back_labelled_with_who_and_when(board):
    board.card("Payment terms for cafés", "done")
    board.card("Payment terms draft", "review")                  # not approved: never past work

    got = past_work.search_past_work(board.db, board.ws, "payment terms for cafés")

    assert got["success"] is True and got["count"] == 1
    item = got["results"][0]
    assert item["label"].startswith("Written by Ledger Clerk on 2026-10-04")
    assert item["owners_facts"] is False and "not the owner's facts" in got["note"]
    assert item["excerpt"] == ANSWER


def test_another_workspaces_cards_are_never_past_work(board, seed_workspace):
    board.card("Payment terms for cafés", "done", workspace=UUID(seed_workspace()))

    assert past_work.search_past_work(board.db, board.ws, "payment terms")["count"] == 0


def test_search_knowledge_gives_past_work_only_when_asked(board):
    from modules.tools.execution import exec_platform

    board.card("Payment terms for cafés", "done")
    searched = []

    async def execute_tool(**kwargs):
        searched.append(kwargs["parameters"])
        return {"success": True, "results": []}                     # the owner's documents hold nothing

    executor = NS(platform_tools=NS(db=board.db, execute_tool=execute_tool))
    plain = asyncio.run(exec_platform.execute_platform_tool(
        executor, "search_knowledge", {"query": "payment terms for cafés"}, board.writer.id))
    asked = asyncio.run(exec_platform.execute_platform_tool(
        executor, "search_knowledge", {"query": "payment terms for cafés", "scope": "past_work"}, board.writer.id))

    assert plain["results"] == []                                    # the default search: none of it
    assert [r["written_by"] for r in asked["results"]] == ["Ledger Clerk"]
    assert len(searched) == 1                                        # past work never searched the documents


def test_platform_search_documents_takes_the_scope(board):
    from modules.tools.discovery.platform_executor import PLATFORM_HANDLERS

    board.card("Payment terms for cafés", "done")
    got = asyncio.run(PLATFORM_HANDLERS["platform_search_documents"](
        board.db, board.ws, {"query": "payment terms", "scope": "past_work"}))

    assert got["scope"] == "past_work" and got["results"][0]["written_on"] == "2026-10-04"


def test_every_search_schema_declares_the_scope():
    from modules.agents.services.agent_platform_tools import AgentPlatformTools
    from modules.tools.discovery.action_registry import ActionRegistry
    from modules.tools.discovery.actions_documents import register_documents_actions
    from modules.tools.registry.tool_registry import ToolRegistry

    chat = AgentPlatformTools.get_available_tools(NS())
    chat_search = next(t for t in chat if t["name"] == "search_knowledge")
    assert chat_search["parameters"]["properties"]["scope"]["enum"] == ["past_work"]

    registry = ActionRegistry()
    register_documents_actions(registry)
    found = registry._actions["platform_search_documents"].parameters["properties"]
    assert found["scope"]["enum"] == ["past_work"] and "query" in found

    spec = ToolRegistry().tools["search_knowledge"]
    assert [p.enum for p in spec.parameters if p.name == "scope"] == [["past_work"]]
