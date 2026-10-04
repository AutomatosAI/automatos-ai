"""F269 and F287 (night 8): what an agent wrote is never handed over as the owner's facts.

- F269: asked where "no Thursday deliveries" came from, Auto answered "your internal
  documents indicate that you generally avoid Thursday deliveries…", quoting #0214.1's
  own draft from twenty minutes earlier. Auto's retrieval-first passages now leave out
  documents an agent wrote (source_type agent_output).
- F287: the mission planner wrote "Friday 27 November from [2] is a good reference"
  into #0352's plan against the owner's "Thursday 10 December", [2] being an agent's
  document, and the email step used it. The planner's knowledge leaves them out too.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

DELIVERY_DAYS = "Wholesale deliveries go out on Wednesdays and Thursdays. Orders by noon the day before."
OLD_DRAFT = "Regarding your requested Thursday delivery, we are unable to accommodate this day."
LAST_ORDERS = "Christmas: the last day for orders is Thursday 10 December."
OLD_PLAN = "Gift boxes: orders close Friday 27 November."


@pytest.fixture
def documents(db_session, seed_workspace):
    """The owner's delivery sheet and Christmas notes, and two documents agents wrote."""
    from core.models.core import Document

    ws = UUID(seed_workspace())

    def document(filename, source_type=None):
        made = Document(filename=filename, workspace_id=ws, status="completed", source_type=source_type)
        db_session.add(made)
        db_session.flush()
        return made

    return NS(db=db_session, ws=ws, sheet=document("delivery-days.md", "upload"),
              draft=document("welcome-email-the-lantern-room.md", "agent_output"),
              christmas=document("christmas-orders.md"), plan=document("gift-box-plan.md", "agent_output"))


def _hit(document, content, similarity):
    return {"filename": document.filename, "source": document.filename, "title": document.filename,
            "similarity": similarity, "content": content, "excerpt": content, "document_id": document.id}


def _auto(monkeypatch, documents, results):
    from config import config
    from consumers.chatbot import knowledge_prefetch as kp
    from consumers.chatbot import service
    from consumers.chatbot.streaming import get_streaming_handler

    async def execute_and_format(**kw):
        return {"success": True, "raw_result": {"success": True, "results": list(results)}, "frontend_data": {}}

    monkeypatch.setattr(kp, "documents_in", lambda db, ws: 4)
    monkeypatch.setattr(kp, "_has_database", lambda db, ws: False)
    monkeypatch.setattr(type(config), "CHATBOT_KNOWLEDGE_PREFETCH", property(lambda self: True))
    monkeypatch.setattr(type(config), "KNOWLEDGE_PREFETCH_MIN_SCORE", property(lambda self: 0.3))
    svc = service.StreamingChatService.__new__(service.StreamingChatService)
    svc.db, svc.workspace_id, svc.streaming_handler = documents.db, documents.ws, get_streaming_handler()
    svc._turn_document_ids, svc._turn_chunk_ids = set(), set()
    svc.tool_router = NS(execute_and_format=execute_and_format)
    return svc


def test_autos_retrieval_first_leaves_out_what_an_agent_wrote(monkeypatch, documents):
    svc = _auto(monkeypatch, documents, [_hit(documents.draft, OLD_DRAFT, 0.91),
                                         _hit(documents.sheet, DELIVERY_DAYS, 0.74)])
    llm_messages, prefetched = [{"role": "system", "content": "You are Auto."}], []

    async def turn():
        return [frame async for frame in svc._retrieval_first(
            "Which day do we deliver to our wholesale cafes?", llm_messages, NS(agent_id=294), "chat-1", prefetched)]

    asyncio.run(turn())

    passages = llm_messages[-1]["content"]
    assert DELIVERY_DAYS in passages and "delivery-days.md" in passages
    assert OLD_DRAFT not in passages and "welcome-email-the-lantern-room.md" not in passages
    assert svc._turn_document_ids == {documents.sheet.id}           # the reply records only what it was given


def test_the_planners_knowledge_leaves_out_what_an_agent_wrote(monkeypatch, documents):
    from modules.context.sections.base import SectionContext
    from modules.context.sections.planning_knowledge import (
        KNOWLEDGE_HEADING,
        OWNERS_WORDS_WIN,
        PlanningKnowledgeSection,
    )
    from modules.rag import service as rag_service
    from modules.rag.budget import assemble_with_citations

    goal = "Christmas gift subscription email to the club"
    chunks = [{"content": LAST_ORDERS, "source_file": "christmas-orders.md", "similarity": 0.8, "tokens": 12,
               "document_id": documents.christmas.id, "metadata": {}},
              {"content": OLD_PLAN, "source_file": "gift-box-plan.md", "similarity": 0.9, "tokens": 10,
               "document_id": None, "metadata": {"external_file_id": str(documents.plan.id)}}]

    class _Rag:
        async def retrieve(self, **kwargs):
            context, sources_map = assemble_with_citations(chunks, kwargs["query"])
            return rag_service.RAGResult(chunks=chunks, formatted_context=context, total_tokens=22,
                                         sources=["christmas-orders.md", "gift-box-plan.md"], query=kwargs["query"],
                                         sources_map=sources_map)

    monkeypatch.setattr(rag_service, "get_rag_service", lambda: _Rag())
    ctx = SectionContext(agent=None, workspace_id=str(documents.ws), db_session=documents.db, task_description=goal)

    knowledge = asyncio.run(PlanningKnowledgeSection()._build(ctx))

    assert LAST_ORDERS in knowledge and "[1] (source: christmas-orders.md)" in knowledge
    assert OLD_PLAN not in knowledge and "gift-box-plan.md" not in knowledge and "[2]" not in knowledge
    assert knowledge.startswith(f"{KNOWLEDGE_HEADING}\n{OWNERS_WORDS_WIN}")      # the owner's date wins over any
    assert "the owner's words win" in OWNERS_WORDS_WIN


def test_without_a_database_session_the_passages_are_as_they_were():
    from services.draft_guides import owners_own

    result = {"raw_result": {"results": [{"document_id": 7, "content": OLD_DRAFT}]}}
    assert owners_own(object(), result, "ws") is result and owners_own(None, result) is result


def _searcher(documents):
    from core.models import Agent

    agent = Agent(name="Auto", agent_type="custom", description="", status="active", configuration={},
                  workspace_id=documents.ws, created_by="test", owner_type="workspace", owner_id=str(documents.ws))
    documents.db.add(agent)
    documents.db.flush()
    return agent


def test_autos_own_search_leaves_out_what_an_agent_wrote(documents):
    """Night 8: the #0214 answer came from Auto's own search_knowledge call, not the
    automatic one, so the draft read as the owner's 'internal documents'. Night 8 marked
    the passage; night 9's #1547 was marked too and still cited as "a document in our
    knowledge base" (chat 9a128fe3), so it is now left out (services/agents_writing)."""
    from modules.tools.execution import exec_platform

    found = {"success": True, "results": [
        {"document_id": documents.draft.id, "content": OLD_DRAFT, "excerpt": OLD_DRAFT, "similarity": 0.91},
        {"document_id": str(documents.sheet.id), "content": DELIVERY_DAYS, "excerpt": DELIVERY_DAYS,
         "similarity": 0.74}]}
    calls = []

    async def execute_tool(**kwargs):
        calls.append(kwargs["tool_name"])
        return found

    executor = NS(platform_tools=NS(db=documents.db, execute_tool=execute_tool))
    agent = _searcher(documents)

    result = asyncio.run(exec_platform.execute_platform_tool(executor, "search_knowledge",
                                                             {"query": "wholesale cafe delivery days"}, agent.id))

    assert [r["content"] for r in result["results"]] == [DELIVERY_DAYS]       # the owner's own, as it was
    assert len(found["results"]) == 2                                          # a new result, not changed in place
    # F325 (night 9b): semantic_search reads the same documents, so it keeps the owner's only too;
    # a search that isn't of the documents (search_codebase) is as it was.
    semantic = asyncio.run(exec_platform.execute_platform_tool(executor, "semantic_search", {"query": "x"}, agent.id))
    assert [r["content"] for r in semantic["results"]] == [DELIVERY_DAYS]
    other = asyncio.run(exec_platform.execute_platform_tool(executor, "search_codebase", {"query": "x"}, agent.id))
    assert other is found and calls == ["search_knowledge", "semantic_search", "search_codebase"]


def test_another_workspaces_search_leaves_out_nothing(documents, seed_workspace):
    from core.models import Agent
    from services.agents_writing import owners_search

    elsewhere = UUID(seed_workspace())
    stranger = Agent(name="Auto", agent_type="custom", description="", status="active", configuration={},
                     workspace_id=elsewhere, created_by="test", owner_type="workspace", owner_id=str(elsewhere))
    documents.db.add(stranger)
    documents.db.flush()
    found = {"results": [{"document_id": documents.draft.id, "content": OLD_DRAFT}]}

    assert owners_search(documents.db, found, stranger.id) is found
    assert owners_search(None, found, _searcher(documents).id) is found
