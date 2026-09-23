"""F085-A (night 3) — retrieval first: a question in a workspace with documents
is searched before the model's first call.

Night 3's RAG test: 40 product questions, the product's own manual in the
workspace, ONE search_knowledge call in 40 and 14/40 right. Now the turn runs
search_knowledge itself when the owner asks a question and the workspace has
documents, and the passages that clear a relevance floor reach the prompt,
cited by file. An instruction, an empty workspace or the dial off: the turn
runs as before.
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from consumers.chatbot import knowledge_prefetch as kp

HIT = {"filename": "brand-voice.md", "source": "brand-voice.md", "title": "brand-voice.md", "similarity": 0.62,
       "content": "Never write 'moreish'.", "excerpt": "Never write 'moreish'.", "document_id": 720}
WEAK = {"filename": "q3-report.md", "source": "q3-report.md", "title": "q3-report.md", "similarity": 0.12,
        "content": "Q3 went well.", "excerpt": "Q3 went well.", "document_id": 900}
ASK = "Which word does the brand voice guide ban?"


class _Search:
    def __init__(self, results):
        self.results, self.calls = results, []

    async def __call__(self, args):
        self.calls.append(args)
        return {"success": True, "raw_result": {"success": True, "results": self.results},
                "frontend_data": {"documents": self.results}}


@pytest.fixture
def db(test_engine):
    with test_engine.connect() as conn:
        session = Session(bind=conn)
        session.execute(text("DROP TABLE IF EXISTS pg_temp.documents"))
        session.execute(text("CREATE TEMP TABLE documents (LIKE public.documents INCLUDING DEFAULTS)"))
        yield session
        session.rollback()
        session.execute(text("DROP TABLE IF EXISTS pg_temp.documents"))
        session.commit()
        session.close()


def _workspace_with_documents(db, n=2):
    ws = str(uuid4())
    for i in range(n):
        db.execute(text("INSERT INTO documents (id, workspace_id, filename, status) "
                        "VALUES (:id, CAST(:ws AS uuid), :name, 'completed')"),
                   {"id": 700 + i, "ws": ws, "name": f"doc-{i}.md"})
    return ws


def _run(db, ws, message, search, *, enabled=True):
    return asyncio.run(kp.prefetch(db, ws, message, search=search, enabled=enabled, limit=5, min_score=0.3))


def test_a_question_in_a_workspace_with_documents_is_searched_first(db):
    ws, search = _workspace_with_documents(db), _Search([HIT, WEAK])
    got = _run(db, ws, ASK, search)
    assert search.calls == [{"query": ASK, "limit": 5}]
    assert (got.passages, got.found, got.files) == (1, 2, ["brand-voice.md"])
    assert "[Source 1: brand-voice.md]" in got.message["content"]            # passages carry their file names
    assert got.message["content"].startswith(kp.PREFETCH_HEADER)
    assert "q3-report.md" not in got.message["content"]                       # under the relevance floor
    assert got.summary == "1 passage from brand-voice.md — searched automatically"


def test_a_workspace_holding_only_agents_reports_is_searched_too(db):
    """The persona's business questions rely on the rulings agents' reports carry."""
    ws = str(uuid4())
    for i in range(3):
        db.execute(text("INSERT INTO documents (id, workspace_id, filename, source_type, status) "
                        "VALUES (:id, CAST(:ws AS uuid), :name, 'agent_output', 'completed')"),
                   {"id": 900 + i, "ws": ws, "name": f"report-{i}.md"})
    search = _Search([HIT])
    assert _run(db, ws, ASK, search).passages == 1 and len(search.calls) == 1


def test_an_instruction_or_an_empty_workspace_is_not_searched(db):
    ws, search = _workspace_with_documents(db), _Search([HIT])
    assert _run(db, ws, "Create an agent called OPS", search) is None
    assert _run(db, ws, "Can you delete the old Christmas sheet?", search) is None
    assert _run(db, str(uuid4()), ASK, search) is None                          # no documents there
    assert search.calls == []


def test_the_dial_off_is_the_path_as_it_was(db):
    ws, search = _workspace_with_documents(db), _Search([HIT])
    assert _run(db, ws, ASK, search, enabled=False) is None
    assert search.calls == []


def test_nothing_above_the_floor_still_says_it_searched(db):
    ws = _workspace_with_documents(db)
    got = _run(db, ws, ASK, _Search([WEAK]))
    assert got.message is None and got.passages == 0
    assert got.summary == "searched automatically — nothing above the relevance floor (1 found)"


def test_a_failed_search_leaves_the_turn_as_it_was(db):
    async def timed_out(_args):
        raise RuntimeError("embedding timed out")

    assert _run(db, _workspace_with_documents(db), ASK, timed_out) is None


@pytest.mark.parametrize("message, asked", [
    ("Will Dropbox files be deleted if I disconnect?", True),
    ("Quick ones — I want to check you know my business. Please answer each from what I've given you", True),
    ("Please look in my documents and tell me the delivery day", True),
    ("how does entity extraction work", True),
    ("Can you tell me what our wholesale terms are?", True),
    ("Can you create an agent called OPS?", False),
    ("Create a report and tell me what it says", False),
    ("Please send the invoice to Declan", False),
    ("thanks!", False),
])
def test_what_counts_as_a_question(message, asked):
    assert kp.is_question(message) is asked


# ── the chat turn ───────────────────────────────────────────────────────────

def _chat(monkeypatch, *, dial_on=True, documents=2, results=(HIT,)):
    from types import SimpleNamespace as NS

    from config import config
    from consumers.chatbot import service
    from consumers.chatbot.streaming import get_streaming_handler

    searched = []

    async def execute_and_format(**kw):
        searched.append(kw)
        return {"success": True, "raw_result": {"success": True, "results": list(results)}, "frontend_data": {}}

    monkeypatch.setattr(kp, "documents_in", lambda db, ws: documents)
    monkeypatch.setattr(type(config), "CHATBOT_KNOWLEDGE_PREFETCH", property(lambda self: dial_on))
    svc = service.StreamingChatService.__new__(service.StreamingChatService)
    svc.db, svc.workspace_id, svc.streaming_handler = object(), "ws", get_streaming_handler()
    svc._turn_document_ids, svc._turn_chunk_ids = set(), set()
    svc.tool_router = NS(execute_and_format=execute_and_format)
    return svc, searched


def _turn(svc, message):
    from types import SimpleNamespace as NS

    llm_messages, prefetched = [{"role": "system", "content": "You are Auto."}], []

    async def collect():
        return [f async for f in svc._retrieval_first(message, llm_messages, NS(agent_id=294), "chat-1", prefetched)]

    return asyncio.run(collect()), llm_messages, prefetched


def test_the_turn_searches_first_shows_it_and_hands_the_passages_to_the_model(monkeypatch):
    svc, searched = _chat(monkeypatch)
    frames, llm_messages, prefetched = _turn(svc, ASK)
    assert searched[0]["tool_name"] == "search_knowledge" and searched[0]["caller_context"]["retrieval_first"]
    assert '"tool-start"' in frames[0] and '"automatic": true' in frames[0]
    assert '"tool-end"' in frames[1] and "searched automatically" in frames[1]
    assert prefetched == [("search_knowledge", searched[0]["tool_args"])]
    assert llm_messages[-1]["role"] == "system" and "[Source 1: brand-voice.md]" in llm_messages[-1]["content"]
    assert svc._turn_document_ids == {720}                      # the reply records what it retrieved


def test_the_dial_off_or_an_instruction_leaves_the_turn_as_it_was(monkeypatch):
    for dial_on, message in ((False, ASK), (True, "Create an agent called OPS")):
        svc, searched = _chat(monkeypatch, dial_on=dial_on)
        frames, llm_messages, prefetched = _turn(svc, message)
        assert (frames, prefetched, searched, len(llm_messages)) == ([], [], [], 1)


def test_the_search_that_ran_counts_in_the_tool_loop():
    import inspect

    from consumers.chatbot import service
    from modules.tools.execution import tool_loop

    turn = inspect.getsource(service.StreamingChatService._stream_response_with_agent_scoped)
    assert turn.index("self._retrieval_first(") < turn.index("self._stream_llm_call(")
    assert "prefetched=_prefetched" in turn
    # a first reply with no tool call may cite the automatic search too (F099's notice)
    assert "unexecuted_claims_notice(\n                        final_text, use_tools, {name for name, _args in _prefetched}" in turn
    assert "executor.tracker.record_execution(_ran_name, _ran_args)" in inspect.getsource(
        service.StreamingChatService._stream_tool_loop)
    assert "self.tracker.tool_counts" in inspect.getsource(tool_loop.ToolLoopExecutor._recover_narrated_actions)
