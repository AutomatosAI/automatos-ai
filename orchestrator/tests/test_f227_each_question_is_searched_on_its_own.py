"""F227/F085 (2 Oct, night 6): a message asking several questions got ONE search,
and under load the model answered all of them from those five passages without
searching (F088KB 3/8, was 8/8). Each question is now searched on its own, under
its own heading, and the model is told to search for itself any question whose
passages do not answer it. The graph section's team filter, which the watchdog
caught stalling a turn, runs off the event loop.
"""
from __future__ import annotations

import asyncio
import threading
import time
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.orm import Session

from consumers.chatbot import knowledge_prefetch as kp

QUICK_ONES = (
    "Quick ones - please answer each:\n"
    "1. Which word does the brand voice guide ban?\n"
    "2. What is the wholesale minimum order?\n"
    "3. Which day do we roast for Thursday deliveries?"
)
BAN = {"filename": "brand-voice.md", "similarity": 0.62, "content": "Never write 'moreish'.",
       "excerpt": "Never write 'moreish'.", "document_id": 720}
MINIMUM = {"filename": "wholesale.md", "similarity": 0.58, "content": "Minimum order: 3 kg.",
           "excerpt": "Minimum order: 3 kg.", "document_id": 721}


@pytest.mark.parametrize("message, questions", [
    (QUICK_ONES, ["Which word does the brand voice guide ban?", "What is the wholesale minimum order?",
                  "Which day do we roast for Thursday deliveries?"]),
    ("- Who is our Clevedon contact?\n- When is their first delivery?",
     ["Who is our Clevedon contact?", "When is their first delivery?"]),
    ("What is the minimum order? And which day is the roast?",
     ["What is the minimum order?", "And which day is the roast?"]),
    ("Which word does the brand voice guide ban?", []),
])
def test_the_questions_of_a_message(message, questions):
    assert kp.split_questions(message) == questions


class _Search:
    """search_knowledge, answering by query: the brand-voice question finds the ban,
    the minimum question the wholesale page, the roast question nothing."""

    def __init__(self):
        self.calls = []

    async def __call__(self, args):
        self.calls.append(args)
        query = args["query"].lower()
        results = [BAN] if "ban" in query else [MINIMUM] if "minimum" in query else []
        return {"success": True, "raw_result": {"success": True, "results": results}, "frontend_data": None}


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


def _workspace_with_documents(db):
    ws = str(uuid4())
    db.execute(text("INSERT INTO documents (id, workspace_id, filename, status) "
                    "VALUES (700, CAST(:ws AS uuid), 'brand-voice.md', 'completed')"), {"ws": ws})
    return ws


def test_each_question_is_searched_and_given_its_own_passages(db):
    search = _Search()
    got = asyncio.run(kp.prefetch(db, _workspace_with_documents(db), QUICK_ONES, search=search,
                                  enabled=True, limit=5, min_score=0.3))

    assert [c["query"] for c in search.calls] == kp.split_questions(QUICK_ONES)   # night 6: one search for all
    assert all(c["limit"] == kp.PER_QUESTION_MIN for c in search.calls)
    content = got.message["content"]
    assert content.startswith(kp.MULTI_HEADER)
    assert content.index("Question 1:") < content.index("brand-voice.md") < content.index("Question 2:")
    assert content.index("Question 2:") < content.index("wholesale.md") < content.index("Question 3:")
    assert content.endswith(kp.NOTHING_FOR_QUESTION)                  # the roast question: search for it
    assert got.searches == search.calls and got.passages == 2


def test_a_single_question_is_searched_as_before(db):
    search = _Search()
    ask = "Which word does the brand voice guide ban?"
    got = asyncio.run(kp.prefetch(db, _workspace_with_documents(db), ask, search=search,
                                  enabled=True, limit=5, min_score=0.3))
    assert search.calls == [{"query": ask, "limit": 5}] and got.searches == search.calls
    assert got.message["content"].startswith(kp.PREFETCH_HEADER)


def test_a_brief_is_searched_whole(db):
    """F201's draft guides search a customer's brief: its questions are the customer's."""
    search = _Search()
    got = asyncio.run(kp.prefetch(db, _workspace_with_documents(db), QUICK_ONES, search=search, enabled=True,
                                  limit=5, min_score=0.3, question_only=False))
    assert search.calls == [{"query": QUICK_ONES, "limit": 5}] and got.searches == search.calls


def test_a_long_run_of_spaces_is_split_in_linear_time():
    """CodeQL py/polynomial-redos on the first patterns: quadratic on these."""
    started = time.monotonic()
    for message in ("* !" + " " * 50_000 + "\x00", "a" + " " * 50_000 + "b"):
        assert kp.split_questions(message) == []
    assert time.monotonic() - started < 1.0


def test_the_graph_is_filtered_and_scored_off_the_event_loop(monkeypatch):
    import networkx as nx

    from modules.context.sections.base import SectionContext
    from modules.context.sections.graph_context import GraphSection

    graph = nx.Graph()
    graph.add_node("margin-target", label="margin target")
    threads = []

    async def _load(workspace_id):
        return graph

    def _view(whole, team):
        threads.append(threading.current_thread() is threading.main_thread())
        return whole

    monkeypatch.setattr("modules.knowledge.graph_service.get_graph_service", lambda: NS(load_graph=_load))
    monkeypatch.setattr("modules.knowledge.graph_service.team_filtered_view", _view)
    ctx = SectionContext(agent=NS(id=7, team="support"), workspace_id=str(uuid4()),
                         messages=[{"role": "user", "content": "What is the margin target?"}])
    asyncio.run(GraphSection().render(ctx))
    assert threads == [False]                                        # a worker thread, never the loop's
