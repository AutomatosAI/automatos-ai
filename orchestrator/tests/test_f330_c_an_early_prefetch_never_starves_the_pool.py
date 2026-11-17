"""F330 (night 9c), voice latency D-2: an early prefetch never starves the pool or stalls the loop.

F330-B pins, with a pool of ONE connection, that a question turn gives its connection back
before its documents are searched. With ``CHAT_EARLY_LOOKUPS`` on, the prefetch starts while
the turn still holds its connection (it is loading tools and building its prompt), so a
question turn can hold several at once. Here, against Postgres:

* with one connection to spare, the early prefetch counts and filters on sessions of its own,
  off the loop, finds the passages, and the loop never stands still;
* with NO connection to spare (the guard bypassed on purpose), the early count waits for the
  pool in its thread, never on the loop, and fails; at the join the turn gives its connection
  back and searches in place, and is handed the same passages as with the switch off;
* with the guard in place, a pool with no room starts no early prefetch at all.

The search opens a session of its own, as the real one does (``tool_router.execute_tool``).
"""
from __future__ import annotations

import asyncio
import gc
import importlib
import time
from types import SimpleNamespace as NS
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from config import config
from consumers.chatbot import early_lookups as el
from consumers.chatbot import knowledge_prefetch as kp

POOL_WAIT_S = 0.5
GIVES_UP_WITHIN_S = 10.0      # far longer than the pool wait: the early count has given up by then
BEAT_S = 0.01
LONGEST_STALL_S = 0.25
QUESTION = "What is our returns policy for opened coffee?"
CHAT_ID = "chat-f330-c"
AGENT = 7
RUNTIME = NS(agent_id=294)
PASSAGE = {"filename": "returns-policy.pdf", "source": "returns-policy.pdf", "title": "returns-policy.pdf",
           "similarity": 0.91, "content": "Opened bags: 14 days.", "excerpt": "Opened bags: 14 days.",
           "document_id": 330}


def _engine(url, pool_size):
    return create_engine(url, pool_size=pool_size, max_overflow=0, pool_timeout=POOL_WAIT_S)


@pytest.fixture
def one_connection(test_db_url):
    eng = _engine(test_db_url, 1)
    yield eng
    eng.dispose()


@pytest.fixture
def two_connections(test_db_url):
    """The turn's, and one to spare."""
    eng = _engine(test_db_url, 2)
    yield eng
    eng.dispose()


def _chat(monkeypatch, engine, turn):
    """A bare service whose turn holds ``turn`` (a session on ``engine``). Every session the
    early prefetch opens for itself, and the search's own, come from ``engine`` too."""
    from consumers.chatbot import service
    from consumers.chatbot.prompt_analyzer import get_prompt_analyzer

    counted = []

    async def execute_and_format(**kw):
        with Session(engine) as own:              # the document search's own session
            own.execute(text("SELECT 1")).all()
        return {"success": True, "raw_result": {"success": True, "results": [dict(PASSAGE)]}, "frontend_data": {}}

    def _count(db, workspace_id):
        counted.append(db)
        return db.execute(text("SELECT 3")).scalar()   # read on the session it is given, as the real count is

    monkeypatch.setattr(kp, "documents_in", _count)
    monkeypatch.setattr(kp, "_has_database", lambda db, ws: False)
    monkeypatch.setattr(type(config), "CHATBOT_KNOWLEDGE_PREFETCH", property(lambda self: True))
    monkeypatch.setattr(config, "KNOWLEDGE_PREFETCH_PASSAGES", 5, raising=False)
    monkeypatch.setattr(config, "KNOWLEDGE_PREFETCH_MIN_SCORE", 0.3, raising=False)
    monkeypatch.setattr(config, "CHAT_EARLY_LOOKUPS", True)
    monkeypatch.setattr(el, "_a_session", lambda: Session(engine))
    chat = service.StreamingChatService.__new__(service.StreamingChatService)
    chat.db, chat.workspace_id, chat.widget_mode = turn, str(uuid4()), False
    chat.prompt_analyzer = get_prompt_analyzer()
    chat._turn_document_ids, chat._turn_chunk_ids = set(), set()
    chat.tool_router = NS(execute_and_format=execute_and_format)
    return chat, counted


def _ask():
    """A question turn that skips its apps: only the prefetch may start early."""
    return el.TurnAsk(chat_id=CHAT_ID, messages=[{"role": "user", "content": QUESTION}], agent_id=AGENT,
                      skip_composio=True, complexity_assessment=None, force_text_only=False)


async def _watched(steps):
    """``await steps()`` and the longest the loop stood still while it ran. The collector is
    emptied first and paused, as in the F105 loop tests: a full collection in a large shard
    takes about half a second, wherever it lands."""
    gaps, done = [], asyncio.Event()

    async def heartbeat():
        last = time.monotonic()
        while not done.is_set():
            await asyncio.sleep(BEAT_S)
            now = time.monotonic()
            gaps.append(now - last)
            last = now

    gc.collect()
    gc.disable()
    try:
        beat = asyncio.create_task(heartbeat())
        await asyncio.sleep(0)
        try:
            result = await steps()
        finally:
            done.set()
            await beat
    finally:
        gc.enable()
    return result, max(gaps)


def _in_place(monkeypatch, engine):
    """What the turn is handed with nothing started early (the switch-off run). Each test runs
    it first, so everything the watched steps import is imported before the loop is timed: a
    cold import is not a stall of the turn's making."""
    importlib.import_module("consumers.chatbot.auto")             # read by early_lookups.apps_key
    with Session(engine) as turn:
        turn.execute(text("SELECT 1")).all()
        chat, _ = _chat(monkeypatch, engine, turn)
        return asyncio.run(el.the_turns_prefetch(chat, QUESTION, RUNTIME, CHAT_ID, {}))


def test_with_a_connection_to_spare_the_early_prefetch_finds_the_passages_and_the_loop_keeps_running(
        two_connections, monkeypatch):
    in_place = _in_place(monkeypatch, two_connections)
    monkeypatch.setattr(el, "pool_has_room", lambda: True)        # two connections is no room by the guard's count
    with Session(two_connections) as turn:
        turn.execute(text("SELECT 1")).all()                      # the turn's opening reads: it holds a connection
        chat, counted = _chat(monkeypatch, two_connections, turn)

        async def steps():
            chat._early = early = el.start(chat, _ask(), RUNTIME)
            await asyncio.wait([early.knowledge])                 # while the turn builds its prompt
            held = turn.in_transaction()
            found = await el.the_turns_prefetch(chat, QUESTION, RUNTIME, CHAT_ID, {})
            use = chat._early.knowledge_use
            await el._drop(chat)
            return held, found, use

        (held, found, use), stood_still = asyncio.run(_watched(steps))

    assert held                                                   # the turn kept its connection all along
    assert use == el.USED and len(counted) == 1 and counted[0] is not turn
    assert found is not None and found.passages == 1 and found.files == ["returns-policy.pdf"]
    assert "[Source 1: returns-policy.pdf]" in found.message["content"]
    assert found.message == in_place.message                      # the same passages as with the switch off
    assert chat._turn_document_ids == {330}
    assert stood_still < LONGEST_STALL_S, f"the loop stood still for {stood_still:.2f} s around an early prefetch"


def test_with_no_connection_to_spare_the_early_count_fails_off_the_loop_and_the_turn_searches_in_place(
        one_connection, monkeypatch):
    in_place = _in_place(monkeypatch, one_connection)
    monkeypatch.setattr(el, "pool_has_room", lambda: True)        # the guard bypassed on purpose
    with Session(one_connection) as turn:
        turn.execute(text("SELECT 1")).all()                      # the turn holds the pool's only connection
        chat, counted = _chat(monkeypatch, one_connection, turn)

        async def steps():
            chat._early = early = el.start(chat, _ask(), RUNTIME)
            await asyncio.wait([early.knowledge], timeout=GIVES_UP_WITHIN_S)   # the turn builds its prompt, connection in hand
            gave_up = early.knowledge.done() and early.knowledge.result().failed
            held = turn.in_transaction()
            found = await el.the_turns_prefetch(chat, QUESTION, RUNTIME, CHAT_ID, {})
            use = chat._early.knowledge_use
            await el._drop(chat)
            return gave_up, held, found, use

        (gave_up, held, found, use), stood_still = asyncio.run(_watched(steps))

    assert gave_up and held                                       # it waited for the pool in its thread, and failed
    assert use == el.REDONE and counted[-1] is turn               # counted again in place, on the turn's session
    assert found is not None and found.passages == 1
    assert found.message == in_place.message                      # the same passages as with the switch off
    assert chat._turn_document_ids == {330}
    assert stood_still < LONGEST_STALL_S, f"the loop stood still for {stood_still:.2f} s behind a full pool"


def test_a_pool_with_no_room_starts_no_early_prefetch(one_connection, monkeypatch):
    monkeypatch.setattr("core.database.database.engine", one_connection)   # the worker's pool, as the guard reads it
    with Session(one_connection) as turn:
        turn.execute(text("SELECT 1")).all()
        chat, counted = _chat(monkeypatch, one_connection, turn)

        async def steps():
            return el.pool_has_room(), el.start(chat, _ask(), RUNTIME)

        room, early = asyncio.run(steps())

    assert room is False
    assert early.knowledge is None and early.knowledge_use == el.NOT_STARTED
    assert early.apps is None and counted == []
