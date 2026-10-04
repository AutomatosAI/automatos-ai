"""F330 (night 9c) — a question turn holds no connection while its documents are searched.

Night 9c's frozen pool held chat turns "idle in transaction" whose last statement
was the retrieval-first count (``knowledge_prefetch.documents_in``): the count
opened the turn's transaction, and the turn kept its connection through the
searches (an embedding call, then a search on sessions of its own) until the
release before its first model call.

Here the pool has ONE connection. The search opens a session of its own, as the
real one does: if the turn still held the connection, the search would wait for
the pool, give up, and the owner's question would go to the model without its
passages. A turn that wrote (here a NOTIFY, which only its commit delivers) keeps
its transaction exactly as before.
"""
from __future__ import annotations

import asyncio
import select
import time

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from consumers.chatbot import knowledge_prefetch as kp

POOL_WAIT_S = 0.5
QUESTION = "What is our returns policy for opened coffee?"
HEADER = "Passages from this workspace's documents."
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
def three_connections(test_db_url):
    """A listener's, the turn's and the search's."""
    eng = _engine(test_db_url, 3)
    yield eng
    eng.dispose()


def _search_on_its_own_session(engine):
    async def search(args):
        with Session(engine) as own:              # the document search's own session
            own.execute(text("SELECT 1")).all()
        return {"raw_result": {"results": [dict(PASSAGE)]}}

    return search


def _ask(turn, engine, monkeypatch):
    # The count, read on the turn's own session as the real one is.
    monkeypatch.setattr(kp, "documents_in", lambda db, ws: db.execute(text("SELECT 3")).scalar())
    return asyncio.run(kp.prefetch(
        turn, "ws-f330", QUESTION, search=_search_on_its_own_session(engine),
        enabled=True, limit=5, min_score=0.5, header=HEADER,
    ))


def test_the_passages_are_found_with_the_only_connection_in_the_pool(one_connection, monkeypatch):
    with Session(one_connection) as turn:
        turn.execute(text("SELECT 1")).all()       # the turn's opening reads
        found = _ask(turn, one_connection, monkeypatch)

    assert found is not None and found.passages == 1
    assert found.files == ["returns-policy.pdf"]
    assert "[Source 1: returns-policy.pdf]" in found.message["content"]   # the owner's passage reaches the model


def _heard(raw, channel: str, within_s: float = 1.0) -> list:
    """What the listener was told on ``channel`` within ``within_s``."""
    deadline, heard = time.monotonic() + within_s, []
    while not heard and time.monotonic() < deadline:
        if select.select([raw.connection], [], [], max(0.0, deadline - time.monotonic()))[0]:
            raw.connection.poll()
            heard = [n.payload for n in raw.connection.notifies if n.channel == channel]
    return heard


def test_a_turn_that_wrote_keeps_its_transaction_and_its_notice(three_connections, monkeypatch):
    raw = three_connections.raw_connection()
    try:
        raw.connection.autocommit = True
        raw.cursor().execute("LISTEN f330_turn")
        with Session(three_connections) as turn:
            turn.execute(text("SELECT pg_notify('f330_turn', 'card moved')"))
            found = _ask(turn, three_connections, monkeypatch)
            assert found is not None and turn.in_transaction()      # kept, not rolled back
            turn.commit()
        assert _heard(raw, "f330_turn") == ["card moved"]
    finally:
        raw.close()
