"""F330 (night 9c) — a database question holds none of the caller's connection while it runs.

``query_database`` (a session's) and ``platform_query_data`` (an agent's) run
NL2SQL in-process with the caller's request session. Night 9c's frozen pool held
29 connections "idle in transaction" for 12+ minutes: each call kept the
request's connection through the query's model calls, while the query opened
sessions of its own (the source, the credentials) on the event loop.

Here the pool has ONE connection and the request has already read on it (the
executor's agent row). The query opens a session of its own, as the real one
does: if the request still held the connection, the query would wait for the
pool, give up, and the agent would read "Database query failed." instead of its
answer. A request that wrote keeps its transaction.
"""
from __future__ import annotations

import asyncio
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from modules.nl2sql.service import DatabaseKnowledgeService
from modules.tools.discovery.handlers_scheduling import query_data

POOL_WAIT_S = 0.5
MODEL_S = 0.05
WS = uuid4()
HARBOURLINE = (36, "harbourline_shop")
QUESTION = {"question": "How many orders shipped last week?", "_user_id": "7"}


@pytest.fixture
def one_connection(test_db_url):
    eng = create_engine(test_db_url, pool_size=1, max_overflow=0, pool_timeout=POOL_WAIT_S)
    yield eng
    eng.dispose()


@pytest.fixture
def two_connections(test_db_url):
    eng = create_engine(test_db_url, pool_size=2, max_overflow=0, pool_timeout=POOL_WAIT_S)
    yield eng
    eng.dispose()


class _Service(DatabaseKnowledgeService):
    """The real resolver (``resolve_source_id`` → ``match_source``); its read and
    the query are faked only as far as their database use, which is real."""

    def __init__(self, engine):
        self.engine = engine

    async def active_sources(self, workspace_id, db_session=None):
        if db_session is not None:
            db_session.execute(text("SELECT 36")).all()
            return [HARBOURLINE]
        with Session(self.engine) as own:
            own.execute(text("SELECT 36")).all()
        return [HARBOURLINE]

    async def query_database(self, **_kwargs):
        await asyncio.sleep(MODEL_S)                       # the model writing the SQL
        with Session(self.engine) as own:                  # the source and credential reads
            own.execute(text("SELECT 412")).all()
        return {"success": True, "sql": "SELECT count(*) AS orders FROM orders",
                "data": [{"orders": 412}], "columns": ["orders"], "row_count": 1}

    async def write_nl_audit(self, **_kwargs):
        return None


def _ask(request, engine, monkeypatch):
    monkeypatch.setattr("modules.nl2sql.get_database_knowledge_service", lambda: _Service(engine))
    return asyncio.run(query_data(request, WS, dict(QUESTION)))


def test_the_answer_comes_back_with_the_only_connection_in_the_pool(one_connection, monkeypatch):
    with Session(one_connection) as request:
        request.execute(text("SELECT 268")).all()          # the executor's agent row
        result = _ask(request, one_connection, monkeypatch)

    assert result["success"] is True, result
    assert result["data"] == [{"orders": 412}]
    assert one_connection.pool.checkedout() == 0


def test_a_request_that_wrote_keeps_its_transaction(two_connections, monkeypatch):
    with Session(two_connections) as request:
        request.execute(text("SELECT pg_advisory_xact_lock(330)"))   # a lock counts as a write
        result = _ask(request, two_connections, monkeypatch)
        assert result["success"] is True, result
        assert request.in_transaction()                    # kept: the lock is still held
        held = request.execute(text(
            "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND objid = 330 AND pid = pg_backend_pid()"
        )).scalar()
        request.rollback()
    assert held == 1
