"""F074 — one tool's failed statement no longer takes down the rest of the turn.

Every tool in a chat turn runs on one request session. Night 1, 19:30:09: the
multimodal search tools caught their own SQL syntax error and returned it as
data, leaving the transaction aborted; query_database (7/7 that night),
platform_shopify_sync_status and COMPOSIO_SEARCH_TAVILY then all failed
"current transaction is aborted". The executor now checks the session after
every tool and rolls a failed transaction back, naming the tool that left it.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

from core.database import session_health
from core.database.session_health import rollback_if_aborted, transaction_aborted
from modules.tools.execution.unified_executor import UnifiedToolExecutor

IN_ERROR, IDLE_IN_TRANSACTION = 3, 2


class _Session:
    def __init__(self, status=IN_ERROR, in_transaction=True, active=True):
        self.status, self._in_tx, self.is_active, self.rollbacks = status, in_transaction, active, 0

    def in_transaction(self):
        return self._in_tx

    def connection(self):
        return NS(connection=NS(dbapi_connection=NS(info=NS(transaction_status=self.status))))

    def rollback(self):
        self.rollbacks += 1
        self.status, self._in_tx = IDLE_IN_TRANSACTION, False


def test_a_failed_transaction_is_detected_and_only_that():
    assert transaction_aborted(_Session(IN_ERROR))
    assert transaction_aborted(_Session(IDLE_IN_TRANSACTION, active=False))    # a failed flush
    assert not transaction_aborted(_Session(IDLE_IN_TRANSACTION))
    assert not transaction_aborted(_Session(IN_ERROR, in_transaction=False))
    assert not transaction_aborted(None)


def test_a_probe_that_raises_is_never_the_failure():
    class Broken(_Session):
        def connection(self):
            raise RuntimeError("pool gone")

    assert transaction_aborted(Broken()) is False


def test_rollback_names_the_culprit(caplog):
    session = _Session(IN_ERROR)
    with caplog.at_level("WARNING", logger=session_health.__name__):
        assert rollback_if_aborted(session, "tool 'search_multimodal'") is True
    assert session.rollbacks == 1 and "search_multimodal" in caplog.text
    assert rollback_if_aborted(session, "tool 'query_database'") is False    # healthy now: untouched
    assert session.rollbacks == 1


# ── the executor: after every tool ──────────────────────────────────────────

def _executor(db):
    ex = UnifiedToolExecutor.__new__(UnifiedToolExecutor)
    ex.composio_actions = {}
    ex.db = db
    return ex


@pytest.fixture
def no_policy_gate(monkeypatch):
    monkeypatch.setattr(UnifiedToolExecutor, "_policy_gate_check", lambda self, *a, **k: None)


def test_the_executor_rolls_back_a_session_a_tool_left_aborted(no_policy_gate):
    db = _Session(IN_ERROR)
    asyncio.run(_executor(db).execute_tool("platform_execute", {}, agent_id=1, trace_id="t-f074"))
    assert db.rollbacks == 1


def test_a_healthy_session_is_left_alone(no_policy_gate):
    db = _Session(IDLE_IN_TRANSACTION)
    asyncio.run(_executor(db).execute_tool("platform_execute", {}, agent_id=1, trace_id="t-f074"))
    assert db.rollbacks == 0


# ── real Postgres: the psycopg2 status is what the probe reads ──────────────

def test_on_postgres_a_failed_statement_is_seen_rolled_back_and_the_session_works(test_engine):
    from sqlalchemy import text
    from sqlalchemy.orm import Session

    with Session(test_engine) as session:
        session.execute(text("SELECT 1"))
        with pytest.raises(Exception):
            session.execute(text("SELECT to_jsonb(:team::text)"), {"team": "sales"})   # what night 1 ran
        assert transaction_aborted(session)
        assert rollback_if_aborted(session, "tool 'search_multimodal'")
        assert session.execute(text("SELECT to_jsonb(CAST(:team AS text))"), {"team": "sales"}).scalar() == "sales"
