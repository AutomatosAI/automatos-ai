"""F156 follow-up (25 Sep, the cleanup batch): two callers at once still file ONE card.

9579ed62d files one blocked [HARNESS] card when a workspace's old ledger file
cannot be read, by looking for the card and then adding it. Two callers at once
(an /approve during the weekly tick, or two workers) could both find no card and
both add one. A transaction-scoped advisory lock on the workspace now makes the
second wait, and then find the first one's card.

Real Postgres, two connections, committed rows (cleaned up after).
"""
from __future__ import annotations

import threading
import uuid

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from core.database.database import get_database_url


@pytest.fixture
def workspace():
    engine = create_engine(get_database_url())
    ws = uuid.uuid4()
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'f156-two-callers')"),
                     {"id": str(ws)})
    yield engine, ws
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM board_tasks WHERE workspace_id = CAST(:id AS uuid)"), {"id": str(ws)})
        conn.execute(text("DELETE FROM workspaces WHERE id = CAST(:id AS uuid)"), {"id": str(ws)})
    engine.dispose()


class _Racing:
    """A session whose look for the card waits (up to 2 s) for the other
    caller's look, so both look before either adds unless something keeps them
    apart."""

    def __init__(self, session, barrier):
        self._session, self._barrier = session, barrier

    def __getattr__(self, name):
        return getattr(self._session, name)

    def query(self, *entities):
        return _Look(self._session.query(*entities), self._barrier)


class _Look:
    def __init__(self, query, barrier):
        self._query, self._barrier = query, barrier

    def filter(self, *criteria):
        return _Look(self._query.filter(*criteria), self._barrier)

    def first(self):
        found = self._query.first()
        try:
            self._barrier.wait(timeout=2)
        except threading.BrokenBarrierError:
            pass
        return found


def test_two_callers_at_once_file_one_card(workspace):
    from core.models.core import BoardTask
    from services.harness_service import UNREADABLE_LEDGER_TAG, HarnessService

    engine, ws = workspace
    make_session = sessionmaker(bind=engine)
    barrier = threading.Barrier(2)
    errors = []

    def _caller():
        session = make_session()
        try:
            HarnessService._file_unreadable_ledger_card(_Racing(session, barrier), ws)
        except Exception as exc:  # noqa: BLE001 -- asserted below
            errors.append(exc)
        finally:
            session.close()

    callers = [threading.Thread(target=_caller) for _ in range(2)]
    for caller in callers:
        caller.start()
    for caller in callers:
        caller.join(timeout=30)

    session = make_session()
    try:
        cards = session.query(BoardTask).filter(BoardTask.workspace_id == ws,
                                                BoardTask.tags.contains([UNREADABLE_LEDGER_TAG])).count()
    finally:
        session.close()
    assert errors == [] and cards == 1
