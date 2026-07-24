"""PRD-142 Wave 1 · WS-D · W1-S8 — get_db() must not linger idle-in-transaction.

The FastAPI ``get_db()`` dependency previously only ``close()``d the session in
its ``finally``. A read-only handler opens an implicit transaction on its first
SELECT; if it never commits, returning the connection to the pool *without*
rolling back can leave it 'idle in transaction' — holding row locks and blocking
DDL (a 9 hr idle SELECT on ``agents`` was observed in production, PRD-135).

The fix aligns ``get_db()`` with the safe ``get_db_session()`` pattern: roll back
before close so no transaction lingers. After a handler that already committed,
the rollback is a harmless no-op (no active transaction to undo).

These tests drive the generator with a recording fake session — no real DB.
"""
import os
import sys
from pathlib import Path

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

# Importing database.py builds the SQLAlchemy engine, which refuses to construct
# without POSTGRES_* creds. These tests never touch a real DB (SessionLocal is
# replaced with a fake); setdefault means a real .env still wins.
for _k, _v in {
    "POSTGRES_USER": "test",
    "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost",
    "POSTGRES_PORT": "5432",
    "POSTGRES_DB": "test",
}.items():
    os.environ.setdefault(_k, _v)

import core.database.database as dbmod  # noqa: E402


class _RecordingSession:
    """Records the lifecycle calls get_db() makes, in order."""

    def __init__(self):
        self.calls = []

    def commit(self):
        self.calls.append("commit")

    def rollback(self):
        self.calls.append("rollback")

    def close(self):
        self.calls.append("close")


def _patch_session_local(rec):
    """Swap SessionLocal for a factory returning ``rec``; return a restore fn."""
    original = dbmod.SessionLocal
    dbmod.SessionLocal = lambda: rec
    return lambda: setattr(dbmod, "SessionLocal", original)


def test_get_db_rolls_back_before_close_on_normal_exit():
    """A read-only handler (no commit) must roll back, then close."""
    rec = _RecordingSession()
    restore = _patch_session_local(rec)
    try:
        gen = dbmod.get_db()
        db = next(gen)
        assert db is rec
        # Exhaust the generator → runs the finally block.
        with pytest.raises(StopIteration):
            next(gen)
    finally:
        restore()
    assert rec.calls == ["rollback", "close"]


def test_get_db_rolls_back_before_close_on_exception():
    """If the handler raises, the finally must still roll back, then close."""
    rec = _RecordingSession()
    restore = _patch_session_local(rec)
    try:
        gen = dbmod.get_db()
        next(gen)
        with pytest.raises(ValueError):
            gen.throw(ValueError("handler boom"))
    finally:
        restore()
    assert rec.calls == ["rollback", "close"]


def test_get_db_rollback_after_commit_is_harmless():
    """A handler that commits is unaffected — rollback runs but is a no-op."""
    rec = _RecordingSession()
    restore = _patch_session_local(rec)
    try:
        gen = dbmod.get_db()
        db = next(gen)
        db.commit()  # handler persisted its work
        with pytest.raises(StopIteration):
            next(gen)
    finally:
        restore()
    # close always follows rollback; the committed work is already durable.
    assert rec.calls == ["commit", "rollback", "close"]
