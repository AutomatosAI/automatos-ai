"""The three ways a system-settings read fails, for the PRD-251 guards that fail closed.

A guard that must fail closed reads its row through the strict
``core.llm.manager.read_system_setting`` (P251-RVW-1), never through
``get_system_setting``, whose catch-all turns "could not read" into the default.
``settings_unreadable`` makes every such read fail in one of three ways:

* SessionLocal itself raises (an exhausted pool);
* the session's query raises (a dropped connection), and every session is still
  closed;
* the real query runs against a database without ``system_settings``.

Used by ``test_prd251_composio_deny.py`` (the Composio deny list, P251-RVW-1)
and ``test_prd251_settings.py`` (the Socials master switch, P251-RVW-8).
"""

from __future__ import annotations

import importlib
from contextlib import contextmanager
from unittest.mock import MagicMock

from sqlalchemy import create_engine
from sqlalchemy import exc as sa_exc
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

SESSION_LOCAL_RAISES = "SessionLocal raises"
QUERY_RAISES = "the query raises"
TABLE_MISSING = "the table is missing"
UNREADABLE_MODES = (SESSION_LOCAL_RAISES, QUERY_RAISES, TABLE_MISSING)


def pool_exhausted():
    """A SessionLocal that cannot hand out a session."""
    raise sa_exc.TimeoutError("QueuePool limit of size 5 overflow 10 reached, connection timed out, timeout 30.00")


class DroppedConnection:
    """A SessionLocal whose sessions fail every query as a dropped connection does."""

    def __init__(self):
        self.sessions = []

    def __call__(self):
        session = MagicMock(name="session")
        session.query.side_effect = sa_exc.OperationalError(
            "SELECT system_settings.value", {}, Exception("server closed the connection unexpectedly"),
        )
        self.sessions.append(session)
        return session


@contextmanager
def settings_unreadable(mode, monkeypatch):
    """Every system-settings read in the block fails as ``mode`` says. On a clean
    exit, the dropped-connection mode also proves every session was closed."""
    # Resolved now, like read_system_setting's own lazy import of SessionLocal.
    database_mod = importlib.import_module("core.database.database")
    engine = None
    dropped = None
    if mode == SESSION_LOCAL_RAISES:
        monkeypatch.setattr(database_mod, "SessionLocal", pool_exhausted)
    elif mode == QUERY_RAISES:
        dropped = DroppedConnection()
        monkeypatch.setattr(database_mod, "SessionLocal", dropped)
    elif mode == TABLE_MISSING:
        engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
        monkeypatch.setattr(database_mod, "SessionLocal", sessionmaker(bind=engine))
    else:
        raise ValueError(f"unknown unreadable-settings mode: {mode!r}")
    try:
        yield mode
    finally:
        if engine is not None:
            engine.dispose()
    if dropped is not None:
        assert dropped.sessions and all(session.close.called for session in dropped.sessions)  # nothing leaks
