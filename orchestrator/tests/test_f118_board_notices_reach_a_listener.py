"""F118 — a board notice reaches a listener after the request ends.

The board endpoints committed, then called notify_board_event /
notify_task_available (pg_notify on the same session). A NOTIFY is delivered
only when its transaction commits, and get_db rolls back when the request ends
— so the Command Centre's sub-second refresh and the dispatch wake-up were
dropped, and both fell back to polling. Every notice in api/board_tasks.py now
goes inside the transaction, before the commit that carries it.
"""
from __future__ import annotations

import asyncio
import json
import re
import select
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text


@pytest.fixture(scope="module")
def engine():
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"needs a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def workspace(engine, new_session):
    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'f118')"), {"id": ws})
    s.commit()
    yield ws
    s = new_session.sweep()
    s.execute(text("DELETE FROM board_tasks WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": ws})
    s.commit()


@pytest.fixture
def listener(engine):
    import psycopg2
    from core.database.database import get_database_url
    from services.board_dispatcher import NOTIFY_CHANNEL as DISPATCH
    from services.board_events import NOTIFY_CHANNEL as BOARD

    conn = psycopg2.connect(get_database_url())
    conn.set_isolation_level(0)   # autocommit: LISTEN takes effect at once
    cur = conn.cursor()
    cur.execute(f'LISTEN "{BOARD}"')
    cur.execute(f'LISTEN "{DISPATCH}"')
    yield conn
    conn.close()


def _heard(conn, seconds=2.0):
    got = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if select.select([conn], [], [], 0.2)[0]:
            conn.poll()
            while conn.notifies:
                note = conn.notifies.pop(0)
                got.append((note.channel, note.payload))
    return got


class _Request:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


def _request(handler, **kwargs):
    """Run an endpoint the way FastAPI does: a get_db session, rolled back and
    closed when the request ends."""
    from core.database.database import get_db

    sessions = get_db()
    db = next(sessions)
    try:
        return asyncio.run(handler(db=db, **kwargs))
    finally:
        sessions.close()


def _ctx(ws):
    return SimpleNamespace(workspace_id=uuid.UUID(ws), user_id=None,
                           user=SimpleNamespace(id=2, clerk_user_id="user_f118"))


def test_a_new_card_reaches_the_command_centre(workspace, listener):
    from api.board_tasks import create_task

    task = _request(create_task, request=_Request({"title": "Chase the Salt Loft invoice"}), ctx=_ctx(workspace))
    board = [json.loads(p) for c, p in _heard(listener) if c == "board_events"]
    assert {"task_id": task["id"], "event": "task_created"}.items() <= next(
        (b for b in board if b["task_id"] == task["id"]), {}).items()


def test_a_status_change_reaches_the_command_centre(workspace, listener, new_session):
    from api.board_tasks import create_task, update_task_status

    task = _request(create_task, request=_Request({"title": "Brew & Bramble reorder"}), ctx=_ctx(workspace))
    _heard(listener, 0.5)                                       # the creation's notice
    _request(update_task_status, task_id=task["id"], ctx=_ctx(workspace),
             request=_Request({"status": "blocked", "blocked_reason": "waiting on the café"}))
    board = [json.loads(p) for c, p in _heard(listener) if c == "board_events"]
    assert any(b["task_id"] == task["id"] and b["event"] == "status_changed" and b["status"] == "blocked"
               for b in board)


def test_every_notice_in_the_board_api_rides_a_commit():
    """No notify_board_event / notify_task_available after the last commit of
    its function — where the request's closing rollback would drop it."""
    lines = (Path(__file__).resolve().parents[1] / "api" / "board_tasks.py").read_text().splitlines()
    late = []
    for i, line in enumerate(lines):
        if not re.search(r"\bnotify_(board_event|task_available)\(", line) or re.match(r"\s*(def|from|import) ", line):
            continue
        for later in lines[i + 1:]:
            if re.match(r"(async def|def|@router)", later):
                late.append(i + 1)
                break
            if "db.commit()" in later:
                break
    assert late == [], f"notices after their function's last commit at lines {late}"
