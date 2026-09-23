"""F119 — board notices outside the board API reach a listener.

The same fault as F118 elsewhere: a service committed, then sent its
pg_notify, and nothing committed after it — the request's closing rollback,
the scheduler loop's close, or a chat turn releasing a "read-only" transaction
before its next model call (F105-B counted SELECT pg_notify as a read) dropped
the notice. Each site below is run in the lifecycle that dropped it, with a
LISTEN connection that must hear it.
"""
from __future__ import annotations

import asyncio
import json
import select
import time
import uuid
from types import SimpleNamespace

import pytest
from sqlalchemy import create_engine, text

BOARD, DISPATCH = "board_events", "board_task_available"


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
def seeded(engine, new_session):
    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'f119')"), {"id": ws})
    agent = s.execute(text(
        "INSERT INTO agents (name, agent_type, workspace_id, status, configuration) VALUES "
        "('ROASTER', 'custom', CAST(:w AS uuid), 'active', CAST(:c AS json)) RETURNING id"),
        {"w": ws, "c": '{"runtime": "cli", "provider": "claude", "model": "sonnet"}'}).fetchone()[0]
    s.commit()
    yield ws, agent
    s = new_session.sweep()
    s.execute(text("DELETE FROM approval_grants WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws})
    s.execute(text("DELETE FROM board_tasks WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws})
    s.execute(text("DELETE FROM agents WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": ws})
    s.commit()


@pytest.fixture
def listener(engine):
    import psycopg2
    from core.database.database import get_database_url

    conn = psycopg2.connect(get_database_url())
    conn.set_isolation_level(0)
    cur = conn.cursor()
    cur.execute(f'LISTEN "{BOARD}"')
    cur.execute(f'LISTEN "{DISPATCH}"')
    yield conn
    conn.close()


def _heard(conn, seconds=2.0):
    got, deadline = [], time.monotonic() + seconds
    while time.monotonic() < deadline:
        if select.select([conn], [], [], 0.2)[0]:
            conn.poll()
            while conn.notifies:
                note = conn.notifies.pop(0)
                got.append((note.channel, note.payload))
    return got


def _board(conn, task_id):
    return [json.loads(p) for c, p in _heard(conn) if c == BOARD and json.loads(p).get("task_id") == task_id]


def _request(fn):
    """A get_db session, rolled back and closed afterwards (FastAPI's lifecycle;
    the scheduler's loop closes its session the same way, with no commit)."""
    from core.database.database import get_db

    sessions = get_db()
    db = next(sessions)
    try:
        out = fn(db)
        return asyncio.run(out) if asyncio.iscoroutine(out) else out
    finally:
        sessions.close()


def _ticket(new_session, ws, agent, *, status="in_progress", ref=None, attempts=0):
    from core.models.core import BoardTask

    s = new_session()
    task = BoardTask(workspace_id=ws, title="Roast day write-up", status=status, priority="medium",
                     assigned_agent_id=agent, source_type="user", attempts=attempts, runtime_ref=ref or {})
    s.add(task)
    s.commit()
    return task.id


def _task(db, task_id):
    from core.models.core import BoardTask

    return db.query(BoardTask).get(task_id)


# ── the CLI host ────────────────────────────────────────────────────────────

def test_a_terminal_opening_on_a_session_ticket_reaches_the_board(seeded, listener, new_session):
    from services import cli_host_service as svc

    ws, agent = seeded
    host = SimpleNamespace(id=uuid.uuid4(), workspace_id=uuid.UUID(ws))
    task_id = _ticket(new_session, ws, agent, status="done", ref={"host_id": str(host.id), "mode": "terminal"})
    _request(lambda db: svc._record_terminal_events(db, host, task_id, [{"hook_event_name": "TerminalOpened"}]))
    assert any(b["event"] == "task_claimed" and b["status"] == "in_progress" for b in _board(listener, task_id))


def test_a_ticket_parked_on_its_sessions_question_reaches_the_board(seeded, listener, new_session):
    from services import cli_host_service as svc

    ws, agent = seeded
    ref = {svc.SESSION_ASKS_KEY: [{"grant_id": 1, "question": "Which café first?"}]}
    task_id = _ticket(new_session, ws, agent, ref=ref)
    _request(lambda db: svc._park_for_answer(db, _task(db, task_id), dict(ref)))
    assert any(b["status"] == "blocked" for b in _board(listener, task_id))


def test_a_ticket_out_of_attempts_reaches_the_board(seeded, listener, new_session):
    from services import cli_host_service as svc

    ws, agent = seeded
    task_id = _ticket(new_session, ws, agent, attempts=9)
    _request(lambda db: svc.park_exhausted(db, _task(db, task_id), "it kept coming back"))
    assert any(b["status"] == "review" for b in _board(listener, task_id))


# ── the CLI ticket lane ─────────────────────────────────────────────────────

def test_a_filed_session_ticket_reaches_the_board_and_wakes_the_dispatcher(seeded, listener):
    from services.cli_ticket_lane import file_cli_ticket

    ws, agent = seeded
    task_id = _request(lambda db: file_cli_ticket(db, workspace_id=ws, agent_id=agent, title="Chase the Salt Loft",
                                                  prompt="Chase the invoice", source_type="chat",
                                                  source_id=f"chat:{uuid.uuid4()}:1").id)
    heard = _heard(listener)
    assert any(c == BOARD and json.loads(p)["task_id"] == task_id for c, p in heard)
    assert any(c == DISPATCH and p.endswith(f":{task_id}") for c, p in heard)


def test_an_opened_runtime_canvas_ticket_reaches_the_board(seeded, listener):
    from core.models import Agent
    from services.cli_ticket_lane import open_session_ticket

    ws, agent_id = seeded
    host = SimpleNamespace(id=uuid.uuid4(), workspace_id=uuid.UUID(ws))

    def open_it(db):
        agent = db.query(Agent).get(agent_id)
        task, created = open_session_ticket(db, workspace_id=ws, agent=agent, chat_id=str(uuid.uuid4()),
                                            host=host, actor=None)
        return task.id

    task_id = _request(open_it)
    assert any(b["event"] == "task_created" for b in _board(listener, task_id))


# ── the scheduler ───────────────────────────────────────────────────────────

def test_a_scheduled_board_ticket_reaches_the_board_and_wakes_the_dispatcher(seeded, listener):
    from core.models.core import BoardTask
    from services.scheduled_task_service import ScheduledTaskService

    ws, agent = seeded
    row = SimpleNamespace(workspace_id=uuid.UUID(ws), description="Weekly stock count", payload={},
                          target_agent_id=agent, created_by_user_id=None, origin_chat_id=None)
    _request(lambda db: ScheduledTaskService._file_board_task(db, row, 4321))

    from core.database.database import SessionLocal

    s = SessionLocal()
    task_id = s.query(BoardTask.id).filter(BoardTask.workspace_id == ws, BoardTask.source_type == "scheduled_task").scalar()
    s.close()
    heard = _heard(listener)
    assert any(c == BOARD and json.loads(p)["task_id"] == task_id and json.loads(p)["event"] == "task_created"
               for c, p in heard)
    assert any(c == DISPATCH and p.endswith(f":{task_id}") for c, p in heard)


# ── a platform tool inside a chat turn ──────────────────────────────────────

def test_a_ticket_filed_by_a_tool_reaches_the_board_across_the_turns_release(seeded, listener):
    """A turn gives back a read-only transaction before its next model call
    (F105-B); a notice sent after the tool's commit sat in exactly that."""
    from core.database.read_release import release_if_read_only
    from modules.tools.discovery.handlers_board_tasks import create_board_task

    ws, _agent = seeded

    def turn(db):
        result = asyncio.run(create_board_task(db, uuid.UUID(ws), {"title": "Order more Guji", "description": "Two sacks of Guji for next week"}))
        release_if_read_only(db)                    # the turn's next model call
        return result["task_id"]

    task_id = _request(turn)
    assert any(b["event"] == "task_created" for b in _board(listener, task_id))


def test_a_notice_is_not_a_plain_read():
    from core.database.read_release import is_plain_read

    assert not is_plain_read("SELECT pg_notify(%(chan)s, %(payload)s)")
    assert is_plain_read("SELECT id FROM board_tasks WHERE id = 1")
