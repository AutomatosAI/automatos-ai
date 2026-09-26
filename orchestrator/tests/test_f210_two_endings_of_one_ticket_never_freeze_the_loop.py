"""F210 — two endings of one ticket never freeze the event loop.

finalize_board_task_run locked the ticket's row (a blocking SELECT … FOR UPDATE)
and held the lock across three awaits: the named-file check, task_complete and
the report. A second ending of the same ticket in the same process (a run's own
result and the stall sweep, or a replaced run) then took the lock synchronously
on the event loop, so the holder could never resume to commit: the loop froze
until a DB timeout, and Postgres waits forever by default.

Nothing awaits under the lock now: the named-file check runs before it, and
task_complete and the report run after the commit that closes the run.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

LOCK_WAIT_MS = 2000   # a blocked lock wait fails after this, so a freeze shows as a stall, not a hang
TICK = 0.02           # the /health stand-in answers this often while the loop runs
SUSPEND = 0.3         # each awaited step yields the loop this long, as real I/O does
WORST_STALL = 1.0


def _engine(**kwargs):
    from core.database.database import get_database_url

    try:
        eng = create_engine(get_database_url(), pool_pre_ping=True, **kwargs)
        with eng.connect() as c:
            c.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"the finalize race needs a reachable Postgres: {exc}")
    return eng


@pytest.fixture(scope="module")
def engine():  # conftest's new_session binds to it
    eng = _engine()
    yield eng
    eng.dispose()


@pytest.fixture(scope="module")
def app_sessions():
    """Sessions made like the app's SessionLocal (autoflush off), on connections
    that give up a lock wait after LOCK_WAIT_MS."""
    eng = _engine(connect_args={"options": f"-c lock_timeout={LOCK_WAIT_MS}"})
    yield sessionmaker(bind=eng, autocommit=False, autoflush=False)
    eng.dispose()


@pytest.fixture
def workspace(new_session):
    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'f210')"), {"id": ws})
    s.commit()
    yield ws
    s = new_session.sweep()
    s.execute(text("DELETE FROM board_tasks WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": ws})
    s.commit()


def _running_ticket(new_session, ws):
    from core.models.core import BoardTask

    s = new_session()
    task = BoardTask(workspace_id=ws, title="Write the shop descriptions", status="in_progress", priority="medium",
                     source_type="user", attempts=0, started_at=datetime.now(timezone.utc) - timedelta(minutes=2))
    s.add(task)
    s.commit()
    return task.id


def test_two_endings_of_one_ticket_never_freeze_the_loop(workspace, new_session, app_sessions):
    from api import board_tasks as bt

    task_id = _running_ticket(new_session, workspace)
    completed, reported = [], []

    async def _files_checked(task, text_, workspace_id, db=None, projects_dir=None):
        await asyncio.sleep(SUSPEND)
        return None

    async def _not_parked(*_a, **_k):
        return False

    async def _completed(db, ws, task):
        await asyncio.sleep(SUSPEND)
        completed.append(task.id)

    async def _reported(db, ws, task, exec_result):
        await asyncio.sleep(SUSPEND)
        reported.append(task.id)

    async def _two_endings():
        stalls, stop = [0.0], asyncio.Event()

        async def _health():  # answers for as long as the event loop runs
            last = time.monotonic()
            while not stop.is_set():
                await asyncio.sleep(TICK)
                now = time.monotonic()
                stalls.append(now - last)
                last = now

        health = asyncio.create_task(_health())
        sessions = [app_sessions(), app_sessions()]
        try:
            endings = await asyncio.gather(*(bt.finalize_board_task_run(
                s, task_id=task_id, workspace_id=workspace, agent_id=None,
                exec_result={"status": "success", "result": "Shop descriptions written."}) for s in sessions),
                return_exceptions=True)
        finally:
            stop.set()
            await health
            for s in sessions:
                s.close()
        return endings, max(stalls)

    with patch("services.result_files.check_named_files", _files_checked), \
            patch("services.ticket_owner_ask.park_if_the_result_asks", _not_parked), \
            patch.object(bt, "_dispatch_task_complete", _completed), \
            patch.object(bt, "_auto_create_task_report", _reported):
        endings, worst_stall = asyncio.run(_two_endings())

    assert worst_stall < WORST_STALL, f"the event loop stalled {worst_stall:.2f}s"
    assert sorted(endings, key=repr) == ["done", None], endings  # one ending wins; the other leaves it
    assert completed == [task_id] and reported == [task_id]    # the owner is told once, one report
    status = new_session().execute(text("SELECT status FROM board_tasks WHERE id = :i"), {"i": task_id}).scalar()
    assert status == "done"
