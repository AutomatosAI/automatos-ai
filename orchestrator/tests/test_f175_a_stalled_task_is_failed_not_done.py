"""F175 — a stalled board task ends 'failed', never 'done'.

Night 6, ticket #1094: status 'done', no result, and error_message "Stalled:
in_progress for >300s with no active execution (source=user)". The board showed
it as delivered. The reconciler's orphan sweep wrote 'done' with a raw UPDATE.

It now closes an orphan through the one completion writer (finalize_board_task_run)
with an error result. The task is 'failed' with the reason on its card, the owner
is told (task_failed), and its report says it stalled, as for any failed run.
"""
from __future__ import annotations

import asyncio
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

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
        pytest.skip(f"the reconciler tests need a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def workspace(engine, new_session):
    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'f175')"), {"id": ws})
    s.commit()
    yield ws
    s = new_session.sweep()
    s.execute(text("DELETE FROM board_tasks WHERE workspace_id = CAST(:w AS uuid)"), {"w": ws})
    s.execute(text("DELETE FROM workspaces WHERE id = CAST(:w AS uuid)"), {"w": ws})
    s.commit()


def _task(new_session, ws, minutes_ago):
    from core.models.core import BoardTask

    s = new_session()
    task = BoardTask(workspace_id=ws, title="Verify the welcome page", status="in_progress", priority="medium",
                     source_type="user", attempts=0,
                     started_at=datetime.now(timezone.utc) - timedelta(minutes=minutes_ago))
    s.add(task)
    s.commit()
    return task.id


def _row(new_session, task_id):
    return new_session().execute(
        text("SELECT status, error_message, result, completed_at FROM board_tasks WHERE id = :i"), {"i": task_id}).first()


def test_a_stalled_task_is_failed_and_says_it_stalled(workspace, new_session):
    from services.task_reconciler import TaskReconciler

    stalled = _task(new_session, workspace, minutes_ago=10)       # #1094's shape: no execution, no lease
    working = _task(new_session, workspace, minutes_ago=1)        # inside the window: left alone
    told, reported = [], []

    async def _told(db, ws, task):                       # read while the reconciler's session is open
        told.append(task.id)

    async def _reported(db, ws, task, exec_result):
        reported.append((task.id, exec_result.get("error") or ""))

    with patch("api.board_tasks._dispatch_task_failed", _told), \
            patch("api.board_tasks._auto_create_task_report", _reported):
        asyncio.run(TaskReconciler()._tick())

    status, error, result, completed_at = _row(new_session, stalled)
    assert status == "failed" and error.startswith("Stalled: in_progress for >")
    assert not result and completed_at is not None
    assert told == [stalled]                                                  # the owner is told it failed
    assert [(t, e.startswith("Stalled:")) for t, e in reported] == [(stalled, True)]  # and the report says why
    assert _row(new_session, working)[0] == "in_progress"


def test_two_writers_never_both_close_a_run(workspace, new_session):
    """Review MEDIUM: finalize read in_progress with no lock, so a run's own result
    and the stall sweep could both close the ticket and the later commit won,
    dropping the other's ending. The row is now locked: the second writer waits,
    sees the first writer's ending, and leaves it."""
    from api.board_tasks import finalize_board_task_run

    task_id = _task(new_session, workspace, minutes_ago=10)
    first = new_session()
    first.execute(text("SELECT id FROM board_tasks WHERE id = :i FOR UPDATE"), {"i": task_id})
    first.execute(text("UPDATE board_tasks SET status = 'failed', error_message = 'Stalled: test' WHERE id = :i"),
                  {"i": task_id})
    outcome = {}

    def _second():
        with patch("api.board_tasks._dispatch_task_complete", _quiet), \
                patch("api.board_tasks._auto_create_task_report", _quiet), \
                patch("api.board_tasks._dispatch_task_failed", _quiet):
            outcome["closed"] = asyncio.run(finalize_board_task_run(
                new_session(), task_id=task_id, workspace_id=workspace, agent_id=None,
                exec_result={"status": "success", "result": "Shop descriptions written."}))

    second = threading.Thread(target=_second)
    second.start()
    time.sleep(0.5)                                     # the second writer is at the row now
    first.commit()
    second.join(15)
    assert outcome == {"closed": None}                  # it saw the first ending and left it
    assert _row(new_session, task_id)[0] == "failed"


async def _quiet(*_a, **_k):
    return None
