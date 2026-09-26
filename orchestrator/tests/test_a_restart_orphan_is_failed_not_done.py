"""F175's twin — a ticket orphaned by a restart ends 'failed', never 'done'.

Refresh 5's boot (26 Sep 07:38Z) closed #1078 and #1091, two mission steps'
cards left in progress when night 6 stopped, as 'done' with
"orphaned_on_restart: executor lost on restart": the boot reaper wrote a raw
'done' ("the board has no 'failed' column" — it has had one since PRD-171).
F175 fixed the same thing in the reconciler's sweep. The boot reaper now closes
an orphan through the one completion writer too: 'failed' with the reason, the
owner told (task_failed), its report written.
"""
from __future__ import annotations

import asyncio
import inspect
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
        pytest.skip(f"the boot reaper tests need a reachable Postgres: {exc}")
    yield eng
    eng.dispose()


@pytest.fixture
def roastery(engine, new_session):
    ws = str(uuid.uuid4())
    s = new_session()
    s.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'restart')"), {"id": ws})
    agent = s.execute(text(
        "INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
        "VALUES ('Roaster', 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json)) RETURNING id"), {"w": ws}).scalar()
    s.commit()
    yield ws, agent
    s = new_session.sweep()
    for table, col in (("board_tasks", "workspace_id"), ("agents", "workspace_id"), ("workspaces", "id")):
        s.execute(text(f"DELETE FROM {table} WHERE {col} = CAST(:w AS uuid)"), {"w": ws})
    s.commit()


def _step_card(session, ws, agent, minutes_ago):
    """#1078's shape: a mission step's card, in progress when the process died."""
    return session.execute(text(
        "INSERT INTO board_tasks (workspace_id, title, status, assigned_agent_id, source_type, started_at) "
        "VALUES (CAST(:w AS uuid), 'Verify Roastery Page Content', 'in_progress', :a, 'orchestration_task', :t) "
        "RETURNING id"), {"w": ws, "a": agent, "t": datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)}
    ).scalar()


def test_a_restart_orphan_is_failed_says_why_and_is_told(roastery, new_session):
    import core.boot.reaper as reaper

    ws, agent = roastery
    seed = new_session()
    orphan, live = _step_card(seed, ws, agent, 90), _step_card(seed, ws, agent, 1)
    seed.commit()
    told, reported = [], []

    async def _told(db, workspace_id, task):
        told.append(task.id)

    async def _reported(db, workspace_id, task, exec_result):
        reported.append(task.id)

    with patch("api.board_tasks._dispatch_task_failed", _told), \
            patch("api.board_tasks._auto_create_task_report", _reported):
        swept = reaper.reap_orphaned_runs(new_session(), now=datetime.now(timezone.utc))
        if inspect.isawaitable(swept):
            asyncio.run(swept)

    rows = dict((r[0], r[1:]) for r in new_session().execute(text(
        "SELECT id, status, error_message FROM board_tasks WHERE id IN (:o, :l)"), {"o": orphan, "l": live}))
    status, error = rows[orphan]
    assert status == "failed"                                    # night: 'done'
    assert error == "orphaned_on_restart: executor lost on restart"
    assert orphan in told and orphan in reported                 # the owner is told, the report written
    assert rows[live][0] == "in_progress" and live not in told   # inside the window: left alone
