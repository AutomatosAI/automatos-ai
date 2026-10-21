"""PRD-266 US-018a (D27): paused scheduled tasks stay in the calendar feed.

``services.activity_scheduled_task_items.scheduled_task_items`` (the source
``ActivityService.get_schedule`` delegates to) reads ``agent_scheduled_tasks``
rows that are active OR paused in one query; every task item carries ``paused``
(true for a paused row). A cancelled task never shows. A paused recurring task
sits at its next cron slot, a paused one-off at its ``next_run_at``.

The fake session answers the task query the way Postgres would: it keeps the rows
whose status the query's own ``t.status IN (...)`` filter names, read from the
SQL and its bound parameters. So the old ``status = 'active'`` query would drop
the paused rows here, and a query that let cancelled rows through would show
them. No database: CI runs these with the rest of the suite.
"""
from __future__ import annotations

import asyncio
import importlib.util as _ilu
import os
import re
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

# Dummy POSTGRES_* satisfies the config import chain (tests/test_schedule_endpoint.py's
# pattern). Nothing here touches a real DB.
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from services.activity_scheduled_task_items import scheduled_task_items  # noqa: E402
from services.activity_service import ActivityService  # noqa: E402

_needs_croniter = pytest.mark.skipif(
    _ilu.find_spec("croniter") is None, reason="croniter not installed (CI has it)"
)

_STATUS_FILTER = re.compile(r"t\.status\s+IN\s*\(([^)]*)\)", re.IGNORECASE)
NOW = datetime(2026, 10, 21, 10, 0, tzinfo=timezone.utc)
HORIZON = NOW + timedelta(days=7)
HOURLY = "0 * * * *"


class _Row:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _Query:
    def filter(self, *a, **k):
        return self

    def all(self):
        return []


def _filter_statuses(sql, params):
    """The statuses the query's ``t.status IN (...)`` filter names, literal or bound."""
    match = _STATUS_FILTER.search(sql)
    assert match, f"the scheduled-task query has no t.status IN (...) filter: {sql}"
    names = [part.strip() for part in match.group(1).split(",")]
    return {params[name[1:]] if name.startswith(":") else name.strip("'") for name in names}


class _TaskDB:
    """The first execute() is the scheduled-task query (get_schedule's order); it keeps
    the rows the query's status filter names, as Postgres would. A row with no status
    attribute passes (the older fakes' rows carry only some columns). Later execute()
    calls (mission and board-task SLAs) answer no rows; query() answers no agents and
    no templates."""

    def __init__(self, tasks):
        self._tasks = tasks
        self.executed = []
        self.rollbacks = 0

    def execute(self, stmt, params=None):
        sql, bound = str(stmt), dict(params or {})
        self.executed.append((sql, bound))
        if len(self.executed) > 1:
            return _Result([])
        statuses = _filter_statuses(sql, bound)
        return _Result([row for row in self._tasks if getattr(row, "status", None) in statuses
                        or not hasattr(row, "status")])

    def query(self, *cols):
        return _Query()

    def rollback(self):
        self.rollbacks += 1


def _task(task_id, status, *, schedule=None, next_run_at=None, **extra):
    return _Row(id=task_id, task_type="reminder", description=f"Task {task_id}", schedule=schedule,
                next_run_at=next_run_at, status=status, deliver_as="chat", payload_title=None,
                target_name="Ops", **extra)


def _by_id(items):
    return {item["scheduled_task_id"]: item for item in items}


# ── The source ──────────────────────────────────────────────────────────────

@_needs_croniter
def test_a_paused_recurring_task_shows_paused_at_its_next_cron_slot():
    db = _TaskDB([_task(1, "paused", schedule=HOURLY)])
    (item,) = scheduled_task_items(db, uuid4(), NOW, HORIZON)
    assert item["paused"] is True
    assert item["id"] == "task-1" and item["type"] == "task"
    assert item["next_run_at"] == (NOW + timedelta(hours=1)).isoformat()
    assert item["frequency"] == HOURLY and item["recurrence"]["cron_expression"] == HOURLY


def test_a_paused_one_off_shows_paused_at_its_next_run_at():
    when = NOW + timedelta(days=2)
    db = _TaskDB([_task(2, "paused", next_run_at=when.replace(tzinfo=None))])
    (item,) = scheduled_task_items(db, uuid4(), NOW, HORIZON)
    assert item["paused"] is True
    assert item["next_run_at"] == when.isoformat()
    assert item["frequency"] == "One-off" and item["recurrence"]["cron_expression"] is None


def test_an_active_task_shows_not_paused():
    db = _TaskDB([_task(3, "active", next_run_at=NOW + timedelta(hours=5))])
    (item,) = scheduled_task_items(db, uuid4(), NOW, HORIZON)
    assert item["paused"] is False


def test_a_cancelled_task_never_shows():
    soon = NOW + timedelta(hours=2)
    db = _TaskDB([
        _task(4, "cancelled", next_run_at=soon),
        _task(5, "active", next_run_at=soon),
        _task(6, "paused", next_run_at=soon),
    ])
    items = _by_id(scheduled_task_items(db, uuid4(), NOW, HORIZON))
    assert set(items) == {5, 6}
    assert items[5]["paused"] is False and items[6]["paused"] is True


def test_a_row_without_a_status_attribute_reads_as_not_paused():
    row = _Row(id=7, task_type="reminder", description="Older fake", schedule=None,
               next_run_at=NOW + timedelta(hours=1), target_name=None)
    (item,) = scheduled_task_items(_TaskDB([row]), uuid4(), NOW, HORIZON)
    assert item["paused"] is False
    assert item["deliver_as"] == "chat"  # unchanged default for a row with no deliver_as


def test_a_paused_task_past_the_horizon_does_not_show():
    db = _TaskDB([_task(8, "paused", next_run_at=HORIZON + timedelta(days=1))])
    assert scheduled_task_items(db, uuid4(), NOW, HORIZON) == []


def test_two_hundred_task_rows_make_one_execute():
    soon = NOW + timedelta(hours=3)
    rows = [_task(i, "paused" if i % 2 else "active", next_run_at=soon) for i in range(200)]
    db = _TaskDB(rows)
    items = scheduled_task_items(db, uuid4(), NOW, HORIZON)
    assert len(items) == 200
    assert len(db.executed) == 1
    assert sum(item["paused"] for item in items) == 100


def test_the_status_filter_names_exactly_active_and_paused():
    db = _TaskDB([])
    workspace_id = uuid4()
    scheduled_task_items(db, workspace_id, NOW, HORIZON)
    ((sql, params),) = db.executed
    assert _filter_statuses(sql, params) == {"active", "paused"}
    assert "t.status = 'active'" not in sql
    assert "t.status" in sql.split("SELECT", 1)[1].split("FROM", 1)[0]  # the status is selected
    assert params["ws"] == str(workspace_id)  # still scoped to the caller's workspace


# ── Through the feed ────────────────────────────────────────────────────────

def test_get_schedule_carries_paused_on_task_items_and_reads_tasks_first():
    soon = datetime.now(timezone.utc) + timedelta(hours=4)  # get_schedule reads the clock itself
    db = _TaskDB([_task(10, "paused", next_run_at=soon), _task(11, "active", next_run_at=soon),
                  _task(12, "cancelled", next_run_at=soon)])
    feed = ActivityService(db, uuid4()).get_schedule(range_days=30)
    tasks = _by_id(item for item in feed["scheduled"] if item["type"] == "task")
    assert set(tasks) == {10, 11}
    assert tasks[10]["paused"] is True and tasks[11]["paused"] is False
    assert "agent_scheduled_tasks" in db.executed[0][0]  # still the first execute() of the feed
    assert sum("agent_scheduled_tasks" in sql for sql, _ in db.executed) == 1


def test_the_platform_get_schedule_tool_sees_paused():
    from modules.tools.discovery.handlers_scheduling import get_schedule as schedule_tool

    db = _TaskDB([_task(20, "paused", next_run_at=datetime.now(timezone.utc) + timedelta(days=1))])
    out = asyncio.run(schedule_tool(db, uuid4(), {}))
    (item,) = [item for item in out["scheduled"] if item["type"] == "task"]
    assert item["scheduled_task_id"] == 20 and item["paused"] is True
