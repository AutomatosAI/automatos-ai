"""PRD-268 US-C11b (§6.3, K8) — a follow-up is its person's own.

A follow-up is one ``agent_scheduled_tasks`` row (``deliver_as 'follow_up'``) whose reason
Auto wrote from a person's private chat. Every generic reader and writer of the table leaves
it out, and the scheduler's own reads keep it, so it still fires. Proven on Postgres, with
one follow-up beside ordinary rows (a chat schedule, an operator's board task, and another
workspace's row):

* ``ScheduledTaskService.list_tasks`` (``platform_list_scheduled_tasks``, ``GET
  /api/v1/scheduled-tasks``) lists only the ordinary rows, and its total counts only them,
  with every filter; a listed row keeps the shape it always had.
* ``_limit_error`` and ``_recurring_limit_error`` never count a follow-up toward
  ``MAX_TASKS_PER_AGENT``, ``MAX_OPERATOR_TASKS_PER_WORKSPACE`` or
  ``MAX_RECURRING_PER_WORKSPACE``; ordinary rows still reach each cap.
* ``update_task_status`` of a follow-up answers "Task <id> not found" and changes nothing
  (no status, no stopped session, no rescheduled job); so does every generic route (PATCH
  status, cancel, pause, resume, delete) and ``platform_cancel_scheduled_task``; an ordinary
  row still changes.
* The Command Center calendar, W1's skipped-run signal and the board snapshot carry only the
  ordinary rows, and a confirmation card never names a follow-up's reason.
* ``reconcile_with_scheduler`` and ``load_active_tasks_to_scheduler`` still register it.

``agent_scheduled_tasks`` is made by migrations only (no model), so the CI database, which
``create_all`` builds, lacks it: each test makes it as a temporary table in the shape this
wave's revision leaves (a temporary table shadows any permanent one), rolled back with the
rest.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException
from sqlalchemy import text

from services import scheduled_task_service as sts
from services.scheduled_task_service import ScheduledTaskService
from services.scheduled_task_steps import DELIVER_BOARD_TASK, DELIVER_CHAT, DELIVER_FOLLOW_UP

# The shape prd77 + calendar_scheduled_board_tasks + prd205_auto_speaks + prd268_w3_follow_ups
# leave. A temporary table may not reference a permanent one, so no foreign keys.
SCHEDULED_TASKS = """
    CREATE TEMP TABLE agent_scheduled_tasks (
        id SERIAL PRIMARY KEY, workspace_id UUID NOT NULL, created_by_agent_id INTEGER,
        target_agent_id INTEGER, task_type VARCHAR(20) NOT NULL DEFAULT 'one_shot',
        description TEXT NOT NULL, schedule VARCHAR(100) NOT NULL, max_runs INTEGER,
        run_count INTEGER NOT NULL DEFAULT 0, status VARCHAR(20) NOT NULL DEFAULT 'active',
        last_run_at TIMESTAMPTZ, next_run_at TIMESTAMPTZ, last_error TEXT,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(), updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        created_by_user_id VARCHAR(255), deliver_as VARCHAR(20) NOT NULL DEFAULT 'chat', payload JSONB,
        origin_chat_id UUID, for_user_id INTEGER,
        CONSTRAINT ck_scheduled_task_deliver_as CHECK (deliver_as IN ('chat', 'board_task', 'follow_up')))
"""
REASON = "Check whether the Lisbon card moved; Ana said the landlord may pull out"
PERSON = 7
HOUR = timedelta(hours=1)
LISTED_KEYS = {
    "id", "task_type", "deliver_as", "title", "payload", "created_by_user_id", "description", "schedule",
    "status", "creator_agent", "target_agent", "run_count", "max_runs", "next_run_at", "last_run_at",
    "last_error", "created_at",
}


@pytest.fixture
def db(db_session):
    db_session.execute(text(SCHEDULED_TASKS))
    return db_session


@pytest.fixture
def now():
    return datetime.now(timezone.utc).replace(microsecond=0)


def _workspace(db) -> UUID:
    ws = uuid4()
    db.execute(text("INSERT INTO workspaces (id, name) VALUES (CAST(:id AS uuid), 'prd268 w3 private')"),
               {"id": str(ws)})
    return ws


def _agent(db, ws: UUID, name: str) -> int:
    return int(db.execute(text(
        "INSERT INTO agents (name, agent_type, workspace_id, status, configuration) "
        "VALUES (:n, 'custom', CAST(:w AS uuid), 'active', CAST('{}' AS json)) RETURNING id"),
        {"n": name, "w": str(ws)}).scalar())


def _row(db, ws: UUID, run_at: datetime, *, deliver_as: str = DELIVER_CHAT, agent=None, user=None,
         task_type: str = "one_shot", description: str = "Send the morning numbers", payload=None,
         for_user_id=None) -> int:
    schedule = "0 9 * * *" if task_type == "recurring" else run_at.isoformat()
    return int(db.execute(text(
        "INSERT INTO agent_scheduled_tasks (workspace_id, created_by_agent_id, target_agent_id, task_type, "
        "description, schedule, next_run_at, deliver_as, payload, created_by_user_id, for_user_id) "
        "VALUES (CAST(:ws AS uuid), :agent, :agent, :type, :what, :schedule, :at, :deliver, CAST(:payload AS jsonb), "
        ":user, :person) RETURNING id"),
        {"ws": str(ws), "agent": agent, "type": task_type, "what": description, "schedule": schedule,
         "at": run_at, "deliver": deliver_as, "payload": json.dumps(payload) if payload else None, "user": user,
         "person": for_user_id}).scalar())


def _follow_up(db, ws: UUID, auto: int, run_at: datetime, **over) -> int:
    return _row(db, ws, run_at, deliver_as=DELIVER_FOLLOW_UP, agent=auto, description=REASON,
                payload={"about": {"kind": "other"}}, for_user_id=PERSON, **over)


def _seed(db, now: datetime) -> SimpleNamespace:
    """One follow-up beside the ordinary rows a workspace has, and another workspace's row."""
    ws = _workspace(db)
    auto = _agent(db, ws, "Auto")
    seed = SimpleNamespace(ws=ws, auto=auto)
    seed.chat = _row(db, ws, now + HOUR, agent=auto)
    seed.board = _row(db, ws, now + 2 * HOUR, deliver_as=DELIVER_BOARD_TASK, user="user_1",
                      description="Review the invoices", payload={"title": "Review invoices", "priority": "high"})
    seed.follow_up = _follow_up(db, ws, auto, now + HOUR)
    seed.elsewhere = _row(db, _workspace(db), now + HOUR)
    return seed


def _status(db, task_id: int) -> str:
    return db.execute(text("SELECT status FROM agent_scheduled_tasks WHERE id = :id"), {"id": task_id}).scalar()


@pytest.fixture
def quiet(monkeypatch):
    """What a status change does beyond the row (the firing in flight, the scheduler's job), recorded."""
    import services.board_cancel as board_cancel

    seen = SimpleNamespace(stopped=[], rescheduled=[])
    monkeypatch.setattr(board_cancel, "stop_scheduled_task_sessions",
                        lambda db, ws, task_id, status: seen.stopped.append(task_id))
    monkeypatch.setattr(ScheduledTaskService, "_reschedule",
                        lambda self, task_id, status: seen.rescheduled.append(task_id))
    return seen


# --------------------------------------------------------------------------- the list


def test_the_list_and_its_total_leave_the_follow_up_out_with_every_filter(db, now):
    seed = _seed(db, now)
    svc = ScheduledTaskService(db, seed.ws)

    every = asyncio.run(svc.list_tasks())
    assert {t["id"] for t in every["tasks"]} == {seed.chat, seed.board} and every["total"] == 2
    by_auto = asyncio.run(svc.list_tasks(agent_id=seed.auto))
    assert [t["id"] for t in by_auto["tasks"]] == [seed.chat] and by_auto["total"] == 1
    active = asyncio.run(svc.list_tasks(status="active", limit=1))
    assert len(active["tasks"]) == 1 and active["total"] == 2
    assert REASON not in json.dumps([every, by_auto, active])


def test_a_listed_row_keeps_the_shape_it_always_had(db, now):
    seed = _seed(db, now)
    listed = asyncio.run(ScheduledTaskService(db, seed.ws).list_tasks())
    rows = {t["id"]: t for t in listed["tasks"]}
    board, chat = rows[seed.board], rows[seed.chat]
    assert set(board) == set(chat) == LISTED_KEYS
    assert (board["deliver_as"], board["title"], board["created_by_user_id"]) == (DELIVER_BOARD_TASK, "Review invoices",
                                                                                   "user_1")
    assert board["payload"] == {"title": "Review invoices", "priority": "high"} and board["target_agent"] is None
    assert (chat["deliver_as"], chat["title"], chat["payload"]) == (DELIVER_CHAT, None, None)
    assert (chat["creator_agent"], chat["target_agent"], chat["status"]) == ("Auto", "Auto", "active")
    assert datetime.fromisoformat(chat["next_run_at"]) == now + HOUR and chat["last_run_at"] is None


def test_the_list_tool_and_route_leave_the_follow_up_out(db, now):
    from api import scheduled_tasks as api
    from modules.tools.discovery.handlers_scheduling import list_scheduled_tasks

    seed = _seed(db, now)
    tool = asyncio.run(list_scheduled_tasks(db, seed.ws, {}))
    route = asyncio.run(api.list_scheduled_tasks(agent_id=None, status=None, limit=20, offset=0,
                                                 ctx=SimpleNamespace(workspace_id=seed.ws), db=db))
    for listed in (tool, route):
        assert {t["id"] for t in listed["tasks"]} == {seed.chat, seed.board} and listed["total"] == 2


# --------------------------------------------------------------------------- the caps


def test_auto_s_follow_ups_never_count_toward_its_own_schedules(db, now, monkeypatch):
    monkeypatch.setattr(sts, "MAX_TASKS_PER_AGENT", 2)
    ws = _workspace(db)
    auto = _agent(db, ws, "Auto")
    for hours in range(1, 6):
        _follow_up(db, ws, auto, now + hours * HOUR)
    _row(db, ws, now + HOUR, agent=auto)
    svc = ScheduledTaskService(db, ws)
    assert svc._limit_error(auto, "one_shot") is None            # five follow-ups and one schedule: under 2
    _row(db, ws, now + 2 * HOUR, agent=auto)
    assert "limit of 2 active tasks" in svc._limit_error(auto, "one_shot")


def test_the_operator_and_recurring_caps_never_count_a_follow_up(db, now, monkeypatch):
    monkeypatch.setattr(sts, "MAX_OPERATOR_TASKS_PER_WORKSPACE", 1)
    monkeypatch.setattr(sts, "MAX_RECURRING_PER_WORKSPACE", 1)
    ws = _workspace(db)
    auto = _agent(db, ws, "Auto")
    _follow_up(db, ws, auto, now + HOUR, user="user_7", task_type="recurring")  # each column a cap reads
    svc = ScheduledTaskService(db, ws)
    assert svc._limit_error(None, "recurring") is None and svc._recurring_limit_error("recurring") is None
    _row(db, ws, now + HOUR, deliver_as=DELIVER_BOARD_TASK, user="user_1", task_type="recurring",
         payload={"title": "Weekly invoices"})
    assert "scheduled board tasks" in svc._limit_error(None, "one_shot")
    assert "recurring tasks" in svc._recurring_limit_error("recurring")


# --------------------------------------------------------------------------- the status writers


@pytest.mark.parametrize("status", ["cancelled", "paused", "active"])
def test_a_follow_up_is_not_found_by_update_task_status_and_nothing_changes(db, now, quiet, status):
    seed = _seed(db, now)
    db.execute(text("UPDATE agent_scheduled_tasks SET status = 'paused' WHERE id = :id"), {"id": seed.chat})
    svc = ScheduledTaskService(db, seed.ws)

    out = asyncio.run(svc.update_task_status(seed.follow_up, status))
    assert out == {"success": False, "error": f"Task {seed.follow_up} not found"}
    assert _status(db, seed.follow_up) == "active" and quiet.stopped == quiet.rescheduled == []

    ordinary = asyncio.run(svc.update_task_status(seed.chat, status))
    assert ordinary == {"success": True, "task_id": seed.chat, "status": status}
    assert _status(db, seed.chat) == status and quiet.stopped == quiet.rescheduled == [seed.chat]


def test_every_generic_route_and_the_cancel_tool_answer_not_found(db, now, quiet):
    from api import scheduled_tasks as api
    from modules.tools.discovery.handlers_scheduling import cancel_scheduled_task

    seed = _seed(db, now)
    ctx = SimpleNamespace(workspace_id=seed.ws)
    calls = (
        lambda: api.update_task_status(seed.follow_up, api.UpdateStatusRequest(status="cancelled"), ctx, db),
        lambda: api.cancel_task(seed.follow_up, ctx, db), lambda: api.pause_task(seed.follow_up, ctx, db),
        lambda: api.resume_task(seed.follow_up, ctx, db), lambda: api.delete_task(seed.follow_up, ctx, db),
    )
    for call in calls:
        with pytest.raises(HTTPException) as refused:
            asyncio.run(call())
        assert refused.value.status_code == 404 and REASON not in str(refused.value.detail)
    tool = asyncio.run(cancel_scheduled_task(db, seed.ws, {"task_id": seed.follow_up}))
    assert tool == {"success": False, "error": f"Task {seed.follow_up} not found"}
    assert _status(db, seed.follow_up) == "active" and quiet.stopped == []
    assert asyncio.run(api.cancel_task(seed.chat, ctx, db))["status"] == "cancelled"


def test_a_confirmation_card_never_names_a_follow_up(db, now):
    from modules.tools.execution.subject_targets import missing_targets_error, resolve_targets

    seed = _seed(db, now)
    action = "platform_cancel_scheduled_task"
    found, missing = resolve_targets(db, seed.ws, {"task_id": seed.follow_up}, action)
    assert found == [] and [(t.ident, t.noun, t.name) for t in missing] == [(seed.follow_up, "scheduled task", None)]
    assert REASON not in json.dumps(missing_targets_error(action, missing))
    found, missing = resolve_targets(db, seed.ws, {"task_id": seed.chat}, action)
    assert [(t.ident, t.name) for t in found] == [(seed.chat, "Send the morning numbers")] and missing == []


class _FollowUpCheckFails:
    """The name reads; the follow-up check errors (the lookup's own failure path)."""

    def execute(self, stmt, params=None):
        if "deliver_as" in str(stmt):
            raise RuntimeError("the follow-up check failed")
        return SimpleNamespace(first=lambda: (REASON,))


def test_a_follow_up_check_that_errors_names_nothing():
    from modules.tools.execution.subject_targets import resolve_targets

    found, missing = resolve_targets(_FollowUpCheckFails(), uuid4(), {"task_id": 41}, "platform_cancel_scheduled_task")
    assert [(t.ident, t.noun, t.name) for t in found] == [(41, "scheduled task", None)] and missing == []


# --------------------------------------------------------------------------- the other readers


def test_the_command_center_calendar_leaves_the_follow_up_out(db, now):
    from services.activity_scheduled_task_items import scheduled_task_items

    seed = _seed(db, now)
    items = scheduled_task_items(db, seed.ws, now, now + timedelta(days=1))
    assert sorted(item["scheduled_task_id"] for item in items) == sorted([seed.chat, seed.board])
    assert REASON not in json.dumps(items)


def test_the_skipped_run_signal_never_carries_a_follow_up(db, now):
    from services.proactive import signals

    ws = _workspace(db)
    auto = _agent(db, ws, "Auto")
    skipped = _row(db, ws, now - HOUR / 2, agent=auto)
    _follow_up(db, ws, auto, now - HOUR / 2)                     # due as long ago, never run
    found = signals._skipped_runs(db, signals._Window(ws, now - HOUR, now + timedelta(minutes=1)))
    assert [signal.ref for signal in found] == [("scheduled_task", str(skipped))]
    assert REASON not in repr(found)


def test_the_board_snapshot_leaves_the_follow_up_out(db, now):
    from modules.tools.discovery.handlers_analytics import _scheduled

    seed = _seed(db, now)
    listed = _scheduled(db, seed.ws)
    assert sorted(item["id"] for item in listed) == sorted([seed.chat, seed.board])
    assert REASON not in json.dumps(listed)


# --------------------------------------------------------------------------- the scheduler's own reads


class _Scheduler:
    running = True

    def get_jobs(self):
        return []

    def remove_job(self, job_id):
        raise AssertionError(f"nothing is removed: {job_id}")


def test_the_scheduler_still_registers_a_follow_up(db, now, monkeypatch):
    seed = _seed(db, now)
    registered = []
    monkeypatch.setattr(ScheduledTaskService, "_register_with_scheduler",
                        lambda self, task_id, task_type, schedule, agent: registered.append(task_id))
    svc = ScheduledTaskService(db, workspace_id=None)

    assert svc.reconcile_with_scheduler(_Scheduler())["added"] == 4
    assert sorted(registered) == sorted([seed.chat, seed.board, seed.follow_up, seed.elsewhere])
    registered.clear()
    assert asyncio.run(svc.load_active_tasks_to_scheduler()) == 4 and seed.follow_up in registered
