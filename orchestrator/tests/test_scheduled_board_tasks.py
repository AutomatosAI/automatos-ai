"""Calendar — a scheduled task can file a board ticket when it fires.

Fake-session tests (the repo's idiom for the raw-DDL ``agent_scheduled_tasks``
table: no model, no DB in CI) for the three surfaces the feature adds:

1. ``ScheduledTaskService.create_task`` accepts ``deliver_as='board_task'`` with
   an operator creator and no target (Inbox), refuses the shapes that cannot
   work, and writes the new columns.
2. ``_file_board_task`` files the ticket the way the HTTP create path does:
   status from the target, SLA from the priority, source = the scheduled task,
   consent for an operator-scheduled assigned ticket, SSE push + dispatcher wake.
3. The tool handler and the executor thread board delivery and the driving
   human through; the REST ``POST /api/v1/scheduled-tasks`` is gated and calls
   the service with board delivery.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from services import scheduled_task_service as sts  # noqa: E402
from services.scheduled_task_service import (  # noqa: E402
    DELIVER_BOARD_TASK,
    DELIVER_CHAT,
    ScheduledTaskService,
)

WS = uuid4()


class _Row:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class _Result:
    def __init__(self, rows=None, scalar=None, one=None):
        self._rows, self._scalar, self._one = rows or [], scalar, one

    def fetchall(self):
        return self._rows

    def scalar(self):
        return self._scalar

    def fetchone(self):
        return self._one


class _FakeDB:
    """Answers by SQL shape: agents lookup → rows, COUNT → scalar, INSERT → the new row."""

    def __init__(self, agents=None, counts=0):
        self.agents = agents if agents is not None else [_Row(id=3, name="Ops")]
        self.counts = counts
        self.calls: list[tuple[str, dict]] = []
        self.commits = 0

    def execute(self, stmt, params=None):
        sql = str(stmt)
        self.calls.append((sql, params or {}))
        if "FROM agents" in sql:
            return _Result(rows=self.agents)
        if "COUNT(*)" in sql:
            return _Result(scalar=self.counts)
        if "INSERT INTO agent_scheduled_tasks" in sql:
            return _Result(one=_Row(id=77, created_at=datetime.now(timezone.utc)))
        return _Result()

    def commit(self):
        self.commits += 1


def _create(db, **over):
    kwargs = dict(
        created_by_agent_id=None,
        target_agent_id=None,
        task_type="one_shot",
        description="Review the invoices",
        schedule=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat(),
        deliver_as=DELIVER_BOARD_TASK,
        payload={"title": "Review invoices", "priority": "high", "review_mode": "auto", "tags": ["finance"]},
        created_by_user_id="user_1",
    )
    kwargs.update(over)
    return asyncio.run(ScheduledTaskService(db, WS).create_task(**kwargs))


# ── 1. create_task ───────────────────────────────────────────────────────────

def test_operator_schedules_an_unassigned_board_task():
    db = _FakeDB()
    out = _create(db)
    assert out["success"] is True, out
    assert out["deliver_as"] == DELIVER_BOARD_TASK
    assert out["title"] == "Review invoices"
    assert out["target_agent"] is None
    assert "Inbox" in out["message"]
    insert = next(p for sql, p in db.calls if "INSERT INTO agent_scheduled_tasks" in sql)
    assert insert["deliver_as"] == DELIVER_BOARD_TASK
    assert insert["created_by"] is None and insert["target"] is None
    assert insert["created_by_user_id"] == "user_1"
    assert json.loads(insert["payload"])["title"] == "Review invoices"
    assert db.commits == 1


def test_operator_rows_share_a_workspace_cap_not_a_per_agent_one():
    db = _FakeDB(counts=sts.MAX_OPERATOR_TASKS_PER_WORKSPACE)
    out = _create(db)
    assert out["success"] is False
    assert "scheduled board tasks" in out["error"]
    count_sql = next(sql for sql, _ in db.calls if "COUNT(*)" in sql)
    assert "created_by_user_id IS NOT NULL" in count_sql


def test_chat_delivery_still_needs_a_target_agent():
    out = _create(_FakeDB(), deliver_as=DELIVER_CHAT, created_by_agent_id=3, payload=None)
    assert out["success"] is False and "target agent" in out["error"]


def test_a_board_task_needs_a_title_and_a_creator():
    assert _create(_FakeDB(), payload={"title": "  "})["success"] is False
    assert _create(_FakeDB(), created_by_user_id=None)["success"] is False


def test_unknown_delivery_mode_is_refused():
    assert _create(_FakeDB(), deliver_as="carrier_pigeon")["success"] is False


def test_assigned_board_task_checks_the_target_lives_in_the_workspace():
    out = _create(_FakeDB(agents=[]), target_agent_id=3)
    assert out["success"] is False and "Target agent 3 not found" in out["error"]


# ── 2. _file_board_task ──────────────────────────────────────────────────────

class _FilingDB:
    def __init__(self):
        self.added = []
        self.commits = 0

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.commits += 1

    def refresh(self, obj):
        obj.id = 501


def _task(**over):
    base = dict(
        id=77, workspace_id=WS, description="Review the invoices\nline two",
        payload={"title": "Review invoices", "priority": "urgent", "review_mode": "human", "tags": ["finance"]},
        target_agent_id=3, created_by_user_id="user_1", created_by_agent_id=None, origin_chat_id=None,
        deliver_as=DELIVER_BOARD_TASK,
    )
    base.update(over)
    return SimpleNamespace(**base)


def _capture_filing(monkeypatch):
    seen = {"consent": [], "events": [], "wakes": [], "messages": []}
    import services.board_consent as consent
    import services.board_dispatcher as dispatcher
    import services.board_events as events
    import services.chat_messenger as messenger

    monkeypatch.setattr(consent, "consent_for_created_ticket",
                        lambda db, **kw: seen["consent"].append(kw) or "created")
    monkeypatch.setattr(events, "notify_board_event", lambda db, **kw: seen["events"].append(kw))
    monkeypatch.setattr(dispatcher, "notify_task_available", lambda db, **kw: seen["wakes"].append(kw))
    monkeypatch.setattr(messenger, "deliver_background_message", lambda db, **kw: seen["messages"].append(kw))
    return seen


def test_filing_an_assigned_ticket_mirrors_the_http_create_path(monkeypatch):
    seen = _capture_filing(monkeypatch)
    db = _FilingDB()
    ScheduledTaskService._file_board_task(db, _task(), 77)

    ticket = db.added[0]
    assert ticket.title == "Review invoices"
    assert ticket.description.startswith("Review the invoices")
    assert ticket.status == "assigned" and ticket.assigned_agent_id == 3
    assert ticket.priority == "urgent" and ticket.review_mode == "human"
    assert ticket.tags == ["finance"]
    assert ticket.source_type == "scheduled_task" and ticket.source_id.startswith("task:77:")
    assert ticket.created_by_type == "user" and ticket.created_by_id == "user_1"
    # urgent = 4h SLA, stamped at filing time
    assert timedelta(hours=3, minutes=59) < ticket.sla_deadline - datetime.now(timezone.utc) <= timedelta(hours=4)
    # consent (the operator scheduled AND assigned it), SSE push, dispatcher wake
    assert seen["consent"][0]["actor"] == "user:user_1"
    assert seen["events"][0]["event"] == "task_created" and seen["events"][0]["task_id"] == 501
    assert seen["wakes"][0]["task_id"] == 501
    assert seen["messages"] == []  # no origin chat → nothing to tell


def test_filing_an_unassigned_ticket_goes_to_the_inbox_without_a_wake(monkeypatch):
    seen = _capture_filing(monkeypatch)
    db = _FilingDB()
    ScheduledTaskService._file_board_task(db, _task(target_agent_id=None, payload={"title": ""}), 77)
    ticket = db.added[0]
    assert ticket.status == "inbox" and ticket.assigned_agent_id is None
    assert ticket.title == "Review the invoices"  # first line of the description
    assert ticket.priority == "medium"  # unknown/missing priority collapses to the default
    assert seen["wakes"] == []
    assert seen["events"][0]["status"] == "inbox"


def test_an_agent_scheduled_ticket_records_no_operator_consent_but_tells_the_chat(monkeypatch):
    seen = _capture_filing(monkeypatch)
    db = _FilingDB()
    ScheduledTaskService._file_board_task(
        db, _task(created_by_user_id=None, created_by_agent_id=9, origin_chat_id="chat-1"), 77,
    )
    ticket = db.added[0]
    assert ticket.created_by_type == "agent" and ticket.created_by_id == "9"
    assert seen["consent"] == []
    assert seen["messages"][0]["chat_id"] == "chat-1"
    assert seen["messages"][0]["link_type"] == "board_task" and seen["messages"][0]["link_id"] == "501"


# ── 3. tool handler, executor, REST ──────────────────────────────────────────

def test_tool_handler_files_an_unassigned_board_task_with_the_driving_human(monkeypatch):
    from modules.tools.discovery import handlers_scheduling as h

    captured = {}

    async def _fake_create(self, **kw):
        captured.update(kw)
        return {"success": True, "task_id": 1}

    monkeypatch.setattr(ScheduledTaskService, "create_task", _fake_create)
    out = asyncio.run(h.schedule_task(SimpleNamespace(), WS, {
        "_agent_id": 5, "_user_id": "user_1", "task_type": "one_shot",
        "description": "Send the weekly report\nwith the numbers", "schedule": "2030-01-01T09:00:00Z",
        "deliver_as": "board_task", "priority": "high", "tags": ["reports"],
    }))
    assert out["success"] is True
    assert captured["deliver_as"] == DELIVER_BOARD_TASK
    assert captured["target_agent_id"] is None  # no name given → Inbox, never self
    assert captured["created_by_agent_id"] == 5 and captured["created_by_user_id"] == "user_1"
    assert captured["payload"] == {"title": "Send the weekly report", "priority": "high",
                                   "review_mode": "auto", "tags": ["reports"]}


def test_tool_handler_chat_delivery_defaults_to_self(monkeypatch):
    from modules.tools.discovery import handlers_scheduling as h

    captured = {}

    async def _fake_create(self, **kw):
        captured.update(kw)
        return {"success": True, "task_id": 1}

    monkeypatch.setattr(ScheduledTaskService, "create_task", _fake_create)
    asyncio.run(h.schedule_task(SimpleNamespace(), WS, {
        "_agent_id": 5, "task_type": "one_shot", "description": "check", "schedule": "2030-01-01T09:00:00Z",
    }))
    assert captured["deliver_as"] == DELIVER_CHAT and captured["target_agent_id"] == 5
    assert captured["payload"] is None and captured["created_by_user_id"] is None


def test_executor_threads_the_driving_human_into_schedule_task():
    from modules.tools.discovery.platform_executor import OPERATOR_CONSENT_ACTIONS

    assert "platform_schedule_task" in OPERATOR_CONSENT_ACTIONS
    assert "platform_create_task" in OPERATOR_CONSENT_ACTIONS  # the PRD-234 originals stay


def test_schedule_task_tool_schema_offers_board_delivery():
    from modules.tools.discovery.action_registry import ActionRegistry
    from modules.tools.discovery.actions_scheduling import register_scheduling_actions

    registry = ActionRegistry()
    register_scheduling_actions(registry)
    action = registry.get("platform_schedule_task")
    props = action.parameters["properties"]
    assert props["deliver_as"]["enum"] == ["chat", "board_task"]
    assert set(props) >= {"title", "priority", "review_mode", "tags"}
    assert "required" in action.parameters and "deliver_as" not in action.parameters["required"]


def test_rest_post_is_gated_and_schedules_a_board_task(monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import api.scheduled_tasks as mod
    from core.database.database import get_db

    app = FastAPI()
    app.include_router(mod.router)
    app.dependency_overrides[get_db] = lambda: SimpleNamespace(name="dummy-session")
    app.dependency_overrides[mod.get_request_context_hybrid] = lambda: SimpleNamespace(
        workspace_id=WS, user=SimpleNamespace(clerk_user_id="user_1", id=1),
    )
    # The route must carry the workspace-permission gate (the PRD-195 authz sweep's
    # class (a)); override that exact dependency so the request reaches the handler.
    route = next(r for r in mod.router.routes if getattr(r, "path", "") == "/api/v1/scheduled-tasks"
                 and "POST" in getattr(r, "methods", set()))
    gates = [d.call for d in route.dependant.dependencies
             if "require_workspace_permission" in getattr(d.call, "__qualname__", "")]
    assert gates, "POST /api/v1/scheduled-tasks must be gated by require_workspace_permission"
    for gate in gates:
        app.dependency_overrides[gate] = lambda: None

    captured = {}

    async def _fake_create(self, **kw):
        captured.update(kw)
        return {"success": True, "task_id": 9, "deliver_as": "board_task", "next_run_at": None}

    monkeypatch.setattr(mod.ScheduledTaskService, "create_task", _fake_create)
    client = TestClient(app)

    r = client.post("/api/v1/scheduled-tasks", json={
        "title": "  Review invoices ", "schedule": "2030-01-01T09:00:00Z",
        "priority": "high", "assigned_agent_id": 3, "tags": ["finance", ""],
    })
    assert r.status_code == 200, r.text
    assert captured["deliver_as"] == DELIVER_BOARD_TASK
    assert captured["created_by_agent_id"] is None and captured["target_agent_id"] == 3
    assert captured["created_by_user_id"] == "user_1"
    assert captured["payload"] == {"title": "Review invoices", "priority": "high", "review_mode": "auto",
                                   "tags": ["finance"]}
    assert captured["description"] == "Review invoices"  # empty description falls back to the title

    assert client.post("/api/v1/scheduled-tasks", json={"title": " ", "schedule": "x"}).status_code == 422
    assert client.post("/api/v1/scheduled-tasks", json={"title": "t", "schedule": "x", "priority": "asap"}).status_code == 422
