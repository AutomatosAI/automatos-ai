"""PRD-141: list_board_tasks projection must surface `tags` and `description`.

The HARNESS command path keys on these two fields: rx_id lives ONLY in a task's
tags (`rx:{id}`), and _parse_harness_task reads Current/Proposed out of the
description. Before this fix the projection dropped both, so _find_task_by_rx
always returned "no pending change" and _parse_harness_task saw an empty body —
the bug only stayed hidden because the unit suites used an in-memory fake
executor instead of this real handler.

The handler is loaded directly from its file: importing the `modules.tools`
package pulls in the RAG/multimodal chain (camelot) that isn't installed in the
unit env. The function's `from core.models.core import BoardTask` runs at call
time and resolves the real ORM class, so the SQLAlchemy class-attribute
expressions are genuine — the fake query just ignores them. The JSONB `tags`
*filter* (`tags @> [...]`) needs a real Postgres and is exercised at the
US-026 live gate, not here.
"""
import asyncio
import importlib.util
import os
import sys
import types
from uuid import UUID

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "test")


def _install_fake_apscheduler():
    """Idempotently provide the apscheduler surface heartbeat_service imports at
    module load. Uses setdefault so a real apscheduler install (CI) always wins;
    only the missing pieces are stubbed. Covers schedulers.asyncio,
    jobstores.memory AND triggers.cron (the last is what a direct file-load of
    services/heartbeat_service.py needs)."""
    aps = sys.modules.setdefault("apscheduler", types.ModuleType("apscheduler"))
    schedulers = sys.modules.setdefault(
        "apscheduler.schedulers", types.ModuleType("apscheduler.schedulers")
    )
    asyncio_mod = sys.modules.setdefault(
        "apscheduler.schedulers.asyncio",
        types.ModuleType("apscheduler.schedulers.asyncio"),
    )
    if not hasattr(asyncio_mod, "AsyncIOScheduler"):
        asyncio_mod.AsyncIOScheduler = type("AsyncIOScheduler", (), {})
    jobstores = sys.modules.setdefault(
        "apscheduler.jobstores", types.ModuleType("apscheduler.jobstores")
    )
    memory_mod = sys.modules.setdefault(
        "apscheduler.jobstores.memory", types.ModuleType("apscheduler.jobstores.memory")
    )
    if not hasattr(memory_mod, "MemoryJobStore"):
        memory_mod.MemoryJobStore = type("MemoryJobStore", (), {})
    triggers = sys.modules.setdefault(
        "apscheduler.triggers", types.ModuleType("apscheduler.triggers")
    )
    cron_mod = sys.modules.setdefault(
        "apscheduler.triggers.cron", types.ModuleType("apscheduler.triggers.cron")
    )
    if not hasattr(cron_mod, "CronTrigger"):
        cron_mod.CronTrigger = type("CronTrigger", (), {})
    aps.schedulers = schedulers
    aps.jobstores = jobstores
    aps.triggers = triggers
    schedulers.asyncio = asyncio_mod
    jobstores.memory = memory_mod
    triggers.cron = cron_mod


_install_fake_apscheduler()


def _load_handler():
    """Load handlers_board_tasks.py directly, skipping the package __init__."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "modules", "tools", "discovery", "handlers_board_tasks.py")
    spec = importlib.util.spec_from_file_location("hbt_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_HANDLER = _load_handler()
_WS_ID = UUID("00000000-0000-0000-0000-000000000001")


class _FakeRow:
    """A BoardTask-shaped row carrying exactly the attrs the projection reads."""

    def __init__(self, **kw):
        self.id = kw.get("id")
        self.title = kw.get("title")
        self.description = kw.get("description")
        self.status = kw.get("status", "review")
        self.priority = kw.get("priority", "medium")
        self.tags = kw.get("tags")
        self.assigned_agent_id = kw.get("assigned_agent_id")  # None -> skip Agent query
        self.created_at = kw.get("created_at")
        self.started_at = kw.get("started_at")
        self.completed_at = kw.get("completed_at")
        self.error_message = kw.get("error_message")


class _FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *a, **k):
        return self

    def order_by(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def all(self):
        return self._rows


class _FakeDB:
    def __init__(self, rows):
        self._rows = rows

    def query(self, model):
        return _FakeQuery(self._rows)


def test_list_board_tasks_projects_tags_and_description():
    """A queued HARNESS task survives the projection with its tags + description
    intact, so the command path can find it by rx tag and parse its body."""
    rows = [
        _FakeRow(
            id=7,
            title="[HARNESS] heartbeat_tune for ScribeAgent",
            description="**Current:** {...}\n**Proposed:** {...}",
            status="review",
            tags=["harness", "org-review", "risk-4", "rx:rx-esc-1"],
            assigned_agent_id=None,
        )
    ]
    db = _FakeDB(rows)

    result = asyncio.run(
        _HANDLER.list_board_tasks(db, _WS_ID, {"tags": ["harness"]})
    )

    assert result["success"] is True
    assert result["total"] == 1
    task = result["tasks"][0]
    # The two fields the HARNESS command path depends on must be present.
    assert task["tags"] == ["harness", "org-review", "risk-4", "rx:rx-esc-1"]
    assert task["description"] == "**Current:** {...}\n**Proposed:** {...}"


def test_list_board_tasks_tags_default_empty_list():
    """A row with NULL tags projects to [] (never None), so `"harness" in tags`
    in _find_task_by_rx is always a safe membership test."""
    rows = [_FakeRow(id=8, title="plain task", description="d", tags=None, assigned_agent_id=None)]
    db = _FakeDB(rows)

    result = asyncio.run(_HANDLER.list_board_tasks(db, _WS_ID, {}))

    assert result["tasks"][0]["tags"] == []


# ===========================================================================
# PRD-154 · P154-S4 — assignment dispatches execution + priority CASE fix
# ===========================================================================
#
# Two breakages from the deep review §2:
#  (1) Assigning a board task started nothing — _launch_task_execution existed
#      but was never called on assignment, so work sat until an opt-in heartbeat
#      most agents lack picked it up. create-with-assignee and the assign
#      endpoint now dispatch the moment a task lands in the 'assigned' state.
#  (2) heartbeat_service ordered assigned tasks by ``BoardTask.priority.desc()``
#      — an alphabetical string sort (urgent, medium, low, high) that buries
#      'high'. Replaced with a data-driven CASE rank (urgent > high > medium > low).
#
# api.board_tasks eagerly builds the SQLAlchemy engine; the POSTGRES_* inert
# creds set at the top of this file satisfy that. Every test mocks the DB, so no
# query ever runs and the executor is mocked — pure dispatch-decision coverage.

def _ns(**kw):
    return types.SimpleNamespace(**kw)


def _stub_task(**kw):
    base = dict(
        id=1,
        status="assigned",
        assigned_agent_id=7,
        source_type="user",
        raw_prompt="do it",
        description=None,
        title="t",
        review_mode="auto",
        attachment_ids=[],
        started_at=None,
    )
    base.update(kw)
    return _ns(**base)


class _FakeQ:
    def __init__(self, result):
        self._r = result

    def filter(self, *a, **k):
        return self

    def first(self):
        return self._r

    def get(self, *_a):
        return self._r


class _FakeSession:
    """Returns a seeded Agent for Agent queries and a seeded BoardTask for
    everything else; records commits. add() captures a freshly-created task."""

    def __init__(self, agent=None, task=None):
        self._agent = agent
        self._task = task
        self.commits = 0

    def query(self, model):
        from core.models import Agent
        if model is Agent:
            return _FakeQ(self._agent)
        return _FakeQ(self._task)

    def add(self, obj):
        self._task = obj

    def commit(self):
        self.commits += 1

    def refresh(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = 4242


class _FakeReq:
    def __init__(self, body):
        self._body = body

    async def json(self):
        return self._body


# --- dispatch is NOTIFY-driven (PRD-161): create/assign fire pg_notify and
#     leave the task 'assigned'; the dispatch loop claims + runs it. The inline
#     _dispatch_on_assign and the _should_dispatch_on_assign predicate are gone.

def test_create_with_assignee_notifies_dispatch(monkeypatch):
    from api import board_tasks as bt
    notifies = []
    monkeypatch.setattr(bt, "notify_task_available", lambda db, **kw: notifies.append(kw))

    ctx = _ns(workspace_id=_WS_ID, user=_ns(clerk_user_id="u1", id=1))
    db = _FakeSession(agent=_ns(id=7))
    req = _FakeReq({"title": "ship it", "assigned_agent_id": 7})

    result = asyncio.run(bt.create_task(req, ctx=ctx, db=db))

    assert len(notifies) == 1, "create-with-assignee must notify the dispatch loop"
    assert notifies[0]["task_id"] == 4242
    # The loop claims it (assigned -> in_progress); the handler leaves it assigned.
    assert result["status"] == "assigned"


def test_assign_endpoint_notifies_dispatch(monkeypatch):
    from api import board_tasks as bt
    from core.models.core import BoardTask
    notifies = []
    monkeypatch.setattr(bt, "notify_task_available", lambda db, **kw: notifies.append(kw))

    task = BoardTask(
        id=5, workspace_id=_WS_ID, title="t", status="inbox",
        source_type="user", review_mode="auto",
    )
    ctx = _ns(workspace_id=_WS_ID, user=_ns(clerk_user_id="u1", id=1))
    db = _FakeSession(agent=_ns(id=9), task=task)
    req = _FakeReq({"assigned_agent_id": 9})

    result = asyncio.run(bt.update_task(5, req, ctx=ctx, db=db))

    assert len(notifies) == 1, "the assign endpoint must notify the dispatch loop"
    assert result["status"] == "assigned"


def test_reassign_running_task_does_not_notify(monkeypatch):
    """A task already in_progress must not be re-dispatched: notify only fires on
    a freshly 'assigned' task (the claim query also filters status='assigned')."""
    from api import board_tasks as bt
    from core.models.core import BoardTask
    notifies = []
    monkeypatch.setattr(bt, "notify_task_available", lambda db, **kw: notifies.append(kw))

    task = BoardTask(
        id=6, workspace_id=_WS_ID, title="t", status="in_progress",
        assigned_agent_id=1, source_type="user", review_mode="auto",
    )
    ctx = _ns(workspace_id=_WS_ID, user=_ns(clerk_user_id="u1", id=1))
    db = _FakeSession(agent=_ns(id=2), task=task)
    req = _FakeReq({"assigned_agent_id": 2})

    asyncio.run(bt.update_task(6, req, ctx=ctx, db=db))

    assert notifies == [], "a running task must not be re-dispatched on re-assign"


def test_assign_on_recipe_mirror_task_does_not_notify(monkeypatch):
    """Recipe-mirror tasks are driven by the recipe executor, never the board."""
    from api import board_tasks as bt
    from core.models.core import BoardTask
    notifies = []
    monkeypatch.setattr(bt, "notify_task_available", lambda db, **kw: notifies.append(kw))

    task = BoardTask(
        id=8, workspace_id=_WS_ID, title="t", status="inbox",
        source_type="recipe", review_mode="auto",
    )
    ctx = _ns(workspace_id=_WS_ID, user=_ns(clerk_user_id="u1", id=1))
    db = _FakeSession(agent=_ns(id=3), task=task)
    req = _FakeReq({"assigned_agent_id": 3})

    asyncio.run(bt.update_task(8, req, ctx=ctx, db=db))

    assert notifies == [], "recipe-mirror tasks are not board-dispatched"


# --- priority ordering moved into the dispatch claim SQL (PRD-161) ---------

def test_dispatch_claim_orders_by_semantic_priority():
    """urgent > high > medium > low, encoded as data in the claim query — NOT an
    alphabetical ``ORDER BY priority`` string sort (which buries 'high' last)."""
    from services import board_dispatcher as bd
    sql = bd._PRIORITY_ORDER_SQL
    assert sql.index("'urgent'") < sql.index("'high'") < sql.index("'medium'")
    assert "THEN 0" in sql and "THEN 1" in sql and "THEN 2" in sql


def test_heartbeat_no_longer_dispatches_board_tasks():
    """The 3-task fold-in is deleted from the heartbeat (no shim): the marker
    string and the priority-order helper it used are both gone."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "services", "heartbeat_service.py")) as f:
        src = f.read()
    assert "ASSIGNED TASKS (Priority Work)" not in src
    assert "_board_task_priority_order" not in src


# --- S3: honest lifecycle (failed state, Q44 rejection feedback) -----------

def test_reject_returns_to_same_agent_with_feedback(monkeypatch):
    """Q44: a rejected review goes back to the SAME agent as 'assigned' with the
    feedback in review_feedback (carried into the redo's context), not to inbox."""
    from api import board_tasks as bt
    from core.models.core import BoardTask
    notifies = []
    monkeypatch.setattr(bt, "notify_task_available", lambda db, **kw: notifies.append(kw))

    task = BoardTask(
        id=11, workspace_id=_WS_ID, title="t", status="review",
        assigned_agent_id=5, source_type="user", review_mode="human",
    )
    ctx = _ns(workspace_id=_WS_ID, user=_ns(clerk_user_id="u1", id=1))
    db = _FakeSession(agent=_ns(id=5), task=task)
    req = _FakeReq({"feedback": "Tighten the headline."})

    result = asyncio.run(bt.reject_task(11, req, ctx=ctx, db=db))

    assert result["status"] == "assigned", "Q44: reject returns to the same agent, not inbox"
    assert result["assigned_agent_id"] == 5
    assert task.review_feedback == "Tighten the headline."
    assert task.attempts == 0, "a human redo is a fresh attempt cycle"
    assert len(notifies) == 1, "reject must re-dispatch the task through the loop"


def test_reject_without_assigned_agent_is_422():
    from api import board_tasks as bt
    from core.models.core import BoardTask
    from fastapi import HTTPException
    import pytest as _pytest

    task = BoardTask(id=12, workspace_id=_WS_ID, title="t", status="review",
                     assigned_agent_id=None, source_type="user")
    ctx = _ns(workspace_id=_WS_ID, user=_ns(clerk_user_id="u1", id=1))
    db = _FakeSession(task=task)
    req = _FakeReq({"feedback": "x"})

    with _pytest.raises(HTTPException) as ei:
        asyncio.run(bt.reject_task(12, req, ctx=ctx, db=db))
    assert ei.value.status_code == 422


def test_task_failed_dispatches_task_failed_notification(monkeypatch):
    """A crashed execution fires a task_failed event (not a silent done)."""
    from api import board_tasks as bt
    import core.services.notification_dispatcher as nd
    from core.models.core import BoardTask
    captured = {}

    class _FakeDispatcher:
        def __init__(self, db, ws):
            pass

        async def dispatch(self, **kw):
            captured.update(kw)
            return {}

    monkeypatch.setattr(nd, "NotificationDispatcher", _FakeDispatcher)

    task = BoardTask(id=13, workspace_id=_WS_ID, title="render report",
                     status="failed", assigned_agent_id=3, error_message="boom")
    db = _FakeSession(agent=_ns(id=3, name="ATLAS"), task=task)

    asyncio.run(bt._dispatch_task_failed(db, _WS_ID, task))

    assert captured.get("event_type") == "task_failed"
    assert captured.get("status") == "error"
    assert "boom" in (captured.get("message") or "")


# --- S4: the blocking Composio call is offloaded so it can't stall the loop --

def test_composio_execute_is_offloaded_to_thread():
    """PRD-161 S4 event-loop guard: the synchronous Composio SDK call runs via
    asyncio.to_thread inside the async executor, so one slow tool call can't
    block other in-flight board-task claims/executions."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "core", "composio", "tool_executor.py")) as f:
        src = f.read()
    # The blocking SDK call is wrapped, not called inline on the event loop.
    assert "asyncio.to_thread(" in src
    assert "self.client.execute_action," in src


# --- S5: Run-Now + SLA archive + SSE auth ----------------------------------

def test_run_now_redispatches_idle_task(monkeypatch):
    from api import board_tasks as bt
    from core.models.core import BoardTask
    notifies = []
    monkeypatch.setattr(bt, "notify_task_available", lambda db, **kw: notifies.append(kw))

    task = BoardTask(id=21, workspace_id=_WS_ID, title="t", status="failed",
                     assigned_agent_id=4, source_type="user", attempts=2)
    ctx = _ns(workspace_id=_WS_ID, user=_ns(clerk_user_id="u1", id=1))
    db = _FakeSession(agent=_ns(id=4), task=task)

    result = asyncio.run(bt.run_task_now(21, ctx=ctx, db=db))

    assert result["status"] == "assigned"
    assert task.attempts == 0, "Run Now resets the attempt cycle"
    assert task.lease_until is None
    assert len(notifies) == 1, "Run Now re-dispatches through the loop"


def test_run_now_requires_an_agent():
    from api import board_tasks as bt
    from core.models.core import BoardTask
    from fastapi import HTTPException
    import pytest as _p

    task = BoardTask(id=22, workspace_id=_WS_ID, title="t", status="failed", assigned_agent_id=None)
    ctx = _ns(workspace_id=_WS_ID, user=_ns(clerk_user_id="u1", id=1))
    db = _FakeSession(task=task)
    with _p.raises(HTTPException) as ei:
        asyncio.run(bt.run_task_now(22, ctx=ctx, db=db))
    assert ei.value.status_code == 422


def test_run_now_rejects_already_running_task():
    from api import board_tasks as bt
    from core.models.core import BoardTask
    from fastapi import HTTPException
    import pytest as _p

    task = BoardTask(id=23, workspace_id=_WS_ID, title="t", status="in_progress", assigned_agent_id=4)
    ctx = _ns(workspace_id=_WS_ID, user=_ns(clerk_user_id="u1", id=1))
    db = _FakeSession(agent=_ns(id=4), task=task)
    with _p.raises(HTTPException) as ei:
        asyncio.run(bt.run_task_now(23, ctx=ctx, db=db))
    assert ei.value.status_code == 409


def test_sse_stream_gated_by_tasks_read_scope():
    """SSE board events ride the read-only PRD-09 TASKS_READ dep — no new scope,
    shared hybrid auth untouched (test_board_sdk_auth.py is not modified)."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(here, "api", "board_tasks.py")) as f:
        src = f.read()
    assert "async def stream_board_events(" in src
    block = src[src.index("async def stream_board_events("):][:400]
    assert "require_task_context(TASKS_READ)" in block
