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


# --- dispatch-on-assign predicate (pure) ---------------------------------

def test_should_dispatch_on_assign_happy_path():
    from api import board_tasks as bt
    assert bt._should_dispatch_on_assign(_stub_task()) is True


def test_should_dispatch_skips_recipe_mirror():
    """Recipe-mirror tasks are driven by the recipe executor, not the board."""
    from api import board_tasks as bt
    assert bt._should_dispatch_on_assign(_stub_task(source_type="recipe")) is False


def test_should_dispatch_skips_already_running():
    """Double-fire guard: an in_progress task is already running."""
    from api import board_tasks as bt
    assert bt._should_dispatch_on_assign(_stub_task(status="in_progress")) is False


def test_should_dispatch_skips_unassigned():
    from api import board_tasks as bt
    assert bt._should_dispatch_on_assign(_stub_task(assigned_agent_id=None)) is False


# --- integration: assign → execution starts same tick (mock executor) -----

def test_create_with_assignee_dispatches_execution(monkeypatch):
    from api import board_tasks as bt
    calls = []
    monkeypatch.setattr(bt, "_launch_task_execution", lambda **kw: calls.append(kw))

    ctx = _ns(workspace_id=_WS_ID, user=_ns(clerk_user_id="u1", id=1))
    db = _FakeSession(agent=_ns(id=7))
    req = _FakeReq({"title": "ship it", "assigned_agent_id": 7})

    result = asyncio.run(bt.create_task(req, ctx=ctx, db=db))

    assert len(calls) == 1, "create-with-assignee must dispatch execution same tick"
    assert calls[0]["agent_id"] == 7
    assert calls[0]["task_id"] == 4242
    assert result["status"] == "in_progress"


def test_assign_endpoint_dispatches_execution(monkeypatch):
    from api import board_tasks as bt
    from core.models.core import BoardTask
    calls = []
    monkeypatch.setattr(bt, "_launch_task_execution", lambda **kw: calls.append(kw))

    task = BoardTask(
        id=5, workspace_id=_WS_ID, title="t", status="inbox",
        source_type="user", review_mode="auto",
    )
    ctx = _ns(workspace_id=_WS_ID, user=_ns(clerk_user_id="u1", id=1))
    db = _FakeSession(agent=_ns(id=9), task=task)
    req = _FakeReq({"assigned_agent_id": 9})

    result = asyncio.run(bt.update_task(5, req, ctx=ctx, db=db))

    assert len(calls) == 1, "the assign endpoint must dispatch execution same tick"
    assert calls[0]["agent_id"] == 9
    assert result["status"] == "in_progress"


def test_reassign_running_task_does_not_double_fire(monkeypatch):
    from api import board_tasks as bt
    from core.models.core import BoardTask
    calls = []
    monkeypatch.setattr(bt, "_launch_task_execution", lambda **kw: calls.append(kw))

    task = BoardTask(
        id=6, workspace_id=_WS_ID, title="t", status="in_progress",
        assigned_agent_id=1, source_type="user", review_mode="auto",
    )
    ctx = _ns(workspace_id=_WS_ID, user=_ns(clerk_user_id="u1", id=1))
    db = _FakeSession(agent=_ns(id=2), task=task)
    req = _FakeReq({"assigned_agent_id": 2})

    result = asyncio.run(bt.update_task(6, req, ctx=ctx, db=db))

    assert calls == [], "a running task must not re-launch on re-assign"
    assert result["status"] == "in_progress"


def test_assign_on_recipe_mirror_task_does_not_dispatch(monkeypatch):
    from api import board_tasks as bt
    from core.models.core import BoardTask
    calls = []
    monkeypatch.setattr(bt, "_launch_task_execution", lambda **kw: calls.append(kw))

    task = BoardTask(
        id=8, workspace_id=_WS_ID, title="t", status="inbox",
        source_type="recipe", review_mode="auto",
    )
    ctx = _ns(workspace_id=_WS_ID, user=_ns(clerk_user_id="u1", id=1))
    db = _FakeSession(agent=_ns(id=3), task=task)
    req = _FakeReq({"assigned_agent_id": 3})

    asyncio.run(bt.update_task(8, req, ctx=ctx, db=db))

    assert calls == [], "recipe-mirror tasks are not board-dispatched"


# --- priority CASE: urgent > high > medium > low --------------------------

def _load_heartbeat_service():
    """Load services/heartbeat_service.py directly from file so services/__init__
    (heavy) never fires. apscheduler is already stubbed at module load."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "services", "heartbeat_service.py")
    spec = importlib.util.spec_from_file_location("hbs_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_priority_rank_is_semantic_not_alphabetical():
    hbs = _load_heartbeat_service()
    rank = hbs._BOARD_TASK_PRIORITY_RANK
    ordered = sorted(["low", "urgent", "medium", "high"], key=rank.get)
    assert ordered == ["urgent", "high", "medium", "low"]
    # The shipped bug: ORDER BY priority DESC is an alphabetical string sort.
    alpha_desc = sorted(["low", "urgent", "medium", "high"], reverse=True)
    assert alpha_desc == ["urgent", "medium", "low", "high"]  # 'high' wrongly last
    assert ordered != alpha_desc


def test_assigned_task_order_uses_priority_case_not_string_desc():
    hbs = _load_heartbeat_service()
    expr = hbs._board_task_priority_order()
    assert type(expr).__name__ == "Case"
    compiled = str(expr.compile(compile_kwargs={"literal_binds": True}))
    for p in ("urgent", "high", "medium", "low"):
        assert p in compiled


def test_heartbeat_source_no_longer_uses_priority_string_desc():
    """Guard: the broken ``BoardTask.priority.desc()`` ordering is gone and the
    CASE helper is what the assigned-task scan orders by."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "services", "heartbeat_service.py")
    with open(path) as f:
        src = f.read()
    assert "BoardTask.priority.desc()" not in src
    assert "_board_task_priority_order()" in src
