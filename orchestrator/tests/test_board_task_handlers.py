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
    if "apscheduler" in sys.modules:
        return
    aps = types.ModuleType("apscheduler")
    schedulers = types.ModuleType("apscheduler.schedulers")
    asyncio_mod = types.ModuleType("apscheduler.schedulers.asyncio")
    asyncio_mod.AsyncIOScheduler = type("AsyncIOScheduler", (), {})
    jobstores = types.ModuleType("apscheduler.jobstores")
    memory_mod = types.ModuleType("apscheduler.jobstores.memory")
    memory_mod.MemoryJobStore = type("MemoryJobStore", (), {})
    aps.schedulers = schedulers
    aps.jobstores = jobstores
    schedulers.asyncio = asyncio_mod
    jobstores.memory = memory_mod
    sys.modules.update({
        "apscheduler": aps,
        "apscheduler.schedulers": schedulers,
        "apscheduler.schedulers.asyncio": asyncio_mod,
        "apscheduler.jobstores": jobstores,
        "apscheduler.jobstores.memory": memory_mod,
    })


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
