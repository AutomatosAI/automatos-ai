"""PRD-142 Wave 4 (W4-S6): platform_create_routing_rule + HARNESS routing_rule_add.

Two units: (1) the new routing-rule handler inserts a workspace-scoped row into
the routing_rules table (read by the UniversalRouter at Tier 2a) and fails closed
on a rule with no target or no matcher; (2) the HARNESS apply branch maps a
routing_rule_add prescription onto that action with the right params. Dummy
POSTGRES_* + the apscheduler stub let the harness_service import chain load
without a DB or the prod-only scheduler.
"""
import asyncio
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

from modules.tools.discovery.handlers_routing import create_routing_rule  # noqa: E402
from services.harness_service import get_harness_service  # noqa: E402

_WS = UUID("00000000-0000-0000-0000-000000000001")


def _run(coro):
    return asyncio.run(coro)


class _FakeDB:
    """Captures the inserted RoutingRule; simulates a PK on refresh."""

    def __init__(self):
        self.added = []
        self.committed = False

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.committed = True

    def refresh(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = 123

    def rollback(self):
        pass


class _FakeExecutor:
    def __init__(self):
        self.calls = []

    async def execute(self, name, params):
        self.calls.append((name, params))
        return {"success": True}


# --- the handler -----------------------------------------------------------

def test_create_routing_rule_writes_row_workspace_scoped():
    db = _FakeDB()
    res = _run(create_routing_rule(db, _WS, {
        "source_channel": "telegram", "target_agent_id": 7, "priority": 5,
    }))
    assert res["success"] is True
    assert db.committed is True
    assert len(db.added) == 1
    rule = db.added[0]
    assert rule.workspace_id == _WS  # scoped from the executor context, never params
    assert rule.source_channel == "telegram"
    assert rule.target_agent_id == 7
    assert rule.priority == 5
    assert rule.is_active is True


def test_create_routing_rule_requires_a_target():
    db = _FakeDB()
    res = _run(create_routing_rule(db, _WS, {"source_channel": "telegram"}))
    assert res["success"] is False
    assert db.added == []  # nothing written on a bad rule


def test_create_routing_rule_requires_a_matcher():
    db = _FakeDB()
    res = _run(create_routing_rule(db, _WS, {"target_agent_id": 7}))
    assert res["success"] is False
    assert db.added == []


def test_create_routing_rule_coerces_bad_priority():
    db = _FakeDB()
    res = _run(create_routing_rule(db, _WS, {
        "source_pattern": "invoice", "target_agent_id": 1, "priority": "not-a-number",
    }))
    assert res["success"] is True
    assert db.added[0].priority == 0  # defaults rather than raising


# --- the HARNESS apply branch ---------------------------------------------

def test_routing_rule_add_maps_to_action():
    svc = get_harness_service()
    ex = _FakeExecutor()
    rx = {"change_type": "routing_rule_add", "proposed_value": {
        "source_pattern": "invoice",
        "target_workflow_id": 3,
        "intent_keywords": ["billing"],
        "priority": 2,
    }}
    res = _run(svc._auto_apply_prescription(ex, rx))
    assert res.get("success") is True
    assert len(ex.calls) == 1
    name, params = ex.calls[0]
    assert name == "platform_create_routing_rule"
    assert params["source_pattern"] == "invoice"
    assert params["target_workflow_id"] == 3
    assert params["intent_keywords"] == ["billing"]
    assert params["priority"] == 2


def test_power_mode_still_refused_while_deferred():
    # power_mode remains net-new work (W4-S5 deferred); the apply path must still
    # refuse it safely rather than silently no-op.
    svc = get_harness_service()
    ex = _FakeExecutor()
    rx = {"change_type": "power_mode_upgrade", "proposed_value": {"power_mode": "performance"}}
    res = _run(svc._auto_apply_prescription(ex, rx))
    assert res.get("success") is False
    assert ex.calls == []  # nothing executed for an unhandled change_type
