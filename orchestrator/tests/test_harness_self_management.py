"""PRD-141 US-020: HARNESS self-management flag + approved-task parser.

Pure unit tests — _parse_harness_task does no I/O, and HarnessService.__init__
takes no DB. Dummy POSTGRES_* satisfies the lazy create_engine in the config
import chain without opening a connection. The self-management flag is popped
before import so test_flag_defaults_false sees the real default.
"""
import json
import os
import sys
import types

os.environ.pop("HARNESS_SELF_MANAGEMENT_ENABLED", None)
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")
os.environ.setdefault("POSTGRES_DB", "test")


def _install_fake_apscheduler():
    """harness_service imports apscheduler at module top for its cron, but the
    parser under test never uses it. Stub the names so import succeeds without
    the (prod-only) dependency installed locally."""
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

from config import config
from services.harness_service import HarnessService


def _harness_task(
    change_type="heartbeat_tune",
    target_name="ScribeAgent",
    current=None,
    proposed=None,
    risk=2,
    task_id="task-1",
    tags=None,
):
    """Build a board task exactly as _phase_apply() produces it."""
    current = {"interval_minutes": 30} if current is None else current
    proposed = {"interval_minutes": 90} if proposed is None else proposed
    return {
        "id": task_id,
        "title": f"[HARNESS] {change_type} for {target_name}",
        "description": (
            f"**Risk Score:** {risk}/5\n\n"
            f"**Change Type:** {change_type}\n\n"
            f"**Current:** {json.dumps(current)}\n\n"
            f"**Proposed:** {json.dumps(proposed)}\n\n"
            f"**Rationale:** because reasons\n\n"
            f"**Expected Improvement:** save tokens"
        ),
        "tags": ["harness", "org-review", f"risk-{risk}"] if tags is None else tags,
    }


def test_parse_harness_task_valid():
    svc = HarnessService()
    rx = svc._parse_harness_task(_harness_task(), agents_by_name={"ScribeAgent": 42})

    assert rx is not None
    assert rx["change_type"] == "heartbeat_tune"
    assert rx["target_name"] == "ScribeAgent"
    assert rx["target_id"] == 42
    assert rx["target_type"] == "agent"
    assert rx["current_value"] == {"interval_minutes": 30}
    assert rx["proposed_value"] == {"interval_minutes": 90}
    assert rx["risk_score"] == 2
    assert rx["rationale"] == "because reasons"
    assert rx["expected_improvement"] == "save tokens"
    assert rx["prescription_id"] == "rx-task-task-1"


def test_parse_harness_task_invalid():
    svc = HarnessService()
    # Non-HARNESS title -> None
    assert svc._parse_harness_task({"id": "t", "title": "Buy milk", "description": ""}) is None
    # HARNESS prefix but missing ' for {target}' -> None
    assert svc._parse_harness_task(
        {"id": "t", "title": "[HARNESS] heartbeat_tune", "description": ""}
    ) is None
    # Empty / missing title -> None
    assert svc._parse_harness_task({"id": "t", "title": "", "description": ""}) is None
    assert svc._parse_harness_task({"id": "t"}) is None


def test_parse_harness_task_unresolved_target_id_is_none():
    """A target_name not present in the agents map yields target_id=None
    (US-021 must treat an unresolved target as non-applicable, never guess)."""
    svc = HarnessService()
    rx = svc._parse_harness_task(
        _harness_task(target_name="GhostAgent"), agents_by_name={"ScribeAgent": 42}
    )
    assert rx is not None
    assert rx["target_name"] == "GhostAgent"
    assert rx["target_id"] is None


def test_flag_defaults_false():
    assert config.HARNESS_SELF_MANAGEMENT_ENABLED is False
