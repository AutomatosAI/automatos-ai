"""PRD-141 US-020: HARNESS self-management flag + approved-task parser.

Pure unit tests — _parse_harness_task does no I/O, and HarnessService.__init__
takes no DB. Dummy POSTGRES_* satisfies the lazy create_engine in the config
import chain without opening a connection. The self-management flag is popped
before import so test_flag_defaults_false sees the real default.
"""
import asyncio
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
    """Build a board task exactly as the weekly tick filed one before PRD-263 US-016
    (an owner's /approve still parses one)."""
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


# ---------------------------------------------------------------------------
# The apply helpers an owner's /approve uses (api/harness_commands.py). The
# weekly tick's card-to-Done path that also used them went in PRD-263 US-016,
# and its cases with it; these pin the guards that path shared with /approve.
# ---------------------------------------------------------------------------

class _FakeExecutor:
    """Records every execute() call and returns canned results for the actions
    _auto_apply_prescription invokes."""

    def __init__(self, tasks, agents):
        self._tasks = tasks
        self._agents = agents
        self.calls = []
        self.db = None

    async def execute(self, action, params, caller_context=None):
        self.calls.append((action, params))
        if action == "platform_list_tasks":
            return {"data": self._tasks}
        if action == "platform_list_agents":
            return {"data": self._agents}
        # apply + status-update actions all report success
        return {"success": True}

    def actions(self):
        return [action for action, _ in self.calls]


def test_snapshot_is_the_parsed_current_value():
    """The pre-change value an applied change records as current_value_before."""
    svc = HarnessService()
    rx = svc._parse_harness_task(_harness_task(current={"interval_minutes": 30}), agents_by_name={"ScribeAgent": 42})
    assert svc._snapshot_current_value(rx) == {"interval_minutes": 30}
    assert svc._snapshot_current_value({"prescription_id": "rx-1", "current_value": "30"}) == {}


def test_placeholder_proposed_value_is_refused():
    """A proposed_value that is the 'review_needed' placeholder is never applied —
    applying it literally would corrupt the agent."""
    svc = HarnessService()
    ex = _FakeExecutor(tasks=[], agents=[])
    rx = svc._parse_harness_task(
        _harness_task(change_type="description_update", current={"description": "old"},
                      proposed={"description": "review_needed"}, task_id=11),
        agents_by_name={"ScribeAgent": 42},
    )
    result = asyncio.run(svc._auto_apply_prescription(ex, rx))

    assert result["success"] is False
    assert "placeholder" in result["error"]
    assert ex.calls == []


# ---------------------------------------------------------------------------
# US-023: expanded prescription vocabulary (correct action names)
# ---------------------------------------------------------------------------


def test_tool_assignment_add_prescription():
    """tool_assignment_add applies via platform_assign_tool_to_agent using the
    VERIFIED param app_name (not tool_name)."""
    svc = HarnessService()
    ex = _FakeExecutor(tasks=[], agents=[])
    rx = {
        "prescription_id": "rx-tool-add",
        "change_type": "tool_assignment_add",
        "target_id": 42,
        "proposed_value": {"app_name": "GMAIL"},
    }
    result = asyncio.run(svc._auto_apply_prescription(ex, rx))

    assert result["success"] is True
    assert (
        "platform_assign_tool_to_agent",
        {"agent_id": 42, "app_name": "GMAIL"},
    ) in ex.calls


def test_tool_assignment_remove_prescription():
    """tool_assignment_remove applies via platform_unassign_tool_from_agent."""
    svc = HarnessService()
    ex = _FakeExecutor(tasks=[], agents=[])
    rx = {
        "prescription_id": "rx-tool-rm",
        "change_type": "tool_assignment_remove",
        "target_id": 42,
        "proposed_value": {"app_name": "GITHUB"},
    }
    result = asyncio.run(svc._auto_apply_prescription(ex, rx))

    assert result["success"] is True
    assert (
        "platform_unassign_tool_from_agent",
        {"agent_id": 42, "app_name": "GITHUB"},
    ) in ex.calls


def test_power_mode_change_type_maps_to_set_power_mode():
    """power_mode_* prescriptions set the WORKSPACE default power mode
    (workspace.settings['power_mode']) via platform_set_power_mode — never a
    per-agent platform_update_agent (power mode is workspace-scoped). W4-S5."""
    svc = HarnessService()
    ex = _FakeExecutor(tasks=[], agents=[])
    for change_type in ("power_mode_upgrade", "power_mode_downgrade"):
        rx = {
            "prescription_id": f"rx-{change_type}",
            "change_type": change_type,
            "target_id": 42,  # ignored — power mode is a workspace knob, not an agent attr
            "proposed_value": {"power_mode": "max"},
        }
        result = asyncio.run(svc._auto_apply_prescription(ex, rx))
        assert result["success"] is True
        assert ("platform_set_power_mode", {"power_mode": "max"}) in ex.calls
    # The workspace knob is set, never a per-agent attribute.
    assert "platform_update_agent" not in ex.actions()


def test_routing_rule_add_maps_to_create_action():
    """routing_rule_add is now implemented (PRD-142 W4-S6) — it maps to
    platform_create_routing_rule (the routing_rules table is router-wired at
    Tier 2a), no longer the 'Unknown auto-apply change_type' refusal."""
    svc = HarnessService()
    ex = _FakeExecutor(tasks=[], agents=[])
    rx = {
        "prescription_id": "rx-route",
        "change_type": "routing_rule_add",
        "target_id": 42,
        "proposed_value": {
            "source_channel": "telegram", "target_agent_id": 7, "priority": 1,
        },
    }
    asyncio.run(svc._auto_apply_prescription(ex, rx))

    name, params = ex.calls[0]
    assert name == "platform_create_routing_rule"
    assert params["source_channel"] == "telegram"
    assert params["target_agent_id"] == 7


def test_temperature_adjust_uses_top_level_param():
    """update_agent reads temperature top-level; a nested model_config is ignored,
    so the handler must pass temperature directly or the change is a silent no-op."""
    svc = HarnessService()
    ex = _FakeExecutor(tasks=[], agents=[])
    rx = {
        "prescription_id": "rx-temp",
        "change_type": "temperature_adjust",
        "target_id": 42,
        "proposed_value": {"temperature": 0.4},
    }
    asyncio.run(svc._auto_apply_prescription(ex, rx))

    assert ("platform_update_agent", {"agent_id": 42, "temperature": 0.4}) in ex.calls


def test_model_change_uses_model_id_param():
    """update_agent reads the new model as top-level model_id (not nested, not
    'model'), so the handler must map proposed['model'] -> model_id."""
    svc = HarnessService()
    ex = _FakeExecutor(tasks=[], agents=[])
    rx = {
        "prescription_id": "rx-model",
        "change_type": "model_change_same_tier",
        "target_id": 42,
        "proposed_value": {"model": "claude-sonnet-4-6"},
    }
    asyncio.run(svc._auto_apply_prescription(ex, rx))

    assert (
        "platform_update_agent",
        {"agent_id": 42, "model_id": "claude-sonnet-4-6"},
    ) in ex.calls
