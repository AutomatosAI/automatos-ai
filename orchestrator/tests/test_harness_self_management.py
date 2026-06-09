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


# ---------------------------------------------------------------------------
# US-021: execute approved board tasks (flag-gated) + snapshot
# ---------------------------------------------------------------------------

_WS_ID = "00000000-0000-0000-0000-000000000001"


def _ledger_path(volume, workspace_id=_WS_ID):
    """The applied-tasks ledger path — _write_applied_tasks writes it and
    _read_applied_tasks reads it (proves they share one on-disk store)."""
    return os.path.join(str(volume), str(workspace_id), "harness", "applied_tasks.json")


class _FakeExecutor:
    """Records every execute() call and returns canned results for the actions
    _apply_approved_board_tasks invokes (list tasks/agents, apply). The ledger
    write no longer goes through the executor — it is a direct disk write — so
    the fake never sees a workspace_write_file call."""

    def __init__(self, tasks, agents):
        self._tasks = tasks
        self._agents = agents
        self.calls = []

    async def execute(self, action, params):
        self.calls.append((action, params))
        if action == "platform_list_tasks":
            return {"data": self._tasks}
        if action == "platform_list_agents":
            return {"data": self._agents}
        # apply + status-update actions all report success
        return {"success": True}

    def actions(self):
        return [action for action, _ in self.calls]


def test_approved_tasks_noop_when_flag_off(monkeypatch):
    """Flag off -> pure no-op: nothing is listed, nothing is applied."""
    monkeypatch.setattr(config, "HARNESS_SELF_MANAGEMENT_ENABLED", False)
    svc = HarnessService()
    ex = _FakeExecutor(
        tasks=[_harness_task(task_id=7)],
        agents=[{"id": 42, "name": "ScribeAgent"}],
    )
    changelog = {}
    asyncio.run(svc._apply_approved_board_tasks(ex, _WS_ID, changelog))

    assert ex.calls == []
    assert changelog == {}


def test_approved_tasks_are_executed(monkeypatch, tmp_path):
    """Flag on -> a done [HARNESS] task is parsed, its target resolved, and the
    change applied via _auto_apply_prescription, then the ledger is persisted to
    disk — the same store _read_applied_tasks reads, so the next tick is idempotent."""
    monkeypatch.setattr(config, "HARNESS_SELF_MANAGEMENT_ENABLED", True)
    monkeypatch.setattr(config, "WORKSPACE_VOLUME_PATH", str(tmp_path))
    svc = HarnessService()
    task = _harness_task(
        change_type="heartbeat_tune",
        target_name="ScribeAgent",
        current={"interval_minutes": 30},
        proposed={"interval_minutes": 90},
        task_id=7,
    )
    ex = _FakeExecutor(tasks=[task], agents=[{"id": 42, "name": "ScribeAgent"}])
    changelog = {}
    asyncio.run(svc._apply_approved_board_tasks(ex, _WS_ID, changelog))

    # The heartbeat change was actually applied to the resolved agent id.
    assert (
        "platform_configure_agent_heartbeat",
        {"agent_id": 42, "interval_minutes": 90},
    ) in ex.calls
    applied = changelog.get("applied_from_approved", [])
    assert len(applied) == 1
    assert applied[0]["task_id"] == "7"
    assert applied[0]["target_id"] == 42
    assert applied[0]["change_type"] == "heartbeat_tune"
    # Idempotency ledger round-tripped to disk: the file exists at the path the
    # reader uses, and _read_applied_tasks sees task 7 as applied.
    assert os.path.exists(_ledger_path(tmp_path))
    ledger = svc._read_applied_tasks(_WS_ID)
    assert "7" in {str(i) for i in ledger["applied_task_ids"]}


def test_snapshot_recorded_before_apply(monkeypatch, tmp_path):
    """The pre-change value is captured as current_value_before so US-022 has a
    rollback target."""
    monkeypatch.setattr(config, "HARNESS_SELF_MANAGEMENT_ENABLED", True)
    monkeypatch.setattr(config, "WORKSPACE_VOLUME_PATH", str(tmp_path))
    svc = HarnessService()
    task = _harness_task(
        change_type="heartbeat_tune",
        target_name="ScribeAgent",
        current={"interval_minutes": 30},
        proposed={"interval_minutes": 90},
        task_id=7,
    )
    ex = _FakeExecutor(tasks=[task], agents=[{"id": 42, "name": "ScribeAgent"}])
    changelog = {}
    asyncio.run(svc._apply_approved_board_tasks(ex, _WS_ID, changelog))

    entry = changelog["applied_from_approved"][0]
    assert entry["current_value_before"] == {"interval_minutes": 30}
    assert entry["proposed_value"] == {"interval_minutes": 90}


def test_already_applied_task_is_skipped(monkeypatch, tmp_path):
    """A task whose id is in the ledger is never re-applied (idempotency)."""
    monkeypatch.setattr(config, "HARNESS_SELF_MANAGEMENT_ENABLED", True)
    monkeypatch.setattr(config, "WORKSPACE_VOLUME_PATH", str(tmp_path))
    svc = HarnessService()
    monkeypatch.setattr(
        svc, "_read_applied_tasks",
        lambda ws: {"applied_task_ids": [7], "entries": []},
    )
    task = _harness_task(change_type="heartbeat_tune", target_name="ScribeAgent", task_id=7)
    ex = _FakeExecutor(tasks=[task], agents=[{"id": 42, "name": "ScribeAgent"}])
    changelog = {}
    asyncio.run(svc._apply_approved_board_tasks(ex, _WS_ID, changelog))

    assert "platform_configure_agent_heartbeat" not in ex.actions()
    assert changelog.get("applied_from_approved", []) == []
    # Nothing applied -> the ledger file is never written.
    assert not os.path.exists(_ledger_path(tmp_path))


def test_unresolved_target_is_skipped_not_applied(monkeypatch, tmp_path):
    """A task whose target_name is not in the agents map is skipped, never
    applied against a guessed/null target."""
    monkeypatch.setattr(config, "HARNESS_SELF_MANAGEMENT_ENABLED", True)
    monkeypatch.setattr(config, "WORKSPACE_VOLUME_PATH", str(tmp_path))
    svc = HarnessService()
    task = _harness_task(change_type="heartbeat_tune", target_name="GhostAgent", task_id=9)
    ex = _FakeExecutor(tasks=[task], agents=[{"id": 42, "name": "ScribeAgent"}])
    changelog = {}
    asyncio.run(svc._apply_approved_board_tasks(ex, _WS_ID, changelog))

    assert "platform_configure_agent_heartbeat" not in ex.actions()
    assert changelog.get("applied_from_approved", []) == []
    assert len(changelog.get("skipped", [])) == 1
    assert changelog["skipped"][0]["task_id"] == "9"


def test_placeholder_proposed_value_is_refused(monkeypatch, tmp_path):
    """An approved task whose proposed_value is the 'review_needed' placeholder
    is never applied — applying it literally would corrupt the agent."""
    monkeypatch.setattr(config, "HARNESS_SELF_MANAGEMENT_ENABLED", True)
    monkeypatch.setattr(config, "WORKSPACE_VOLUME_PATH", str(tmp_path))
    svc = HarnessService()
    task = _harness_task(
        change_type="description_update",
        target_name="ScribeAgent",
        current={"description": "old"},
        proposed={"description": "review_needed"},
        task_id=11,
    )
    ex = _FakeExecutor(tasks=[task], agents=[{"id": 42, "name": "ScribeAgent"}])
    changelog = {}
    asyncio.run(svc._apply_approved_board_tasks(ex, _WS_ID, changelog))

    assert "platform_update_agent" not in ex.actions()
    assert changelog.get("applied_from_approved", []) == []
    assert len(changelog.get("failed", [])) == 1
    assert "placeholder" in changelog["failed"][0]["error"]
    # Not applied -> the ledger file is never written.
    assert not os.path.exists(_ledger_path(tmp_path))


# ---------------------------------------------------------------------------
# US-022: auto-rollback on regression
# ---------------------------------------------------------------------------


def test_rollback_on_regression():
    """An auto-applied change whose target agent is now REGRESSION yields an
    auto_applied_regression issue with a rollback_spec; a non-regressing target
    yields nothing."""
    svc = HarnessService()
    baseline = {
        "applied_changes": [
            {
                "target_id": 42,
                "target_name": "ScribeAgent",
                "change_type": "heartbeat_tune",
                "current_value_before": {"interval_minutes": 30},
            },
        ]
    }
    cards = {"42": {"agent_id": "42", "agent_name": "ScribeAgent", "classification": "REGRESSION"}}
    issues = svc._detect_auto_applied_regressions(baseline, cards)

    assert len(issues) == 1
    assert issues[0]["root_cause"] == "auto_applied_regression"
    assert issues[0]["rollback_spec"]["change_type"] == "heartbeat_tune"
    assert issues[0]["rollback_spec"]["revert_to"] == {"interval_minutes": 30}

    # A STABLE target produces no rollback.
    cards["42"]["classification"] = "STABLE"
    assert svc._detect_auto_applied_regressions(baseline, cards) == []

    # An entry with no snapshot is skipped (nothing safe to revert to).
    cards["42"]["classification"] = "REGRESSION"
    baseline["applied_changes"][0]["current_value_before"] = {}
    assert svc._detect_auto_applied_regressions(baseline, cards) == []


def test_rollback_reverts_to_snapshot():
    """_phase_prescribe turns an auto_applied_regression issue into a risk-1
    prescription whose proposed_value is the snapshot."""
    svc = HarnessService()
    diagnosis = {
        "is_first_run": False,
        "health_cards": {},
        "issues": [{
            "agent_id": "42",
            "agent_name": "ScribeAgent",
            "root_cause": "auto_applied_regression",
            "severity": "high",
            "detail": "ScribeAgent regressed",
            "rollback_spec": {
                "change_type": "heartbeat_tune",
                "target_id": 42,
                "revert_to": {"interval_minutes": 30},
            },
        }],
    }
    prescriptions = asyncio.run(
        svc._phase_prescribe(_WS_ID, diagnosis, {"agents": []}, db=None)
    )

    rollbacks = [p for p in prescriptions if p.get("risk_score") == 1]
    assert len(rollbacks) == 1
    assert rollbacks[0]["change_type"] == "heartbeat_tune"
    assert rollbacks[0]["target_id"] == 42
    assert rollbacks[0]["proposed_value"] == {"interval_minutes": 30}


def test_duplicate_applied_changes_emit_single_rollback():
    """The same field can land in applied_changes twice in one tick (a low-risk
    auto-apply plus an approved-task apply). A regressing target must yield ONE
    rollback for that field, not a duplicate."""
    svc = HarnessService()
    baseline = {
        "applied_changes": [
            {
                "target_id": 42,
                "target_name": "ScribeAgent",
                "change_type": "heartbeat_tune",
                "current_value_before": {"interval_minutes": 30},
            },
            {
                "target_id": 42,
                "target_name": "ScribeAgent",
                "change_type": "heartbeat_tune",
                "current_value_before": {"interval_minutes": 45},
            },
        ]
    }
    cards = {"42": {"agent_id": "42", "agent_name": "ScribeAgent", "classification": "REGRESSION"}}
    issues = svc._detect_auto_applied_regressions(baseline, cards)

    assert len(issues) == 1
    # First-seen snapshot wins — the true pre-change value, before any same-tick edit.
    assert issues[0]["rollback_spec"]["revert_to"] == {"interval_minutes": 30}

    # Distinct change_types on the same agent each get their own rollback.
    baseline["applied_changes"][1]["change_type"] = "description_update"
    baseline["applied_changes"][1]["current_value_before"] = {"description": "old"}
    assert len(svc._detect_auto_applied_regressions(baseline, cards)) == 2


def test_applied_rollback_is_not_itself_rolled_back():
    """Oscillation guard: an applied rollback is recorded with current_value={},
    so even if its target stays REGRESSION on the next tick it must NOT spawn
    another rollback (rollback-of-rollback)."""
    svc = HarnessService()
    # This is what a previously-applied rollback looks like in applied_changes:
    # _snapshot_current_value({"current_value": {}}) -> {} stored as the snapshot.
    baseline = {
        "applied_changes": [
            {
                "target_id": 42,
                "target_name": "ScribeAgent",
                "change_type": "heartbeat_tune",
                "current_value_before": {},
            },
        ]
    }
    cards = {"42": {"agent_id": "42", "agent_name": "ScribeAgent", "classification": "REGRESSION"}}
    # Still REGRESSION, but the empty snapshot makes it ineligible — no rollback.
    assert svc._detect_auto_applied_regressions(baseline, cards) == []


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


def test_power_mode_change_type_is_not_implemented():
    """Agents have no power_mode attribute (it is mission-run scoped), so
    power_mode_* prescriptions are refused, not applied to platform_update_agent."""
    svc = HarnessService()
    ex = _FakeExecutor(tasks=[], agents=[])
    for change_type in ("power_mode_upgrade", "power_mode_downgrade"):
        rx = {
            "prescription_id": f"rx-{change_type}",
            "change_type": change_type,
            "target_id": 42,
            "proposed_value": {"power_mode": "max"},
        }
        result = asyncio.run(svc._auto_apply_prescription(ex, rx))
        assert result["success"] is False
        assert "Unknown auto-apply change_type" in result["error"]
    # Nothing was written to any agent.
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


# ---------------------------------------------------------------------------
# US-024: high-risk escalation
# ---------------------------------------------------------------------------


def _high_risk_rx(rx_id="rx-esc", risk=4):
    return {
        "prescription_id": rx_id,
        "change_type": "model_change_same_tier",
        "target_name": "ScribeAgent",
        "current_value": {"model": "haiku"},
        "proposed_value": {"model": "opus"},
        "risk_score": risk,
    }


def test_escalation_sends_telegram(monkeypatch):
    """risk>=4 with a connected channel -> notification sent + escalated entry."""
    import core.services.notification_service as notif_mod
    sent = []

    async def _fake_send(workspace_id, message, channel=None):
        sent.append((workspace_id, message, channel))
        return True

    monkeypatch.setattr(notif_mod, "send_workspace_notification", _fake_send)

    svc = HarnessService()
    svc._workspace_has_channel = lambda db, ws: True
    changelog = {"escalated": []}
    asyncio.run(svc._maybe_escalate(None, _WS_ID, _high_risk_rx(), changelog))

    assert len(sent) == 1
    _, message, _ = sent[0]
    # Approval moved to the authenticated Command Center (W4-S1); the channel
    # notice points there and carries the prescription id, not a /approve command.
    assert "Command Center" in message and "rx-esc" in message
    assert len(changelog["escalated"]) == 1
    assert changelog["escalated"][0]["notified"] is True
    assert changelog["escalated"][0]["prescription_id"] == "rx-esc"


def test_escalation_skipped_no_channel(monkeypatch):
    """No connected channel -> no send, no phantom escalated entry."""
    import core.services.notification_service as notif_mod
    sent = []

    async def _fake_send(workspace_id, message, channel=None):
        sent.append((workspace_id, message, channel))
        return True

    monkeypatch.setattr(notif_mod, "send_workspace_notification", _fake_send)

    svc = HarnessService()
    svc._workspace_has_channel = lambda db, ws: False
    changelog = {"escalated": []}
    asyncio.run(svc._maybe_escalate(None, _WS_ID, _high_risk_rx(), changelog))

    assert sent == []
    assert changelog["escalated"] == []


def test_low_risk_is_not_escalated(monkeypatch):
    """A channel exists but risk < 4 -> not escalated (only high-risk nags)."""
    import core.services.notification_service as notif_mod
    sent = []

    async def _fake_send(workspace_id, message, channel=None):
        sent.append(1)
        return True

    monkeypatch.setattr(notif_mod, "send_workspace_notification", _fake_send)

    svc = HarnessService()
    svc._workspace_has_channel = lambda db, ws: True
    changelog = {"escalated": []}
    asyncio.run(svc._maybe_escalate(None, _WS_ID, _high_risk_rx(risk=2), changelog))

    assert sent == []
    assert changelog["escalated"] == []


def test_escalation_message_points_to_command_center():
    """Approval moved to the authenticated Command Center (PRD-142 W4-S1) — the
    escalation message points there and no longer carries a channel /approve
    command, while still carrying the prescription id, risk, and change_type."""
    svc = HarnessService()
    msg = svc._build_escalation_message(_high_risk_rx(rx_id="rx-99", risk=5), 5)
    assert "Command Center" in msg
    assert "rx-99" in msg
    assert "risk 5/5" in msg
    assert "model_change_same_tier" in msg
    assert "/approve" not in msg  # approval is no longer a channel command
