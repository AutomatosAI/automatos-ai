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
# A path that does not exist, so _read_applied_tasks finds no ledger and treats
# every task as un-applied. The ledger WRITE goes through the fake executor's
# workspace_write_file (in-memory), so no real filesystem I/O occurs.
_MISSING_VOLUME = "/tmp/harness-self-mgmt-test-no-such-volume"


class _FakeExecutor:
    """Records every execute() call and returns canned results for the actions
    _apply_approved_board_tasks invokes (list tasks/agents, apply, write file)."""

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
        # apply actions + workspace_write_file all report success
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


def test_approved_tasks_are_executed(monkeypatch):
    """Flag on -> a done [HARNESS] task is parsed, its target resolved, and the
    change applied via _auto_apply_prescription, then the ledger is persisted."""
    monkeypatch.setattr(config, "HARNESS_SELF_MANAGEMENT_ENABLED", True)
    monkeypatch.setattr(config, "WORKSPACE_VOLUME_PATH", _MISSING_VOLUME)
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
    # Idempotency ledger was persisted via the workspace file store.
    assert "workspace_write_file" in ex.actions()


def test_snapshot_recorded_before_apply(monkeypatch):
    """The pre-change value is captured as current_value_before so US-022 has a
    rollback target."""
    monkeypatch.setattr(config, "HARNESS_SELF_MANAGEMENT_ENABLED", True)
    monkeypatch.setattr(config, "WORKSPACE_VOLUME_PATH", _MISSING_VOLUME)
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


def test_already_applied_task_is_skipped(monkeypatch):
    """A task whose id is in the ledger is never re-applied (idempotency)."""
    monkeypatch.setattr(config, "HARNESS_SELF_MANAGEMENT_ENABLED", True)
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
    # Nothing applied -> no ledger write.
    assert "workspace_write_file" not in ex.actions()


def test_unresolved_target_is_skipped_not_applied(monkeypatch):
    """A task whose target_name is not in the agents map is skipped, never
    applied against a guessed/null target."""
    monkeypatch.setattr(config, "HARNESS_SELF_MANAGEMENT_ENABLED", True)
    monkeypatch.setattr(config, "WORKSPACE_VOLUME_PATH", _MISSING_VOLUME)
    svc = HarnessService()
    task = _harness_task(change_type="heartbeat_tune", target_name="GhostAgent", task_id=9)
    ex = _FakeExecutor(tasks=[task], agents=[{"id": 42, "name": "ScribeAgent"}])
    changelog = {}
    asyncio.run(svc._apply_approved_board_tasks(ex, _WS_ID, changelog))

    assert "platform_configure_agent_heartbeat" not in ex.actions()
    assert changelog.get("applied_from_approved", []) == []
    assert len(changelog.get("skipped", [])) == 1
    assert changelog["skipped"][0]["task_id"] == "9"


def test_placeholder_proposed_value_is_refused(monkeypatch):
    """An approved task whose proposed_value is the 'review_needed' placeholder
    is never applied — applying it literally would corrupt the agent."""
    monkeypatch.setattr(config, "HARNESS_SELF_MANAGEMENT_ENABLED", True)
    monkeypatch.setattr(config, "WORKSPACE_VOLUME_PATH", _MISSING_VOLUME)
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
    # Not applied -> not ledgered.
    assert "workspace_write_file" not in ex.actions()


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
