"""PRD-141 US-025: /approve /reject command handler WITH authorization.

The security-critical story: every command must enforce a workspace-ADMIN check
before any mutation, and an unauthorized caller must change nothing. These are
unit tests — the DB is faked down to the two queries the handler makes
(WorkspaceMember for authz, BoardTask for reject), and the PlatformActionExecutor
is replaced with the in-memory fake from the US-021 suite. Dummy POSTGRES_* and
the apscheduler stub let the harness_service import chain load without a real DB
or the prod-only scheduler dependency.
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
import api.harness_commands as hc

_WS_ID = "00000000-0000-0000-0000-000000000001"
_MISSING_VOLUME = "/tmp/harness-commands-test-no-such-volume"
_RX_ID = "rx-esc-1"


def _harness_task(
    change_type="heartbeat_tune",
    target_name="ScribeAgent",
    current=None,
    proposed=None,
    risk=4,
    task_id=7,
    rx_id=_RX_ID,
):
    """A queued [HARNESS] board task exactly as US-024 tags it for escalation."""
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
        "tags": ["harness", "org-review", f"risk-{risk}", f"rx:{rx_id}"],
    }


class _FakeExecutor:
    """Records execute() calls; returns canned results for the actions the
    command handler invokes (list tasks/agents, apply, update status, write)."""

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
        return {"success": True}

    def actions(self):
        return [action for action, _ in self.calls]


class _FakeQuery:
    def __init__(self, result):
        self._result = result

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self._result


class _FakeBoardTask:
    def __init__(self, task_id):
        self.id = task_id
        self.workspace_id = _WS_ID
        self.status = "review"
        self.blocked_reason = None
        self.blocked_at = None


class _FakeDB:
    """Answers the two queries the handler makes, by model name."""

    def __init__(self, member=None, board_task=None):
        self._member = member
        self._board_task = board_task
        self.committed = False

    def query(self, model):
        name = getattr(model, "__name__", "")
        if name == "WorkspaceMember":
            return _FakeQuery(self._member)
        if name == "BoardTask":
            return _FakeQuery(self._board_task)
        return _FakeQuery(None)

    def commit(self):
        self.committed = True


_ADMIN_MEMBER = object()  # truthy sentinel: an active owner/admin row exists
_ADMIN = {"user_id": 5}
_NON_ADMIN = {"user_id": 99}


def _patch_executor(monkeypatch, ex):
    monkeypatch.setattr(hc, "_make_executor", lambda db, workspace_id: ex)


def test_handle_approve_command(monkeypatch):
    """Admin /approve applies the prescription now and records it in the ledger."""
    monkeypatch.setattr(config, "HARNESS_SELF_MANAGEMENT_ENABLED", True)
    monkeypatch.setattr(config, "WORKSPACE_VOLUME_PATH", _MISSING_VOLUME)
    task = _harness_task(task_id=7)
    ex = _FakeExecutor(tasks=[task], agents=[{"id": 42, "name": "ScribeAgent"}])
    _patch_executor(monkeypatch, ex)
    db = _FakeDB(member=_ADMIN_MEMBER)

    result = asyncio.run(
        hc.handle_harness_command(db, _WS_ID, "/approve", _RX_ID, _ADMIN)
    )

    assert result["success"] is True
    # The change was actually applied to the resolved agent id.
    assert (
        "platform_configure_agent_heartbeat",
        {"agent_id": 42, "interval_minutes": 90},
    ) in ex.calls
    # The board task was marked done and the ledger persisted.
    assert ("platform_update_task_status", {"task_id": 7, "status": "done"}) in ex.calls
    assert "workspace_write_file" in ex.actions()


def test_handle_reject_command(monkeypatch):
    """Admin /reject flips the board task to 'rejected' and commits, so
    _get_rejected_signatures suppresses re-proposal next tick."""
    monkeypatch.setattr(config, "HARNESS_SELF_MANAGEMENT_ENABLED", True)
    task = _harness_task(task_id=7)
    ex = _FakeExecutor(tasks=[task], agents=[{"id": 42, "name": "ScribeAgent"}])
    _patch_executor(monkeypatch, ex)
    board_row = _FakeBoardTask(7)
    db = _FakeDB(member=_ADMIN_MEMBER, board_task=board_row)

    result = asyncio.run(
        hc.handle_harness_command(db, _WS_ID, "/reject", _RX_ID, _ADMIN)
    )

    assert result["success"] is True
    assert board_row.status == "rejected"
    assert db.committed is True
    # Nothing was applied on a reject.
    assert "platform_configure_agent_heartbeat" not in ex.actions()


def test_handle_unknown_rx_id(monkeypatch):
    """An rx id with no matching queued task returns not-found, no mutation."""
    monkeypatch.setattr(config, "HARNESS_SELF_MANAGEMENT_ENABLED", True)
    monkeypatch.setattr(config, "WORKSPACE_VOLUME_PATH", _MISSING_VOLUME)
    task = _harness_task(task_id=7, rx_id="rx-different")
    ex = _FakeExecutor(tasks=[task], agents=[{"id": 42, "name": "ScribeAgent"}])
    _patch_executor(monkeypatch, ex)
    db = _FakeDB(member=_ADMIN_MEMBER)

    result = asyncio.run(
        hc.handle_harness_command(db, _WS_ID, "/approve", "rx-missing", _ADMIN)
    )

    assert result["success"] is False
    assert "no pending harness change" in result["message"].lower()
    assert "platform_configure_agent_heartbeat" not in ex.actions()


def test_handle_already_applied(monkeypatch):
    """A second /approve is idempotent: the ledger already has the task id, so
    nothing is applied again."""
    monkeypatch.setattr(config, "HARNESS_SELF_MANAGEMENT_ENABLED", True)
    task = _harness_task(task_id=7)
    ex = _FakeExecutor(tasks=[task], agents=[{"id": 42, "name": "ScribeAgent"}])
    _patch_executor(monkeypatch, ex)
    db = _FakeDB(member=_ADMIN_MEMBER)
    # The shared US-021 ledger already records this task as applied.
    monkeypatch.setattr(
        hc.get_harness_service(),
        "_read_applied_tasks",
        lambda workspace_id: {"applied_task_ids": ["7"], "entries": []},
    )

    result = asyncio.run(
        hc.handle_harness_command(db, _WS_ID, "/approve", _RX_ID, _ADMIN)
    )

    assert result["success"] is True
    assert result.get("already_applied") is True
    assert "platform_configure_agent_heartbeat" not in ex.actions()
    assert "workspace_write_file" not in ex.actions()


def test_unauthorized_caller_rejected(monkeypatch):
    """A non-admin caller is refused before ANY state is read or written —
    no task lookup, no apply, no commit. This is the US-026 security boundary."""
    monkeypatch.setattr(config, "HARNESS_SELF_MANAGEMENT_ENABLED", True)
    task = _harness_task(task_id=7)
    ex = _FakeExecutor(tasks=[task], agents=[{"id": 42, "name": "ScribeAgent"}])
    _patch_executor(monkeypatch, ex)
    # member=None -> the WorkspaceMember admin query finds no row -> not authorized.
    board_row = _FakeBoardTask(7)
    db = _FakeDB(member=None, board_task=board_row)

    result = asyncio.run(
        hc.handle_harness_command(db, _WS_ID, "/approve", _RX_ID, _NON_ADMIN)
    )

    assert result["success"] is False
    assert result.get("unauthorized") is True
    # Refused before any executor call or DB mutation.
    assert ex.calls == []
    assert db.committed is False
    assert board_row.status == "review"


def test_malformed_identity_shapes_refused(monkeypatch):
    """Every malformed/hostile caller_identity is fail-closed refused before any
    state is touched — including a bool (which would coerce to user 1) and a
    non-positive id. An admin member row exists, so only the identity shape can
    grant or deny access here."""
    monkeypatch.setattr(config, "HARNESS_SELF_MANAGEMENT_ENABLED", True)
    for identity in (None, {}, {"user_id": None}, {"user_id": 0}, {"user_id": True},
                     {"user_id": -1}, {"user_id": "abc"}, "not-a-dict"):
        ex = _FakeExecutor(tasks=[_harness_task()], agents=[{"id": 42, "name": "ScribeAgent"}])
        _patch_executor(monkeypatch, ex)
        db = _FakeDB(member=_ADMIN_MEMBER)
        result = asyncio.run(
            hc.handle_harness_command(db, _WS_ID, "/approve", _RX_ID, identity)
        )
        assert result["success"] is False, identity
        assert result.get("unauthorized") is True, identity
        assert ex.calls == [], identity


def test_disabled_flag_is_noop(monkeypatch):
    """Flag off -> every command is inert, regardless of caller."""
    monkeypatch.setattr(config, "HARNESS_SELF_MANAGEMENT_ENABLED", False)
    ex = _FakeExecutor(tasks=[_harness_task()], agents=[])
    _patch_executor(monkeypatch, ex)
    db = _FakeDB(member=_ADMIN_MEMBER)

    result = asyncio.run(
        hc.handle_harness_command(db, _WS_ID, "/approve", _RX_ID, _ADMIN)
    )

    assert result["success"] is False
    assert "disabled" in result["message"].lower()
    assert ex.calls == []
