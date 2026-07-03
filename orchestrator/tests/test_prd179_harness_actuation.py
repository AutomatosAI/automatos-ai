"""PRD-179 S3 (F048) — HARNESS prescription actuation through the policy plane.

Today a human-approved prescription either dead-ends (the flag is off by default,
so the approve endpoint returns HTTP 409) or, with the flag on, actuates via a
DARK direct call to ``_auto_apply_prescription`` that bypasses the Wave-4 policy
plane entirely. F048 (adjusted): make approval a GOVERNED ACTIVATION — the
prescription's actuation is routed through the same ``evaluate_approval`` ask
verdict every other governed action passes through (Wave 4), the board task is
marked done WITH a result, and there is no 409.

Reuses ``core.services.approval_policy.evaluate_approval`` — no parallel approval
path. The admin's explicit approve is the per-request override the plane already
models. Enabled behind the existing ``HARNESS_SELF_MANAGEMENT_ENABLED`` flag.

Unit-level: the DB is faked to the handler's two queries and the executor is the
in-memory fake from the US-021 suite; the policy plane is real (pure function).
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

from config import config  # noqa: E402

# config caches POSTGRES_* at import; a sibling conftest may have imported it
# with no env set (local dev). Backfill so the lazy engine builder in the
# harness import chain does not refuse on missing creds. No-op on CI.
for _k in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB"):
    if not getattr(config, _k, None):
        setattr(config, _k, os.environ[_k])

import api.harness_commands as hc  # noqa: E402

_WS_ID = "00000000-0000-0000-0000-000000000001"
_RX_ID = "rx-actuate-1"


def _ledger_path(volume, workspace_id=_WS_ID):
    return os.path.join(str(volume), str(workspace_id), "harness", "applied_tasks.json")


def _harness_task(task_id=7, rx_id=_RX_ID, risk=2):
    return {
        "id": task_id,
        "title": f"[HARNESS] heartbeat_tune for ScribeAgent",
        "description": (
            f"**Risk Score:** {risk}/5\n\n"
            f"**Change Type:** heartbeat_tune\n\n"
            f"**Current:** {json.dumps({'interval_minutes': 30})}\n\n"
            f"**Proposed:** {json.dumps({'interval_minutes': 90})}\n\n"
            f"**Rationale:** because reasons\n\n"
            f"**Expected Improvement:** save tokens"
        ),
        "tags": ["harness", "org-review", f"risk-{risk}", f"rx:{rx_id}"],
    }


class _FakeExecutor:
    """Records execute() calls. The prescription actuation and the board-task
    status update BOTH return a non-null result payload, so the test can assert
    the task completes with a result (status=done, result != null)."""

    def __init__(self, tasks, agents):
        self._tasks = tasks
        self._agents = agents
        self.calls = []
        self.task_status = {}
        self.task_result = {}

    async def execute(self, action, params):
        self.calls.append((action, params))
        if action == "platform_list_tasks":
            return {"data": self._tasks}
        if action == "platform_list_agents":
            return {"data": self._agents}
        if action == "platform_update_task_status":
            self.task_status[params.get("task_id")] = params.get("status")
            if params.get("result") is not None:
                self.task_result[params.get("task_id")] = params.get("result")
            return {"success": True, "data": {"status": params.get("status")}}
        if action == "platform_configure_agent_heartbeat":
            # The actuation returns a concrete, non-null result.
            return {"success": True, "data": {"agent_id": params.get("agent_id"),
                                              "interval_minutes": params.get("interval_minutes")}}
        return {"success": True}

    def actions(self):
        return [action for action, _ in self.calls]


class _FakeBoardTask:
    """Stands in for the BoardTask ORM row the actuation writes its result onto."""

    def __init__(self, task_id):
        self.id = task_id
        self.workspace_id = _WS_ID
        self.status = "review"
        self.result = None
        self.completed_at = None


class _FakeQuery:
    def __init__(self, result):
        self._result = result

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self._result


class _FakeDB:
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


_ADMIN_MEMBER = object()
_ADMIN = {"user_id": 5}


def _patch_executor(monkeypatch, ex):
    monkeypatch.setattr(hc, "_make_executor", lambda db, workspace_id: ex)


def test_harness_prescription_actuates(monkeypatch, tmp_path):
    """Approve → routed through the policy-plane ask verdict → actuates → board
    task done with a non-null result. No 409, no dark direct-apply."""
    monkeypatch.setattr(config, "HARNESS_SELF_MANAGEMENT_ENABLED", True)
    monkeypatch.setattr(config, "WORKSPACE_VOLUME_PATH", str(tmp_path))

    # Spy on the policy plane so we prove the actuation is GOVERNED — the same
    # evaluate_approval verdict every other governed action passes through.
    import core.services.approval_policy as ap
    verdicts = []
    real_eval = ap.evaluate_approval

    def _spy(db, workspace_id, cost, **kw):
        d = ap.ApprovalDecision(
            auto_approve=True, reason="harness admin approve (per-request override)",
            policy="override", ceiling=None, estimated_cost=cost, countdown_seconds=None,
        )
        verdicts.append((workspace_id, kw))
        return d

    monkeypatch.setattr(hc, "evaluate_approval", _spy, raising=False)
    # Also patch at the source in case the handler imports it lazily by module.
    monkeypatch.setattr(ap, "evaluate_approval", _spy)

    task = _harness_task(task_id=7)
    ex = _FakeExecutor(tasks=[task], agents=[{"id": 42, "name": "ScribeAgent"}])
    _patch_executor(monkeypatch, ex)
    board_row = _FakeBoardTask(7)
    db = _FakeDB(member=_ADMIN_MEMBER, board_task=board_row)

    result = asyncio.run(hc.handle_harness_command(db, _WS_ID, "/approve", _RX_ID, _ADMIN))

    # 1. It succeeded (no 409 dead-end).
    assert result["success"] is True, result

    # 2. It went through the policy plane's ask verdict (governed activation).
    assert verdicts, "actuation did not route through evaluate_approval (policy plane)"

    # 3. The actuation actually ran against the resolved agent.
    assert (
        "platform_configure_agent_heartbeat",
        {"agent_id": 42, "interval_minutes": 90},
    ) in ex.calls

    # 4. The board task completed with status=done AND a non-null result.
    assert ex.task_status.get(7) == "done", ex.task_status
    assert board_row.result is not None, "board task completed with a null result"
    assert board_row.completed_at is not None


def test_actuation_blocked_when_policy_denies(monkeypatch, tmp_path):
    """If the policy plane denies the verdict, the prescription does NOT actuate —
    governed activation means the plane can still say no."""
    monkeypatch.setattr(config, "HARNESS_SELF_MANAGEMENT_ENABLED", True)
    monkeypatch.setattr(config, "WORKSPACE_VOLUME_PATH", str(tmp_path))

    import core.services.approval_policy as ap

    def _deny(db, workspace_id, cost, **kw):
        return ap.ApprovalDecision(
            auto_approve=False, reason="denied by policy", policy="always_ask",
            ceiling=None, estimated_cost=cost, countdown_seconds=None,
        )

    monkeypatch.setattr(hc, "evaluate_approval", _deny, raising=False)
    monkeypatch.setattr(ap, "evaluate_approval", _deny)

    task = _harness_task(task_id=7)
    ex = _FakeExecutor(tasks=[task], agents=[{"id": 42, "name": "ScribeAgent"}])
    _patch_executor(monkeypatch, ex)
    db = _FakeDB(member=_ADMIN_MEMBER)

    result = asyncio.run(hc.handle_harness_command(db, _WS_ID, "/approve", _RX_ID, _ADMIN))

    assert result["success"] is False, "a denied verdict must not actuate"
    assert "platform_configure_agent_heartbeat" not in ex.actions()
    assert ex.task_status.get(7) != "done"
    assert not os.path.exists(_ledger_path(tmp_path))


def test_disabled_flag_still_dark(monkeypatch):
    """Flag off → the whole path stays inert (feature dark by default)."""
    monkeypatch.setattr(config, "HARNESS_SELF_MANAGEMENT_ENABLED", False)
    ex = _FakeExecutor(tasks=[_harness_task()], agents=[])
    _patch_executor(monkeypatch, ex)
    db = _FakeDB(member=_ADMIN_MEMBER)

    result = asyncio.run(hc.handle_harness_command(db, _WS_ID, "/approve", _RX_ID, _ADMIN))

    assert result["success"] is False
    assert "disabled" in result["message"].lower()
    assert ex.calls == []
