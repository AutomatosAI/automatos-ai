"""PRD-143 S8 — governance positive path: full operator surface at
autonomy=full, with the gates-and-logs safety net proven LIVE.

Rev 2 inverts exclusion for everything but the obs tier, so this suite proves
"open ≠ unguarded" (PRD-143 US-008 positive half):

  * a representative ``permission_level='destructive'`` OPERATOR action
    (``platform_delete_agent``, real catalogue) executes WITHOUT confirmation
    under full autonomy for a plain member principal — while the PRD-140
    hierarchy gate is still consulted and the destructive backstop stays
    satisfied (``requires_confirmation=True`` on the def);
  * the dial works both ways — the same action stops with
    ``requires_confirmation`` at standard autonomy;
  * the kill-switch halts mid-flow — flipping the autonomy dial off re-imposes
    confirmation on the very next call (no caching), and flipping
    ``HARNESS_SELF_MANAGEMENT_ENABLED`` off refuses further auto-apply;
  * every autonomous invocation lands in the Wave 4 audit trail
    (``tool_execution_logs.router_decision->>'autonomous'``) distinctly and
    queryably.

Idioms: S2's fake-POSTGRES preamble (closed port 59432 so a wedged local
proxy can't hang the fail-soft import-time connect), the manifest suite's
real-catalogue registry fixture, W3's autonomy/rate-limit patches, and the
Wave 4 harness helpers from tests/test_harness_self_management.py. No DB.
"""
from __future__ import annotations

import importlib.util as _ilu
import json
import os
import sys
import types
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

os.environ.pop("HARNESS_SELF_MANAGEMENT_ENABLED", None)
os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
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


# Lean-venv shim: modules/tools/__init__ pulls modules.rag's ingestion chain
# (camelot at module top). Stub the missing leaf only when truly absent.
def _camelot_unlocatable() -> bool:  # pragma: no cover - env-dependent
    try:
        return _ilu.find_spec("camelot") is None
    except ValueError:
        return False


if _camelot_unlocatable():  # pragma: no cover - env-dependent
    sys.modules.setdefault("camelot", types.ModuleType("camelot"))

import modules.tools.discovery.platform_executor as pe  # noqa: E402
from core.security.hierarchy_permissions import (  # noqa: E402
    PermissionDecision,
    TARGET_AGENT,
)
from modules.tools.discovery.action_registry import ActionRegistry  # noqa: E402
from modules.tools.discovery.platform_actions import register_all_actions  # noqa: E402
from modules.tools.discovery.platform_executor import PlatformActionExecutor  # noqa: E402
from modules.tools.execution.telemetry import write_telemetry  # noqa: E402

pytestmark = pytest.mark.asyncio

# Representative destructive OPERATOR action — real catalogue, in
# _HIERARCHY_TARGETS (TARGET_AGENT, "agent_id") so the PRD-140 gate runs.
_DESTRUCTIVE = "platform_delete_agent"
_READ = "platform_list_agents"
_MEMBER_CTX = {"workspace_role": "member", "user_id": 7}
_WS_ID = "00000000-0000-0000-0000-000000000001"


@pytest.fixture(scope="module")
def real_registry() -> ActionRegistry:
    """The REAL catalogue, registered directly (not the singleton) so
    _ensure_initialized cannot trigger a second full init."""
    reg = ActionRegistry()
    register_all_actions(reg)
    reg._initialized = True
    return reg


def _registry_patch(reg: ActionRegistry):
    return patch("modules.tools.discovery.get_action_registry", return_value=reg)


def _no_rate_limit():
    return patch(
        "core.security.rate_limiter.check_rate_limit",
        new=AsyncMock(return_value=None),
    )


def _allow_decision() -> PermissionDecision:
    return PermissionDecision(allowed=True, reason="s8_test_allow")


def _executor() -> PlatformActionExecutor:
    return PlatformActionExecutor(MagicMock(), uuid.uuid4())


def _stub_handler(ex: PlatformActionExecutor, action: str):
    sentinel = {"success": True, "_sentinel": "handler-ran"}
    ex._handlers[action] = AsyncMock(return_value=sentinel)
    return sentinel


# ---------------------------------------------------------------------------
# 1. Full surface positive: destructive operator action at autonomy=full
# ---------------------------------------------------------------------------

async def test_operator_destructive_action_executes_at_full_autonomy(real_registry):
    """A plain member principal + full autonomy: the destructive action runs
    without confirmation, the PRD-140 hierarchy gate is CONSULTED (not
    bypassed) and allows, and the destructive backstop stays satisfied."""
    action_def = real_registry.get(_DESTRUCTIVE)
    # The Rev 2 facts this test rides on: operator tier, destructive,
    # requires_confirmation=True (so the backstop never fires on it).
    assert action_def is not None
    assert action_def.permission_level == "destructive"
    assert action_def.requires_confirmation is True
    assert action_def.super_admin_only is False
    assert action_def.admin_only is False

    ex = _executor()
    sentinel = _stub_handler(ex, _DESTRUCTIVE)
    with _registry_patch(real_registry), patch.object(
        PlatformActionExecutor, "_full_autonomy", return_value=True
    ), patch.object(
        pe, "can_actor_modify", return_value=_allow_decision()
    ) as gate, _no_rate_limit():
        result = await ex.execute(
            _DESTRUCTIVE, {"agent_id": 42, "_agent_id": 9}, dict(_MEMBER_CTX)
        )

    assert result.get("success") is True
    assert result.get("_sentinel") == sentinel["_sentinel"]
    assert "requires_confirmation" not in result
    ex._handlers[_DESTRUCTIVE].assert_awaited_once()
    # The safety net replacing exclusion: the hierarchy gate ran on this call.
    gate.assert_called_once()
    gate_kwargs = gate.call_args.kwargs
    assert gate_kwargs["actor_agent_id"] == 9
    assert gate_kwargs["target_type"] == TARGET_AGENT
    assert gate_kwargs["target_id"] == 42
    assert gate_kwargs["change_type"] == "delete"
    assert gate_kwargs["source"] == "platform_tool"


# ---------------------------------------------------------------------------
# 2. The dial works both ways
# ---------------------------------------------------------------------------

async def test_same_action_requires_confirmation_when_autonomy_standard(real_registry):
    """Standard autonomy: the SAME action stops at the confirmation gate
    before the hierarchy gate or handler are ever reached."""
    ex = _executor()
    _stub_handler(ex, _DESTRUCTIVE)
    with _registry_patch(real_registry), patch.object(
        PlatformActionExecutor, "_full_autonomy", return_value=False
    ), patch.object(
        pe, "can_actor_modify", return_value=_allow_decision()
    ) as gate, _no_rate_limit():
        result = await ex.execute(
            _DESTRUCTIVE, {"agent_id": 42, "_agent_id": 9}, dict(_MEMBER_CTX)
        )

    assert result["success"] is False
    assert result.get("requires_confirmation") is True
    assert result.get("action") == _DESTRUCTIVE
    assert result.get("permission_level") == "destructive"
    ex._handlers[_DESTRUCTIVE].assert_not_awaited()
    gate.assert_not_called()


# ---------------------------------------------------------------------------
# 3. Kill-switch: both flags halt mid-flow
# ---------------------------------------------------------------------------

def _harness_task(task_id=7, current=None, proposed=None):
    """Board task exactly as HARNESS _phase_apply() produces it (Wave 4 idiom,
    see tests/test_harness_self_management.py)."""
    current = {"interval_minutes": 30} if current is None else current
    proposed = {"interval_minutes": 90} if proposed is None else proposed
    return {
        "id": task_id,
        "title": "[HARNESS] heartbeat_tune for ScribeAgent",
        "description": (
            "**Risk Score:** 2/5\n\n"
            "**Change Type:** heartbeat_tune\n\n"
            f"**Current:** {json.dumps(current)}\n\n"
            f"**Proposed:** {json.dumps(proposed)}\n\n"
            "**Rationale:** because reasons\n\n"
            "**Expected Improvement:** save tokens"
        ),
        "tags": ["harness", "org-review", "risk-2"],
    }


class _FakeHarnessExecutor:
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


async def test_kill_switch_halts(monkeypatch, tmp_path, real_registry):
    """Flipping either kill-switch flag off MID-FLOW takes effect on the very
    next call — the autonomy dial re-imposes confirmation, the HARNESS flag
    refuses auto-apply. Neither is cached."""
    # --- Half 1: the autonomy dial, read fresh through the canonical service
    # on EVERY execute() — flipping it off mid-flow halts the next action.
    import core.services.auto_autonomy as autonomy_mod

    state = {"full": True}
    monkeypatch.setattr(
        autonomy_mod, "is_full_autonomy", lambda db, ws: state["full"]
    )

    ex = _executor()
    sentinel = _stub_handler(ex, _DESTRUCTIVE)
    with _registry_patch(real_registry), patch.object(
        pe, "can_actor_modify", return_value=_allow_decision()
    ), _no_rate_limit():
        first = await ex.execute(
            _DESTRUCTIVE, {"agent_id": 42, "_agent_id": 9}, dict(_MEMBER_CTX)
        )
        assert first.get("_sentinel") == sentinel["_sentinel"]

        state["full"] = False  # the human flips the dial mid-flow

        second = await ex.execute(
            _DESTRUCTIVE, {"agent_id": 42, "_agent_id": 9}, dict(_MEMBER_CTX)
        )

    assert second["success"] is False
    assert second.get("requires_confirmation") is True
    ex._handlers[_DESTRUCTIVE].assert_awaited_once()  # only the first ran

    # --- Half 2: the HARNESS self-management flag — on applies, off refuses.
    from config import config
    from services.harness_service import HarnessService

    monkeypatch.setattr(config, "HARNESS_SELF_MANAGEMENT_ENABLED", True)
    monkeypatch.setattr(config, "WORKSPACE_VOLUME_PATH", str(tmp_path))
    svc = HarnessService()
    agents = [{"id": 42, "name": "ScribeAgent"}]

    fake_on = _FakeHarnessExecutor(tasks=[_harness_task(task_id=7)], agents=agents)
    changelog_on = {}
    await svc._apply_approved_board_tasks(fake_on, _WS_ID, changelog_on)
    assert (
        "platform_configure_agent_heartbeat",
        {"agent_id": 42, "interval_minutes": 90},
    ) in fake_on.calls

    monkeypatch.setattr(config, "HARNESS_SELF_MANAGEMENT_ENABLED", False)
    fake_off = _FakeHarnessExecutor(tasks=[_harness_task(task_id=8)], agents=agents)
    changelog_off = {}
    await svc._apply_approved_board_tasks(fake_off, _WS_ID, changelog_off)
    assert fake_off.calls == []  # flag off → no listing, no apply
    assert changelog_off == {}


# ---------------------------------------------------------------------------
# 4. Wave 4 audit trail: autonomous invocations are distinct and queryable
# ---------------------------------------------------------------------------

async def test_audit_row_written_for_autonomous_action(real_registry):
    """An invocation that ran ONLY because the full dial skipped confirmation
    is (a) marked ``autonomous`` at the executor and (b) persisted to
    tool_execution_logs with router_decision->>'autonomous' = true — so the
    audit trail distinguishes autonomous actions queryably. Non-autonomous
    invocations carry no marker (distinctness cuts both ways)."""
    ws_id = uuid.uuid4()

    # (a) the executor marks the confirmation-skipped invocation...
    ex = PlatformActionExecutor(MagicMock(), ws_id)
    _stub_handler(ex, _DESTRUCTIVE)
    _stub_handler(ex, _READ)
    with _registry_patch(real_registry), patch.object(
        PlatformActionExecutor, "_full_autonomy", return_value=True
    ), patch.object(
        pe, "can_actor_modify", return_value=_allow_decision()
    ), _no_rate_limit():
        autonomous_result = await ex.execute(
            _DESTRUCTIVE, {"agent_id": 42, "_agent_id": 9}, dict(_MEMBER_CTX)
        )
        # ...and a read that never needed confirmation is NOT marked, even
        # at full autonomy — the marker means "the dial bypassed the gate".
        read_result = await ex.execute(_READ, {}, dict(_MEMBER_CTX))

    assert autonomous_result.get("success") is True
    assert autonomous_result.get("autonomous") is True
    assert "autonomous" not in read_result

    # (b) the universal telemetry hook persists the marker on the row.
    db = MagicMock()
    await write_telemetry(
        session_factory=lambda: db,
        tool_name=_DESTRUCTIVE,
        parameters={"agent_id": 42},
        agent_id=9,
        workspace_id=ws_id,
        result=autonomous_result,
        execution_time_ms=5,
        caller_context={"user_id": 7},
    )
    db.add.assert_called_once()
    row = db.add.call_args[0][0]
    assert row.action_name == _DESTRUCTIVE
    assert row.workspace_id == ws_id
    assert (row.router_decision or {}).get("autonomous") is True
    db.commit.assert_called_once()

    # Control row: a non-autonomous result writes NO autonomous marker.
    db2 = MagicMock()
    await write_telemetry(
        session_factory=lambda: db2,
        tool_name=_READ,
        parameters={},
        agent_id=9,
        workspace_id=ws_id,
        result=read_result,
        execution_time_ms=3,
        caller_context={"user_id": 7},
    )
    row2 = db2.add.call_args[0][0]
    assert not (row2.router_decision or {}).get("autonomous")
