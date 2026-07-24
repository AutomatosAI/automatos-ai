"""PRD-174 W4 — executor chokepoint characterization (§6.1/§6.2, safety rule #1/#2).

The headline, at the real integration point (``UnifiedToolExecutor.execute_tool``):

- **Flag OFF ⇒ byte-for-byte** — ``_policy_gate_check`` returns ``None`` (a no-op),
  so dispatch proceeds exactly as it did before PRD-174. Composio/workspace/
  registry routing is unchanged.
- **Flag ON + deny ⇒ the tool NEVER dispatches** and the caller gets an
  errors-as-data result (``policy_error`` block the model can read).

``UnifiedToolExecutor`` imports the DB stack, so this suite is **CI-gated**: it
is skipped when the DB isn't configured (the local ``py_compile`` + pure-unit
tests cover the logic; CI runs this against a real DB). We monkeypatch the gate
verdict + the downstream dispatch so no tool actually executes and no network is
touched — we assert on *whether dispatch was reached*.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

# CI gate: import the executor or skip the whole module. The executor pulls the
# DB stack, which raises ValueError (not ImportError) when creds are absent, so
# importorskip alone won't catch it — guard on any exception.
try:
    import modules.tools.execution.unified_executor as unified_executor  # noqa: E402
except Exception as _exc:  # pragma: no cover - environment-dependent skip
    pytest.skip(
        f"UnifiedToolExecutor unavailable (CI-gated): {type(_exc).__name__}",
        allow_module_level=True,
    )

from modules.policy.types import PolicyError, Verdict  # noqa: E402


class _FakeSession:
    """No-op DB session — the gate is monkeypatched so it's never really used."""

    def query(self, *a, **k):
        raise AssertionError("gate was monkeypatched; DB should not be queried")


def _make_executor():
    return unified_executor.UnifiedToolExecutor(db_session=_FakeSession())


@pytest.mark.asyncio
async def test_flag_off_gate_is_noop(monkeypatch):
    """Mode off ⇒ _policy_gate_check returns None (byte-for-byte, no gate).

    PRD-192 S1: the executor branches on the staged mode dial
    (``policy_plane_mode``), not the derived boolean.
    """
    monkeypatch.setattr("modules.policy.policy_plane_mode", lambda: "off", raising=False)
    ex = _make_executor()
    blocked = ex._policy_gate_check(
        "platform_delete_agent", {"id": 1}, agent_id=1,
        workspace_id="ws", caller_context=None, trace="t",
    )
    assert blocked is None  # no policy result → dispatch proceeds as before


@pytest.mark.asyncio
async def test_flag_on_deny_blocks_before_dispatch(monkeypatch):
    """Mode on + deny ⇒ execute_tool returns the errors-as-data block and the
    tool is NEVER dispatched. (PRD-192 S1: on = enforce-all stage.)"""
    monkeypatch.setattr("modules.policy.policy_plane_mode", lambda: "on", raising=False)

    # Force the gate to deny, regardless of DB/registry.
    deny = Verdict.deny(PolicyError(
        code="approval_required",
        message_for_model="Blocked: needs approval; NOT executed.",
        remediation="approve in the queue", retryable=True,
    ))
    monkeypatch.setattr(
        "modules.policy.PolicyGate.check", lambda self, call: deny, raising=False
    )

    ex = _make_executor()

    # Trip-wire: if dispatch is reached, fail loudly. It must not be.
    async def _tripwire(*a, **k):
        raise AssertionError("denied call reached dispatch — it must not execute")

    monkeypatch.setattr(ex, "_execute_platform_action", _tripwire)

    result = await ex.execute_tool(
        "platform_delete_agent", {"id": 1}, agent_id=1, workspace_id="ws",
    )
    assert result["success"] is False
    assert result.get("policy_error", {}).get("code") == "approval_required"
    assert "NOT executed" in result["llm_context"]


@pytest.mark.asyncio
async def test_flag_on_allow_reaches_dispatch(monkeypatch):
    """Mode on + allow ⇒ the gate is a pass-through; dispatch is reached."""
    monkeypatch.setattr("modules.policy.policy_plane_mode", lambda: "on", raising=False)
    monkeypatch.setattr(
        "modules.policy.PolicyGate.check",
        lambda self, call: Verdict.allow("fine"), raising=False,
    )
    ex = _make_executor()

    reached = {"dispatched": False}

    async def _dispatch(*a, **k):
        reached["dispatched"] = True
        return {"success": True, "tool": "platform_list_agents"}

    monkeypatch.setattr(ex, "_execute_platform_action", _dispatch)

    await ex.execute_tool("platform_list_agents", {}, agent_id=1, workspace_id="ws")
    assert reached["dispatched"] is True
