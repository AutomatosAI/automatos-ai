"""PRD-192 S5 (P2-11) — playbook Composio steps + widget email ride the spine.

The two remaining Composio side-doors stop executing raw:

- **Playbook lane** — the recipe LLM-step's per-action block dispatches through
  ``execute_and_format`` → ``UnifiedToolExecutor`` (composio_execute meta-tool,
  per-action name in ``action``), so the gate/telemetry/outcome-capture govern
  the daily-failing cron lane.
- **Widget email** — the four raw ``client.execute_action`` sites route through
  the same executor with the HUMAN-DIRECT actor marker (locked #6): the user's
  click IS the approval, so the route gate treats the ask as satisfied while
  budget admission and the Art.12 audit row still apply; agent-initiated calls
  on the same integration still ask.

Pure at the boundaries: the gate is driven directly with in-memory fakes
(the ``test_prd174_policy_gate`` harness shape); the widget/recipe wiring is
pinned by source-grep guards + a CI-gated helper test.
"""
from __future__ import annotations

import re
import sys
import types as _types
from pathlib import Path

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

_LEAKED_PARENT_STUBS = {}
for _pkg in ("modules", "modules.tools", "modules.tools.execution"):
    if _pkg not in sys.modules:
        _stub = _types.ModuleType(_pkg)
        _stub.__path__ = [str(_ORCH / _pkg.replace(".", "/"))]
        sys.modules[_pkg] = _stub
        _LEAKED_PARENT_STUBS[_pkg] = _stub


def teardown_module(module):
    for _name, _stub in _LEAKED_PARENT_STUBS.items():
        if sys.modules.get(_name) is _stub:
            del sys.modules[_name]


from modules.policy import gate as gate_mod  # noqa: E402
from modules.policy import policy_document as pd  # noqa: E402
from modules.policy.budget import BudgetDecision  # noqa: E402
from modules.policy.gate import PolicyGate, ToolCall  # noqa: E402
from modules.policy.types import Decision  # noqa: E402


# ---------------------------------------------------------------------------
# The human-direct actor rule at the gate (pure)
# ---------------------------------------------------------------------------


def _gate(monkeypatch, *, budget_calls: list):
    g = PolicyGate(db="fake-db")
    monkeypatch.setattr(g, "_lookup_action", lambda name: None)
    monkeypatch.setattr(g, "_full_autonomy", lambda ws: False)
    monkeypatch.setattr(
        gate_mod._policy_doc, "load_policy_document",
        lambda db, ws: pd.PolicyDocument(pd.BALANCED, False, {}),
    )

    def _budget(db, ws, **kw):
        budget_calls.append(kw)
        return BudgetDecision(True, "within budget")

    monkeypatch.setattr(gate_mod._budget, "check_budget", _budget)
    return g


def test_widget_send_human_direct_not_ask_blocked(monkeypatch):
    """A user-direct external send is NOT ask-gated — the click is the
    approval (locked #6) — and budget admission still evaluated."""
    budget_calls: list = []
    g = _gate(monkeypatch, budget_calls=budget_calls)

    verdict = g.check(ToolCall(
        tool_name="GMAIL_SEND_EMAIL",
        parameters={"recipient_email": "x@y.z"},
        workspace_id="ws-1",
        caller_context={"actor_type": "user_direct", "user_id": "user_abc"},
        is_composio=True,
    ))
    assert verdict.decision is Decision.ALLOW
    assert "human-direct" in verdict.reason
    assert len(budget_calls) == 1  # budget still applies to user-direct calls


def test_agent_send_still_asks(monkeypatch):
    """The exemption is ACTOR-scoped, not lane-scoped: the same call without
    the marker (an agent) still routes external → ask under Balanced."""
    budget_calls: list = []
    g = _gate(monkeypatch, budget_calls=budget_calls)

    verdict = g.check(ToolCall(
        tool_name="GMAIL_SEND_EMAIL",
        parameters={"recipient_email": "x@y.z"},
        workspace_id="ws-1",
        caller_context={"user_id": "user_abc"},  # no user_direct marker
        is_composio=True,
    ))
    assert verdict.decision is Decision.ASK


def test_human_direct_does_not_bypass_budget_deny(monkeypatch):
    """Budget still binds for user-direct calls — the marker satisfies ONLY
    the ask gate."""
    g = PolicyGate(db="fake-db")
    monkeypatch.setattr(g, "_lookup_action", lambda name: None)
    monkeypatch.setattr(g, "_full_autonomy", lambda ws: False)
    monkeypatch.setattr(
        gate_mod._policy_doc, "load_policy_document",
        lambda db, ws: pd.PolicyDocument(pd.BALANCED, False, {}),
    )
    monkeypatch.setattr(
        gate_mod._budget, "check_budget",
        lambda db, ws, **kw: BudgetDecision(False, "over ceiling"),
    )

    verdict = g.check(ToolCall(
        tool_name="GMAIL_SEND_EMAIL",
        parameters={},
        workspace_id="ws-1",
        caller_context={"actor_type": "user_direct"},
        is_composio=True,
    ))
    assert verdict.decision is Decision.DENY
    assert verdict.error.code == "budget_exceeded"


# ---------------------------------------------------------------------------
# Source-grep guards — no raw execute_action side-doors in the closed lanes
# ---------------------------------------------------------------------------


def test_no_raw_execute_action_in_lanes():
    """No direct ``.execute_action(`` remains in api/widget_email.py or the
    recipe LLM-step lane (the spine is the only execution path).

    Note: ``get_tools_for_step`` (discovery) is not execution; the LinkedIn
    image workaround (``execute_linkedin_image_post``) is a named temporary
    upstream-bug bypass, tracked for removal when Composio fixes #3094/#3113.
    """
    widget_src = (_ORCH / "api" / "widget_email.py").read_text()
    assert not re.search(r"\.execute_action\(", widget_src), (
        "raw Composio client execution remains in api/widget_email.py"
    )
    assert "get_composio_client" not in widget_src

    recipe_src = (_ORCH / "api" / "recipe_executor.py").read_text()
    assert not re.search(r"tool_service\.execute_action\(", recipe_src), (
        "the playbook LLM-step still calls ComposioToolService.execute_action raw"
    )


def test_widget_email_routes_through_spine():
    """The widget helper dispatches composio_execute with the user_direct
    actor marker (locked #6) via the module-level execute_tool chokepoint."""
    widget_src = (_ORCH / "api" / "widget_email.py").read_text()
    assert '"composio_execute"' in widget_src
    assert '"actor_type": "user_direct"' in widget_src
    assert "from modules.tools.tool_router import execute_tool" in widget_src


def test_playbook_step_rides_spine_source():
    """The recipe per-action block dispatches through execute_and_format as
    the composio_execute meta-tool with the playbook identity threaded."""
    recipe_src = (_ORCH / "api" / "recipe_executor.py").read_text()
    assert 'tool_name="composio_execute"' in recipe_src
    assert '"playbook_execution_id": recipe_execution_id' in recipe_src


# ---------------------------------------------------------------------------
# Widget helper wiring (CI-gated import)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_widget_send_dispatches_spine_with_marker(monkeypatch):
    """_execute_email_action calls the spine with the meta-tool wrap + the
    human-direct marker + the workspace identity."""
    try:
        import api.widget_email as widget_mod
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"widget_email unavailable (CI-gated): {type(exc).__name__}")

    captured = {}

    async def _spine(tool_name, tool_args, agent_id=0, **kw):
        captured.update(
            tool_name=tool_name, tool_args=tool_args, agent_id=agent_id, **kw
        )
        return {"success": True, "data": {"id": "m-1"}, "error": None}

    monkeypatch.setattr("modules.tools.tool_router.execute_tool", _spine)

    from types import SimpleNamespace

    ctx = SimpleNamespace(
        workspace_id="ws-1",
        user=SimpleNamespace(id="user_abc"),
        auth_type="clerk",
    )
    result = await widget_mod._execute_email_action(
        ctx, "GMAIL_SEND_EMAIL", {"recipient_email": "x@y.z"}
    )
    assert result["success"] is True
    assert captured["tool_name"] == "composio_execute"
    assert captured["tool_args"]["action"] == "GMAIL_SEND_EMAIL"
    assert captured["caller_context"]["actor_type"] == "user_direct"
    assert captured["caller_context"]["user_id"] == "user_abc"
    assert captured["workspace_id"] == "ws-1"
