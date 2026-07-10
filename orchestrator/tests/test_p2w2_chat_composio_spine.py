"""PRD-192 S4 (P2-11) — the chat per-action Composio shortcut rides the spine.

The legacy `_tool_callback` shortcut called ``ComposioToolService.execute_action``
raw and returned ``{"success": True, ...}`` UNCONDITIONALLY — no policy gate, no
telemetry, no outcome capture, no scope enforcement, and a dishonest envelope
(tool-runtime C.3a). This suite pins the rewire:

- per-action Composio calls dispatch through ``execute_and_format`` →
  ``UnifiedToolExecutor`` as the ``composio_execute`` meta-tool (the per-action
  name travels in ``action`` — the tracker, the routing graph, and the policy
  gate's effective-name resolution all see ``GMAIL_SEND_EMAIL``);
- the policy gate governs the lane (a deny blocks BEFORE the Composio dispatch);
- failures surface honestly (``success: False`` reaches the loop);
- the raw shortcut (``_execute_composio_action`` / in-package
  ``.execute_action(`` calls) is DELETED — source-grep guard;
- the error-recovery hook reads the same envelope for both call shapes.

NOTE (PRD drift, recorded): the PRD expected per-action names to route via the
executor's ``composio_actions`` dict — grep proves that dict is populated
nowhere in the tree, so the honest spine route is the registry-backed
``composio_execute`` meta-tool, which resolves the same per-action name.
"""
from __future__ import annotations

import re
import sys
import types as _types
from pathlib import Path
from unittest.mock import MagicMock

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


_SERVICE_SRC = (_ORCH / "consumers" / "chatbot" / "service.py").read_text()


# ---------------------------------------------------------------------------
# Source-grep guards — the side-door is deleted, the spine dispatch exists
# ---------------------------------------------------------------------------


def test_no_direct_composio_shortcut_remains():
    """No raw ComposioToolService.execute_action call in the chat package.

    (``get_tools_for_step`` — discovery, not execution — is the one legitimate
    ComposioToolService use left.)
    """
    chat_dir = _ORCH / "consumers" / "chatbot"
    offenders = []
    for path in chat_dir.rglob("*.py"):
        text = path.read_text(errors="ignore")
        if re.search(r"\.execute_action\(", text):
            offenders.append(str(path.relative_to(_ORCH)))
    assert offenders == [], f"raw Composio execution side-door(s) remain: {offenders}"

    # The deleted method must not resurface under its old name (no shim, §5).
    assert "async def _execute_composio_action" not in _SERVICE_SRC


def test_chat_composio_dispatches_meta_tool():
    """The callback rewrites per-action names onto the composio_execute spine
    (the per-action name rides in `action`), instead of short-circuiting."""
    assert 'dispatch_name = "composio_execute"' in _SERVICE_SRC
    assert '"action": name,' in _SERVICE_SRC
    # The dishonest unconditional-success envelope is gone.
    assert '{"success": True, "llm_context": llm_context, "raw_result": {}}' not in _SERVICE_SRC


def test_recovery_hook_reads_meta_shape():
    """_on_tool_result maps a per-action name to the meta-tool for recovery, so
    invalid-parameter recovery keeps firing on the rewired lane."""
    assert '"composio_execute" if _is_chat_composio_action(name) else name' in _SERVICE_SRC


# ---------------------------------------------------------------------------
# The spine governs what chat now dispatches (executor-level, CI-gated)
# ---------------------------------------------------------------------------

try:
    import modules.tools.execution.unified_executor as unified_executor
    _EXECUTOR_AVAILABLE = True
    _EXECUTOR_SKIP = ""
except Exception as _exc:  # pragma: no cover
    _EXECUTOR_AVAILABLE = False
    _EXECUTOR_SKIP = f"UnifiedToolExecutor unavailable (CI-gated): {type(_exc).__name__}"


class _FakeSession:
    def query(self, *a, **k):
        raise AssertionError("gate is monkeypatched; DB must not be queried")


@pytest.mark.skipif(not _EXECUTOR_AVAILABLE, reason=_EXECUTOR_SKIP or "executor unavailable")
@pytest.mark.asyncio
async def test_chat_composio_rides_spine(monkeypatch):
    """The meta-tool dispatch chat now uses is consulted at the policy gate
    with the PER-ACTION effective name, and a deny blocks BEFORE the Composio
    executor is reached."""
    from modules.policy.types import PolicyError, Verdict

    monkeypatch.setattr("modules.policy.policy_plane_mode", lambda: "on", raising=False)
    ex = unified_executor.UnifiedToolExecutor(db_session=_FakeSession())
    monkeypatch.setattr(ex, "_policy_action_def", lambda name: None)

    seen = {}

    def _check(self, call):
        seen["tool_name"] = call.tool_name
        seen["is_composio"] = call.is_composio
        return Verdict.deny(PolicyError(
            code="approval_required",
            message_for_model="Needs approval; NOT executed.",
            retryable=True,
        ))

    monkeypatch.setattr("modules.policy.PolicyGate.check", _check, raising=False)
    monkeypatch.setattr(ex, "_fire_policy_bus", lambda *a, **k: None)

    async def _tripwire(*a, **k):
        raise AssertionError("denied Composio call reached dispatch — must not execute")

    monkeypatch.setattr(ex, "_execute_composio_execute", _tripwire)

    result = await ex.execute_tool(
        "composio_execute",
        {"action": "GMAIL_SEND_EMAIL", "params": {"recipient_email": "x@y.z"}},
        agent_id=1,
        workspace_id="ws-1",
    )
    assert seen["tool_name"] == "GMAIL_SEND_EMAIL"  # gate judges the ACTION
    assert seen["is_composio"] is True
    assert result["success"] is False
    assert result["policy_error"]["code"] == "approval_required"


# ---------------------------------------------------------------------------
# Honest failure envelope through the router (CI-gated)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_chat_composio_failure_is_honest(monkeypatch):
    """An executor failure surfaces success:False + error_type to the loop —
    previously this lane returned success:True structurally."""
    try:
        import modules.tools.tool_router as tool_router_mod
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"tool_router unavailable (CI-gated): {type(exc).__name__}")

    async def _failing_execute_tool(tool_name, tool_args, agent_id=1, **kw):
        return {
            "success": False,
            "error": "'GMAIL' is not connected for this workspace.",
            "error_type": "composio_not_connected",
            "tool": tool_name,
        }

    monkeypatch.setattr(tool_router_mod, "execute_tool", _failing_execute_tool)

    router = tool_router_mod.ToolRouter()
    result = await router.execute_and_format(
        tool_name="composio_execute",
        tool_args={"action": "GMAIL_SEND_EMAIL", "params": {}},
        agent_id=1,
        workspace_id=None,
    )
    assert result["success"] is False
    assert result["error_type"] == "composio_not_connected"
    assert "not connected" in result["raw_result"]["error"]


# ---------------------------------------------------------------------------
# Error recovery still fires off the honest envelope (CI-gated)
# ---------------------------------------------------------------------------


def test_composio_error_recovery_still_fires():
    try:
        from consumers.chatbot.service import ChatbotService
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"chat service unavailable (CI-gated): {type(exc).__name__}")

    fake_self = MagicMock()
    result = {
        "success": False,
        "error_type": "invalid_parameters",
        "raw_result": {"error": "missing recipient_email"},
    }
    recovery = ChatbotService._handle_composio_error_recovery(
        fake_self,
        result,
        "composio_execute",  # the recovery name the rewired hook passes
        llm_messages=[],
        agent_runtime=MagicMock(),
        action_not_mapped_retry_budget=1,
        invalid_parameters_retry_budget=1,
        followup_system_messages=[],
    )
    assert recovery == {"invalid_parameters_retry_budget": 0}
    fake_self._build_composio_param_recovery.assert_called_once()
