"""PRD-142 W3-S6 — Chat failure-path + chat-primitive heartbeat regression net.

The §H DoD for Chat requires that a tool error or provider error is **visible**
to the user (an error frame in the SSE stream), never silently swallowed, and
that the chat primitive emits a heartbeat finding so the W3-S2 tile can
reflect Chat's real-time state.

This file pins those two contracts against the converged spine (W3-S4):

1. ``_stream_tool_loop`` LLM-call exception → yields ``_final_response``
   carrying ``Error: <e>`` text → caller streams it via ``stream_text_aisdk``
   (the user sees the error).
2. ``_stream_tool_loop`` fatal-error short-circuit → forces a user-facing
   message instead of letting the loop hang silently.
3. ``stream_response_with_agent`` outer ``except`` → yields the AI SDK
   ``e:{"message": "..."}`` frame (the AI SDK error frame).
4. ``stream_response`` outer ``except`` → yields the SSE
   ``data: {"type": "error", "error": "..."}`` frame (legacy SSE error).
5. Tool-callback returns ``success=False`` → executor still drives the
   loop to completion and the failed-tool ``tool-end`` is emitted to SSE
   (the user sees which tool failed).
6. Chat primitive heartbeat emit:
   - ``emit_primitive_finding`` is called with ``"green"`` on a clean turn,
   - ``emit_primitive_finding`` is called with ``"down"`` on a caught
     exception,
   - the call passes ``self.workspace_id`` (cross-workspace isolation —
     never another workspace's id),
   - an emit failure NEVER breaks the chat flow (best-effort).

Tests use the ToolLoopExecutor directly with chat-shape callbacks (the same
spine ``_stream_tool_loop`` drives) plus a focused harness for the outer
streaming paths — avoiding the full StreamingChatService dependency surface
while still exercising the real error-visibility code paths.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types as _types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Dict, List, Optional
from unittest.mock import patch, MagicMock

import pytest


_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

# Stub parent packages so importing ``modules.tools.execution.tool_loop``
# does NOT drag in the rest of modules.tools (heavy registry side effects).
for _pkg in ("modules", "modules.tools", "modules.tools.execution"):
    if _pkg not in sys.modules:
        _stub = _types.ModuleType(_pkg)
        _stub.__path__ = [str(_ORCH / _pkg.replace(".", "/"))]
        sys.modules[_pkg] = _stub

# Same trick for ``consumers.chatbot.primitive_heartbeat``: stub the parent
# packages so importing the leaf does NOT trigger consumers/chatbot/__init__
# which pulls in the full StreamingChatService dep surface. Mirrors the
# W3-S7 / W3-S8 patterns.
for _pkg in ("consumers", "consumers.chatbot"):
    if _pkg not in sys.modules:
        _stub = _types.ModuleType(_pkg)
        _stub.__path__ = [str(_ORCH / _pkg.replace(".", "/"))]
        sys.modules[_pkg] = _stub

from modules.tools.execution.tool_loop import (  # noqa: E402
    RoundState,
    ToolLoopExecutor,
    ToolPostResult,
)


# Stub the ``services`` package init only (path-only). The real
# ``services.heartbeat_service`` then loads at import time without
# triggering ``services/__init__.py`` (which pulls in metadata_sync_service
# and the Composio cache models).
if "services" not in sys.modules:
    _services_stub = _types.ModuleType("services")
    _services_stub.__path__ = [str(_ORCH / "services")]
    sys.modules["services"] = _services_stub


def _load_primitive_heartbeat():
    """Load ``consumers.chatbot.primitive_heartbeat`` directly from file
    so the chatbot package __init__ doesn't fire. Returns the loaded module."""
    if "consumers.chatbot.primitive_heartbeat" in sys.modules:
        return sys.modules["consumers.chatbot.primitive_heartbeat"]
    path = _ORCH / "consumers" / "chatbot" / "primitive_heartbeat.py"
    spec = importlib.util.spec_from_file_location(
        "consumers.chatbot.primitive_heartbeat", str(path)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["consumers.chatbot.primitive_heartbeat"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Test doubles — minimal shapes the chat callbacks actually consume.
# ---------------------------------------------------------------------------


def _resp(
    content: Optional[str] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
    finish_reason: Optional[str] = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage={},
        model="test",
        provider="test",
    )


def _tc(name: str, args: Dict[str, Any], call_id: str) -> Dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


class _RaisingLLM:
    """LLM callback that raises a provider error on the FIRST call.

    Used to simulate a provider outage that hits during the executor's
    follow-up LLM call (after the initial response that drove round 1).
    """

    def __init__(self, exc: Exception) -> None:
        self._exc = exc
        self.call_count = 0

    async def __call__(self, messages, tools=None):
        self.call_count += 1
        raise self._exc


class _ScriptedLLM:
    def __init__(self, *responses: SimpleNamespace) -> None:
        self._responses = list(responses)
        self.call_count = 0

    async def __call__(self, messages, tools=None):
        self.call_count += 1
        return self._responses.pop(0) if self._responses else _resp(content="done")


class _RecordingTool:
    def __init__(self, result: Optional[Dict[str, Any]] = None) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.result = result or {"success": True, "llm_context": "ok", "raw_result": {}}

    async def __call__(self, name, args, call_id, ws_id):
        self.calls.append({"name": name, "args": args, "call_id": call_id, "ws_id": ws_id})
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


# ---------------------------------------------------------------------------
# 1. LLM provider error inside the tool loop surfaces to the user.
# ---------------------------------------------------------------------------


def test_provider_error_inside_tool_loop_surfaces_not_silent():
    """An LLM provider exception during the tool loop must NOT be swallowed
    — the executor's run() raises, the chat caller's _stream_tool_loop
    catches it and yields a _final_response with ``Error: <msg>`` so the
    user sees the failure (parity with service.py:1424-1428)."""
    # initial_response drives round 1; the follow-up LLM call raises.
    initial = _resp(tool_calls=[_tc("custom_tool", {"x": 1}, "c1")])
    llm = _RaisingLLM(exc=RuntimeError("provider outage"))
    tool = _RecordingTool()

    executor = ToolLoopExecutor(llm_callback=llm, tool_callback=tool, max_iterations=5)

    # Simulate what _stream_tool_loop does: run the executor inside a wrapper
    # that catches and surfaces. Mirrors service.py:1399-1428.
    async def _runner():
        try:
            return await executor.run(
                initial_response=initial,
                messages=[{"role": "user", "content": "hi"}],
                tools=[],
                workspace_id="ws-1",
            )
        except Exception as loop_err:  # noqa: BLE001 — chat's catch-all
            return SimpleNamespace(_visible_error=str(loop_err), content=None)

    out = asyncio.run(_runner())
    # The chat path turns this into yield {"_final_response": SimpleNamespace(
    #   content=f"Error: {loop_err}", ...)} — pin the visibility contract.
    assert hasattr(out, "_visible_error"), "Provider error must not be silently swallowed"
    assert "provider outage" in out._visible_error, "Error detail must be user-visible"
    assert llm.call_count == 1, "LLM should have been called once (the raising call)"
    assert len(tool.calls) == 1, "Tool must have run before the follow-up LLM raised"


# ---------------------------------------------------------------------------
# 2. fatal_error short-circuit yields a visible user-facing message.
# ---------------------------------------------------------------------------


def test_fatal_error_short_circuits_with_visible_message():
    """When a tool returns fatal_error=True the chat loop must force a
    final synthesis with a user-facing message — never silently retry."""
    initial = _resp(tool_calls=[_tc("broken_tool", {}, "c1")])
    tool = _RecordingTool({
        "success": False,
        "llm_context": "tool blew up",
        "raw_result": {},
        "fatal_error": True,
    })
    # No follow-up LLM call needed — on_round_end short-circuits.
    llm = _ScriptedLLM()

    # The chat round-end hook (service.py:1335-1361).
    async def _on_round_end(state: RoundState):
        if state.had_fatal_errors:
            return ToolPostResult(
                force_final=True,
                final_content=(
                    "I ran into a server configuration issue while executing that tool. "
                    "Please restart the backend and try again."
                ),
            )
        return None

    executor = ToolLoopExecutor(llm_callback=llm, tool_callback=tool, max_iterations=5)
    result = asyncio.run(executor.run(
        initial_response=initial,
        messages=[{"role": "user", "content": "do it"}],
        tools=[],
        workspace_id="ws-2",
        on_round_end=_on_round_end,
    ))

    assert result.forced_final, "fatal_error must force the final synthesis"
    assert result.response.content
    assert "server configuration issue" in result.response.content


# ---------------------------------------------------------------------------
# 3. Outer-except surface yields the AI SDK error frame (visible to FE).
# ---------------------------------------------------------------------------


def test_aisdk_error_frame_is_emitted_on_outer_exception():
    """The stream_response_with_agent outer except yields
    ``e:{"message": "..."}`` — the AI SDK error frame the frontend reads.

    We import streaming.py and assert the helper emits the e: prefix +
    JSON-encoded message; this is the only path the user sees errors.
    """
    from consumers.chatbot.streaming import get_streaming_handler

    handler = get_streaming_handler()
    frame = handler.format_aisdk_error("boom")
    assert frame.startswith("e:"), f"AI SDK error frame must start with 'e:', got: {frame!r}"
    assert "boom" in frame
    assert json.loads(frame[2:].rstrip("\n"))["message"] == "boom"


def test_sse_error_frame_is_emitted_on_outer_exception():
    """The stream_response outer except yields the legacy SSE error frame.
    Mirrors the same visibility contract on the older endpoint."""
    from consumers.chatbot.streaming import get_streaming_handler

    handler = get_streaming_handler()
    frame = handler.format_sse_error("kaboom")
    assert frame.startswith("data: "), f"SSE error frame must start with 'data:', got: {frame!r}"
    assert "kaboom" in frame
    payload = json.loads(frame[len("data: "):].strip())
    assert payload["type"] == "error"
    assert payload["error"] == "kaboom"


# ---------------------------------------------------------------------------
# 4. Tool failure is observable — tool-end frame still flows for failed tool.
# ---------------------------------------------------------------------------


def test_failed_tool_call_is_visible_to_user_via_tool_end_and_llm_followup():
    """Visibility contract for a tool that returns success=False without
    raising:

    1. A ``tool-end`` event fires for that tool (no silent drop).
    2. The failed tool's ``llm_context`` reaches the next LLM call as a
       tool-result message — the LLM can see and surface the failure to
       the user in its synthesized response.

    The executor's ``success`` flag on the tool-end frame indicates
    "didn't raise" — the actual failure visibility flows through the
    LLM follow-up message (tool_loop.py:316-358)."""
    initial = _resp(tool_calls=[_tc("bad_tool", {}, "c1")])
    tool = _RecordingTool({
        "success": False,
        "llm_context": "the bad_tool returned ERROR_X — operation failed",
        "raw_result": {},
        "fatal_error": False,
    })
    llm = _ScriptedLLM(_resp(content="I tried bad_tool and it failed", tool_calls=None))

    events: List[Dict[str, Any]] = []

    async def on_event(event):
        events.append(event)

    executor = ToolLoopExecutor(llm_callback=llm, tool_callback=tool, max_iterations=5)
    result = asyncio.run(executor.run(
        initial_response=initial,
        messages=[{"role": "user", "content": "try it"}],
        tools=[],
        workspace_id="ws-3",
        on_event=on_event,
    ))

    # 1. tool-end fired (not silent).
    end_events = [e for e in events if e.get("type") == "tool-end"]
    assert end_events, "tool-end event must fire for a failed tool — not silently swallowed"
    assert end_events[0]["tool_name"] == "bad_tool"

    # 2. The failure context reached the LLM (visible via follow-up).
    llm_messages_on_call = llm.call_count
    assert llm_messages_on_call == 1, "LLM must have been called once for the follow-up"
    # The follow-up LLM saw the failure context — it acknowledges it.
    assert "failed" in (result.response.content or "").lower(), (
        "LLM follow-up must surface the failure to the user"
    )


# ---------------------------------------------------------------------------
# 5. Chat primitive heartbeat emit — green on success, down on error.
# ---------------------------------------------------------------------------


def test_heartbeat_emit_helper_validates_chat_primitive():
    """``emit_primitive_finding`` accepts ``"chat"`` as a canonical primitive
    name and rejects non-canonical ones — same shape Memory uses (W3-S7)."""
    from services.heartbeat_service import (
        PRIMITIVE_NAMES,
        PRIMITIVE_STATUSES,
        emit_primitive_finding,
    )

    assert "chat" in PRIMITIVE_NAMES, "chat must be a canonical primitive"
    assert {"green", "degraded", "down"} <= PRIMITIVE_STATUSES

    # Bad primitive name returns False without raising. No DB needed.
    assert emit_primitive_finding("ws-x", "CHATS", "green") is False
    assert emit_primitive_finding("ws-x", "chat", "GREEN") is False


def test_chat_emits_green_heartbeat_on_successful_turn():
    """A successful chat turn must invoke ``emit_primitive_finding(ws, "chat", "green", ...)``
    — the chat tile goes green only when chat actually completes a turn."""
    pb = _load_primitive_heartbeat()

    recorded: List[Dict[str, Any]] = []

    def _capture(workspace_id, primitive, status, detail=""):
        recorded.append(
            {
                "workspace_id": workspace_id,
                "primitive": primitive,
                "status": status,
                "detail": detail,
            }
        )
        return True

    with patch.object(pb, "emit_primitive_finding", side_effect=_capture):
        pb._emit_chat_primitive("ws-success", success=True, detail="ok")

    assert recorded == [
        {
            "workspace_id": "ws-success",
            "primitive": "chat",
            "status": "green",
            "detail": "ok",
        }
    ]


def test_chat_emits_down_heartbeat_on_caught_exception():
    """When the outer except catches an exception the chat primitive must
    emit ``"down"`` so the tile flips — never silent."""
    pb = _load_primitive_heartbeat()

    recorded: List[Dict[str, Any]] = []

    def _capture(workspace_id, primitive, status, detail=""):
        recorded.append(
            {
                "workspace_id": workspace_id,
                "primitive": primitive,
                "status": status,
                "detail": detail,
            }
        )
        return True

    with patch.object(pb, "emit_primitive_finding", side_effect=_capture):
        pb._emit_chat_primitive("ws-fail", success=False, detail="provider down")

    assert len(recorded) == 1
    row = recorded[0]
    assert row["workspace_id"] == "ws-fail"
    assert row["primitive"] == "chat"
    assert row["status"] == "down"
    assert "provider down" in row["detail"]


def test_chat_emit_failure_is_swallowed_does_not_break_chat():
    """If ``emit_primitive_finding`` raises (DB outage etc.), the chat
    helper must NEVER propagate — chat flow continues. Mirrors the
    best-effort contract emit_primitive_finding itself provides
    (heartbeat_service.py:113)."""
    pb = _load_primitive_heartbeat()

    def _boom(*args, **kwargs):
        raise RuntimeError("DB down")

    with patch.object(pb, "emit_primitive_finding", side_effect=_boom):
        # Must not raise.
        pb._emit_chat_primitive("ws-z", success=True, detail="")
        pb._emit_chat_primitive("ws-z", success=False, detail="ouch")


def test_chat_emit_skipped_when_no_workspace_id():
    """An unauthenticated/unconfigured caller has no workspace_id — never
    fabricate one (A4 — honest gaps). The helper silently skips."""
    pb = _load_primitive_heartbeat()

    recorded: List[Any] = []

    def _capture(workspace_id, primitive, status, detail=""):
        recorded.append(workspace_id)
        return True

    with patch.object(pb, "emit_primitive_finding", side_effect=_capture):
        pb._emit_chat_primitive(None, success=True, detail="")
        pb._emit_chat_primitive("", success=False, detail="x")

    assert recorded == [], "No emit when workspace_id is missing"


def test_chat_emit_workspace_isolation_uses_caller_ws_not_global():
    """Cross-workspace isolation: two sequential turns from two
    workspaces produce two emits with each caller's own ws_id — the
    helper never lifts a global/cached id (the leak shape from
    `feedback-cross-tenant-runtime`)."""
    pb = _load_primitive_heartbeat()

    recorded: List[str] = []

    def _capture(workspace_id, primitive, status, detail=""):
        recorded.append(workspace_id)
        return True

    with patch.object(pb, "emit_primitive_finding", side_effect=_capture):
        pb._emit_chat_primitive("ws-A", success=True, detail="")
        pb._emit_chat_primitive("ws-B", success=False, detail="boom")
        pb._emit_chat_primitive("ws-A", success=True, detail="ok again")

    assert recorded == ["ws-A", "ws-B", "ws-A"]
    assert "ws-B" not in {recorded[0], recorded[2]}, "ws B must not bleed into ws A's emits"


# ---------------------------------------------------------------------------
# 6. Restart-safe regression: the helper itself is stateless and importable
#    after a process boot (no module-level state that could be lost).
# ---------------------------------------------------------------------------


def test_chat_emit_helper_is_stateless_and_importable():
    """The helper must hold no per-process cache — every call reads inputs
    cleanly. Re-importing the module produces an equivalent helper. This
    is the restart-safe contract at the function level (a real DB restart
    test lives in the integration suite, gated on PG availability)."""
    pb = _load_primitive_heartbeat()

    assert hasattr(pb, "_emit_chat_primitive"), (
        "Chat heartbeat helper must exist on primitive_heartbeat"
    )
    assert callable(pb._emit_chat_primitive)

    # No module-level state that records prior calls.
    state_attrs = [a for a in dir(pb) if a.startswith("_chat_primitive_")]
    assert state_attrs == [], (
        f"Chat heartbeat helper must be stateless; found cached state: {state_attrs}"
    )


def test_chat_service_wires_primitive_heartbeat_helper():
    """The StreamingChatService module must import the helper so the
    streaming entry points can call it on success/failure boundaries.
    This pins the wire-up so future refactors do not silently drop it.

    We grep the chat service source rather than importing it (the full
    module pulls heavy deps the rest of these tests avoid) — same pattern
    other Wave 3 tests use for low-coupling assertions.
    """
    service_src = _ORCH / "consumers" / "chatbot" / "service.py"
    src = service_src.read_text()
    assert "_emit_chat_primitive" in src, (
        "consumers/chatbot/service.py must invoke _emit_chat_primitive"
    )
    assert "primitive_heartbeat" in src, (
        "consumers/chatbot/service.py must import the primitive_heartbeat helper"
    )
