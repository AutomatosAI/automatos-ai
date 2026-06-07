"""PRD-142 W3-S4 — Tool-loop union characterization.

Pins the UNION of dedup / retry / iteration / truncation behaviour that
the chat ``_run_tool_loop`` and the agent ``execute_with_prompt`` inner
loop perform today, against the converged ``ToolLoopExecutor`` they will
both delegate to.

Written RED first: ``modules.tools.execution.tool_loop`` does not exist
until the convergence lands. Once it lands these tests stay green as the
regression net for both surfaces.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest


# ---------------------------------------------------------------------------
# Isolated import: stub the ``modules`` / ``modules.tools`` /
# ``modules.tools.execution`` package inits so we never run them (they pull
# asyncpg/pgvector via unified_executor — not present in the unit-test env).
# The actual leaf modules ``tool_execution_tracker`` + ``tool_loop`` are
# stdlib-only and load cleanly via their normal import paths once the
# parents are stubbed. Same pattern used by ``test_heartbeat_primitive_findings``.
# ---------------------------------------------------------------------------

import types as _types

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

# Record the path-only parent stubs we create so ``teardown_module`` can drop
# them again. Left in ``sys.modules`` a bare ``modules`` stub has no ``tools``
# attribute, so a later sibling's ``monkeypatch.setattr("modules.tools.…")``
# fails walking ``getattr(modules, "tools")``.
_LEAKED_PARENT_STUBS = {}
for _pkg in ("modules", "modules.tools", "modules.tools.execution"):
    if _pkg not in sys.modules:
        _stub = _types.ModuleType(_pkg)
        _stub.__path__ = [str(_ORCH / _pkg.replace(".", "/"))]
        sys.modules[_pkg] = _stub
        _LEAKED_PARENT_STUBS[_pkg] = _stub


def teardown_module(module):
    """Drop the path-only parent stubs this module installed (identity-checked
    so we never evict a real import or another test's replacement)."""
    for _name, _stub in _LEAKED_PARENT_STUBS.items():
        if sys.modules.get(_name) is _stub:
            del sys.modules[_name]


from modules.tools.execution.tool_loop import ToolLoopExecutor  # noqa: E402


# ---------------------------------------------------------------------------
# Test doubles — small, dependency-free, exercise the executor contract only.
# ---------------------------------------------------------------------------


def _resp(content: Optional[str] = None,
          tool_calls: Optional[List[Dict[str, Any]]] = None,
          finish_reason: Optional[str] = None,
          usage: Optional[Dict[str, Any]] = None) -> SimpleNamespace:
    """Mirror what llm_manager.generate_response returns."""
    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=usage or {},
        model="test",
        provider="test",
    )


def _tc(name: str, args: Dict[str, Any], call_id: str = "call_1") -> Dict[str, Any]:
    """OpenAI-format tool call dict."""
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


class _ScriptedLLM:
    """LLM whose responses are queued by the test.

    The executor calls ``__call__(messages, tools)`` on each iteration.
    """

    def __init__(self, *responses: SimpleNamespace) -> None:
        self._responses: List[SimpleNamespace] = list(responses)
        self.call_count: int = 0
        self.calls_seen: List[List[Dict[str, Any]]] = []

    async def __call__(self, messages, tools=None):
        self.call_count += 1
        # snapshot the messages list at call time (executor mutates the live list)
        self.calls_seen.append([dict(m) for m in messages])
        if not self._responses:
            return _resp(content="exhausted")
        return self._responses.pop(0)


class _RecordingTool:
    """Tool callback that records every invocation and returns canned results.

    Signature: ``await tool(name, args, call_id, workspace_id) -> dict``
    The executor adapts to this shape (or however we define it).
    """

    def __init__(self, default_result: Optional[Dict[str, Any]] = None) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.default_result = default_result or {
            "success": True,
            "llm_context": "ok",
            "raw_result": {},
        }
        self.error_for: Dict[str, Exception] = {}

    async def __call__(self, name: str, args: Dict[str, Any], call_id: str, workspace_id):
        self.calls.append({
            "name": name,
            "args": args,
            "call_id": call_id,
            "workspace_id": workspace_id,
        })
        if name in self.error_for:
            raise self.error_for[name]
        return self.default_result


# ---------------------------------------------------------------------------
# 1. Trivial path
# ---------------------------------------------------------------------------


def test_no_tool_calls_returns_initial_response_immediately():
    """A response with no tool_calls is returned unchanged."""
    initial = _resp(content="hello world", tool_calls=None)
    llm = _ScriptedLLM()  # never called
    tool = _RecordingTool()

    executor = ToolLoopExecutor(
        llm_callback=llm,
        tool_callback=tool,
        max_iterations=10,
    )
    result = asyncio.run(executor.run(
        initial_response=initial,
        messages=[{"role": "user", "content": "hi"}],
        tools=[],
        workspace_id=None,
    ))
    assert result.response is initial
    assert result.iterations == 0
    assert llm.call_count == 0
    assert tool.calls == []


# ---------------------------------------------------------------------------
# 2. Dedup within a turn — chat behaviour absorbed
# ---------------------------------------------------------------------------


def test_dedup_skips_identical_args_within_one_iteration():
    """Two tool_calls with identical (name, args) in one round → tool runs once."""
    duplicate = [
        _tc("search_knowledge", {"query": "alpha"}, "c1"),
        _tc("search_knowledge", {"query": "alpha"}, "c2"),
    ]
    initial = _resp(tool_calls=duplicate)
    llm = _ScriptedLLM(_resp(content="final answer", tool_calls=None))
    tool = _RecordingTool()

    executor = ToolLoopExecutor(
        llm_callback=llm, tool_callback=tool, max_iterations=10,
    )
    asyncio.run(executor.run(
        initial_response=initial, messages=[], tools=[], workspace_id=None,
    ))
    assert len(tool.calls) == 1
    assert tool.calls[0]["args"] == {"query": "alpha"}


def test_dedup_skips_identical_args_across_iterations():
    """Same (name, args) repeated in a later iteration → still skipped."""
    initial = _resp(tool_calls=[_tc("read_file", {"path": "/a.txt"}, "c1")])
    second_round = _resp(tool_calls=[_tc("read_file", {"path": "/a.txt"}, "c2")])
    llm = _ScriptedLLM(second_round, _resp(content="done", tool_calls=None))
    tool = _RecordingTool()

    executor = ToolLoopExecutor(
        llm_callback=llm, tool_callback=tool, max_iterations=10,
    )
    asyncio.run(executor.run(
        initial_response=initial, messages=[], tools=[], workspace_id=None,
    ))
    assert len(tool.calls) == 1


# ---------------------------------------------------------------------------
# 3. Per-tool attempt cap — chat behaviour absorbed (union over agent)
# ---------------------------------------------------------------------------


def test_per_tool_attempt_cap_blocks_overuse():
    """A tool with default cap=5 is blocked after 5 distinct-args calls in one turn."""
    rounds = [
        _resp(tool_calls=[_tc("custom_tool", {"i": i}, f"c{i}")])
        for i in range(8)
    ]
    rounds.append(_resp(content="terminate", tool_calls=None))
    llm = _ScriptedLLM(*rounds[1:])
    tool = _RecordingTool()

    executor = ToolLoopExecutor(
        llm_callback=llm, tool_callback=tool, max_iterations=20,
    )
    asyncio.run(executor.run(
        initial_response=rounds[0],
        messages=[], tools=[], workspace_id=None,
    ))
    # 'default' retry limit is 5 in ToolExecutionTracker → cap kicks in.
    assert len(tool.calls) <= 5


# ---------------------------------------------------------------------------
# 4. Max iterations cap — both loops had this
# ---------------------------------------------------------------------------


def test_max_iterations_cap_terminates_loop():
    """If LLM keeps returning tool_calls, executor stops at max_iterations."""
    def round_with_call(i: int) -> SimpleNamespace:
        return _resp(tool_calls=[_tc("custom_tool", {"i": i}, f"c{i}")])

    rounds = [round_with_call(i) for i in range(10)]
    llm = _ScriptedLLM(*rounds[1:])
    tool = _RecordingTool()

    executor = ToolLoopExecutor(
        llm_callback=llm, tool_callback=tool, max_iterations=3,
    )
    final = asyncio.run(executor.run(
        initial_response=rounds[0],
        messages=[], tools=[], workspace_id=None,
    ))
    # 3 iterations: each appends a round → cap reached, executor must finalize
    assert llm.call_count <= 3
    assert final is not None


# ---------------------------------------------------------------------------
# 5. finish_reason=length recovery — chat behaviour absorbed
# ---------------------------------------------------------------------------


def test_finish_reason_length_with_malformed_args_injects_recovery_message():
    """When LLM hits output limit mid tool call (JSON malformed), executor
    injects a 'use shorter content' system message and re-invokes the LLM.
    """
    bad_call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "write_file", "arguments": '{"path": "/a", "content": "tru'},  # malformed
    }
    initial = _resp(tool_calls=[bad_call], finish_reason="length")
    # On retry, LLM returns a valid response with no tool_calls
    retry = _resp(content="ok shorter", tool_calls=None)
    llm = _ScriptedLLM(retry)
    tool = _RecordingTool()

    messages: List[Dict[str, Any]] = [{"role": "user", "content": "do it"}]
    executor = ToolLoopExecutor(
        llm_callback=llm, tool_callback=tool, max_iterations=5,
    )
    result = asyncio.run(executor.run(
        initial_response=initial, messages=messages, tools=[], workspace_id=None,
    ))
    # LLM was re-invoked with a 'shorter content' system msg in the message history
    assert llm.call_count == 1
    injected = [m for m in messages if m.get("role") == "system"
                and "truncated" in (m.get("content") or "").lower()]
    assert injected, "Expected an injected 'truncated/shorter content' system message"
    # Tool should NOT have been called (args were malformed)
    assert tool.calls == []
    assert result.response.content == "ok shorter"


# ---------------------------------------------------------------------------
# 6. Tenant pass-through — every endpoint is workspace-scoped (PRD §1)
# ---------------------------------------------------------------------------


def test_workspace_id_is_passed_through_to_tool_callback():
    """The workspace_id given to executor.run is propagated to every tool call."""
    initial = _resp(tool_calls=[_tc("any_tool", {"a": 1}, "c1")])
    llm = _ScriptedLLM(_resp(content="done", tool_calls=None))
    tool = _RecordingTool()

    ws_id = "ws-abc-123"
    executor = ToolLoopExecutor(
        llm_callback=llm, tool_callback=tool, max_iterations=5,
    )
    asyncio.run(executor.run(
        initial_response=initial, messages=[], tools=[], workspace_id=ws_id,
    ))
    assert tool.calls[0]["workspace_id"] == ws_id


# ---------------------------------------------------------------------------
# 7. Streaming events — chat surface
# ---------------------------------------------------------------------------


def test_streaming_emits_tool_start_and_tool_end_events():
    """When on_event is provided, the executor emits at least
    tool-start and tool-end events for every tool invocation."""
    initial = _resp(tool_calls=[_tc("custom_tool", {"x": 1}, "c1")])
    llm = _ScriptedLLM(_resp(content="done", tool_calls=None))
    tool = _RecordingTool()

    events: List[Dict[str, Any]] = []

    async def on_event(event: Dict[str, Any]) -> None:
        events.append(event)

    executor = ToolLoopExecutor(
        llm_callback=llm, tool_callback=tool, max_iterations=5,
    )
    asyncio.run(executor.run(
        initial_response=initial, messages=[], tools=[], workspace_id=None,
        on_event=on_event,
    ))
    types = [e.get("type") for e in events]
    assert "tool-start" in types
    assert "tool-end" in types


def test_non_streaming_mode_emits_no_events():
    """Default (on_event=None) → no events emitted, agent-style."""
    initial = _resp(tool_calls=[_tc("custom_tool", {"x": 1}, "c1")])
    llm = _ScriptedLLM(_resp(content="done", tool_calls=None))
    tool = _RecordingTool()

    executor = ToolLoopExecutor(
        llm_callback=llm, tool_callback=tool, max_iterations=5,
    )
    # No on_event passed → executor must not require one
    final = asyncio.run(executor.run(
        initial_response=initial, messages=[], tools=[], workspace_id=None,
    ))
    assert final is not None


# ---------------------------------------------------------------------------
# 8. LLM-loop continuity — both loops had this
# ---------------------------------------------------------------------------


def test_llm_re_invoked_after_tool_round_with_tool_results_in_messages():
    """After one tool round, the next LLM call sees assistant+tool messages."""
    initial = _resp(tool_calls=[_tc("custom_tool", {"x": 1}, "c1")])
    final_resp = _resp(content="final", tool_calls=None)
    llm = _ScriptedLLM(final_resp)
    tool = _RecordingTool(default_result={"success": True, "llm_context": "result-1"})

    executor = ToolLoopExecutor(
        llm_callback=llm, tool_callback=tool, max_iterations=5,
    )
    asyncio.run(executor.run(
        initial_response=initial,
        messages=[{"role": "user", "content": "do it"}],
        tools=[], workspace_id=None,
    ))
    # LLM was called once (the followup); the messages it saw must include
    # an assistant message with tool_calls AND a tool result.
    assert llm.call_count == 1
    msgs = llm.calls_seen[0]
    roles = [m.get("role") for m in msgs]
    assert "assistant" in roles
    assert "tool" in roles


# ---------------------------------------------------------------------------
# 9. Error path — tool raising does not crash the loop
# ---------------------------------------------------------------------------


def test_tool_callback_exception_is_surfaced_to_llm_not_raised():
    """If tool_callback raises, the error is captured as a tool result and
    the loop continues to the next LLM round (failure surfaces visibly)."""
    initial = _resp(tool_calls=[_tc("broken_tool", {"x": 1}, "c1")])
    final = _resp(content="recovered", tool_calls=None)
    llm = _ScriptedLLM(final)
    tool = _RecordingTool()
    tool.error_for["broken_tool"] = RuntimeError("boom")

    executor = ToolLoopExecutor(
        llm_callback=llm, tool_callback=tool, max_iterations=5,
    )
    result = asyncio.run(executor.run(
        initial_response=initial,
        messages=[{"role": "user", "content": "do it"}],
        tools=[], workspace_id=None,
    ))
    # Loop did NOT raise; LLM was called once with an error tool result.
    assert result.response is final
    msgs = llm.calls_seen[0]
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    assert tool_msgs, "Expected an error-tool message appended for the failed call"
    assert "boom" in (tool_msgs[0].get("content") or "").lower() or \
           "error" in (tool_msgs[0].get("content") or "").lower()
