"""PRD-142 W3-S4 — Converged tool-loop regression net.

Proves the SAME :class:`ToolLoopExecutor` handles both shapes the legacy
chat and agent loops needed:

1. Chat shape: streaming on, ``on_round_end`` returns ``force_final`` on
   dedup/fatal-error — exactly the chat loop's old behaviour.
2. Agent shape: streaming off, no per-round force-synth — exactly what
   ``execute_with_prompt`` did internally.

Plus a cross-tenant pass-through test (every endpoint stays workspace-scoped).
"""
from __future__ import annotations

import asyncio
import json
import sys
import types as _types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional


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


from modules.tools.execution.tool_loop import (  # noqa: E402
    RoundState,
    ToolLoopExecutor,
    ToolLoopResult,
    ToolPostResult,
)


# ---------------------------------------------------------------------------
# Reusable test doubles (same shape as characterization tests).
# ---------------------------------------------------------------------------


def _resp(content: Optional[str] = None,
          tool_calls: Optional[List[Dict[str, Any]]] = None,
          finish_reason: Optional[str] = None) -> SimpleNamespace:
    return SimpleNamespace(
        content=content, tool_calls=tool_calls,
        finish_reason=finish_reason, usage={},
        model="test", provider="test",
    )


def _tc(name: str, args: Dict[str, Any], call_id: str) -> Dict[str, Any]:
    return {
        "id": call_id, "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


class _ScriptedLLM:
    def __init__(self, *responses: SimpleNamespace) -> None:
        self._responses = list(responses)
        self.call_count = 0
        self.calls_seen: List[List[Dict[str, Any]]] = []

    async def __call__(self, messages, tools=None):
        self.call_count += 1
        self.calls_seen.append([dict(m) for m in messages])
        return self._responses.pop(0) if self._responses else _resp(content="exhausted")


class _RecordingTool:
    def __init__(self, result: Optional[Dict[str, Any]] = None) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.result = result or {"success": True, "llm_context": "ok", "raw_result": {}}

    async def __call__(self, name, args, call_id, ws_id):
        self.calls.append({"name": name, "args": args, "call_id": call_id, "ws_id": ws_id})
        return self.result


# ---------------------------------------------------------------------------
# 1. Chat shape — streaming + force-synth on dedup
# ---------------------------------------------------------------------------


def test_chat_shape_streaming_and_force_synth_on_dedup():
    """Two duplicate tool_calls in one round → executor emits tool events to
    the SSE callback, marks ``had_skips=True`` in the round state, and the
    caller's on_round_end forces final synthesis (chat parity)."""
    duplicate = [
        _tc("search_knowledge", {"query": "alpha"}, "c1"),
        _tc("search_knowledge", {"query": "alpha"}, "c2"),
    ]
    initial = _resp(tool_calls=duplicate)
    # Force-synth follow-up LLM call returns the final.
    synth_resp = _resp(content="here is the synthesized final answer", tool_calls=None)
    llm = _ScriptedLLM(synth_resp)
    tool = _RecordingTool()

    events: List[Dict[str, Any]] = []

    async def on_event(event):
        events.append(event)

    async def on_round_end(state: RoundState):
        if state.had_skips:
            return ToolPostResult(force_final=True)
        return None

    executor = ToolLoopExecutor(
        llm_callback=llm, tool_callback=tool, max_iterations=10,
    )
    result = asyncio.run(executor.run(
        initial_response=initial, messages=[{"role": "user", "content": "hi"}],
        tools=[], workspace_id="ws-chat",
        on_event=on_event, on_round_end=on_round_end,
    ))

    assert result.forced_final, "Chat-shape should force-final on dedup"
    assert len(tool.calls) == 1, "Dedup means only one tool actually runs"
    types_seen = [e.get("type") for e in events]
    assert "tool-start" in types_seen and "tool-end" in types_seen
    assert "synthesized" in (result.response.content or "")


# ---------------------------------------------------------------------------
# 2. Agent shape — no streaming, no force-synth, simple completion
# ---------------------------------------------------------------------------


def test_agent_shape_non_streaming_runs_to_completion():
    """Agent path: one tool round, then LLM returns content → loop ends
    cleanly. No on_event, no on_round_end — the executor's default path
    matches what ``execute_with_prompt``'s inner while-loop did."""
    initial = _resp(tool_calls=[_tc("custom_tool", {"x": 1}, "c1")])
    follow = _resp(content="agent done", tool_calls=None)
    llm = _ScriptedLLM(follow)
    tool = _RecordingTool()

    executor = ToolLoopExecutor(
        llm_callback=llm, tool_callback=tool, max_iterations=10,
    )
    result = asyncio.run(executor.run(
        initial_response=initial, messages=[{"role": "user", "content": "go"}],
        tools=[], workspace_id="ws-agent",
    ))

    assert not result.forced_final
    assert not result.max_iterations_reached
    assert result.iterations == 1
    assert result.response.content == "agent done"
    assert len(tool.calls) == 1
    assert tool.calls[0]["ws_id"] == "ws-agent"


# ---------------------------------------------------------------------------
# 3. Cross-tenant pass-through — every workspace stays isolated
# ---------------------------------------------------------------------------


def test_workspace_isolation_across_two_executor_runs():
    """Two separate executor runs with two workspace_ids must never bleed
    — each tool callback sees only its own ws_id (executor passes through)."""
    ws_a_calls: List[Any] = []
    ws_b_calls: List[Any] = []

    async def tool_cb_for(target_calls):
        async def cb(name, args, call_id, ws_id):
            target_calls.append(ws_id)
            return {"success": True, "llm_context": str(ws_id), "raw_result": {}}
        return cb

    # Workspace A
    llm_a = _ScriptedLLM(_resp(content="a done", tool_calls=None))
    initial_a = _resp(tool_calls=[_tc("custom_tool", {"x": 1}, "ca")])
    exec_a = ToolLoopExecutor(
        llm_callback=llm_a,
        tool_callback=asyncio.run(tool_cb_for(ws_a_calls)),
        max_iterations=5,
    )
    asyncio.run(exec_a.run(
        initial_response=initial_a, messages=[], tools=[], workspace_id="ws-A",
    ))

    # Workspace B
    llm_b = _ScriptedLLM(_resp(content="b done", tool_calls=None))
    initial_b = _resp(tool_calls=[_tc("custom_tool", {"x": 2}, "cb")])
    exec_b = ToolLoopExecutor(
        llm_callback=llm_b,
        tool_callback=asyncio.run(tool_cb_for(ws_b_calls)),
        max_iterations=5,
    )
    asyncio.run(exec_b.run(
        initial_response=initial_b, messages=[], tools=[], workspace_id="ws-B",
    ))

    assert ws_a_calls == ["ws-A"], f"workspace A leak: {ws_a_calls}"
    assert ws_b_calls == ["ws-B"], f"workspace B leak: {ws_b_calls}"
    assert "ws-B" not in ws_a_calls
    assert "ws-A" not in ws_b_calls


# ---------------------------------------------------------------------------
# 4. Same executor instance runs chat-style then agent-style — convergence
#    proof: ONE class handles both call shapes.
# ---------------------------------------------------------------------------


def test_one_executor_class_handles_both_shapes():
    """Build two distinct executors with different callbacks; same class.
    Both reach the right terminal state — chat forces synthesis, agent
    runs to natural completion."""
    # Chat-style
    llm_chat = _ScriptedLLM(_resp(content="chat-synth", tool_calls=None))
    chat_initial = _resp(tool_calls=[
        _tc("read_file", {"path": "/x"}, "c1"),
        _tc("read_file", {"path": "/x"}, "c2"),  # duplicate → forces synth
    ])

    async def chat_round_end(state: RoundState):
        return ToolPostResult(force_final=True) if state.had_skips else None

    chat_exec = ToolLoopExecutor(
        llm_callback=llm_chat, tool_callback=_RecordingTool(), max_iterations=10,
    )
    chat_res = asyncio.run(chat_exec.run(
        initial_response=chat_initial, messages=[], tools=[],
        workspace_id="chat-ws", on_round_end=chat_round_end,
    ))
    assert chat_res.forced_final

    # Agent-style
    llm_agent = _ScriptedLLM(_resp(content="agent-done", tool_calls=None))
    agent_initial = _resp(tool_calls=[_tc("workspace_grep", {"q": "abc"}, "a1")])
    agent_exec = ToolLoopExecutor(
        llm_callback=llm_agent, tool_callback=_RecordingTool(), max_iterations=10,
    )
    agent_res = asyncio.run(agent_exec.run(
        initial_response=agent_initial, messages=[], tools=[],
        workspace_id="agent-ws",
    ))
    assert not agent_res.forced_final
    assert agent_res.response.content == "agent-done"

    # Same class — assertions just to make the convergence intent visible.
    assert isinstance(chat_exec, ToolLoopExecutor)
    assert isinstance(agent_exec, ToolLoopExecutor)
    assert type(chat_exec) is type(agent_exec)


# ---------------------------------------------------------------------------
# 5. max_iterations cap signal — agent path uses this for the "Hit max"
#    warning, chat path uses it to emit limit_reached SSE.
# ---------------------------------------------------------------------------


def test_max_iterations_reached_is_signalled_in_result():
    """When the loop terminates because of the iteration cap (LLM still
    wants more tools), ``ToolLoopResult.max_iterations_reached`` is True."""
    keep_going = [
        _resp(tool_calls=[_tc("custom_tool", {"i": i}, f"c{i}")])
        for i in range(10)
    ]
    llm = _ScriptedLLM(*keep_going[1:])  # the executor calls LLM after round 1
    tool = _RecordingTool()

    executor = ToolLoopExecutor(
        llm_callback=llm, tool_callback=tool, max_iterations=2,
    )
    result = asyncio.run(executor.run(
        initial_response=keep_going[0], messages=[], tools=[], workspace_id=None,
    ))
    assert result.iterations == 2
    assert result.max_iterations_reached is True
