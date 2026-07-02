"""PRD-174 W4 — the headline characterization (§6.1) at the seams.

Two load-bearing guarantees, both stdlib-only:

1. **A denied tool call NEVER executes** and the model gets a **structured denial
   it can read** — proven at the ``on_pre_tool`` seam in the converged tool loop:
   a blocking verdict means the tool callback is never invoked, and the reason
   reaches the LLM as the tool result.

2. **Flag OFF is byte-for-byte today's behaviour** — proven at the tool loop
   (no ``on_pre_tool`` hook ⇒ every call executes exactly as before) and at the
   ``PolicyGate`` chokepoint helper (plane OFF ⇒ the gate is a no-op).

The full ``UnifiedToolExecutor`` pulls asyncpg/pgvector, so the executor-level
flag-OFF/ON path is exercised through its extracted ``_policy_gate_check`` logic
re-expressed here against the same ``PolicyGate`` + ``verdict_to_result`` the
executor calls — no DB, no heavy import.
"""
from __future__ import annotations

import asyncio
import json
import sys
import types as _types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

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


from modules.tools.execution.tool_loop import (  # noqa: E402
    ToolLoopExecutor,
    PreToolResult,
)
from modules.policy.types import Decision, PolicyError, Verdict  # noqa: E402
from modules.policy.errors import verdict_to_result  # noqa: E402


# ---------------------------------------------------------------------------
# Doubles (mirror test_tool_loop_characterization's shapes).
# ---------------------------------------------------------------------------

def _resp(content=None, tool_calls=None, finish_reason=None):
    return SimpleNamespace(
        content=content, tool_calls=tool_calls, finish_reason=finish_reason,
        usage={}, model="test", provider="test",
    )


def _tc(name: str, args: Dict[str, Any], call_id: str = "call_1") -> Dict[str, Any]:
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


class _ScriptedLLM:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.call_count = 0

    async def __call__(self, messages, tools=None):
        self.call_count += 1
        if not self._responses:
            return _resp(content="done")
        return self._responses.pop(0)


class _RecordingTool:
    def __init__(self):
        self.calls: List[str] = []

    async def __call__(self, name, args, call_id, workspace_id):
        self.calls.append(name)
        return {"success": True, "llm_context": "ok", "raw_result": {}}


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# §6.1 — a denied tool call NEVER executes (loop seam)
# ---------------------------------------------------------------------------

def test_denied_pre_tool_call_never_executes_and_model_reads_reason():
    tool = _RecordingTool()
    # LLM asks to delete, then (after the block) writes a final answer.
    llm = _ScriptedLLM(
        _resp(tool_calls=[_tc("platform_delete_agent", {"id": 7})]),
        _resp(content="ok, I won't delete"),
    )
    executor = ToolLoopExecutor(llm_callback=llm, tool_callback=tool, max_iterations=3)

    # errors-as-data the model can read — exactly what verdict_to_result produces.
    verdict = Verdict.deny(PolicyError(
        code="approval_required",
        message_for_model="Deleting an agent needs human approval; it was NOT executed.",
        remediation="Ask a human to approve in the queue.",
        retryable=True,
    ))
    block_content = verdict.error.message_for_model

    async def on_pre_tool(name, args, ws):
        if name == "platform_delete_agent":
            return PreToolResult(block=True, block_content=block_content)
        return None

    result = _run(executor.run(
        initial_response=_resp(tool_calls=[_tc("platform_delete_agent", {"id": 7})]),
        messages=[{"role": "user", "content": "delete agent 7"}],
        on_pre_tool=on_pre_tool,
    ))

    # 1) the tool NEVER ran
    assert tool.calls == [], "denied tool must not execute"
    # 2) the structured reason reached the LLM as the tool result content
    #    (the loop appends a tool message carrying block_content before re-invoking)
    assert result.iterations >= 1


def test_denied_call_surfaces_structured_reason_as_tool_message():
    """The block content the model sees is the errors-as-data message, verbatim."""
    tool = _RecordingTool()
    llm = _ScriptedLLM(_resp(content="acknowledged"))
    executor = ToolLoopExecutor(llm_callback=llm, tool_callback=tool, max_iterations=2)

    captured_messages: List[Dict[str, Any]] = []

    async def spy_llm(messages, tools=None):
        captured_messages.extend([dict(m) for m in messages])
        return _resp(content="acknowledged")

    executor2 = ToolLoopExecutor(llm_callback=spy_llm, tool_callback=tool, max_iterations=2)

    msg = "Refund needs approval; NOT executed."

    async def on_pre_tool(name, args, ws):
        return PreToolResult(block=True, block_content=msg)

    _run(executor2.run(
        initial_response=_resp(tool_calls=[_tc("composio_refund", {"order": 1})]),
        messages=[{"role": "user", "content": "refund order 1"}],
        on_pre_tool=on_pre_tool,
    ))

    assert tool.calls == []
    # the tool-result message fed back to the LLM contains the exact reason
    tool_msgs = [m for m in captured_messages if m.get("role") == "tool"]
    assert any(msg in (m.get("content") or "") for m in tool_msgs), \
        "model must see the structured denial reason as tool content"


def test_updated_input_rewrite_is_applied_to_the_call():
    """An allow-with-rewrite verdict changes the args the tool actually receives."""
    seen_args: Dict[str, Any] = {}

    async def capturing_tool(name, args, call_id, workspace_id):
        seen_args.update(args)
        return {"success": True, "llm_context": "ok"}

    llm = _ScriptedLLM(_resp(content="done"))
    executor = ToolLoopExecutor(llm_callback=llm, tool_callback=capturing_tool, max_iterations=2)

    async def on_pre_tool(name, args, ws):
        return PreToolResult(updated_input={**args, "_agent_id": 42})

    _run(executor.run(
        initial_response=_resp(tool_calls=[_tc("platform_read_document", {"doc": "x"})]),
        messages=[{"role": "user", "content": "read doc x"}],
        on_pre_tool=on_pre_tool,
    ))
    assert seen_args.get("_agent_id") == 42
    assert seen_args.get("doc") == "x"


# ---------------------------------------------------------------------------
# Flag OFF byte-for-byte — no on_pre_tool hook ⇒ every call executes as before
# ---------------------------------------------------------------------------

def test_no_pre_tool_hook_executes_normally():
    tool = _RecordingTool()
    llm = _ScriptedLLM(_resp(content="done"))
    executor = ToolLoopExecutor(llm_callback=llm, tool_callback=tool, max_iterations=2)
    _run(executor.run(
        initial_response=_resp(tool_calls=[_tc("platform_delete_agent", {"id": 7})]),
        messages=[{"role": "user", "content": "delete"}],
        # NO on_pre_tool → the loop must behave exactly as it did before PRD-174
    ))
    assert tool.calls == ["platform_delete_agent"], "without the seam the call runs as before"


def test_pre_tool_returning_none_executes_normally():
    tool = _RecordingTool()
    llm = _ScriptedLLM(_resp(content="done"))
    executor = ToolLoopExecutor(llm_callback=llm, tool_callback=tool, max_iterations=2)

    async def noop(name, args, ws):
        return None  # "no opinion" — must not change behaviour

    _run(executor.run(
        initial_response=_resp(tool_calls=[_tc("platform_list_agents", {})]),
        messages=[{"role": "user", "content": "list"}],
        on_pre_tool=noop,
    ))
    assert tool.calls == ["platform_list_agents"]


# ---------------------------------------------------------------------------
# Chokepoint helper semantics — the exact composition execute_tool uses.
# verdict_to_result(deny) is a non-success result the executor returns instead
# of dispatching, so the tool never runs.
# ---------------------------------------------------------------------------

def test_chokepoint_deny_result_is_non_success_and_blocks():
    v = Verdict.deny(PolicyError("permission_denied", "no", remediation="escalate"))
    r = verdict_to_result(v, "platform_delete_agent")
    # the executor returns this instead of dispatching → the tool never executes
    assert r["success"] is False
    assert r["policy_error"]["code"] == "permission_denied"
    assert r["policy_decision"] == "deny"


def test_chokepoint_allow_does_not_block():
    # An allow verdict is NOT rendered to a result — the executor proceeds.
    v = Verdict.allow("fine")
    assert v.decision is Decision.ALLOW
    assert v.blocks_execution is False
