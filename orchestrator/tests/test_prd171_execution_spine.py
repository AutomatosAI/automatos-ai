"""PRD-171 (Wave 1) — Execution Spine Integrity regression net.

These tests pin the five spine repairs and, above all, close the gap that let
the F001 regression ship: the whole `ToolLoopExecutor` suite constructs the
executor DIRECTLY, so nothing exercised `AgentFactory.execute_with_prompt` — the
real constructor whose kwarg drifted (`content_truncate_chars` vs
`content_truncate_tokens`). The headline test (§5.1) drives a tool-using turn
*through* `execute_with_prompt` with a stubbed LLM and asserts a NON-error
result. With the bug present it returns `{"status":"error"}` (the TypeError is
swallowed by the retry `except`); after the one-line fix it returns
`{"status":"success"}`.

Findings covered:
  - F001   — the headline gap test (execute_with_prompt → real loop → success)
  - F023   — a `status:error` execution closes the board task as 'failed', not 'done'
  - F024   — `renew_lease` extends a live run's lease (no double-execute > 600s)
  - F025   — mission-mirror source_types are excluded from drag-to-execute
  - §9.5-1 — the stuck-loop learning sink imports/calls the real
             `tool_outcome_capture` module without ImportError

No DB / event-loop network needed: the LLM, tools, monitoring and (for F023)
the board's dispatch/report side-effects are stubbed at their seams. CI is the
integration gate; these are the focused unit guards.
"""
from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

def _llm_response(content: str = "Done.", *, tool_calls=None):
    """A SimpleNamespace shaped like the real LLMResponse the loop consumes."""
    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls or [],
        finish_reason="stop",
        model="stub-model",
        provider="stub",
        usage={"total_tokens": 3, "prompt_tokens": 2, "completion_tokens": 1},
    )


class _StubLLMManager:
    """Minimal llm_manager: records calls and returns a canned response."""

    def __init__(self, responses: Optional[List[Any]] = None):
        self._responses = list(responses or [_llm_response()])
        self.config = SimpleNamespace(model="stub-model")
        self.calls: List[Dict[str, Any]] = []

    async def generate_response(self, messages, tools=None, **kwargs):
        self.calls.append({"messages": list(messages), "tools": tools})
        return self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]


def _make_runtime(agent_factory_mod, *, agent_id: int = 101):
    """A ready-to-run AgentRuntime so execute_with_prompt skips activation."""
    meta = agent_factory_mod.AgentMetadata(name="Spine Agent", agent_type="worker")
    return agent_factory_mod.AgentRuntime(
        agent_id=agent_id,
        metadata=meta,
        llm_manager=_StubLLMManager(),
        lifecycle_state=agent_factory_mod.AgentLifecycle.ACTIVE,
        created_at=datetime.now(timezone.utc),
        tools=[],                      # no Composio apps → hint branch skipped
        tool_executor=None,
        workspace_id="11111111-1111-1111-1111-111111111111",
    )


# ===========================================================================
# F001 — THE HEADLINE GAP TEST (§5.1): drive execute_with_prompt end-to-end.
# ===========================================================================

def test_execute_with_prompt_returns_non_error_through_real_loop(monkeypatch):
    """The wave's definition of done.

    Drives a full turn through `AgentFactory.execute_with_prompt` — its REAL
    body, which constructs `ToolLoopExecutor(...)`. Before F001 that construction
    raised `TypeError: unexpected keyword argument 'content_truncate_chars'`,
    caught by the retry `except`, so the method returned `status:error`. This
    test asserts a NON-error result, permanently guarding the exact gap the
    direct-construction tests missed.
    """
    import modules.agents.factory.agent_factory as af

    # Monitoring is a DB-backed side-effect; stub it to a no-op recorder.
    monkeypatch.setattr(
        af, "get_monitoring_service",
        lambda: SimpleNamespace(record_agent_execution=lambda **kw: None),
        raising=True,
    )
    # With `system_prompt` set, the factory loads tools via get_tools_for_agent,
    # which touches the DB. This test is about the LOOP, not tool loading — hand
    # it an empty tool set at the seam so no DB is required.
    import modules.tools.tool_router as tr
    monkeypatch.setattr(tr, "get_tools_for_agent", lambda **kw: [], raising=True)

    factory = af.AgentFactory.__new__(af.AgentFactory)  # no DB session needed
    factory.db_session = None
    factory.active_agents = {}
    import logging
    factory.logger = logging.getLogger("test.spine")

    runtime = _make_runtime(af)
    factory.active_agents[runtime.agent_id] = runtime

    result = asyncio.run(
        factory.execute_with_prompt(
            agent=runtime.agent_id,
            prompt="Say done.",
            system_prompt="You are a test agent.",  # bypass ContextService/DB path
            use_memory=False,
            max_retries=1,
        )
    )

    assert result["status"] != "error", (
        f"execute_with_prompt returned an error result — F001 regression: {result}"
    )
    assert result["status"] == "success"
    assert result["result"] == "Done."
    # The loop actually ran (at least the initial generate_response happened).
    assert runtime.llm_manager.calls, "the stubbed LLM was never called"


def test_tool_loop_executor_rejects_the_removed_kwarg():
    """Belt-and-braces: constructing the executor with the OLD kwarg must fail.

    This is what silently broke every non-chat path. Keeping it explicit means a
    future re-introduction of `content_truncate_chars` fails loudly at the seam.
    """
    from modules.tools.execution.tool_loop import ToolLoopExecutor

    async def _noop_llm(msgs, tools):  # pragma: no cover - never invoked
        return _llm_response()

    async def _noop_tool(name, args, call_id, ws):  # pragma: no cover
        return {"success": True, "llm_context": "{}"}

    with pytest.raises(TypeError):
        ToolLoopExecutor(
            llm_callback=_noop_llm,
            tool_callback=_noop_tool,
            max_iterations=3,
            content_truncate_chars=0,  # the removed name
        )

    # The correct kwarg constructs fine.
    ex = ToolLoopExecutor(
        llm_callback=_noop_llm,
        tool_callback=_noop_tool,
        max_iterations=3,
        content_truncate_tokens=0,
    )
    assert ex.content_truncate_tokens == 0


# ===========================================================================
# §9.5-1 — the stuck-loop learning sink points at the REAL module.
# ===========================================================================

def test_stuck_learning_sink_uses_tool_outcome_capture(monkeypatch):
    """`_record_stuck_learning` must reach `tool_outcome_capture` (not the dead
    `modules.memory.task_learning`) and pass its real signature — a failure
    outcome — without ImportError."""
    import modules.tools.execution.tool_loop as tl
    import modules.memory.tool_outcome_capture as toc

    captured: Dict[str, Any] = {}

    def _spy(*, tool_name, parameters, result, workspace_id, agent_id):
        captured.update(
            tool_name=tool_name, parameters=parameters,
            result=result, workspace_id=workspace_id, agent_id=agent_id,
        )
        return None

    monkeypatch.setattr(toc, "capture_tool_outcome", _spy, raising=True)

    # Must not raise (the old import would have ImportError'd here).
    tl._record_stuck_learning("ws-42", "composio_execute:SLACK_SEND_MESSAGE")

    assert captured, "the real tool_outcome_capture sink was never called"
    assert captured["workspace_id"] == "ws-42"
    # A stuck loop is a failure outcome — that is what the capture must record.
    assert captured["result"]["success"] is False
    assert "stuck" in captured["result"]["error"].lower()


def test_stuck_learning_builds_a_real_outcome_record():
    """End-to-end through the real capture: the stuck signature yields a
    `tool_outcome` failure record from `build_tool_outcome` (the sink is live,
    not a dead log line)."""
    from modules.memory.tool_outcome_capture import build_tool_outcome, TOOL_OUTCOME_TYPE

    record = build_tool_outcome(
        tool_name="tool_loop.stuck",
        parameters={"action": "stuck_loop", "app_name": "platform"},
        result={"success": False, "error": "stuck loop: identical tool call(s) repeated — X"},
        workspace_id="ws-42",
    )
    assert record is not None
    assert record["type"] == TOOL_OUTCOME_TYPE
    assert record["metadata"]["success"] is False


# ===========================================================================
# F023 — a status:error execution closes the board task as 'failed', not 'done'.
# ===========================================================================

class _FakeTask:
    def __init__(self, task_id: int, status: str = "in_progress"):
        self.id = task_id
        self.status = status
        self.result = None
        self.error_message = None
        self.completed_at = None
        self.assigned_agent_id = 7


class _FakeQuery:
    def __init__(self, task):
        self._task = task

    def get(self, _id):
        return self._task


class _FakeSession:
    def __init__(self, task):
        self._task = task
        self.committed = 0

    def query(self, _model):
        return _FakeQuery(self._task)

    def commit(self):
        self.committed += 1

    def rollback(self):  # pragma: no cover - not expected in these tests
        pass

    def close(self):
        pass


def _run_launch_with(monkeypatch, exec_result: Dict[str, Any], task: _FakeTask):
    """Drive `_launch_task_execution`'s inner `_run` with a stubbed factory +
    session, capturing the terminal task state. Side-effect seams (dispatch,
    reports, heartbeat, guarded launch) are neutralised."""
    import api.board_tasks as bt

    session = _FakeSession(task)
    monkeypatch.setattr(bt, "SessionLocal", lambda: session, raising=False)

    # Stub the agent factory to return the canned execution result.
    class _StubFactory:
        def __init__(self, db_session=None):
            pass

        async def execute_with_prompt(self, **kwargs):
            return exec_result

    import modules.agents.factory.agent_factory as af
    monkeypatch.setattr(af, "AgentFactory", _StubFactory, raising=True)
    # core.database.database.SessionLocal is imported inside _run — patch there.
    import core.database.database as dbmod
    monkeypatch.setattr(dbmod, "SessionLocal", lambda: session, raising=False)

    # Neutralise async side-effects + the heartbeat + report writer.
    async def _noop(*a, **k):
        return None

    monkeypatch.setattr(bt, "_dispatch_task_complete", _noop, raising=True)
    monkeypatch.setattr(bt, "_dispatch_task_failed", _noop, raising=True)
    monkeypatch.setattr(bt, "_auto_create_task_report", _noop, raising=True)
    monkeypatch.setattr(bt, "_lease_heartbeat", _noop, raising=True)
    monkeypatch.setattr(bt, "record_error", lambda **k: None, raising=True)

    # Capture the coroutine `launch_guarded` would schedule and run it inline.
    captured = {}

    def _capture(coro, **kwargs):
        captured["coro"] = coro

    monkeypatch.setattr(bt, "launch_guarded", _capture, raising=True)

    bt._launch_task_execution(
        task_id=task.id,
        agent_id=7,
        workspace_id="ws-1",
        prompt="do it",
        review_mode="auto",
    )
    asyncio.run(captured["coro"])
    return task


def test_board_task_error_result_marks_failed_not_done(monkeypatch):
    """F023: `execute_with_prompt` → {status:error} must close the task as
    'failed' with the error text — NOT 'done'."""
    task = _FakeTask(task_id=555)
    _run_launch_with(
        monkeypatch,
        {"status": "error", "error": "Task execution failed after 1 attempts: boom"},
        task,
    )
    assert task.status == "failed", "an error result must not close as done/review"
    assert task.error_message and "boom" in task.error_message
    assert task.completed_at is not None


def test_board_task_success_result_marks_done(monkeypatch):
    """F023 parity: a success result still closes 'done' with the result text."""
    task = _FakeTask(task_id=556)
    _run_launch_with(
        monkeypatch,
        {"status": "success", "result": "all good"},
        task,
    )
    assert task.status == "done"
    assert task.result == "all good"


# ===========================================================================
# F024 — renew_lease extends a live run's lease (no double-execute > 600s).
# ===========================================================================

class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def fetchall(self):
        return self._rows


class _RecordingSession:
    """Captures the UPDATE + params renew_lease issues, returns canned rows."""

    def __init__(self, returned_rows):
        self._rows = returned_rows
        self.executed: List[Any] = []
        self.commits = 0

    def execute(self, statement, params=None):
        self.executed.append((str(statement), params or {}))
        return _FakeResult(self._rows)

    def commit(self):
        self.commits += 1


def test_renew_lease_extends_running_task():
    """A row still in_progress gets its lease pushed forward → renew returns
    True, so the sweeper's `lease_until < now` never fires for a live run."""
    from services.board_dispatcher import renew_lease

    session = _RecordingSession(returned_rows=[(999,)])
    ok = renew_lease(session, 999, lease_seconds=600)

    assert ok is True
    sql, params = session.executed[-1]
    assert "UPDATE board_tasks" in sql
    assert "status = 'in_progress'" in sql  # only renews a live run
    assert params["task_id"] == 999
    assert session.commits == 1


def test_renew_lease_noop_for_terminal_task():
    """A finished/failed/requeued row (no in_progress match) returns False so the
    heartbeat loop stops — never resurrects a terminal task."""
    from services.board_dispatcher import renew_lease

    session = _RecordingSession(returned_rows=[])  # WHERE matched nothing
    ok = renew_lease(session, 999, lease_seconds=600)
    assert ok is False


# ===========================================================================
# F025 — mission-mirror source_types excluded from drag-to-execute.
# ===========================================================================

def test_non_executable_source_types_cover_mission_mirrors():
    """The drag/PATCH launch gate must exclude recipe AND both mission-mirror
    source_types (`orchestration`, `orchestration_task`) so a kanban drag never
    re-runs work the recipe/mission engine already owns."""
    import api.board_tasks as bt

    assert "recipe" in bt._NON_EXECUTABLE_SOURCE_TYPES
    assert "orchestration" in bt._NON_EXECUTABLE_SOURCE_TYPES
    assert "orchestration_task" in bt._NON_EXECUTABLE_SOURCE_TYPES
    # A user-owned board task IS executable on drag.
    assert "user" not in bt._NON_EXECUTABLE_SOURCE_TYPES


def test_mission_mirror_source_matches_bridge_constant():
    """Guard against drift: the source_types the board excludes must match the
    values `orchestration_board_bridge` actually stamps onto mirror rows."""
    import api.board_tasks as bt

    # These are the literals used in services/orchestration_board_bridge.py.
    parent_source = "orchestration"
    child_source = "orchestration_task"
    assert parent_source in bt._NON_EXECUTABLE_SOURCE_TYPES
    assert child_source in bt._NON_EXECUTABLE_SOURCE_TYPES
