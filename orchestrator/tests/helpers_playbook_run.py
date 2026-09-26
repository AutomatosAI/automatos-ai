"""A two-step playbook ("Monday dispatch") through the real step loop and the
real board bridge, faking only the edges: the session, the step's agent call,
the clock, the log upload and the notifications. Shared by F123, F125 and F130;
``patch_edges`` is the same faking for a test that brings its own steps
(PRD-251 US-117)."""
from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace
from uuid import UUID

from api import recipe_executor as rex
from config import config as app_config
from core.models import Agent
from core.models.core import BoardTask, RecipeExecution, WorkflowTemplate
from core.models.workspaces import Workspace
from services import board_task_bridge, playbook_engine_heartbeat

WS = UUID("00000000-0000-0000-0000-0000000000c1")
TOKENS = 107_543
LOG = "https://logs.example/exec-120/step-{}"


class _Query:
    def __init__(self, rows):
        self._rows = rows

    def filter(self, *args, **kwargs):
        return self

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)


class _Session:
    """The rows one run reads: its playbook, execution, workspace, agent, card."""

    def __init__(self, rows):
        self._rows = rows

    def query(self, model):
        return _Query(self._rows.get(model, []))

    def commit(self):
        pass

    def rollback(self):
        pass

    def expire(self, obj):
        pass

    def refresh(self, obj):
        pass

    def close(self):
        pass


class _Pad:
    def __init__(self, execution_id, redis_client=None):
        pass

    def write_inputs(self, input_data):
        pass

    def write_meta(self, playbook_id, total_steps):
        pass

    def write_step_results(self, **kwargs):
        pass

    def get_exports(self):
        return {}

    def step_exports(self, step_order):
        return {}

    def _hgetall(self):
        return {}

    def cleanup(self):
        pass


class _Memory:
    def __init__(self, db=None):
        pass

    async def retrieve_relevant_memories(self, **kwargs):
        return None

    async def store_execution_memory(self, *args, **kwargs):
        return None


async def _nothing(*args, **kwargs):
    return None


def done(output, tokens=TOKENS):
    """A step the agent completed."""
    return {"status": "success", "result": output, "execution": {"tokens_used": tokens, "tool_calls": []}}


MONDAY_DISPATCH = [
    {"step_id": "s1", "order": 1, "agent_id": 7, "prompt_template": "Redo the club list.",
     "error_handling": "skip", "max_retries": 0},
    {"step_id": "s2", "order": 2, "agent_id": 7, "prompt_template": "Send Callum the Monday sheet.",
     "error_handling": "stop", "max_retries": 0},
]


def run_playbook(monkeypatch, *, outcomes, step_seconds, exec_config, steps=None, input_data=None, calls=None,
                 agent_status="active", execution_metadata=None, on_step=None, after_run=None, inputs=None):
    """Run a playbook (default: the two-step Monday dispatch) through the real
    loop; return (execution, card). ``calls`` collects what each step was sent;
    ``on_step`` is called inside each step, ``after_run`` in the run's task once
    the run returns. ``inputs`` is the playbook's declared inputs (F182)."""
    clock = [1_000_000.0]
    results = iter(outcomes)

    async def _step(**kwargs):
        clock[0] += step_seconds
        if calls is not None:
            calls.append(kwargs)
        if on_step is not None:
            on_step(kwargs)
        return next(results)

    steps = steps if steps is not None else MONDAY_DISPATCH
    execution = SimpleNamespace(
        execution_id="exec-120", recipe_id=79, workspace_id=WS, status="pending", current_step=0,
        step_results=None, error_message=None, completed_at=None, started_at=None, output_data=None,
        execution_metadata=dict(execution_metadata or {}),
    )
    card = SimpleNamespace(id=760, status="in_progress", result=None, error_message=None,
                           review_feedback=None, completed_at=None)
    session = _Session({
        WorkflowTemplate: [SimpleNamespace(id=79, name="Monday dispatch", steps=steps, execution_config=exec_config,
                                           inputs=inputs)],
        RecipeExecution: [execution],
        Workspace: [SimpleNamespace(deleted_at=None, paused_at=None, paused_reason=None)],
        Agent: [SimpleNamespace(id=7, name="CLUB SECRETARY", configuration={}, status=agent_status)],
        BoardTask: [card],
    })
    patch_edges(monkeypatch, session=session, step=_step, clock=clock)

    async def _run():
        await rex._execute_recipe_inner("exec-120", 79, WS, input_data or {}, None)
        if after_run is not None:
            after_run()

    asyncio.run(_run())
    return execution, card


def patch_edges(monkeypatch, *, session, step, pad=_Pad, clock=None):
    """Fake one run's edges: its session (rows by model), the agent step call
    ``step(**kwargs)``, the scratchpad class ``pad``, memory, the clock (a
    one-item list, when given), the log upload, notifications, the board bridge."""
    pad_mod = types.ModuleType("core.services.playbook_scratchpad")
    pad_mod.PlaybookScratchpad = pad
    mem_mod = types.ModuleType("core.services.playbook_memory_service")
    mem_mod.PlaybookMemoryService = _Memory
    monkeypatch.setitem(sys.modules, pad_mod.__name__, pad_mod)
    monkeypatch.setitem(sys.modules, mem_mod.__name__, mem_mod)
    # The loop reads this system_setting eagerly, and the test DB seeds none.
    monkeypatch.setattr(type(app_config), "RECIPE_DEFAULT_MAX_ITERATIONS", 3)
    monkeypatch.setattr(rex, "SessionLocal", lambda: session)
    if clock is not None:
        monkeypatch.setattr(rex, "time", SimpleNamespace(time=lambda: clock[0]))
    monkeypatch.setattr(rex, "_execute_step", step)
    monkeypatch.setattr(rex, "_is_session_step", lambda db, agent: False)
    monkeypatch.setattr(rex, "_upload_step_log_to_s3", lambda ws, ex, order, log: LOG.format(order))
    monkeypatch.setattr(rex, "_dispatch_playbook_event", _nothing)
    monkeypatch.setattr(rex, "_auto_create_playbook_report", _nothing)
    monkeypatch.setattr(rex, "_ingest_playbook_terminal_watch", lambda *a, **k: None)
    monkeypatch.setattr(rex, "_update_agent_performance_metrics", lambda *a, **k: None)
    monkeypatch.setattr(board_task_bridge, "create_recipe_board_task", lambda *a, **k: None)
    monkeypatch.setattr(board_task_bridge, "update_recipe_board_task_progress", lambda *a, **k: None)
    monkeypatch.setattr(playbook_engine_heartbeat, "_emit_playbooks_primitive", lambda *a, **k: None)
