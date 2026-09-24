"""A two-step playbook ("Monday dispatch") through the real step loop and the
real board bridge, faking only the edges: the session, the step's agent call,
the clock, the log upload and the notifications. Shared by F123, F125 and F130."""
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
                 agent_status="active"):
    """Run a playbook (default: the two-step Monday dispatch) through the real
    loop; return (execution, card). ``calls`` collects what each step was sent."""
    clock = [1_000_000.0]
    results = iter(outcomes)

    async def _step(**kwargs):
        clock[0] += step_seconds
        if calls is not None:
            calls.append(kwargs)
        return next(results)

    steps = steps if steps is not None else MONDAY_DISPATCH
    execution = SimpleNamespace(
        execution_id="exec-120", recipe_id=79, workspace_id=WS, status="pending", current_step=0,
        step_results=None, error_message=None, completed_at=None, started_at=None, output_data=None,
        execution_metadata={},
    )
    card = SimpleNamespace(id=760, status="in_progress", result=None, error_message=None,
                           review_feedback=None, completed_at=None)
    session = _Session({
        WorkflowTemplate: [SimpleNamespace(id=79, name="Monday dispatch", steps=steps, execution_config=exec_config)],
        RecipeExecution: [execution],
        Workspace: [SimpleNamespace(deleted_at=None, paused_at=None, paused_reason=None)],
        Agent: [SimpleNamespace(id=7, name="CLUB SECRETARY", configuration={}, status=agent_status)],
        BoardTask: [card],
    })
    pad_mod = types.ModuleType("core.services.playbook_scratchpad")
    pad_mod.PlaybookScratchpad = _Pad
    mem_mod = types.ModuleType("core.services.playbook_memory_service")
    mem_mod.PlaybookMemoryService = _Memory
    monkeypatch.setitem(sys.modules, pad_mod.__name__, pad_mod)
    monkeypatch.setitem(sys.modules, mem_mod.__name__, mem_mod)
    # The loop reads this system_setting eagerly, and the test DB seeds none.
    monkeypatch.setattr(type(app_config), "RECIPE_DEFAULT_MAX_ITERATIONS", 3)
    monkeypatch.setattr(rex, "SessionLocal", lambda: session)
    monkeypatch.setattr(rex, "time", SimpleNamespace(time=lambda: clock[0]))
    monkeypatch.setattr(rex, "_execute_step", _step)
    monkeypatch.setattr(rex, "_is_session_step", lambda db, agent: False)
    monkeypatch.setattr(rex, "_upload_step_log_to_s3", lambda ws, ex, order, log: LOG.format(order))
    monkeypatch.setattr(rex, "_dispatch_playbook_event", _nothing)
    monkeypatch.setattr(rex, "_auto_create_playbook_report", _nothing)
    monkeypatch.setattr(rex, "_ingest_playbook_terminal_watch", lambda *a, **k: None)
    monkeypatch.setattr(rex, "_update_agent_performance_metrics", lambda *a, **k: None)
    monkeypatch.setattr(board_task_bridge, "create_recipe_board_task", lambda *a, **k: None)
    monkeypatch.setattr(board_task_bridge, "update_recipe_board_task_progress", lambda *a, **k: None)
    monkeypatch.setattr(playbook_engine_heartbeat, "_emit_playbooks_primitive", lambda *a, **k: None)
    asyncio.run(rex._execute_recipe_inner("exec-120", 79, WS, input_data or {}, None))
    return execution, card
