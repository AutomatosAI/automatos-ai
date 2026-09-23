"""F123 — a run out of budget after finished work fails honestly.

Run 4's execution 120: step 1, a session step, completed after 51 minutes. The
check before step 2 then found the 40-minute total budget spent and failed the
run "at step 2". Board card #760 went to failed with an empty result: the owner
saw a failure and nothing of the finished work. A budget stop after a completed
step still fails the run, but its error_message says what happened, and the
card goes to review with every completed step named and the last one's output.
The dollar ceiling stops the same way. With no completed step, today's failure
stands. The card's cap names its cut, on the success path too.

These run the real step loop and the real board bridge, faking only the edges:
the session, the step's agent call, the clock, the log upload, notifications.
"""
from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace
from uuid import UUID

import pytest

from api import recipe_executor as rex
from config import config as app_config
from core.models import Agent
from core.models.core import BoardTask, RecipeExecution, WorkflowTemplate
from core.models.workspaces import Workspace
from services import board_task_bridge, playbook_engine_heartbeat

WS = UUID("00000000-0000-0000-0000-0000000000c1")
OUTPUT = "I've redone the list and Callum's Monday sheet from your export: 14 boxes, 3 changes."
TOKENS = 107_543
SESSION_STEP_S = 3082      # execution 120's step 1: 51 min 22 s
BUDGET_S = 2400            # the 40-minute total budget
LOG = "https://logs.example/exec-120/step-{}"
CUT = "[Cut at 4,000 characters. The full output: {}]"   # what the owner reads at the cap


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


def _done(output=OUTPUT):
    return {"status": "success", "result": output, "execution": {"tokens_used": TOKENS, "tool_calls": []}}


def _run(monkeypatch, *, outcomes, step_seconds, exec_config):
    """Run a two-step playbook through the real loop; return (execution, card)."""
    clock = [1_000_000.0]
    results = iter(outcomes)

    async def _step(**kwargs):
        clock[0] += step_seconds
        return next(results)

    steps = [
        {"step_id": "s1", "order": 1, "agent_id": 7, "prompt_template": "Redo the club list.",
         "error_handling": "skip", "max_retries": 0},
        {"step_id": "s2", "order": 2, "agent_id": 7, "prompt_template": "Send Callum the Monday sheet.",
         "error_handling": "stop", "max_retries": 0},
    ]
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
        Agent: [SimpleNamespace(id=7, name="CLUB SECRETARY", configuration={})],
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
    asyncio.run(rex._execute_recipe_inner("exec-120", 79, WS, {}, None))
    return execution, card


def _spent():
    usd = rex._tokens_to_usd(TOKENS, None)
    assert usd > 0, "the flat price must make step 1 cost something"
    return usd


def _stop(kind):
    """(step seconds, execution_config, the reason the run must give)."""
    finished = "Step 2 never started. Step 1 completed; its output is below."
    if kind == "time":
        return SESSION_STEP_S, {"total_timeout": BUDGET_S}, f"Out of time after step 1 of 2 (51 min; budget 40 min). {finished}"
    ceiling = _spent() * 1.5           # step 1 fits; step 1's cost again would not
    reason = f"Budget ceiling ${ceiling:.2f} reached after step 1 of 2 (${_spent():.4f} spent). {finished}"
    return 5, {"total_timeout": BUDGET_S, "cost_ceiling": ceiling}, reason


@pytest.mark.parametrize("kind", ["time", "ceiling"])
def test_a_budget_stop_after_a_finished_step_fails_honestly_with_the_work_on_the_card(monkeypatch, kind):
    step_seconds, exec_config, reason = _stop(kind)
    execution, card = _run(monkeypatch, outcomes=[_done()], step_seconds=step_seconds, exec_config=exec_config)

    assert execution.status == "failed"
    assert execution.error_message.splitlines()[0] == reason
    assert execution.step_results[0]["status"] == "completed"
    assert card.status == "review" and card.review_feedback is None   # the reviewer's channel stays theirs
    assert card.result.splitlines()[0] == reason
    assert f"Step 1 (CLUB SECRETARY): completed in {'51 min' if kind == 'time' else '5 s'}, {TOKENS:,} tokens" in card.result
    assert LOG.format(1) in card.result and OUTPUT in card.result


def test_with_no_finished_step_the_failure_is_unchanged(monkeypatch):
    execution, card = _run(monkeypatch, outcomes=[{"status": "error", "error": "the sheet would not open"}],
                           step_seconds=SESSION_STEP_S, exec_config={"total_timeout": BUDGET_S})

    message = f"Total execution timeout ({float(BUDGET_S)}s) exceeded after {SESSION_STEP_S}s at step 2"
    assert execution.status == "failed" and execution.error_message == message
    assert execution.step_results[0]["status"] == "failed"
    assert card.status == "failed" and card.error_message == message
    assert card.result is None and card.review_feedback is None


def test_a_long_output_on_the_stopped_card_says_where_it_was_cut(monkeypatch):
    long_output = "Row " * 2000
    execution, card = _run(monkeypatch, outcomes=[_done(long_output)], step_seconds=SESSION_STEP_S,
                           exec_config={"total_timeout": BUDGET_S})

    assert card.status == "review" and long_output not in card.result
    assert card.result.endswith(CUT.format(LOG.format(1)))


def test_the_success_path_names_its_cut_too(monkeypatch):
    long_output = "Box " * 2000
    execution, card = _run(monkeypatch, outcomes=[_done(), _done(long_output)], step_seconds=5,
                           exec_config={"total_timeout": BUDGET_S})

    assert execution.status == "completed" and card.status == "done"
    assert card.result.startswith("Box Box")
    assert card.result.endswith(CUT.format(LOG.format(2)))
