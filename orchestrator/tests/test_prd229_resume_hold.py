"""PRD-229 P229-RVW-6 — a parked task is HELD, not clobbered/completed (pure).

The reachable harm the finding names: after an agent escalates via
``platform_ask_orchestrator``, ``escalate_clarification`` writes a labelled DRAFT
+ ``PENDING_KEY`` and returns ``{parked}``, but the coordinator UNCONDITIONALLY
records the completion — ``record_task_completion`` FULL-REPLACES
``output_metadata`` (wiping the draft) and transitions the task to COMPLETED,
ORPHANING the durable human ask and DESTROYING Gerard's baked draft-on-park.

These tests pin the two guards that close the reachability:
  * completion-side — a parked task's draft is PRESERVED (merge, not replace) and
    the task is HELD QUEUED (not finalized);
  * dispatch-side — a ``PENDING_KEY`` task is not re-dispatched, failed, or
    stalled; a runnable sibling still makes progress.

No DB / LLM / network: transition_task + sync_board_status are patched, the task
is a mock so the JSONB writes are inspected directly. The answer→resume BRIDGE
(apply_answered_clarification on 225's answer path) is US-005.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from core.models.orchestration_enums import TaskState  # noqa: E402
from modules.coordination.dispatcher import MissionDispatcher  # noqa: E402
from services.clarification_ladder import DRAFT_KEY, PENDING_KEY  # noqa: E402

_SUCCESS = {
    "status": "success",
    "result": "I parked this and asked a human.",
    "execution": {"model": "m", "provider": "p", "time": 1.0, "tokens_used": 42},
}


def _parked_task(*, task_id=None, draft="the half-finished section", ask_id=99):
    task = MagicMock()
    task.id = task_id or uuid4()
    task.state = TaskState.RUNNING.value
    task.assigned_agent_id = 7
    task.tokens_used = 0
    task.input_context = {PENDING_KEY: {"ask_id": ask_id, "question": "Ship A or B?"}}
    task.output_metadata = {
        DRAFT_KEY: {"label": "draft — parked", "ask_id": ask_id, "partial_output": draft}
    }
    return task


def _ready_task(run_id, *, seq, parked, task_id=None):
    task = MagicMock()
    task.id = task_id or uuid4()
    task.run_id = run_id
    task.sequence_number = seq
    task.state = TaskState.PENDING.value
    task.agent_role = "researcher"
    task.version_id = 1
    task.failure_reason_code = None
    task.failure_detail = None
    task.assigned_agent_id = None
    task.title = f"Task {seq}"
    task.description = f"Description {seq}"
    task.input_context = (
        {PENDING_KEY: {"ask_id": 99, "question": "Ship A or B?"}} if parked else {}
    )
    return task


# ---------------------------------------------------------------------------
# completion-side guard — preserve the draft, hold QUEUED, do NOT finalize
# ---------------------------------------------------------------------------

@patch("modules.coordination.dispatcher.sync_board_status")
@patch("modules.coordination.dispatcher.transition_task")
def test_record_completion_holds_parked_task_preserves_draft(mock_transition, mock_sync):
    task = _parked_task()

    MissionDispatcher.record_task_completion(MagicMock(), task, _SUCCESS)

    # draft PRESERVED (merge, not the finding's full-replace)
    assert DRAFT_KEY in task.output_metadata
    assert task.output_metadata[DRAFT_KEY]["partial_output"] == "the half-finished section"
    # execution metadata merged alongside the draft; tokens accrued
    assert task.output_metadata["model"] == "m"
    assert task.tokens_used == 42
    # NOT finalized — held QUEUED, never COMPLETED
    assert mock_transition.call_count == 1
    assert mock_transition.call_args.kwargs["new_state"] == TaskState.QUEUED


@patch("modules.coordination.dispatcher.sync_board_status")
@patch("modules.coordination.dispatcher.transition_task")
def test_record_completion_completes_normal_task(mock_transition, mock_sync):
    # a task with NO pending clarification still completes normally — the guard is
    # scoped to parked tasks, so the happy path is untouched.
    task = MagicMock()
    task.id = uuid4()
    task.state = TaskState.RUNNING.value
    task.assigned_agent_id = 7
    task.tokens_used = 0
    task.input_context = {}
    task.output_metadata = None

    MissionDispatcher.record_task_completion(MagicMock(), task, _SUCCESS)

    assert task.output == "I parked this and asked a human."
    assert DRAFT_KEY not in (task.output_metadata or {})
    assert mock_transition.call_args.kwargs["new_state"] == TaskState.COMPLETED


@patch("modules.coordination.dispatcher.sync_board_status")
@patch("modules.coordination.dispatcher.transition_task")
def test_parked_ask_subject_survives_completion(mock_transition, mock_sync):
    # AC2 — the durable human ask (RVW-5) is NOT orphaned by destruction: after
    # completion recording the parked task still carries PENDING_KEY + DRAFT_KEY,
    # so the US-005 answer bridge can reach it, and it is held (QUEUED) not final.
    task = _parked_task()

    MissionDispatcher.record_task_completion(MagicMock(), task, _SUCCESS)

    assert task.input_context[PENDING_KEY]["ask_id"] == 99   # ask still reachable
    assert DRAFT_KEY in task.output_metadata                 # draft preserved
    assert mock_transition.call_args.kwargs["new_state"] == TaskState.QUEUED


# ---------------------------------------------------------------------------
# dispatch-side HOLD — a parked task is skipped, the sibling still runs
# ---------------------------------------------------------------------------

@patch("modules.coordination.dispatcher.sync_board_status")
@patch("modules.coordination.dispatcher.create_task_board_task")
@patch("modules.coordination.dispatcher.emit_event")
@patch("modules.coordination.dispatcher.transition_task")
@patch("modules.coordination.dispatcher.AgentMatcher")
@patch("modules.coordination.dispatcher.DependencyResolver")
def test_dispatch_ready_holds_parked_task(
    mock_dep, mock_matcher, mock_transition, mock_emit, mock_board, mock_sync,
):
    run = MagicMock()
    run.id = uuid4()
    run.max_concurrent = 3
    run.token_budget_estimate = None
    run.tokens_used = 0
    agents = [MagicMock(id=1, name="Agent One")]

    parked = _ready_task(run.id, seq=1, parked=True)
    runnable = _ready_task(run.id, seq=2, parked=False)

    db = MagicMock()
    count_q = MagicMock()
    count_q.filter.return_value = count_q
    count_q.count.return_value = 0
    actionable_q = MagicMock()
    actionable_q.filter.return_value = actionable_q
    actionable_q.order_by.return_value = actionable_q
    actionable_q.all.return_value = []
    db.query.side_effect = [count_q, actionable_q]

    mock_dep.get_ready_tasks.return_value = [parked, runnable]
    mr = MagicMock()
    mr.agent_id = 1
    mr.agent_name = "Agent One"
    mr.total_score = 0.9
    mock_matcher.match.return_value = mr
    db.execute.return_value = MagicMock(rowcount=1)
    db.expire = MagicMock()

    results = MissionDispatcher.dispatch_ready(db, run, agents)

    dispatched = [r for r in results if r.dispatched]
    # only the runnable sibling dispatches; the parked task is HELD OUT of dispatch
    assert len(dispatched) == 1
    assert dispatched[0].task_id == runnable.id
    # the parked task was NEVER dispatched (held), and dispatch_ready returned
    # normally — it was not failed or counted as a stall.
    assert all(r.task_id != parked.id for r in dispatched)


@patch("modules.coordination.dispatcher.sync_board_status")
@patch("modules.coordination.dispatcher.create_task_board_task")
@patch("modules.coordination.dispatcher.emit_event")
@patch("modules.coordination.dispatcher.transition_task")
@patch("modules.coordination.dispatcher.AgentMatcher")
@patch("modules.coordination.dispatcher.DependencyResolver")
def test_dispatch_ready_all_parked_is_no_ready_not_stall(
    mock_dep, mock_matcher, mock_transition, mock_emit, mock_board, mock_sync,
):
    # a run whose only candidate is parked → dispatch_ready reports no_ready_tasks
    # (the run waits for the answer), NOT a failure or budget stall.
    run = MagicMock()
    run.id = uuid4()
    run.max_concurrent = 3
    run.token_budget_estimate = None
    run.tokens_used = 0

    parked = _ready_task(run.id, seq=1, parked=True)

    db = MagicMock()
    count_q = MagicMock()
    count_q.filter.return_value = count_q
    count_q.count.return_value = 0
    actionable_q = MagicMock()
    actionable_q.filter.return_value = actionable_q
    actionable_q.order_by.return_value = actionable_q
    actionable_q.all.return_value = []
    db.query.side_effect = [count_q, actionable_q]
    mock_dep.get_ready_tasks.return_value = [parked]

    results = MissionDispatcher.dispatch_ready(db, run, [MagicMock(id=1)])

    assert len(results) == 1
    assert results[0].dispatched is False
    assert results[0].skipped_reason == "no_ready_tasks"
    mock_matcher.match.assert_not_called()  # the parked task was never matched/dispatched
