"""PRD-229 US-005 — the resume loop wired end-to-end (pure).

park → human answers → task resumes. Three parts, all exercised here without a
DB / LLM / network:
  * HOLD — dispatch_ready skips a PENDING_KEY task (built in P229-RVW-6; the
    dispatchable-after-clear half is pinned here).
  * BRIDGE — api/approval_grants._requeue_subject now calls
    apply_answered_clarification for a clarification-park tool_call grant (the
    missing production caller; no parallel resume path).
  * RESUME — once RESUME_KEY is set the held task drops out of the hold and its
    next prompt carries render_resume_block's Q&A + preserved draft.

225's ask_human is stubbed (reuse proof), the event trail is stubbed, and the
grant/task are plain namespaces so the JSONB writes are inspected directly.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import api.approval_grants as ag  # noqa: E402
import modules.tools.discovery.handlers_asks as ha  # noqa: E402
import services.clarification_ladder as cl  # noqa: E402
from api.approval_grants import _requeue_subject, _resume_clarification_if_parked  # noqa: E402
from core.models.approval_grants import GrantStatus, SUBJECT_TOOL_CALL  # noqa: E402
from core.models.orchestration_enums import TaskState  # noqa: E402
from modules.coordination.dispatcher import MissionDispatcher  # noqa: E402
from services.clarification_ladder import (  # noqa: E402
    DRAFT_KEY,
    PENDING_KEY,
    RESUME_KEY,
    escalate_clarification,
    pending_ask_id,
    render_resume_block,
)
from services.orchestrator_answers import ClarificationSubject  # noqa: E402


class _One:
    def __init__(self, obj):
        self._obj = obj

    def filter(self, *a, **k):
        return self

    def first(self):
        return self._obj


class _BridgeDB:
    """Dispatches query(OrchestrationTask) → the parked task and
    query(ApprovalGrant) → the granted grant (ids are read by the bridge/apply
    logic, so the filter is a no-op here). commit/rollback are inert."""

    def __init__(self, task, grant):
        self._task = task
        self._grant = grant

    def query(self, model):
        name = getattr(model, "__name__", "")
        if name == "OrchestrationTask":
            return _One(self._task)
        if name == "ApprovalGrant":
            return _One(self._grant)
        return _One(None)

    def commit(self):
        pass

    def rollback(self):
        pass


@pytest.fixture
def stub_ask_human(monkeypatch):
    async def _ask(db, workspace_id, params):
        return {"success": True, "ask_id": 99, "parked": True}

    monkeypatch.setattr(ha, "ask_human", _ask)
    monkeypatch.setattr(cl, "emit_event", lambda *a, **k: SimpleNamespace(id=uuid4()))


def _granted_grant(*, ask_id=99, subject_id="task-9", answer="Ship variant B."):
    return SimpleNamespace(
        id=ask_id,
        subject_type=SUBJECT_TOOL_CALL,
        subject_id=subject_id,
        status=GrantStatus.GRANTED.value,
        question_md="Ship A or B?",
        answer_text=answer,
        details=None,
    )


# ---------------------------------------------------------------------------
# BRIDGE — apply_answered_clarification is called from 225's answer path
# ---------------------------------------------------------------------------

def test_bridge_resumes_clarification_park():
    task = SimpleNamespace(
        id="task-9",
        input_context={PENDING_KEY: {"ask_id": 99, "question": "Ship A or B?"}},
        output_metadata={DRAFT_KEY: {"partial_output": "half-finished", "question": "Ship A or B?"}},
    )
    grant = _granted_grant(subject_id="task-9")

    handled = _resume_clarification_if_parked(_BridgeDB(task, grant), grant)

    assert handled is True
    assert PENDING_KEY not in task.input_context                 # hold released
    assert task.input_context[RESUME_KEY]["answer"] == "Ship variant B."


def test_bridge_declines_non_clarification_tool_call():
    # the subject task is parked on a DIFFERENT ask (7, not this grant's 99) → the
    # bridge declines so the PRD-193 stored-call path handles it.
    task = SimpleNamespace(id="task-9", input_context={PENDING_KEY: {"ask_id": 7}})
    grant = _granted_grant(subject_id="task-9")
    assert _resume_clarification_if_parked(_BridgeDB(task, grant), grant) is False
    # and a subject with no orchestration task at all → decline
    assert _resume_clarification_if_parked(_BridgeDB(None, grant), grant) is False


@pytest.mark.asyncio
async def test_requeue_falls_through_to_resume_tool_call_for_stored_call(monkeypatch):
    # a PRD-193 stored-call grant (no parked task) must still route to
    # _resume_tool_call — the clarification bridge does not hijack it.
    grant = _granted_grant(subject_id="board-5")
    called = {"n": 0}

    async def _resume(db, g):
        called["n"] += 1

    monkeypatch.setattr(ag, "_resume_tool_call", _resume)
    await _requeue_subject(_BridgeDB(None, grant), grant)
    assert called["n"] == 1


# ---------------------------------------------------------------------------
# FULL LOOP — escalate (park+draft) → answer via the production bridge → resume
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_loop_park_answer_resume_via_production_bridge(stub_ask_human):
    # 1. escalate → park + draft (RVW-5) on the task's EXISTING JSONB
    task = SimpleNamespace(id="task-9", output="the half-finished section",
                           input_context=None, output_metadata=None)
    subject = ClarificationSubject(
        run_id="run-1", workspace_id="ws-1", task_id="task-9", task=task, agent_id=5,
    )
    grant = _granted_grant(subject_id="task-9")
    db = _BridgeDB(task, grant)

    await escalate_clarification(db, subject, "Ship A or B?")
    assert task.input_context[PENDING_KEY]["ask_id"] == 99                 # parked+held
    assert task.output_metadata[DRAFT_KEY]["partial_output"] == "the half-finished section"

    # 2. the human answers → the PRODUCTION bridge (225 answer path → _requeue_subject)
    await _requeue_subject(db, grant)

    # 3. RESUME_KEY set, PENDING_KEY cleared → the task drops out of dispatch_ready's hold
    assert PENDING_KEY not in task.input_context
    assert pending_ask_id(task) is None
    assert task.input_context[RESUME_KEY]["answer"] == "Ship variant B."

    # 4. the re-run prompt carries the human answer AND the preserved draft
    block = render_resume_block(task)
    assert "Ship variant B." in block
    assert "the half-finished section" in block


# ---------------------------------------------------------------------------
# RUN-LEVEL — parked held while sibling runs; resumes once, no duplicate exec
# ---------------------------------------------------------------------------

def _ready_task(run_id, *, seq, pending):
    task = MagicMock()
    task.id = uuid4()
    task.run_id = run_id
    task.sequence_number = seq
    task.state = TaskState.QUEUED.value
    task.agent_role = "researcher"
    task.version_id = 1
    task.failure_reason_code = None
    task.failure_detail = None
    task.assigned_agent_id = None
    task.title = f"Task {seq}"
    task.description = f"Description {seq}"
    task.input_context = {PENDING_KEY: {"ask_id": 99}} if pending else {RESUME_KEY: {"answer": "B"}}
    return task


def _dispatch(run, tasks, monkeypatch):
    """Run dispatch_ready over `tasks` with the heavy collaborators patched."""
    import modules.coordination.dispatcher as d

    monkeypatch.setattr(d, "sync_board_status", lambda *a, **k: None)
    monkeypatch.setattr(d, "create_task_board_task", lambda *a, **k: None)
    monkeypatch.setattr(d, "emit_event", lambda *a, **k: None)
    monkeypatch.setattr(d, "transition_task", lambda *a, **k: None)
    mr = MagicMock(agent_id=1, agent_name="A", total_score=0.9)
    monkeypatch.setattr(d.AgentMatcher, "match", lambda *a, **k: mr)
    monkeypatch.setattr(d.DependencyResolver, "get_ready_tasks", staticmethod(lambda db, rid: tasks))

    db = MagicMock()
    count_q = MagicMock()
    count_q.filter.return_value = count_q
    count_q.count.return_value = 0
    actionable_q = MagicMock()
    actionable_q.filter.return_value = actionable_q
    actionable_q.order_by.return_value = actionable_q
    actionable_q.all.return_value = []
    db.query.side_effect = [count_q, actionable_q]
    db.execute.return_value = MagicMock(rowcount=1)
    db.expire = MagicMock()
    return MissionDispatcher.dispatch_ready(db, run, [MagicMock(id=1)])


def test_parked_task_resumes_once_after_answer_no_dup(monkeypatch):
    run = MagicMock(id=uuid4(), max_concurrent=3, token_budget_estimate=None, tokens_used=0)

    # while PENDING_KEY is set the task is HELD (RVW-6) — not dispatched.
    parked = _ready_task(run.id, seq=1, pending=True)
    held = _dispatch(run, [parked], monkeypatch)
    assert all(not r.dispatched for r in held)

    # after the bridge clears PENDING_KEY (RESUME_KEY set), the SAME task is now
    # dispatchable and resumes exactly once (no duplicate execution).
    resumed_task = _ready_task(run.id, seq=1, pending=False)
    resumed = _dispatch(run, [resumed_task], monkeypatch)
    dispatched = [r for r in resumed if r.dispatched]
    assert len(dispatched) == 1
    assert dispatched[0].task_id == resumed_task.id


def test_run_keeps_progress_parked_plus_runnable_sibling(monkeypatch):
    run = MagicMock(id=uuid4(), max_concurrent=3, token_budget_estimate=None, tokens_used=0)
    parked = _ready_task(run.id, seq=1, pending=True)
    runnable = _ready_task(run.id, seq=2, pending=False)

    results = _dispatch(run, [parked, runnable], monkeypatch)

    dispatched = [r for r in results if r.dispatched]
    assert len(dispatched) == 1                      # the run keeps making progress
    assert dispatched[0].task_id == runnable.id      # via the runnable sibling
    assert all(r.task_id != parked.id for r in dispatched)  # parked stays held
