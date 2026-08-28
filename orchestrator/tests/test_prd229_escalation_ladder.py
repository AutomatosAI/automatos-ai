"""PRD-229 US-003 — the escalation ladder (pure tests).

No DB / LLM / network: PRD-225's ``ask_human`` is stubbed to prove REUSE (not a
parallel ask path), the task is a plain namespace so the park/draft writes are
inspected directly, and the full loop is driven through the real
``apply_answered_clarification`` bridge against a granted grant.

Covers: cannot_answer/escalate_directly → ask via 225 shared internals + park +
labelled draft; governance escalates directly with zero answering; full loop
(ask → park+draft → answer → resume with Q&A + draft in the next-run context);
escalation is NOT limited by CLARIFICATION_BUDGET.
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

import modules.tools.discovery.handlers_asks as ha  # noqa: E402
import modules.tools.discovery.handlers_clarify as hc  # noqa: E402
import services.clarification_ladder as cl  # noqa: E402
import services.orchestrator_answers as oa  # noqa: E402
from core.models.orchestration_enums import EventType  # noqa: E402
from services.clarification_ladder import (  # noqa: E402
    DRAFT_KEY,
    PENDING_KEY,
    RESUME_KEY,
    apply_answered_clarification,
    escalate_clarification,
    render_resume_block,
)
from services.orchestrator_answers import ClarificationSubject  # noqa: E402


def _task(task_id="task-1", output="partial draft so far", input_context=None, output_metadata=None):
    return SimpleNamespace(
        id=task_id,
        output=output,
        output_metadata=output_metadata,
        input_context=input_context,
    )


def _subject(task):
    return ClarificationSubject(
        run_id=uuid4(), workspace_id=uuid4(), task_id=task.id, task=task, agent_id=5,
    )


@pytest.fixture
def spy_events(monkeypatch):
    calls = []

    def _rec(db, run_id, event_type, actor_type, actor_id=None, task_id=None, payload=None):
        calls.append(SimpleNamespace(event_type=event_type, payload=payload or {}))
        return SimpleNamespace(id=uuid4())

    monkeypatch.setattr(cl, "emit_event", _rec)
    return calls


@pytest.fixture
def stub_ask_human(monkeypatch):
    """Stub 225's ask_human, recording the params it was called with (reuse proof)."""
    seen = {}

    async def _ask(db, workspace_id, params):
        seen["params"] = params
        seen["workspace_id"] = workspace_id
        return {"success": True, "ask_id": 99, "parked": True}

    monkeypatch.setattr(ha, "ask_human", _ask)
    return seen


# ---------------------------------------------------------------------------
# escalate: reuse 225 ask_human, park, labelled draft
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_escalate_reuses_ask_human_and_parks_with_draft(stub_ask_human, spy_events):
    task = _task()
    subject = _subject(task)

    result = await escalate_clarification(MagicMock(), subject, "Which vendor?", agent_name="QUILL")

    # reuse: the SAME 225 handler, subject_type tool_call carrying the task id
    assert stub_ask_human["params"]["subject_type"] == "tool_call"
    assert stub_ask_human["params"]["subject_id"] == "task-1"
    assert stub_ask_human["params"]["question"] == "Which vendor?"

    assert result == {"parked": True, "ask_id": 99, "message": result["message"]}

    # labelled draft on the EXISTING result JSONB
    draft = task.output_metadata[DRAFT_KEY]
    assert "draft" in draft["label"].lower()
    assert draft["ask_id"] == 99
    assert draft["partial_output"] == "partial draft so far"
    # awaiting marker on input_context
    assert task.input_context[PENDING_KEY]["ask_id"] == 99
    # recorded on the run event trail
    assert [c.event_type for c in spy_events] == [EventType.CLARIFICATION_ESCALATED]
    assert spy_events[0].payload["ask_id"] == 99


@pytest.mark.asyncio
async def test_escalate_passes_partial_output_when_given(stub_ask_human, spy_events):
    task = _task(output="")
    subject = _subject(task)
    await escalate_clarification(
        MagicMock(), subject, "Q?", partial_output="the agent's in-progress work",
    )
    assert task.output_metadata[DRAFT_KEY]["partial_output"] == "the agent's in-progress work"


# ---------------------------------------------------------------------------
# handler routing: cannot_answer AND escalate_directly (governance) → escalate
# ---------------------------------------------------------------------------

def _server_params(**over):
    p = {"question": "Q?", "_run_id": "r", "_task_id": "task-1", "_agent_id": 5, "_agent_name": "QUILL"}
    p.update(over)
    return p


@pytest.mark.asyncio
async def test_handler_escalates_on_cannot_answer(monkeypatch):
    async def _cannot(db, subject, question, *, category=None):
        return {"cannot_answer": True, "reason": "unretrievable"}

    seen = {}

    async def _escalate(db, subject, question, *, category=None, partial_output=None, agent_name=None):
        seen["called"] = True
        return {"parked": True, "ask_id": 7, "message": "parked"}

    monkeypatch.setattr(oa, "answer_clarification", _cannot)
    monkeypatch.setattr(cl, "escalate_clarification", _escalate)
    result = await hc.ask_orchestrator(MagicMock(), uuid4(), _server_params())

    assert seen.get("called") is True
    assert result["parked"] is True
    assert result["ask_id"] == 7
    assert "proceed_with_assumption" not in result


@pytest.mark.asyncio
async def test_handler_escalates_directly_on_governance(monkeypatch):
    answered = {"n": 0}

    async def _gov(db, subject, question, *, category=None):
        answered["n"] += 1
        return {"escalate_directly": True, "reason": "governance", "category": "spend"}

    captured = {}

    async def _escalate(db, subject, question, *, category=None, partial_output=None, agent_name=None):
        captured["category"] = category
        return {"parked": True, "ask_id": 8, "message": "parked"}

    monkeypatch.setattr(oa, "answer_clarification", _gov)
    monkeypatch.setattr(cl, "escalate_clarification", _escalate)
    result = await hc.ask_orchestrator(MagicMock(), uuid4(), _server_params(category="spend"))

    assert result["parked"] is True
    assert result["ask_id"] == 8
    # governance category carried into the escalation
    assert captured["category"] == "spend"


# ---------------------------------------------------------------------------
# escalation is NOT budget-limited
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_escalation_works_when_budget_exhausted(monkeypatch):
    # answer_clarification returns cannot_answer(budget) — budget spent — and the
    # ladder STILL escalates (escalations are never budget-limited).
    async def _budget(db, subject, question, *, category=None):
        return {"cannot_answer": True, "reason": "budget"}

    seen = {}

    async def _escalate(db, subject, question, *, category=None, partial_output=None, agent_name=None):
        seen["called"] = True
        return {"parked": True, "ask_id": 11, "message": "parked"}

    monkeypatch.setattr(oa, "answer_clarification", _budget)
    monkeypatch.setattr(cl, "escalate_clarification", _escalate)
    result = await hc.ask_orchestrator(MagicMock(), uuid4(), _server_params())

    assert seen.get("called") is True
    assert result["parked"] is True


# ---------------------------------------------------------------------------
# full loop: ask → park+draft → answer (225 shared) → resume w/ Q&A + draft
# ---------------------------------------------------------------------------

class _GrantQuery:
    def __init__(self, grant):
        self._grant = grant

    def filter(self, *a, **k):
        return self

    def first(self):
        return self._grant


class _GrantDB:
    def __init__(self, grant):
        self._grant = grant

    def query(self, model):
        return _GrantQuery(self._grant)


@pytest.mark.asyncio
async def test_full_loop_ask_park_answer_resume(stub_ask_human, spy_events):
    # 1. escalate → ask created (225 stub) + park + draft
    task = _task(output="the half-finished section")
    subject = _subject(task)
    esc = await escalate_clarification(MagicMock(), subject, "Ship A or B?")
    assert task.input_context[PENDING_KEY]["ask_id"] == 99
    assert task.output_metadata[DRAFT_KEY]["partial_output"] == "the half-finished section"

    # 2. the human answers — PRD-225's answer path sets the grant granted + answer
    #    (this is exactly what apply_question_answer persists; unchanged here).
    from core.models.approval_grants import GrantStatus
    grant = SimpleNamespace(
        id=99, status=GrantStatus.GRANTED.value,
        question_md="Ship A or B?", answer_text="Ship variant B.",
    )

    # 3. resume: bridge the answer into the task's next-run context
    resumed = apply_answered_clarification(_GrantDB(grant), task)
    assert resumed is True
    assert PENDING_KEY not in task.input_context           # awaiting marker cleared
    resume = task.input_context[RESUME_KEY]
    assert resume["answer"] == "Ship variant B."
    assert resume["draft"] == "the half-finished section"

    # 4. the re-run prompt reads the Q&A + the preserved draft
    block = render_resume_block(task)
    assert "Ship variant B." in block
    assert "the half-finished section" in block


@pytest.mark.asyncio
async def test_resume_waits_while_grant_still_pending(stub_ask_human):
    task = _task()
    subject = _subject(task)
    await escalate_clarification(MagicMock(), subject, "Q?")

    from core.models.approval_grants import GrantStatus
    pending_grant = SimpleNamespace(id=99, status=GrantStatus.PENDING.value, question_md="Q?", answer_text=None)
    # still pending → not resumed, marker intact
    assert apply_answered_clarification(_GrantDB(pending_grant), task) is False
    assert PENDING_KEY in task.input_context


def test_render_resume_block_none_without_answer():
    assert render_resume_block(_task(input_context={})) is None
    assert render_resume_block(_task(input_context={RESUME_KEY: {"answer": ""}})) is None
