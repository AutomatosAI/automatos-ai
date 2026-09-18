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
    ClarificationAskNotPlaced,
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
def stub_stage_question(monkeypatch):
    """Stub 225's SHARED ``stage_question``, recording the kwargs it was called
    with (reuse proof). The ladder reaches this directly, NOT through the
    ``platform_ask_human`` tool — that tool refuses every non-board_task subject
    (see test_ask_human_still_refuses_a_tool_call_subject), which is what left a
    clarification park stranded behind a question nobody was asked."""
    seen = {}

    async def _stage(db, workspace_id, **kwargs):
        seen.update(kwargs)
        seen["workspace_id"] = workspace_id
        return {"success": True, "ask_id": 99, "parked": True}

    monkeypatch.setattr(ha, "stage_question", _stage)
    return seen


@pytest.fixture
def stub_stage_question_returning_nothing(monkeypatch):
    """The stager answers without an ask_id — no question row was filed."""
    async def _stage(db, workspace_id, **kwargs):
        return {"success": False, "error": "refused"}

    monkeypatch.setattr(ha, "stage_question", _stage)


# ---------------------------------------------------------------------------
# escalate: reuse 225 ask_human, park, labelled draft
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_escalate_reuses_the_shared_stager_and_parks_with_draft(stub_stage_question, spy_events):
    task = _task()
    subject = _subject(task)

    result = await escalate_clarification(MagicMock(), subject, "Which vendor?", agent_name="QUILL")

    # reuse: the SAME 225 internals, subject_type tool_call carrying the task id
    assert stub_stage_question["subject_type"] == "tool_call"
    assert stub_stage_question["subject_id"] == "task-1"
    assert stub_stage_question["question"] == "Which vendor?"
    assert stub_stage_question["asked_by_agent_id"] == 5
    assert stub_stage_question["agent_name"] == "QUILL"
    # park=None: the ladder parks its OWN OrchestrationTask below. ask_human's
    # park flips a BoardTask status, which this subject is not.
    assert stub_stage_question["park"] is None

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
async def test_escalate_passes_partial_output_when_given(stub_stage_question, spy_events):
    task = _task(output="")
    subject = _subject(task)
    await escalate_clarification(
        MagicMock(), subject, "Q?", partial_output="the agent's in-progress work",
    )
    assert task.output_metadata[DRAFT_KEY]["partial_output"] == "the agent's in-progress work"


# ---------------------------------------------------------------------------
# the ask has to actually reach a human (the regression these tests missed)
#
# The suite above stubs the ask, so for two PRDs it proved the ladder CALLS the
# ask internals without ever proving the internals ACCEPT what it sends. They
# did not: platform_ask_human refuses every non-board_task subject, the ladder
# read ask_id off a refusal dict as None, parked the task anyway and told the
# agent "asked a human (ask #None)". No grant row, so nothing in the Questions
# tab, no Telegram message, and no answer could ever resume it. The three tests
# below pin the contract from both sides.
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_ask_human_still_refuses_a_tool_call_subject():
    """The REAL tool, unstubbed: why the ladder must not route through it.

    ``platform_ask_human``'s refusal is deliberate (P225-RVW-11) and stays —
    ITS answer path re-dispatches a stored call, which a clarification park has
    none of. The ladder's answer path is different and real
    (``_resume_clarification_if_parked``), which is why it reaches the shared
    stager directly instead of asking this tool to relax.
    """
    result = await ha.ask_human(MagicMock(), uuid4(), {
        "subject_type": "tool_call",
        "subject_id": "task-1",
        "question": "Which vendor?",
    })
    assert result["success"] is False
    assert result["parked"] is False
    assert "ask_id" not in result


class _FakeSession:
    """Enough session to run the REAL ``create_grant``: rows in a list, ids on add."""

    def __init__(self):
        self.rows = []
        self.commits = 0

    def add(self, obj):
        if getattr(obj, "id", None) is None:
            obj.id = len(self.rows) + 1
        self.rows.append(obj)

    def flush(self):
        pass

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass

    def refresh(self, obj):
        pass

    def query(self, model):
        # Only the cascade count reaches here, and its caller swallows the error
        # on purpose — a count must never unfile an ask that is already committed.
        raise RuntimeError("no queries in this fake")


@pytest.mark.asyncio
async def test_the_real_ask_internals_accept_a_tool_call_subject(monkeypatch):
    """The other half of the fix, and the half the old tests never had.

    Every other test here stubs ``stage_question`` with a fake that accepts
    anything, so they prove the ladder CALLS it — not that it is ALLOWED to. That
    is the exact gap that let the bug ship: the collaborator was stubbed, so the
    refusal on the other side was invisible for two PRDs. This runs the REAL
    ``stage_question`` (and the real ``create_grant`` under it) with the subject
    the ladder actually sends, so a subject-type whitelist appearing in
    ``create_grant`` — the standing precedent is ``handlers_asks._SUBJECTS`` —
    fails HERE, loudly, instead of silently degrading every escalation to
    proceed-with-assumption.
    """
    from core.models.approval_grants import KIND_QUESTION

    bell, telegram = [], []

    async def _bell(db, workspace_id, **kw):
        bell.append(kw)

    async def _telegram(db, workspace_id, grant, **kw):
        telegram.append(grant.id)

    monkeypatch.setattr(ha, "_dispatch_question_pending", _bell)
    monkeypatch.setattr(ha, "_capture_question_telegram", _telegram)
    # The cascade count imports inside the function, so it cannot be patched from
    # here — it runs for real against the fake session, raises, and is swallowed
    # by the guard that exists so a count can never unfile a committed ask.

    db = _FakeSession()
    ws = uuid4()

    res = await ha.stage_question(
        db, ws,
        subject_type="tool_call", subject_id="task-1",
        question="Which vendor?", asked_by_agent_id=5, agent_name="QUILL", park=None,
    )

    # a real row, not a refusal dict read as None
    assert res["success"] is True
    assert isinstance(res["ask_id"], int)
    # park=None: the ladder parks its own OrchestrationTask, nothing here does
    assert res["parked"] is False

    grants = [r for r in db.rows if getattr(r, "kind", None) == KIND_QUESTION]
    assert len(grants) == 1
    assert grants[0].subject_type == "tool_call"
    assert grants[0].subject_id == "task-1"
    assert grants[0].id == res["ask_id"]
    # the human is actually told
    assert bell and telegram == [res["ask_id"]]


@pytest.mark.asyncio
async def test_escalate_does_not_park_when_no_question_row_was_filed(
    stub_stage_question_returning_nothing, spy_events,
):
    """No ask id means nobody was asked — raise instead of parking."""
    task = _task()
    subject = _subject(task)

    with pytest.raises(ClarificationAskNotPlaced):
        await escalate_clarification(MagicMock(), subject, "Which vendor?")

    # the task is untouched: no draft, no awaiting-answer marker, no trail entry
    assert DRAFT_KEY not in (task.output_metadata or {})
    assert PENDING_KEY not in (task.input_context or {})
    assert spy_events == []


@pytest.mark.asyncio
async def test_handler_falls_back_when_the_ask_internals_file_nothing(
    stub_stage_question_returning_nothing, monkeypatch,
):
    """End to end through the REAL ladder: a task that cannot be asked about
    proceeds with an assumption rather than parking behind a phantom ask."""
    async def _cannot(db, subject, question, *, category=None):
        return {"cannot_answer": True, "reason": "no_context"}

    monkeypatch.setattr(oa, "answer_clarification", _cannot)
    task = _task()
    monkeypatch.setattr(hc, "_load_task", lambda db, run_id, workspace_id, task_id: task)

    out = await hc.ask_orchestrator(MagicMock(), uuid4(), _server_params())

    assert out["proceed_with_assumption"]
    assert "parked" not in out
    assert PENDING_KEY not in (task.input_context or {})


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
async def test_full_loop_ask_park_answer_resume(stub_stage_question, spy_events):
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
async def test_resume_waits_while_grant_still_pending(stub_stage_question):
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


# ---------------------------------------------------------------------------
# P229-RVW-5 — park+draft is COMMITTED (survives a sibling rollback) and the
# escalation is exception-safe once the ask is placed (never double-asks)
# ---------------------------------------------------------------------------

class _CommitAwareSession:
    """Models commit/rollback on the task JSONB: rollback() reverts UNcommitted
    mutations back to the last committed snapshot, exactly as SQLAlchemy would on
    the shared per-tick session. Proves the park+draft is COMMITTED (durable), not
    merely flushed — without escalate_clarification's db.commit(), the sibling
    rollback below wipes the draft and the assertions fail."""

    def __init__(self, task):
        import copy

        self._task = task
        self._copy = copy.deepcopy
        self._committed = self._snapshot()

    def _snapshot(self):
        return (
            self._copy(getattr(self._task, "output_metadata", None)),
            self._copy(getattr(self._task, "input_context", None)),
        )

    def commit(self):
        self._committed = self._snapshot()

    def rollback(self):
        om, ic = self._committed
        self._task.output_metadata = self._copy(om)
        self._task.input_context = self._copy(ic)

    def flush(self):
        pass


@pytest.mark.asyncio
async def test_escalate_commits_draft_survives_sibling_rollback(stub_stage_question, spy_events):
    # escalate parks + drafts, then a SIBLING task's tool error rolls back the
    # SHARED session (platform_executor.py:1245). The draft must survive because
    # escalate_clarification COMMITTED it (P229-RVW-5) — as durable as the ask.
    task = _task()
    db = _CommitAwareSession(task)

    await escalate_clarification(db, _subject(task), "Which vendor?")

    db.rollback()  # a sibling's failure rolls back the shared per-tick session

    assert task.output_metadata[DRAFT_KEY]["ask_id"] == 99      # draft survived
    assert task.input_context[PENDING_KEY]["ask_id"] == 99      # marker survived


@pytest.mark.asyncio
async def test_escalate_marks_ask_placed_when_park_fails_after_ask(stub_stage_question, spy_events, monkeypatch):
    # A throw AFTER ask_human has placed the (committed) ask must NOT surface as a
    # failure — the human WAS asked, and a bare failure would make a retrying agent
    # file a DUPLICATE ask. escalate swallows it, discards the half-written park,
    # and still returns {parked, ask_id}.
    def _boom(*a, **k):
        raise RuntimeError("JSONB write failed mid-park")

    monkeypatch.setattr(cl, "_park_task_with_draft", _boom)
    db = MagicMock()

    result = await escalate_clarification(db, _subject(_task()), "Q?")

    assert result["parked"] is True          # ask marked placed, not a failure
    assert result["ask_id"] == 99
    db.commit.assert_not_called()            # commit never reached...
    db.rollback.assert_called_once()         # ...the half-written park is discarded


@pytest.mark.asyncio
async def test_handler_falls_back_when_ask_never_placed(monkeypatch):
    # escalate_clarification RAISES only when ask_human itself failed — i.e. NO ask
    # was placed. The handler must fall back to proceed-with-assumption (nothing to
    # orphan; a retry re-attempts a fresh ask, it does not double-ask).
    async def _cannot(db, subject, question, *, category=None):
        return {"cannot_answer": True, "reason": "unretrievable"}

    async def _boom(db, subject, question, *, category=None, partial_output=None, agent_name=None):
        raise RuntimeError("ask_human DB error — no ask was placed")

    monkeypatch.setattr(oa, "answer_clarification", _cannot)
    monkeypatch.setattr(cl, "escalate_clarification", _boom)

    result = await hc.ask_orchestrator(MagicMock(), uuid4(), _server_params())

    assert result["success"] is True
    assert "proceed_with_assumption" in result
    assert "parked" not in result
