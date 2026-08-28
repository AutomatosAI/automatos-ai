"""PRD-225 US-003 — answer → resume, dismiss, and the approve/answer split.

Pure tests over a fake session (the test_p2w2_grant_resume pattern), calling the
endpoint functions directly with monkeypatched board-dispatch + chat seams:

  - answering a board-task question re-queues the parked task and lands the Q&A
    in planning_data.human_qa (rebuild-don't-mutate);
  - answering a tool_call question runs the _resume_tool_call path with the Q&A
    on the resume payload;
  - dismissing (deny) a question leaves the subject BLOCKED, trail intact;
  - approving a question is rejected — /answer is its only completion path;
  - the chat confirmation fires via deliver_background_message.
"""
from __future__ import annotations

import os
import sys
import uuid
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

from pathlib import Path

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from fastapi import HTTPException

from core.models.approval_grants import ApprovalGrant, GrantStatus, KIND_QUESTION
from core.models.core import BoardTask


# ---------------------------------------------------------------------------
# Fake session — .filter().first() (for _load_grant) + .get(pk) (board task).
# ---------------------------------------------------------------------------

class _Query:
    def __init__(self, rows):
        self._rows = list(rows)

    def get(self, pk):
        for r in self._rows:
            if getattr(r, "id", None) == pk:
                return r
        return None

    def filter(self, *conds):
        rows = self._rows
        for cond in conds:
            key = cond.left.key
            value = getattr(cond.right, "value", None)
            rows = [r for r in rows if str(getattr(r, key, None)) == str(value)]
        return _Query(rows)

    def order_by(self, *a):
        return _Query(list(reversed(self._rows)))

    def first(self):
        return self._rows[0] if self._rows else None

    def all(self):
        return list(self._rows)


class _FakeSession:
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

    def query(self, model):
        return _Query([r for r in self.rows if isinstance(r, model)])


@pytest.fixture()
def ws():
    return uuid.uuid4()


@pytest.fixture()
def ctx(ws):
    return SimpleNamespace(workspace_id=ws, user_id=5, internal_user_id=5)


@pytest.fixture()
def confirmed(monkeypatch):
    """Capture the chat confirmation without touching the DB."""
    calls = []
    monkeypatch.setattr(
        "services.chat_messenger.deliver_background_message",
        lambda db, **kw: calls.append(kw),
    )
    return calls


@pytest.fixture(autouse=True)
def _no_board_notify(monkeypatch):
    """The board re-queue's NOTIFY is infra — stub it for the pure test."""
    monkeypatch.setattr(
        "services.board_dispatcher.notify_task_available",
        lambda *a, **k: None,
    )


def _question(db, ws, *, subject_type, subject_id, question="Ship A or B?", options=None):
    g = ApprovalGrant(
        workspace_id=ws, subject_type=subject_type, subject_id=str(subject_id),
        kind=KIND_QUESTION, question_md=question, options=options,
        status=GrantStatus.PENDING.value,
    )
    db.add(g)
    return g


# ===========================================================================
# 1. Answer a board-task question → re-queue + Q&A in planning_data
# ===========================================================================

@pytest.mark.asyncio
async def test_answer_requeues_board_task_and_records_qa(ws, ctx, confirmed):
    from api.approval_grants import answer_question, AnswerRequest

    db = _FakeSession()
    task = BoardTask(
        id=42, workspace_id=ws, title="T", status="blocked",
        blocked_reason="Awaiting human answer (ask #1)",
    )
    task.blocked_at = None
    db.rows.append(task)
    grant = _question(db, ws, subject_type="board_task", subject_id="42")

    res = await answer_question(grant.id, AnswerRequest(answer_text="Ship A"), ctx, db)

    assert grant.status == GrantStatus.GRANTED.value
    assert grant.answer_text == "Ship A"
    assert grant.answered_by == "user:5"
    assert grant.answered_at is not None
    # The parked task is re-queued.
    assert task.status == "assigned"
    assert task.blocked_reason is None
    # The Q&A landed in the run context (rebuild-don't-mutate).
    qa = task.planning_data["human_qa"]
    assert qa == [{"q": "Ship A or B?", "a": "Ship A", "answered_by": "user:5", "at": qa[0]["at"]}]
    assert res["grant"]["status"] == GrantStatus.GRANTED.value


@pytest.mark.asyncio
async def test_answer_appends_to_existing_human_qa(ws, ctx, confirmed):
    """A re-ask appends — the prior Q&A is preserved, not overwritten."""
    from api.approval_grants import answer_question, AnswerRequest

    db = _FakeSession()
    task = BoardTask(id=7, workspace_id=ws, title="T", status="blocked")
    task.planning_data = {"human_qa": [{"q": "old", "a": "prior", "answered_by": "user:1", "at": "t0"}]}
    db.rows.append(task)
    grant = _question(db, ws, subject_type="board_task", subject_id="7", question="new?")

    await answer_question(grant.id, AnswerRequest(answer_text="fresh"), ctx, db)
    qa = task.planning_data["human_qa"]
    assert len(qa) == 2
    assert qa[0]["a"] == "prior" and qa[1]["a"] == "fresh"


# ===========================================================================
# 2. Answer a tool_call question → _resume_tool_call path + Q&A on payload
# ===========================================================================

@pytest.mark.asyncio
async def test_answer_resumes_tool_call(ws, ctx, confirmed):
    from api.approval_grants import answer_question, AnswerRequest

    db = _FakeSession()
    grant = _question(db, ws, subject_type="tool_call", subject_id="call-1")

    await answer_question(grant.id, AnswerRequest(answer_text="use vendor X"), ctx, db)

    assert grant.status == GrantStatus.GRANTED.value
    # The Q&A rode the resume payload (details), rebuild-don't-mutate.
    assert grant.details["human_qa"][0]["a"] == "use vendor X"
    # _resume_tool_call ran (no stored action ⇒ an honest no-op summary).
    assert "executed_result" in grant.details


# ===========================================================================
# 3. Chosen option must be a real choice; option answers work
# ===========================================================================

@pytest.mark.asyncio
async def test_option_answer_and_validation(ws, ctx, confirmed):
    from api.approval_grants import answer_question, AnswerRequest

    db = _FakeSession()
    grant = _question(db, ws, subject_type="tool_call", subject_id="c2", options=["A", "B"])

    ok = await answer_question(grant.id, AnswerRequest(option="B"), ctx, db)
    assert ok["grant"]["answer_text"] == "B"

    db2 = _FakeSession()
    g2 = _question(db2, ws, subject_type="tool_call", subject_id="c3", options=["A", "B"])
    with pytest.raises(HTTPException) as ei:
        await answer_question(g2.id, AnswerRequest(option="Z"), ctx, db2)
    assert ei.value.status_code == 422
    assert "option" in ei.value.detail


@pytest.mark.asyncio
async def test_empty_answer_rejected(ws, ctx, confirmed):
    from api.approval_grants import answer_question, AnswerRequest

    db = _FakeSession()
    grant = _question(db, ws, subject_type="tool_call", subject_id="c4")
    with pytest.raises(HTTPException) as ei:
        await answer_question(grant.id, AnswerRequest(answer_text="   "), ctx, db)
    assert ei.value.status_code == 422


# ===========================================================================
# 4. Dismiss (deny) a question → subject stays BLOCKED, trail intact
# ===========================================================================

@pytest.mark.asyncio
async def test_dismiss_leaves_subject_blocked(ws, ctx):
    from api.approval_grants import deny_approval

    db = _FakeSession()
    task = BoardTask(id=99, workspace_id=ws, title="T", status="blocked",
                     blocked_reason="Awaiting human answer (ask #1)")
    db.rows.append(task)
    grant = _question(db, ws, subject_type="board_task", subject_id="99")

    res = await deny_approval(grant.id, ctx, db)

    assert grant.status == GrantStatus.DENIED.value
    # Baked decision: dismiss does NOT fail the subject — it stays blocked.
    assert task.status == "blocked"
    assert task.blocked_reason == "Awaiting human answer (ask #1)"
    # Trail intact: the question row survives with no fabricated answer.
    assert grant.question_md == "Ship A or B?"
    assert grant.answer_text is None
    assert res["grant"]["status"] == GrantStatus.DENIED.value


# ===========================================================================
# 5. Approve on a question is rejected — /answer is the only completion path
# ===========================================================================

@pytest.mark.asyncio
async def test_grant_rejects_question(ws, ctx):
    from api.approval_grants import grant_approval

    db = _FakeSession()
    grant = _question(db, ws, subject_type="tool_call", subject_id="c5")
    with pytest.raises(HTTPException) as ei:
        await grant_approval(grant.id, ctx, db)
    assert ei.value.status_code == 422
    assert "question" in ei.value.detail.lower()


@pytest.mark.asyncio
async def test_answer_rejects_non_question(ws, ctx, confirmed):
    """Answering an approval-kind grant is a 422 — the split holds both ways."""
    from api.approval_grants import answer_question, AnswerRequest

    db = _FakeSession()
    approval = ApprovalGrant(
        workspace_id=ws, subject_type="board_task", subject_id="1",
        status=GrantStatus.PENDING.value,  # kind defaults to 'approval'
    )
    db.add(approval)
    with pytest.raises(HTTPException) as ei:
        await answer_question(approval.id, AnswerRequest(answer_text="x"), ctx, db)
    assert ei.value.status_code == 422


# ===========================================================================
# 6. Chat confirmation fires via deliver_background_message
# ===========================================================================

@pytest.mark.asyncio
async def test_answer_confirms_into_chat(ws, ctx, confirmed):
    """A board-task answer genuinely resumes the parked task, so the confirmation
    says 'resuming' and carries the question link."""
    from api.approval_grants import answer_question, AnswerRequest

    db = _FakeSession()
    task = BoardTask(id=6, workspace_id=ws, title="T", status="blocked")
    db.rows.append(task)
    grant = _question(db, ws, subject_type="board_task", subject_id="6")
    await answer_question(grant.id, AnswerRequest(answer_text="go"), ctx, db)

    assert len(confirmed) == 1
    call = confirmed[0]
    assert call["link_type"] == "question"
    assert call["link_id"] == str(grant.id)
    assert "resuming" in call["text"].lower()


@pytest.mark.asyncio
async def test_confirmation_is_honest_when_no_resume(ws, ctx, confirmed):
    """A channel trust-gate hold has NO resume path: answering it records the
    answer, but the chat confirmation must NOT claim the work is 'resuming' —
    it says the answer was recorded with nothing to auto-resume (P225-RVW-11)."""
    from api.approval_grants import answer_question, AnswerRequest

    db = _FakeSession()
    grant = _question(db, ws, subject_type="channel", subject_id="chan-1")
    await answer_question(grant.id, AnswerRequest(answer_text="route it"), ctx, db)

    assert grant.status == GrantStatus.GRANTED.value  # the answer is recorded
    assert len(confirmed) == 1
    text = confirmed[0]["text"].lower()
    assert "resuming" not in text     # never a false resume claim
    assert "recorded" in text         # honest: recorded, nothing to auto-resume
