"""F036 — a person's stop is a fact the machine may not undo.

Night 1 (2026-09-18): the persona blocked 13 tickets. The host received all 13
stops and killed each session — and then the tickets came back. Ticket 136 was
blocked at 18:18:06 UTC and re-claimed with a fresh session at 18:19:17, one
second after someone answered a question its dead session had asked. Ticket 231
came back at 23:56 off an approval granted before the stop.

``blocked`` meant two things with one spelling — a machine PARK (waiting for an
answer, an approval, the spend window) and a person's STOP — and every
auto-resume path read only the status. These tests pin the rule: an explicit
stop is recorded, the auto-resume paths refuse a stopped ticket, and a plain
machine park still resumes exactly as it did.
"""
from __future__ import annotations

from types import SimpleNamespace as NS

from services import cli_host_service as svc
from services.operator_stop import (
    OPERATOR_STOP_KEY,
    apply_explicit_status,
    operator_stop,
)


def _task(status="blocked", ref=None, task_id=136):
    return NS(id=task_id, workspace_id="ws-c1", status=status, assigned_agent_id=58,
              runtime_ref=dict(ref or {}), blocked_at="earlier", blocked_reason=None,
              lease_until=None, raw_prompt=None, description="roast schedule",
              title="OPS", review_feedback=None)


class _Db:
    """One task, counted commits; answers both query shapes the paths use."""

    def __init__(self, task=None):
        self.task = task
        self.commits = 0

    def query(self, _model):
        return self

    def filter(self, *_conds):
        return self

    def first(self):
        return self.task

    def get(self, _id):
        return self.task

    def commit(self):
        self.commits += 1

    def rollback(self):
        pass


def _answered_ask_grant(grant_id=86, task_id=136):
    return NS(id=grant_id, workspace_id="ws-c1",
              details={svc.SESSION_ASK_MARKER: {"task_id": task_id}},
              answer_text="Thursday is fine.", answered_by="user:1", kind="question")


# ── recording the stop ───────────────────────────────────────────────────────

def test_an_explicit_block_records_who_stopped_it_and_why():
    task = _task(status="in_progress")
    apply_explicit_status(task, "in_progress", "blocked", "not what I asked for", by="operator")
    stop = operator_stop(task)
    assert stop["status"] == "blocked"
    assert stop["reason"] == "not what I asked for"
    assert stop["by"] == "operator" and stop["at"]


def test_cancel_is_a_stop_too():
    task = _task(status="in_progress")
    apply_explicit_status(task, "in_progress", "cancelled", None, by="platform_tool")
    assert operator_stop(task)["status"] == "cancelled"


def test_only_a_person_moving_it_on_lifts_the_stop():
    task = _task(status="blocked")
    apply_explicit_status(task, "in_progress", "blocked", "hold", by="operator")
    apply_explicit_status(task, "blocked", "assigned", None, by="operator")
    assert operator_stop(task) is None
    assert OPERATOR_STOP_KEY not in task.runtime_ref


def test_recording_a_stop_rebuilds_runtime_ref_rather_than_mutating_it():
    original = {"session_id": "s-1"}
    task = _task(status="in_progress", ref=original)
    apply_explicit_status(task, "in_progress", "blocked", None, by="operator")
    assert OPERATOR_STOP_KEY not in original, "JSONB must be reassigned, never edited in place"
    assert task.runtime_ref["session_id"] == "s-1"


# ── night 1, ticket 136: the answer must not undo the stop ───────────────────

def test_ticket_136_an_answer_to_a_dead_sessions_question_does_not_undo_a_stop():
    # 18:17:21 — the session asks; the ask is on the ticket, the ticket still runs
    ref = svc.record_session_ask({"cli_session_id": "claude-136", "host_id": "h1"},
                                 grant_id=86, question="Which roast day?")
    task = _task(status="in_progress", ref=ref)
    # 18:18:06 — a person blocks it (the host then kills the session)
    apply_explicit_status(task, "in_progress", "blocked", "stop — wrong brief", by="operator")
    task.status = "blocked"
    # 18:19:16 — the question is answered
    resumed = svc.answer_session_ask(_Db(task), _answered_ask_grant())
    # 18:19:17 used to be a fresh session. Now:
    assert resumed is False
    assert task.status == "blocked"
    answered = svc.session_asks(task.runtime_ref)[0]
    assert answered["answer"] == "Thursday is fine." and answered["answered_at"], \
        "the answer is still recorded — it just does not restart the work"
    assert operator_stop(task) is not None, "the stop survives the answer"


def test_a_ticket_parked_for_its_question_still_resumes_on_the_answer():
    """The regression guard: the machine park is untouched by this fix."""
    ref = svc.record_session_ask({"cli_session_id": "claude-117", "host_id": "h1"},
                                 grant_id=86, question="Which notes?")
    task = _task(status="blocked", ref=ref)
    task.blocked_reason = svc.PARKED_FOR_ANSWER_REASON.format(grant_id=86)
    assert svc.answer_session_ask(_Db(task), _answered_ask_grant()) is True
    assert task.status == "assigned"


# ── night 1, ticket 231: a grant must not undo the stop ──────────────────────

def test_ticket_231_a_resolved_approval_does_not_requeue_a_stopped_ticket():
    from api.approval_grants import _requeue_blocked_task

    task = _task(status="in_progress", task_id=231)
    apply_explicit_status(task, "in_progress", "blocked", "hold everything", by="platform_tool")
    task.status = "blocked"
    assert _requeue_blocked_task(_Db(task), "ws-c1", 231) is False
    assert task.status == "blocked"


def test_a_ticket_parked_for_approval_still_requeues_when_granted():
    from api.approval_grants import _requeue_blocked_task

    task = _task(status="blocked", task_id=231)          # a machine park: no stop recorded
    assert _requeue_blocked_task(_Db(task), "ws-c1", 231) is True
    assert task.status == "assigned"
