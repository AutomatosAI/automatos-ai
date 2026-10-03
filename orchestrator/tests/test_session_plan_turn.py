"""PRD-253 Wave P — Plan on every CLI: the plan is a card, the answer resumes the session.

#845 gave sessions Claude Code's four permission modes, but Plan worked on Claude
Code only: its own plan mode presented the plan, and an approval inside the turn
let it carry on. Every other CLI ran Plan as Edit automatically, and a Claude
card nobody answered in time sent the ticket to review.

Now a plan turn on any CLI ends with the plan, the host sends it as a
``PlanReady`` event, and the backend files it as ONE Plan card — the Questions
tab, the bell, Telegram — and the turn's end parks the ticket on it. Approve
resumes the same session as Edit automatically with the plan in its prompt; the
operator's own words resume it planning again; Reject sends it to review. Each
claim reads the plan ledger for its mode, so the host never guesses.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

from core.session_permission_modes import claim_permission_mode
from services import cli_host_service as svc
from services import session_plans as plans

PLAN = "1. Add hello.txt\n2. Verify with cat"


def _task(status="in_progress", ref=None, task_id=301):
    return NS(id=task_id, workspace_id="ws-c1", status=status, assigned_agent_id=58,
              runtime_ref=dict(ref or {}), blocked_at=None, blocked_reason=None,
              lease_until="soon", raw_prompt=None, description="write hello.txt",
              title="Say hi", review_feedback=None, completed_at=None)


class _Db:
    """One ticket, one agent, counted commits and rollbacks."""

    def __init__(self, task=None, agent=None):
        self.task = task
        self.agent = agent
        self.commits = 0
        self.rollbacks = 0
        self._model = None

    def query(self, model):
        self._model = model
        return self

    def filter(self, *_conds):
        return self

    def first(self):
        return self.agent if getattr(self._model, "__name__", "") == "Agent" else self.task

    def refresh(self, _row):
        pass

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def _plan_event(text=PLAN, approved_in_turn=False):
    return {"event": plans.PLAN_EVENT, "text": text, "approved_in_turn": approved_in_turn}


@pytest.fixture
def staged(monkeypatch):
    """``stage_question`` as the Plan card meets it: records the call, files ask #900, #901, …"""
    calls = []

    async def fake_stage(_db, workspace_id, **kwargs):
        calls.append({"workspace_id": workspace_id, **kwargs})
        return {"success": True, "ask_id": 899 + len(calls), "parked": False}

    import modules.tools.discovery.handlers_asks as handlers
    monkeypatch.setattr(handlers, "stage_question", fake_stage)
    return calls


def _raise(task, events, db=None):
    db = db or _Db(task, NS(name="CODER"))
    return asyncio.run(plans.raise_session_plan(db, task, dict(task.runtime_ref), events))


def _awaiting(version=1, grant_id=900, **ref):
    entry = {"kind": plans.PLAN_KIND, "version": version, "attempt": version, "plan": PLAN,
             "grant_id": grant_id, "asked_at": "2026-10-02T10:00:00+00:00"}
    return {"cli_session_id": "codex-1", "host_id": "h1", "attempt": version, plans.PLANS_KEY: [entry], **ref}


def _answered(answer, version=1, grant_id=900):
    ref = _awaiting(version, grant_id)
    ref[plans.PLANS_KEY][0] = {**ref[plans.PLANS_KEY][0], "answer": answer, "answered_at": "2026-10-02T10:05:00+00:00"}
    return ref


def _grant(answer, grant_id=900, task_id=301, version=1):
    return NS(id=grant_id, workspace_id="ws-c1", answer_text=answer, kind="question",
              details={plans.PLAN_MARKER: {"task_id": task_id, "version": version, "attempt": version}})


# ── where a ticket's plan stands, and the mode each claim runs in ────────────

@pytest.mark.parametrize("answer, last_round, verdict", [
    ("Approve", False, plans.STATE_APPROVED), ("approved — keep the old API", False, plans.STATE_APPROVED),
    ("Reject", False, plans.STATE_REJECTED), ("rejected: wrong module", False, plans.STATE_REJECTED),
    ("Use the existing helper instead", False, plans.STATE_DISCUSSING),
    ("Use the existing helper instead", True, plans.STATE_REJECTED),      # the last round: only Approve carries on
    ("Approve", True, plans.STATE_APPROVED),
])
def test_the_answer_decides_the_verdict(answer, last_round, verdict):
    assert plans.plan_verdict(answer, last_round=last_round) == verdict


def test_the_latest_plan_decides_the_state():
    assert plans.plan_state({}) == plans.STATE_PLANNING
    assert plans.plan_state(_awaiting()) == plans.STATE_AWAITING
    assert plans.plan_state(_answered("Approve")) == plans.STATE_APPROVED
    assert plans.plan_state(_answered("split it in two")) == plans.STATE_DISCUSSING
    assert plans.plan_state(_answered("split it in two", version=plans.MAX_PLAN_ROUNDS)) == plans.STATE_REJECTED


@pytest.mark.parametrize("ticket_mode, approved, expected", [
    ("plan", False, "plan"), ("plan", True, "edits"),
    ("edits", True, "edits"), ("manual", True, "manual"), ("auto", False, "auto"),
])
def test_a_plan_ticket_plans_until_its_plan_is_approved(ticket_mode, approved, expected):
    assert claim_permission_mode(ticket_mode, approved) == expected


# ── the plan arrives with the turn's last events ─────────────────────────────

def test_a_plan_from_the_turn_becomes_one_plan_card(staged):
    task = _task(ref={"attempt": 3, "host_id": "h1"})
    ref = _raise(task, [{"event": "Stop"}, _plan_event()])
    assert len(staged) == 1
    card = staged[0]
    # filed UNPARKED through PRD-225's shared internals, with the marker the answer reads
    assert card["park"] is None and card["subject_type"] == "board_task" and card["subject_id"] == "301"
    assert card["details"] == {plans.PLAN_MARKER: {"task_id": 301, "version": 1, "attempt": 3}}
    assert card["options"] == ["Approve", "Reject"]
    assert "CODER has a plan for ticket #301 — Say hi" in card["question"] and PLAN in card["question"]
    assert "answer in your own words" in card["question"]          # a Telegram reply sees no buttons
    entry = plans.session_plans(ref)[0]
    assert entry["grant_id"] == 900 and entry["version"] == 1 and entry["plan"] == PLAN
    assert not entry.get("answered_at")
    assert plans.plan_state(task.runtime_ref) == plans.STATE_AWAITING


def test_a_reflushed_plan_files_nothing_twice(staged):
    task = _task(ref={"attempt": 3})
    _raise(task, [_plan_event()])
    _raise(task, [_plan_event()])                                    # the host retried the same batch
    assert len(staged) == 1 and len(plans.session_plans(task.runtime_ref)) == 1


def test_a_flush_without_a_plan_changes_nothing(staged):
    task = _task(ref={"attempt": 3})
    assert _raise(task, [{"event": "PreToolUse", "tool_name": "Read"}, _plan_event(text="   ")]) == {"attempt": 3}
    assert staged == []


def test_a_plan_approved_inside_the_turn_is_recorded_with_no_card(staged):
    """Claude Code's ExitPlanMode card, answered in time: the session is already at work."""
    task = _task(ref={"attempt": 3})
    _raise(task, [_plan_event(approved_in_turn=True)])
    assert staged == []
    entry = plans.session_plans(task.runtime_ref)[0]
    assert entry["approved_in_turn"] is True and entry["folded_at"]
    assert plans.plan_state(task.runtime_ref) == plans.STATE_APPROVED
    assert svc._park_for_answer(_Db(task), task, dict(task.runtime_ref)) is None   # it finishes the normal way


def test_a_card_that_cannot_be_filed_never_breaks_the_flush(monkeypatch):
    async def boom(*_a, **_k):
        raise RuntimeError("the grants table is unhappy")

    import modules.tools.discovery.handlers_asks as handlers
    monkeypatch.setattr(handlers, "stage_question", boom)
    task = _task(ref={"attempt": 3})
    db = _Db(task)
    assert _raise(task, [_plan_event()], db) == {"attempt": 3}
    assert db.rollbacks == 1 and plans.session_plans(task.runtime_ref) == []


def test_the_last_round_says_so(staged):
    earlier = [{"kind": plans.PLAN_KIND, "version": v, "attempt": v, "plan": "p", "grant_id": 800 + v,
                "answer": "tighter", "answered_at": "t", "folded_at": "t"} for v in range(1, plans.MAX_PLAN_ROUNDS)]
    task = _task(ref={"attempt": 9, plans.PLANS_KEY: earlier})
    _raise(task, [_plan_event()])
    assert f"round {plans.MAX_PLAN_ROUNDS} of {plans.MAX_PLAN_ROUNDS}" in staged[0]["question"]
    assert "This is the last round" in staged[0]["question"]


def test_a_newer_plan_supersedes_a_card_still_open(staged, monkeypatch):
    closed = []
    import core.services.approval_grants as grants
    monkeypatch.setattr(grants, "expire_pending_grants", lambda db, ws, ids, **kw: closed.append(list(ids)) or len(ids))
    task = _task(ref=_awaiting(version=1, grant_id=800, attempt=4))   # Run Now on a ticket parked on plan 1
    _raise(task, [_plan_event(text="a better plan")])
    assert closed == [[800]]
    old, new = plans.session_plans(task.runtime_ref)
    assert old["answer"] == plans.SUPERSEDED_ANSWER and old["folded_at"]          # nothing resumes on it
    assert new["version"] == 2 and new["grant_id"] == 900 and plans.plan_state(task.runtime_ref) == plans.STATE_AWAITING


# ── the turn's end parks on it ───────────────────────────────────────────────

def test_a_turn_that_ends_with_its_plan_open_parks_on_the_plan_card():
    task = _task(ref=_awaiting())
    assert svc._park_for_answer(_Db(task), task, dict(task.runtime_ref)) == "blocked"
    assert task.blocked_reason == plans.PARKED_FOR_PLAN_REASON.format(grant_id=900)
    assert task.runtime_ref["resume_session_id"] == "codex-1" and task.lease_until is None


def test_a_plan_approved_while_the_turn_ran_goes_straight_back_to_work():
    task = _task(ref=_answered("Approve"))
    assert svc._park_for_answer(_Db(task), task, dict(task.runtime_ref)) == "assigned"


def test_a_plan_rejected_while_the_turn_ran_goes_to_review():
    task = _task(ref=_answered("Reject — wrong module"))
    assert svc._park_for_answer(_Db(task), task, dict(task.runtime_ref)) == "review"
    assert task.review_feedback == "Plan rejected: Reject — wrong module"
    assert plans.session_plans(task.runtime_ref)[0]["folded_at"]


def test_a_mid_turn_answer_to_the_plan_is_folded_in_before_the_park_decides():
    task = _task(ref=_awaiting())
    answered = _answered("Approve")[plans.PLANS_KEY]

    class _RowDb(_Db):
        def execute(self, *_a, **_k):        # what the answer route committed meanwhile
            return NS(first=lambda: ({plans.PLANS_KEY: answered},))

    db = _RowDb(task)
    merged = svc._merge_fresh_session_asks(db, task, dict(task.runtime_ref))
    assert plans.plan_state(merged) == plans.STATE_APPROVED
    assert svc._park_for_answer(db, task, merged) == "assigned"


# ── the operator answers ─────────────────────────────────────────────────────

def test_approve_resumes_the_session_that_made_the_plan():
    task = _task(status="blocked", ref=_awaiting())
    assert plans.answer_session_plan(_Db(task), _grant("Approve")) is True
    assert task.status == "assigned" and task.blocked_reason is None
    assert task.runtime_ref["resume_session_id"] == "codex-1"
    assert plans.plan_state(task.runtime_ref) == plans.STATE_APPROVED


def test_the_operators_own_words_resume_it_planning_again():
    task = _task(status="blocked", ref=_awaiting())
    assert plans.answer_session_plan(_Db(task), _grant("Reuse the existing helper")) is True
    assert task.status == "assigned"
    assert plans.plan_state(task.runtime_ref) == plans.STATE_DISCUSSING


def test_reject_sends_the_ticket_to_review():
    task = _task(status="blocked", ref=_awaiting())
    assert plans.answer_session_plan(_Db(task), _grant("Reject: not this quarter")) is False
    assert task.status == "review" and task.review_feedback == "Plan rejected: Reject: not this quarter"
    assert task.blocked_reason is None and "resume_session_id" not in task.runtime_ref


def test_on_the_last_round_anything_but_approve_rejects():
    task = _task(status="blocked", ref=_awaiting(version=plans.MAX_PLAN_ROUNDS, grant_id=905))
    assert plans.answer_session_plan(_Db(task), _grant("one more tweak", grant_id=905)) is False
    assert task.status == "review"


def test_an_answer_while_the_turn_still_runs_is_only_recorded():
    task = _task(status="in_progress", ref=_awaiting())
    assert plans.answer_session_plan(_Db(task), _grant("Approve")) is False
    assert task.status == "in_progress" and plans.plan_state(task.runtime_ref) == plans.STATE_APPROVED


def test_a_card_that_is_not_the_open_plan_changes_nothing():
    task = _task(status="blocked", ref=_awaiting(version=2, grant_id=901))
    before = dict(task.runtime_ref)
    assert plans.answer_session_plan(_Db(task), _grant("Approve", grant_id=900)) is False      # plan 1's card
    assert task.runtime_ref == before and task.status == "blocked"
    assert plans.answer_session_plan(_Db(None), _grant("Approve", grant_id=901)) is False      # another workspace


def test_only_a_plan_card_takes_the_plan_path():
    assert plans.plan_marker(_grant("x"))["task_id"] == 301
    assert plans.plan_marker(NS(details={plans.PLAN_MARKER: {}})) is None
    assert plans.plan_marker(NS(details=None)) is None
    ask = NS(details={svc.SESSION_ASK_MARKER: {"task_id": 301}})
    assert plans.plan_marker(ask) is None and svc.session_ask_marker(_grant("x")) is None


def test_the_answer_route_sends_a_plan_card_to_the_plan_path(monkeypatch):
    from api.approval_grants import _requeue_subject

    seen = []
    monkeypatch.setattr(plans, "answer_session_plan", lambda db, grant: seen.append(grant.id) or True)
    grant = _grant("Approve")
    grant.subject_type = "board_task"
    assert asyncio.run(_requeue_subject(_Db(None), grant)) is True and seen == [900]


# ── the session that picks the work up ───────────────────────────────────────

def test_an_approved_plan_is_in_the_prompt_once():
    task = _task(ref=_answered("Approve — keep the old API"))
    prompt = svc._ticket_prompt(task)
    assert "## Your plan was approved — implement it now" in prompt
    assert "**The operator added:** keep the old API" in prompt and PLAN in prompt
    folded = {**task.runtime_ref, plans.PLANS_KEY: svc._mark_answers_folded(task.runtime_ref[plans.PLANS_KEY])}
    assert plans.plan_fold_in(folded) == ""


def test_feedback_on_the_plan_is_in_the_prompt():
    prompt = svc._ticket_prompt(_task(ref=_answered("Reuse the existing helper")))
    assert "## Feedback on your plan — revise it and present it again" in prompt
    assert "**The operator's answer:** Reuse the existing helper" in prompt and PLAN in prompt
    assert "## Your plan was approved" not in prompt
    assert plans.plan_fold_in(_awaiting()) == ""                      # an open card says nothing yet


def _claim(monkeypatch, ref, workspace_mode="plan"):
    task = _task(status="assigned", ref=ref)
    task.attempts, task.review_mode, task.attachment_ids = 2, "auto", []
    monkeypatch.setattr(svc, "claim_tasks", lambda db, **kw: [task])
    monkeypatch.setattr(svc, "_blocked_pending_approval", lambda db, t: False)
    monkeypatch.setattr(svc, "served_providers_of", lambda h: None)
    monkeypatch.setattr(svc, "session_permission_mode", lambda db, ws: workspace_mode)
    monkeypatch.setattr(svc, "default_session_folder", lambda db, ws: "/tmp/projects")
    monkeypatch.setattr(svc, "explorer_root_for", lambda *a, **k: None)
    monkeypatch.setattr(svc, "_session_system_prompt", lambda agent: "")
    monkeypatch.setattr(svc, "session_tool_names", lambda: ("board_summary",))
    return task, svc.claim_for_host(_Db(None), NS(id="h1", workspace_id="ws-c1"), limit=1)["tasks"][0]


def test_the_claim_after_approve_works_as_edit_automatically_with_the_plan(monkeypatch):
    task, claimed = _claim(monkeypatch, _answered("Approve"))
    assert claimed["permission_mode"] == "edits" and claimed["plan_approved"] is True
    assert "## Your plan was approved — implement it now" in claimed["prompt"] and PLAN in claimed["prompt"]
    kept = plans.session_plans(task.runtime_ref)
    assert [p["grant_id"] for p in kept] == [900] and kept[0]["folded_at"]         # carried, and shown once


@pytest.mark.parametrize("ref", [{}, _answered("Reuse the existing helper")])
def test_a_plan_ticket_plans_until_approved(monkeypatch, ref):
    _, claimed = _claim(monkeypatch, ref)
    assert claimed["permission_mode"] == "plan" and claimed["plan_approved"] is False


def test_other_modes_are_untouched_by_a_plan(monkeypatch):
    _, claimed = _claim(monkeypatch, _answered("Approve"), workspace_mode="manual")
    assert claimed["permission_mode"] == "manual" and claimed["plan_approved"] is True
