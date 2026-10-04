"""F259 (night 7, "Moving cards by hand") — every drag that needs a decision names its button.

PRD-252 R6 refused only Review → Done ("Use Approve") and Review → Assigned ("Use
Reject"). The board still took:

1. Inbox → Done on a ticket nobody had worked on (#0044): Done, no result.
2. Inbox → Review on an empty ticket (#0045): a review of nothing, counted by Needs-you.
3. In progress → Review or Done mid-run (#0094 at 8 s, #0111 at 5 s): the agent kept
   working, billed, and its answer never reached the card.
4. Blocked → In progress (#0059): the move was the owner's approval (PRD-252 R6,
   PRD-234), but nothing said so, and "Awaiting human approval" stayed on the card to
   Done. Underneath it, the same consent "granted" ANY pending grant on the ticket,
   so an agent's open question would have left the Questions tab unanswered.
5. A failed playbook card → In progress (#0082): "Assign an agent first", though a
   playbook's card never has one.
6. Done → Inbox kept its completion time (#0044).
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace as NS

import pytest
from fastapi import HTTPException

from services.board_drag_rules import NO_AGENT_NO_PROGRESS, drag_refusal
from tests import test_f243_a_redo_runs_on_the_same_card as f243
from tests import test_prd252_drags_match_buttons as r6

board = r6.board              # PRD-252 R6's drag route over a seeded ticket
engine = f243.engine          # F116's Postgres, for the playbook card and the grants below
workspace = f243.workspace
launched = f243.launched
CTX, _ticket = r6.CTX, r6._ticket

APPROVAL_BLOCK = "Awaiting human approval (grant #1142): board task requires approval under 'always_ask' policy"


def _never_worked(status="inbox", **over):
    return _ticket(status, **{"workspace_seq": 44, "assigned_agent_id": None, "result": None,
                              "completed_at": None, **over})


def _running(**over):
    lease = datetime.now(timezone.utc) + timedelta(minutes=10)   # the dispatch lease a live run renews
    return _ticket("in_progress", **{"workspace_seq": 94, "lease_until": lease, "completed_at": None,
                                     "result": None, **over})


# --- the rules ----------------------------------------------------------------------------


@pytest.mark.parametrize("to, what", [("done", "approve"), ("review", "review")])
def test_a_ticket_nobody_worked_on_has_nothing_to_approve_or_review(to, what):
    refusal = drag_refusal(_never_worked(), to, running=False, mission_ticket=False)

    assert refusal.startswith(f"No one has worked on ticket #0044 yet, so there is nothing to {what}")
    # F294 (night 8): Assign starts the card, so "assign an agent and use Run now" was a step too many
    assert "give it to an agent, which starts it" in refusal and "Cancel" in refusal


def test_a_failed_run_with_nothing_left_points_to_run_now():
    failed = _ticket("failed", workspace_seq=52, result=None, error_message="402: out of credit")

    refusal = drag_refusal(failed, "done", running=False, mission_ticket=False)

    assert refusal == "Ticket #0052's run failed and left nothing to approve: use Run now to try again, or Cancel it."


@pytest.mark.parametrize("to", ["review", "done", "blocked", "assigned", "inbox"])
def test_a_running_ticket_waits_for_its_result_or_is_cancelled_first(to):
    refusal = drag_refusal(_running(), to, running=True, mission_ticket=False)

    assert refusal == ("Ticket #0094 is running: wait for its result, which lands on the card, "
                       "or use Cancel to stop it first.")


@pytest.mark.parametrize("to", ["cancelled", "in_progress"])
def test_cancel_and_a_repeat_still_reach_a_running_ticket(to):
    assert drag_refusal(_running(), to, running=True, mission_ticket=False) is None


def test_a_card_that_holds_an_answer_can_still_be_filed():
    # night 7, iteration 7 (#0129): an Inbox card holding the agent's draft, filed as Done
    holding = _ticket("inbox", workspace_seq=129, assigned_agent_id=None, result="The right draft.")

    assert drag_refusal(holding, "done", running=False, mission_ticket=False) is None


def test_a_missions_ticket_keeps_its_own_refusals():
    # F294 (night 8): one with no answer has nothing to approve (test_f294_a); this one has its draft
    step = _never_worked(source_type="orchestration_task", assigned_agent_id=5, result="Step 1's draft.")

    assert drag_refusal(step, "done", running=True, mission_ticket=True) is None    # mission_runs_it says where
    assert drag_refusal(_ticket("review"), "done", running=False, mission_ticket=True).startswith("Use Approve")


def test_a_playbook_card_has_no_agent_by_design():
    card = _ticket("failed", assigned_agent_id=None, source_type="recipe", source_id="exec-82")
    plain = _ticket("failed", assigned_agent_id=None)

    assert drag_refusal(card, "in_progress", running=False, mission_ticket=False) is None
    assert drag_refusal(plain, "in_progress", running=False, mission_ticket=False) == NO_AGENT_NO_PROGRESS


# --- the drag route -----------------------------------------------------------------------


@pytest.mark.parametrize("to", ["done", "review"])
def test_dragging_an_untouched_ticket_into_a_finished_column_changes_nothing(board, to):
    task = _never_worked()

    with pytest.raises(HTTPException) as refused:
        board.drag(task, to)

    assert refused.value.status_code == 409 and "give it to an agent, which starts it" in refused.value.detail  # F294
    assert task.status == "inbox" and task.completed_at is None
    assert board.completed == []                     # #0044: Done, with a completion time and no result


def test_a_running_ticket_keeps_running_and_keeps_its_lease(board):
    task = _running()
    lease = task.lease_until

    with pytest.raises(HTTPException) as refused:
        board.drag(task, "done")

    assert refused.value.status_code == 409 and "use Cancel to stop it first" in refused.value.detail
    assert task.status == "in_progress" and task.lease_until == lease   # #0111: Done at 5 s, answer lost


def test_a_ticket_moved_back_out_of_done_is_not_finished(board):
    task = _ticket("done")

    board.drag(task, "inbox")

    assert task.status == "inbox" and task.completed_at is None          # #0044 kept a completion time


def test_the_drag_that_approves_a_ticket_says_so_and_clears_the_old_block(board, monkeypatch):
    from api import board_tasks as bt

    monkeypatch.setattr(bt, "record_operator_consent", lambda *a, **kw: "granted")
    task = _ticket("blocked", workspace_seq=59, blocked_reason=APPROVAL_BLOCK,
                   blocked_at=datetime(2026, 10, 3, 2, 3, tzinfo=timezone.utc))

    out = board.drag(task, "in_progress")

    assert out["message"] == "Ticket #0059 started. It was waiting for your approval; starting it gave it."
    assert task.status == "assigned" and (task.blocked_reason, task.blocked_at) == (None, None)


def test_a_ticket_waiting_for_an_answer_is_answered_not_started_over(board, monkeypatch):
    import core.services.approval_grants as grants
    from api import board_tasks as bt
    from tests.test_board_task_handlers import _FakeSession

    question = NS(id=1150, kind="question", status="pending")
    monkeypatch.setattr(grants, "find_pending_grant",
                        lambda db, ws, **kw: question if kw.get("kind") == "question" else None)
    task = _ticket("blocked", workspace_seq=59, blocked_reason="Waiting for your answer (question 1150)")

    for start in (lambda: board.drag(task, "in_progress"),
                  lambda: asyncio.run(bt.run_task_now(task.id, ctx=CTX, db=_FakeSession(agent=NS(id=5), task=task)))):
        with pytest.raises(HTTPException) as refused:
            start()
        assert refused.value.status_code == 409
        assert refused.value.detail.startswith("Ticket #0059 is waiting for your answer to its question")

    assert task.status == "blocked" and question.status == "pending" and board.consent == []


# --- consent is an approval, never an answer ------------------------------------------------


def test_consent_never_grants_an_open_question(monkeypatch):
    import core.services.approval_grants as grants
    from services import board_consent

    question = NS(id=1150, kind="question", status="pending")
    asked_for = []
    created = []

    def _pending(db, ws, **kw):
        asked_for.append(kw.get("kind"))
        return question if kw.get("kind") in (None, "question") else None

    monkeypatch.setattr(grants, "find_active_grant", lambda db, ws, **kw: None)
    monkeypatch.setattr(grants, "find_pending_grant", _pending)
    monkeypatch.setattr(grants, "create_grant", lambda db, ws, **kw: created.append(NS(**kw)) or created[-1])
    monkeypatch.setattr(grants, "grant_grant", lambda g, *, granted_by, now=None: setattr(g, "status", "granted"))
    db = NS(commit=lambda: None, rollback=lambda: None)

    outcome = board_consent.record_operator_consent(db, workspace_id="ws", task_id=1206, agent_id=5,
                                                    actor="user:2", why="moved to in progress")

    assert asked_for == ["approval"] and outcome == "created"
    assert question.status == "pending"              # #0059's night: it left the Questions tab unanswered


def test_find_pending_grant_narrows_by_kind(engine, workspace, new_session):
    from sqlalchemy import text

    from core.models.approval_grants import KIND_APPROVAL, KIND_QUESTION, SUBJECT_BOARD_TASK
    from core.services.approval_grants import create_grant, find_pending_grant

    s = new_session()
    try:
        create_grant(s, workspace, subject_type=SUBJECT_BOARD_TASK, subject_id="1206",
                     kind=KIND_QUESTION, question_md="Which café, Harbour Street or the market?")
        s.commit()

        assert find_pending_grant(s, workspace, subject_type=SUBJECT_BOARD_TASK, subject_id="1206",
                                  kind=KIND_APPROVAL) is None
        found = find_pending_grant(s, workspace, subject_type=SUBJECT_BOARD_TASK, subject_id="1206",
                                   kind=KIND_QUESTION)
        assert found is not None and found.kind == KIND_QUESTION
        assert find_pending_grant(s, workspace, subject_type=SUBJECT_BOARD_TASK, subject_id="1206") is not None
    finally:
        s.execute(text("DELETE FROM approval_grants WHERE workspace_id = CAST(:w AS uuid)"), {"w": workspace})
        s.commit()


# --- a playbook's card runs its playbook again --------------------------------------------


def test_dragging_a_failed_playbook_card_to_in_progress_runs_its_playbook_again(workspace, new_session, launched):
    from api.board_tasks import update_task_status

    pb = f243._finished_playbook(new_session, workspace, status="failed", run_status="failed")

    out = asyncio.run(update_task_status(pb.card, f243._body({"status": "in_progress"}),
                                         ctx=f243._owner(workspace), db=new_session()))

    card = f243._card(new_session, pb.card)
    assert out["started"] is True and launched == [card.source_id]        # #0082: "Assign an agent first"
    assert card.status == "in_progress"
