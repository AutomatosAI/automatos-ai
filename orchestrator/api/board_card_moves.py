"""F294 (night 8): what the board's own buttons say when there is nothing for them to do.

- Cancel on a Done card answered 200 ``{"status": "done", "applied": false}`` with no
  words (#0422), while a drag to Cancelled moved the same card with no note. Both now
  call the card off the same way: Cancelled, with who and when on it
  (``services.board_cancel.call_off_done``), its answer kept. A card already cancelled
  or closed is left as it is, and the answer says so.
- Approve with a note on a card that had finished by itself (review off) answered the
  raw "Task must be in review status (currently: done)" (#0345): "a card that finishes
  by itself can't take a next-time note; only a Reject reaches the agent". The note
  is now kept on the card as an Approve note, which its agent's next runs carry when
  it says what to do next time (``services.ticket_redo.agent_lessons``). Approve in
  any other column says, in words, what it needs.
- The board's Assign left no note on the card (#0313, #0332), while Auto's does
  ("Auto · Gave this to Analyst, in chat."). It now says "you · Gave this to …".
- PRD-256 US-005 (#0065): a person's Approve closes a card that lacks the Socials post or
  the Deliverable it asked for, and the card says what was missing (``done_by_hand``);
  only a caller that is not a person is refused.
- Ticket PR fields: with ``TICKET_APPROVE_MERGES_PR`` on, a person's Approve merges the pull
  request its session reported, and the card says whether it did (``after_its_action``,
  ``services.ticket_pr_merge``); once merged, a later failure of the approval leaves it Done
  (``its_merge_stands``). This is the one module that imports ``services.ticket_pr_merge``.

It lives here because api/board_tasks.py is over 800 lines and does not grow.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from fastapi import HTTPException
from sqlalchemy.orm import Session

from services.ticket_numbers import ticket_label, ticket_number

logger = logging.getLogger(__name__)

CALLED_OFF_REASON = "called off on the board"
# How the board names a ticket's column in words.
COLUMN_WORDS = {"inbox": "in the Inbox", "assigned": "assigned and waiting to start", "in_progress": "running",
                "blocked": "blocked", "failed": "failed", "cancelled": "cancelled", "closed": "closed",
                "done": "done", "review": "in Review"}
# What Approve needs, by the column the ticket is in.
APPROVE_NEEDS = {
    "inbox": "no one has worked on it yet, so there is nothing to approve",
    "assigned": "its agent hasn't answered yet, so there is nothing to approve",
    "in_progress": "wait for its answer, which lands on the card and comes to Review",
    "blocked": "it is waiting for something before it can go on; its card says what",
    "failed": "its run failed: use Run now to try again, or Cancel it",
    "cancelled": "it was called off; Run now starts it again",
    "closed": "it is finished and filed",
}
GAVE_THIS_TO = "Gave this to {name}"
TOOK_ITS_AGENT_OFF = "Took {name} off it"


def label_of(db: Session, task: Any, *, capital: bool = False) -> str:
    return ticket_label(task, ticket_number(db, task), capital=capital)


def call_off(db: Session, task: Any, *, by: str) -> Dict[str, Any]:
    """A Done card the owner no longer wants is Cancelled, with who and when on it, from
    the Cancel button or a drag (#0422). Commits."""
    from services.board_cancel import call_off_done

    call_off_done(db, task, by=by, reason=CALLED_OFF_REASON)
    db.commit()
    db.refresh(task)
    return {"id": task.id, "status": task.status, "applied": True, "previous_status": "done",
            "message": f"{label_of(db, task, capital=True)} was done: it is cancelled now, and its answer is kept."}


def left_as_it_is(db: Session, task: Any) -> Dict[str, Any]:
    """Cancel on a card already cancelled or closed: nothing changes, and the answer says so."""
    return {"id": task.id, "status": task.status, "applied": False,
            "message": (f"{label_of(db, task, capital=True)} is already {COLUMN_WORDS.get(task.status, task.status)}: "
                        "there is nothing to cancel.")}


def approve_outside_review(db: Session, task: Any, note: str) -> Dict[str, Any]:
    """Approve on a ticket that is not in Review. A Done one keeps the note as an
    Approve note (#0345); any other column is a 422 that says what Approve needs.
    Commits the note."""
    from services.ticket_verdict import keep_approval_note

    label = label_of(db, task, capital=True)
    if task.status != "done":
        needs = APPROVE_NEEDS.get(task.status, "only a ticket in Review can be approved")
        raise HTTPException(status_code=422, detail=f"{label} is {COLUMN_WORDS.get(task.status, task.status)}: "
                                                    f"{needs}.")
    if not note:
        return {"success": True, "task_id": task.id, "status": task.status, "applied": False,
                "message": f"{label} is done already: there is nothing to approve. A note with Approve is kept on it."}
    from services.ticket_redo import NEXT_TIME

    keep_approval_note(db, task_id=task.id, workspace_id=task.workspace_id, note=note)
    db.commit()
    logger.info("[BoardTasks] Task %d was done already; the owner's Approve note is kept on it", task.id)
    teaches = ", and its agent's next work carries it" if NEXT_TIME.search(note) else ""
    return {"success": True, "task_id": task.id, "status": task.status, "applied": True,
            "message": f"{label} was done already: your note is kept on it{teaches}."}


def done_by_hand(db: Session, ctx: Any, task: Any) -> bool:
    """PRD-256 US-005 (#0065): Approve's move of ``task`` to Done, before anything moves. True
    for a person's, which closes the card whatever it lacks (what was missing is read once the
    approval's own action has run: ``done_needs_an_artifact.closed_without``). Any other
    caller's (a request with no credential; an in-process call for Auto, with no person's lane),
    short of the card's artifact, is a 409 in the refusal's words, as Auto's status tool gets."""
    from services.done_needs_an_artifact import a_persons_request, missing_artifact

    if a_persons_request(ctx):
        return True
    refused = missing_artifact(db, task)
    if refused:
        raise HTTPException(status_code=409, detail=refused)
    return False


async def after_its_action(db: Session, ctx: Any, task: Any, *, by_a_person: bool, decided_at: Any,
                           shown: Any = None) -> Tuple[Optional[str], Optional[str]]:
    """What a person's Approve leaves on the card once the approval's own action has run, before
    its result is written (``ticket_verdict.record_approval``): what the card lacked
    (``done_needs_an_artifact.closed_without``: a Deliverable the action filed counts) and the note
    still to write about its pull request (``ticket_pr_merge.merge_on_approve``: why it was not
    merged; None with the setting off, no pull request, or a merge already noted on its own).
    ``shown`` is the pull request the Approve's page showed. (None, None) for any caller that is not
    a person."""
    if not by_a_person:
        return None, None
    from services.done_needs_an_artifact import closed_without
    from services.ticket_pr_merge import merge_on_approve

    return closed_without(db, task), await merge_on_approve(db, ctx, task.id, decided_at=decided_at, shown=shown)


def its_merge_stands(db: Session, task_id: int, workspace_id: Any, *, decided_at: Any) -> bool:
    """The approval failed after this approval merged the ticket's pull request: the ticket stays
    Done (the merge cannot be undone), and the caller does not reopen Review. Rolls back first."""
    from services.ticket_pr_merge import merged_by_this_approval

    return merged_by_this_approval(db, workspace_id, task_id, decided_at)


def assign_note(db: Session, workspace_id: Any, was: Optional[int], now: Optional[int]) -> Optional[str]:
    """The note the board's Assign leaves (#0313, #0332): who the card went to, or whose
    it no longer is. None when the agent is the same."""
    if was == now:
        return None
    from core.models import Agent

    def name(agent_id: Optional[int]) -> Optional[str]:
        row = db.query(Agent.name).filter(Agent.id == agent_id, Agent.workspace_id == workspace_id).first() \
            if agent_id else None
        return getattr(row, "name", None) if row else None

    if now is not None:
        return GAVE_THIS_TO.format(name=name(now) or f"agent {now}") + "."
    return TOOK_ITS_AGENT_OFF.format(name=name(was) or "its agent") + "."


__all__ = ["after_its_action", "approve_outside_review", "assign_note", "call_off", "done_by_hand", "its_merge_stands",
           "label_of", "left_as_it_is"]
