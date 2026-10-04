"""The general PATCH /{task_id}'s checks and writes (F259).

``api/board_tasks.update_task`` reads the body, refuses what it can't store or what
the board would refuse, writes the fields, and moves the ticket the board's way
(``_move_by_hand``, shared with the drag's PATCH /status). The checks and the writes
are here. Every refusal comes before anything is written: a field the route can't
store as sent is a 422, then a status change the board refuses is a 409, judged on
the ticket as the PATCH leaves its agent and its result (#1094).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from core.models import Agent
from core.models.core import BoardTask
from services.board_drag_rules import as_patched, drag_refusal

# The fields a PATCH writes as they were sent (checked by check_fields first).
FIELDS_AS_SENT = ("description", "priority", "review_mode", "result", "error_message", "tags", "planning_data")


def patch_status(db: Session, task: BoardTask, body: Dict[str, Any]) -> Optional[str]:
    """The status this PATCH sets, checked, or None when it sets none."""
    from api import board_tasks as bt

    if body.get("status") == "in_progress" and task.status != "in_progress":
        bt._hold_before_starting(db, task)  # F209: a claim that landed while the body arrived wins

    # F060: this route accepted any key, returned 200 and echoed the task back,
    # while storing only the eleven fields below. `review_feedback` — the field
    # a reviewer's verdict travels in — went in and vanished, so a rejection
    # looked accepted and nothing was rejected. A PATCH that silently drops
    # what it was given is worse than one that refuses: refuse.
    unknown = sorted(set(body) - bt.PATCHABLE_TASK_FIELDS)
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unknown field(s) for a board task: {unknown}. "
                f"This route stores exactly: {sorted(bt.PATCHABLE_TASK_FIELDS)}. "
                "Nothing was changed."
            ),
        )
    if "status" not in body:
        return None
    # F195's candidates: a status is checked before anything compares it (a list was a 500).
    if not bt._one_of(body["status"], bt.VALID_STATUSES):
        raise HTTPException(status_code=422, detail=f"Invalid status: {body['status']}")
    return body["status"]


def patch_agent(db: Session, workspace_id: Any, task: BoardTask, body: Dict[str, Any]) -> Optional[int]:
    """The ticket's agent once this PATCH lands: the one the body sets, read as an
    id first (#1094 review LOW: never by the truth of what was sent) and found in
    this workspace, or the ticket's own."""
    from api import board_tasks as bt

    if "assigned_agent_id" not in body:
        return task.assigned_agent_id
    agent_id = bt._agent_id_of(body["assigned_agent_id"])
    if agent_id is not None and not db.query(Agent).filter(
        Agent.id == agent_id,
        Agent.workspace_id == workspace_id,
    ).first():
        raise HTTPException(status_code=404, detail="Assigned agent not found in workspace")
    return agent_id


def refuse_the_patch(db: Session, task: BoardTask, body: Dict[str, Any], new_status: Optional[str],
                     agent: Optional[int]) -> None:
    """Every refusal of the PATCH, before anything is written. A repeat of the
    ticket's own status moves nothing, so nothing refuses it."""
    from api import board_tasks as bt

    check_fields(body)
    from api.board_mission_card import starts_its_mission

    starts = new_status in bt.STARTING_STATUSES and not starts_its_mission(db, task, new_status)  # F291
    owned = bt.mission_runs_it(db, task) if starts else None
    if owned:
        raise HTTPException(status_code=409, detail=owned)
    if new_status is None or new_status == task.status:
        return
    after = as_patched(task, assigned_agent_id=agent, result=body.get("result", task.result))
    refusal = drag_refusal(after, new_status, running=bt._running_now(db, task),
                           mission_ticket=task.source_type in bt.MISSION_TICKET_TYPES)
    if refusal:
        raise HTTPException(status_code=409, detail=refusal)


def check_fields(body: Dict[str, Any]) -> None:
    """A field the PATCH can't store as sent is a 422."""
    from api import board_tasks as bt

    if "title" in body and not bt._text_of(body["title"], "title"):
        raise HTTPException(status_code=422, detail="title cannot be empty")
    if "priority" in body and not bt._one_of(body["priority"], bt.VALID_PRIORITIES):
        raise HTTPException(status_code=422, detail=f"Invalid priority: {body['priority']}")
    if "review_mode" in body and not bt._one_of(body["review_mode"], bt.VALID_REVIEW_MODES):
        raise HTTPException(status_code=422, detail=f"Invalid review_mode: {body['review_mode']}")


def patch_fields(task: BoardTask, body: Dict[str, Any]) -> None:
    """Write the fields the PATCH sets, all but its agent and its status."""
    from api import board_tasks as bt

    if "review_feedback" in body:
        # The reviewer's verdict. The dispatcher folds it into the prompt of the
        # next attempt (see _ticket_prompt) and clears it once consumed.
        feedback = body["review_feedback"]
        task.review_feedback = str(feedback)[:bt.MAX_REVIEW_FEEDBACK_CHARS] if feedback else None
    if "title" in body:
        task.title = bt._text_of(body["title"], "title")
    for field in FIELDS_AS_SENT:
        if field in body:
            setattr(task, field, body[field])
    if body.get("note"):
        add_operator_note(task, body["note"])


def note_the_assign(db: Session, task: BoardTask, agent: Optional[int]) -> None:
    """F294 (#0313, #0332): the board's Assign says on the card who it went to, as
    Auto's does ("Auto · Gave this to Analyst, in chat."). Nothing when the agent is
    the same."""
    from api.board_card_moves import assign_note

    said = assign_note(db, task.workspace_id, task.assigned_agent_id, agent)
    if said:
        add_operator_note(task, said)


def add_operator_note(task: BoardTask, note: Any) -> None:
    """An operator note — a remark on the ticket that is NOT a rejection.

    Night 1 (2026-09-18): the only way to say anything to a ticket was to
    reject it into a redo, so a correction and a comment were the same gesture.
    Notes land beside the session's own progress notes, in the same list the
    card already renders.
    """
    from api import board_tasks as bt

    note_text = str(note).strip()[:bt.MAX_TASK_NOTE_CHARS]
    if not note_text:
        return
    ref = dict(task.runtime_ref or {})
    ref["session_notes"] = (ref.get("session_notes") or []) + [{
        "note": note_text,
        "at": datetime.now(timezone.utc).isoformat(),
        "by": "you",
    }]
    task.runtime_ref = ref   # rebuilt, never mutated in place (JSONB)


def patch_after_the_move(task: BoardTask, body: Dict[str, Any], was: str, new_status: Optional[str]) -> None:
    """What the PATCH keeps once the ticket has moved."""
    if new_status == "in_progress":
        # F190: a move to in progress clears the last run's result and error; a
        # PATCH that sets its own keeps them.
        for field in ("result", "error_message"):
            if field in body:
                setattr(task, field, body[field])
    if new_status == "blocked" and (body.get("blocked_reason") or was != "blocked"):
        # A person blocking a ticket that a machine had ALREADY parked used to
        # record nothing — the `blocked_at is None` guard kept the park's
        # reason, so the person's intent was invisible to everything after.
        task.blocked_reason = body.get("blocked_reason")
    if "assigned_agent_id" in body and task.assigned_agent_id and task.status == "inbox":
        task.status = "assigned"  # an agent set on an inbox ticket assigns it
