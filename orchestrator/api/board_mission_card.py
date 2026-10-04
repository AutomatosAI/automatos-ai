"""F291 (night 8): a mission's card waiting for its plan's approval is approved from the board.

Needs you lists a mission waiting for its plan's approval as an approval, under its
card's number. Neither of the card's own controls in the API gave it. Its Approve was
refused ("This is a mission's card: approve or change its plan on the mission's
page…"), and a drag into In progress said "Assign an agent first: a ticket with no
agent cannot be in progress." (11 of 11 on night 8). Following that advice put an
agent on the mission's card, and the next drag was refused for another reason: "the
mission runs its steps, not the board" (#0282).

The card's Approve, and a drag into In progress, now approve the plan as the
mission's own Approve does (``CoordinatorService.approve_plan``, which the board's
verdict buttons already call), and the mission starts. An Approve's note is kept on
the mission for every step (``modules.coordination.owner_note``), and the card says
who approved the plan and when. A Reject on the card, and either control on a
mission in any other state, is refused with the mission named and its page linked:
the plan is turned down or changed there, and its steps are the mission's to run.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from services.ticket_numbers import ticket_label, ticket_number

logger = logging.getLogger(__name__)

MISSION_CARD = "orchestration"
GOAL_SHOWN_CHARS = 80
PLAN_APPROVED_NOTE = "Approved the plan"
REJECT_ON_THE_MISSION = ("{label} is the mission{goal}: its plan is turned down or changed on the mission's page "
                         "(/missions/{run_id}). Approve on the card approves the plan as it stands.")
DECIDED_ON_THE_MISSION = ("{label} is the mission{goal}, and its plan is not waiting for approval (the mission is "
                          "{state}): the mission runs its steps, so act on it from its page (/missions/{run_id}).")
STATE_WORDS = {"pending": "being set up", "planning": "being planned", "running": "running", "paused": "paused",
               "replanning": "being re-planned", "verifying": "checking its work",
               "awaiting_human": "waiting for your review", "completed": "completed", "failed": "failed",
               "cancelled": "cancelled"}


def is_mission_card(task: Any) -> bool:
    """A mission's own card on the board (not one of its steps)."""
    return getattr(task, "source_type", None) == MISSION_CARD


def mission_of(db: Session, task: Any) -> Optional[Any]:
    """The mission ``task`` is the card of, in the card's workspace; None for any other ticket."""
    if not is_mission_card(task):
        return None
    from services.run_cancel import mission_run_of

    return mission_run_of(db, task)


def plan_waits(run: Any) -> bool:
    from core.models.orchestration_enums import RunState

    return run is not None and run.state == RunState.AWAITING_APPROVAL.value


def approve_the_plan(db: Session, ctx: Any, task: Any, run: Any, *, note: str = "") -> Dict[str, Any]:
    """Approve ``run``'s plan from its card, as the mission's Approve does; the note is
    kept on the mission and the card says who approved it. Commits. A plan decided
    meanwhile is a 409 that names the mission."""
    from modules.coordination.owner_note import keep_owner_note, note_text
    from services.cli_host_service import append_session_note
    from services.coordinator_service import get_coordinator_service
    from services.orchestration_state import ConflictError, InvalidTransitionError
    from services.ticket_verdict import OPERATOR_NOTE_BY

    kept = keep_owner_note(run, note)
    try:
        get_coordinator_service().approve_plan(db=db, run_id=run.id, actor_id=ctx.user.id or "unknown")
        db.flush()
    except (ConflictError, InvalidTransitionError) as decided:
        db.rollback()
        db.refresh(run)
        raise HTTPException(status_code=409, detail=mission_card_refusal(db, task, run)) from decided
    said = f"{PLAN_APPROVED_NOTE}: {note_text(note)}" if kept else f"{PLAN_APPROVED_NOTE}."
    append_session_note(db, task_id=task.id, workspace_id=task.workspace_id, note=said, by=OPERATOR_NOTE_BY)
    db.commit()
    db.refresh(task)
    logger.info("[BoardTasks] Mission %s: plan approved from its card %d%s", run.id, task.id,
                " with the owner's note" if kept else "")
    with_note = " Your note goes to every step." if kept else ""
    return {"success": True, "task_id": task.id, "status": task.status, "mission_id": str(run.id),
            "message": f"Approved the plan of {_label(db, task)}{_goal(run)}: the mission is starting.{with_note}"}


def approve_from_its_card(db: Session, ctx: Any, task: Any, *, note: str = "") -> Dict[str, Any]:
    """The card's Approve: the mission's plan, while it waits for approval. Any other
    state of the mission is a 409 that names it and links its page."""
    run = mission_of(db, task)
    if not plan_waits(run):
        raise HTTPException(status_code=409, detail=mission_card_refusal(db, task, run))
    return approve_the_plan(db, ctx, task, run, note=note)


def starts_its_mission(db: Session, task: Any, new_status: Any) -> bool:
    """A move into In progress of a mission's card whose plan waits: the move approves it."""
    return new_status == "in_progress" and plan_waits(mission_of(db, task))


def start_its_mission(db: Session, ctx: Any, task: Any, new_status: Any) -> Optional[Dict[str, Any]]:
    """A drag into In progress of a mission's card whose plan waits approves the plan,
    and the answer says so; None for any other move, which the board judges as before."""
    if not starts_its_mission(db, task, new_status):
        return None
    return approve_the_plan(db, ctx, task, mission_of(db, task))


def mission_card_refusal(db: Session, task: Any, run: Any = None) -> str:
    """Why a mission's card can't take this decision on the board, naming the mission and
    linking its page: a waiting plan is turned down or changed there (Approve on the card
    approves it); in any other state the mission runs its steps."""
    from api.board_tasks import MISSION_CARD_VERDICT

    run = run if run is not None else mission_of(db, task)
    if run is None:
        return MISSION_CARD_VERDICT
    label, goal = _label(db, task, capital=True), _goal(run)
    if plan_waits(run):
        return REJECT_ON_THE_MISSION.format(label=label, goal=goal, run_id=run.id)
    return DECIDED_ON_THE_MISSION.format(label=label, goal=goal, run_id=run.id,
                                         state=STATE_WORDS.get(str(run.state), str(run.state)))


def _label(db: Session, task: Any, *, capital: bool = False) -> str:
    return ticket_label(task, ticket_number(db, task), capital=capital)


def _goal(run: Any) -> str:
    goal = (getattr(run, "goal", None) or "").strip()[:GOAL_SHOWN_CHARS]
    return f" “{goal}”" if goal else ""


__all__ = ["approve_from_its_card", "approve_the_plan", "is_mission_card", "mission_card_refusal", "mission_of",
           "plan_waits", "start_its_mission", "starts_its_mission"]
