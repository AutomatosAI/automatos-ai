"""F241 (night 7b): a mission tool given a card's number acts on that card's mission.

Night 7b, Auto sent the owner's card numbers as mission ids, and every call failed:
- platform_approve_mission("0177") and platform_cancel_mission("0193"): both task
  cards, both "invalid mission ID";
- platform_get_mission(188) for mission #0188's own card, and
  platform_get_mission("0188.3") for its third step: "can't find a mission".

A mission's id is a UUID, so anything else in ``mission_id`` is a card's number
(``takes_card_numbers`` reads it as the ticket tools do, F241 night 7):
- a mission's card (#0188, "0188", 188) is its mission;
- a mission step's card (#0188.3, 188.3) is its mission for reading it, pausing,
  resuming or replanning it, or editing its plan. Approving, rejecting or cancelling
  a step is refused: a step's work is approved or sent back on its own card, and a
  step stops with its mission. Reading a mission by a step names the step. Night 9
  (F308): once the mission has started, approving or rejecting a step is that step's
  Approve or Reject, through its mission (mission_step_verdicts);
- any other card is no mission. Reading it gives the card itself; anything else is
  refused, naming the card and the call that does what was asked.

A public widget turn never reads a number (F155): its call reaches the tool as sent.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

Handler = Callable[[Session, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

MISSION_CARD, STEP_CARD, RUN_CARD = "orchestration", "orchestration_task", "recipe"

# What each mission tool does with a mission, by the card a number names.
READS = "read"          # platform_get_mission
RUNS = "run"            # pause, resume, replan, edit the plan: the mission's, whichever card
DECIDES = "decide"      # approve, reject, cancel: a step or a task card has its own way

NOT_A_MISSION = "{label} ('{title}') is a {kind}, not a mission, so nothing was done. {instead}"
A_STEP = ("{label} ('{title}') is a step of mission {mission}, so nothing was done. {instead} "
          "To act on the whole mission, call this again with mission_id \"{mission}\".")
KINDS = {RUN_CARD: "playbook run's card", MISSION_CARD: "mission's card"}
TASK_KIND = "task card"
ASKED_ABOUT_STEP = "{label} is this mission's step '{title}'. Its card in full: platform_get_task(\"{label}\")."
A_CARD_NOT_A_MISSION = "{label} is a {kind}, not a mission: this is the card."

# The call that does on a card what the mission tool was asked to do.
ON_ITS_CARD = {
    "platform_approve_mission": ("Approve it on its card: platform_update_task_status with task_id \"{number}\", "
                                 "status \"done\" and the owner's words in note."),
    "platform_reject_mission": ("Send it back on its card: platform_update_task_status with task_id \"{number}\", "
                                "status \"assigned\" and the owner's words in note."),
    "platform_cancel_mission": "Cancel it: platform_update_task_status with task_id \"{number}\" and status \"cancelled\".",
}
A_STEP_STOPS_WITH_ITS_MISSION = "A step stops with its mission."
# F308 (night 9): once its mission has started, a step is approved or sent back through it.
STEP_VERDICTS = ("platform_approve_mission", "platform_reject_mission")
OTHERWISE = ("Act on it with the ticket tools: platform_update_task_status (run it again with status "
             "\"in_progress\", or cancel it) or platform_update_task (change its brief).")
NOT_IN_REVIEW = " It is {status}, not in Review, so it has nothing to approve yet."


def takes_card_numbers(does: str) -> Callable[[Handler], Handler]:
    """Let a mission tool take a card's number in ``mission_id`` (see the module)."""
    def decorate(handler: Handler) -> Handler:
        @functools.wraps(handler)
        async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
            from core.security.surface import widget_turn

            said = (params or {}).get("mission_id")
            if said in (None, "") or _is_uuid(said) or widget_turn():
                return await handler(db, workspace_id, params)
            card, refusal = _card_named(db, workspace_id, said)
            if refusal:
                return {"success": False, "error": refusal}
            run_id = _mission_of(db, card)
            step = _decided_on_its_mission(db, card, run_id, handler, does)
            if step is not None:  # F308 (night 9): approving or sending back a started mission's step
                return await handler(db, workspace_id, {**params, "mission_id": str(run_id), "step": step})
            if run_id is not None and (card.source_type == MISSION_CARD or does != DECIDES):
                out = await handler(db, workspace_id, {**params, "mission_id": str(run_id)})
                return _naming_the_step(db, out, card) if card.source_type == STEP_CARD else out
            if does == READS:
                return await _the_card(db, workspace_id, card)
            return {"success": False, "error": _refusal(db, card, handler, run_id)}
        return wrapped
    return decorate


def _decided_on_its_mission(db: Session, card: Any, run_id: Optional[UUID], handler: Handler,
                            does: str) -> Optional[str]:
    """F308 (night 9): the step's number when approve or reject names a step of a mission
    that has started; its mission decides it as the board's Approve or Reject on the
    step's card would (mission_step_verdicts). None otherwise."""
    if does != DECIDES or card.source_type != STEP_CARD or run_id is None or _action_of(handler) not in STEP_VERDICTS:
        return None
    from modules.tools.discovery.mission_step_verdicts import step_of_a_started_mission

    return step_of_a_started_mission(db, card, run_id)


def _is_uuid(value: Any) -> bool:
    if isinstance(value, UUID):
        return True
    try:
        UUID(str(value))
    except (ValueError, TypeError, AttributeError):
        return False
    return True


def _card_named(db: Session, workspace_id: Any, said: Any) -> Tuple[Any, Optional[str]]:
    """The card ``said`` names in this workspace, or why there is none."""
    from core.models.core import BoardTask
    from services.ticket_refs import ticket_id_named

    task_id, error = ticket_id_named(db, workspace_id, said)
    if error:
        return None, error
    card = db.query(BoardTask).filter(BoardTask.id == task_id, BoardTask.workspace_id == workspace_id).first()
    return card, None


def _mission_of(db: Session, card: Any) -> Optional[UUID]:
    """The mission a mission's card or step belongs to; None for any other card."""
    from core.models.core import BoardTask

    if card.source_type not in (MISSION_CARD, STEP_CARD):
        return None
    if card.orchestration_run_id is not None or card.parent_task_id is None:
        return card.orchestration_run_id
    parent = db.query(BoardTask.orchestration_run_id).filter(BoardTask.id == card.parent_task_id).first()
    return parent.orchestration_run_id if parent else None


def _naming_the_step(db: Session, out: Dict[str, Any], step: Any) -> Dict[str, Any]:
    """A mission read by one of its steps says which step was meant."""
    from services.ticket_numbers import ticket_number

    if not (isinstance(out, dict) and out.get("success")):
        return out
    label = ticket_number(db, step) or f"ticket {step.id}"
    return {**out, "asked_about": ASKED_ABOUT_STEP.format(label=label, title=step.title)}


async def _the_card(db: Session, workspace_id: Any, card: Any) -> Dict[str, Any]:
    """A read of a card that is no mission answers with the card."""
    from modules.tools.discovery.handlers_board_tasks import get_board_task
    from services.ticket_numbers import ticket_number

    out = await get_board_task(db, workspace_id, {"task_id": card.id})
    if not (isinstance(out, dict) and out.get("success")):
        return out
    label = ticket_number(db, card) or f"ticket {card.id}"
    return {**out, "note": A_CARD_NOT_A_MISSION.format(label=label, kind=KINDS.get(card.source_type, TASK_KIND))}


def _refusal(db: Session, card: Any, handler: Handler, run_id: Optional[UUID]) -> str:
    """Why the tool did nothing to ``card``, and the call that does what was asked."""
    from services.ticket_numbers import ticket_number

    action = _action_of(handler)
    number = ticket_number(db, card) or str(card.id)
    instead = ON_ITS_CARD.get(action, OTHERWISE).format(number=number)
    if action == "platform_approve_mission" and card.status != "review":
        instead += NOT_IN_REVIEW.format(status=card.status)
    if card.source_type == STEP_CARD:
        if action == "platform_cancel_mission":
            instead = A_STEP_STOPS_WITH_ITS_MISSION
        mission = _mission_number(db, card) or str(run_id)
        return A_STEP.format(label=number, title=card.title, mission=mission, instead=instead)
    kind = KINDS.get(card.source_type, TASK_KIND)
    return NOT_A_MISSION.format(label=number, title=card.title, kind=kind, instead=instead)


def _mission_number(db: Session, step: Any) -> Optional[str]:
    """The number of a step's mission card (#0188 for #0188.3)."""
    from core.models.core import BoardTask
    from services.ticket_numbers import format_number

    parent = db.query(BoardTask.workspace_seq).filter(BoardTask.id == step.parent_task_id).first()
    return format_number(parent.workspace_seq) if parent else None


def _action_of(handler: Handler) -> str:
    """The platform action a mission handler serves: approve_mission → platform_approve_mission."""
    return f"platform_{getattr(handler, '__name__', '')}"


__all__ = ["DECIDES", "READS", "RUNS", "takes_card_numbers"]
