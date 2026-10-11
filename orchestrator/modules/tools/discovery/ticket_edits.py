"""``platform_update_task`` in a chat a person drives (F241, night 7b).

- The owner's note: "Approve #0198 with this note: Right, 204 bags a sack" ended with
  the owner's note on #0198 credited to "an agent". Auto writes a note in a chat
  because the person behind the chat said it, so it is theirs: the board says "you",
  as it does for a note the owner types on the card. A note from an agent's own lane
  (a heartbeat, a ticket, a playbook step), where no person drives the call, still
  says "an agent".
- The re-brief: "Update #0199 with that brief and send it back" was refused twice
  ("platform_update_task doesn't allow me to change the brief"), and once Auto moved
  the card with no brief at all, so it re-ran the old one.

Night 8 (F279): ``send_back`` replaced the card's brief with the owner's correction,
7 times of 7 (twice after "keep its brief as it is"): #0347's "take out the line It
starts with To:" became the whole brief, and the agent deleted the To: line; #0402's
redo invented a shop notice; #0376, #0380 and #0449 asked for their own draft. The
board's Reject keeps the brief and adds the owner's words. So now, as on the board:

- ``send_back`` is the board's Reject (``ticket_moves``, through
  platform_update_task_status): the brief stays, and the owner's words (``note``, or
  the ``description`` the call put them in) are what the redo fixes;
- a new ``description`` on a card its agent has worked on is the board's Re-brief
  (``api.board_task_rebrief.rebrief``): the card goes back to its agent with it, and
  the old brief and the last draft stay on record (night 8: Auto's "update" changed
  #0377's description, kept no old brief and re-ran nothing);
- any other edit is the handler's.

Night 9 (F309): a status on this call is the card's move (``ticket_edit_moves``), its
note kept as platform_update_task_status keeps one: #1866's approval note was lost when
the edit tool refused ``status`` and Auto split the call in two. Beside a Re-brief the
status is dropped and the answer says so (PRD-256 FX-013).

``_user_id`` is the server-injected driver (platform executor, OPERATOR_CONSENT_ACTIONS),
never a model argument.
"""
from __future__ import annotations

import functools
import logging
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict

from sqlalchemy.orm import Session

from modules.tools.discovery.owner_only import CLICKED_BY

logger = logging.getLogger(__name__)

Handler = Callable[[Session, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

NOTE = "note"
BY_THE_PERSON = "you"
# update_board_task's answer when a call carries nothing it changes.
NOTHING_TO_CHANGE = "Nothing to change"
SEND_BACK = "send_back"
DESCRIPTION = "description"
# The fields update_board_task changes besides the description.
OTHER_FIELDS = ("title", "priority", "review_mode", "tags", NOTE)
BY_AN_AGENT = "platform_tool"
NEEDS_THE_WORDS = ("send_back sends the card back the way the board's Reject does: its brief stays, and the owner's "
                   "words are what the redo fixes. Put their words in note, as they wrote them. Nothing was done.")
BOTH_GIVEN = ("send_back keeps the card's brief, so it takes no description. Put the owner's words in note to send "
              "it back, or send a description without send_back to give it a new brief. Nothing was done.")
# A card its agent has worked on (or is working on): a new brief there is the board's Re-brief.
WORKED = ("review", "done", "failed", "blocked", "in_progress")
SEND_BACK_STATUS = "assigned"
SENT_BACK = "Gave this a new brief and sent it back"
REBRIEFED = ("{label} has the new brief and is back with its agent, who redoes it on this card. The old brief and "
             "the last draft are kept on the card.")


def notes_say_who_asked(handler: Handler) -> Handler:
    """A note in a call a person drives is written as theirs; anything else is the handler's."""
    @functools.wraps(handler)
    async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        note = str((params or {}).get(NOTE) or "").strip()
        if not note or not params.get("_user_id"):
            return await handler(db, workspace_id, params)
        out = await handler(db, workspace_id, {k: v for k, v in params.items() if k != NOTE})
        only_the_note = isinstance(out, dict) and str(out.get("error") or "").startswith(NOTHING_TO_CHANGE)
        if not isinstance(out, dict) or (out.get("success") is not True and not only_the_note):
            return out
        _write_note(db, workspace_id, params["task_id"], note)
        updated = {**(out.get("updated") or {}), NOTE: "added"}
        return {"success": True, "task_id": out.get("task_id") or params["task_id"], "updated": updated}
    return wrapped


def _write_note(db: Session, workspace_id: Any, task_id: Any, note: str) -> None:
    from api.board_tasks import MAX_TASK_NOTE_CHARS
    from services.cli_host_service import append_session_note

    append_session_note(db, task_id=int(task_id), workspace_id=workspace_id,
                        note=note[:MAX_TASK_NOTE_CHARS], by=BY_THE_PERSON)
    db.commit()


def briefs_like_the_board(handler: Handler) -> Handler:
    """``send_back`` is the board's Reject, and a new description on a worked card the
    board's Re-brief (see the module). Any other edit is the handler's."""
    @functools.wraps(handler)
    async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        from modules.tools.discovery.new_card_checks import without_status_orders
        from modules.tools.discovery.ticket_edit_moves import edited_then_moved, moves_the_card

        params = params or {}
        if moves_the_card(params) and not params.get(SEND_BACK):   # F309 (9): a status here is the card's move
            return await edited_then_moved(wrapped, db, workspace_id, params)
        if params.get(SEND_BACK):
            return await _sent_back(handler, db, workspace_id, params)
        brief, _ = without_status_orders(params.get(DESCRIPTION))   # F265: the board moves the card
        edit = {k: v for k, v in params.items() if k != SEND_BACK}
        edit = {**edit, DESCRIPTION: brief} if DESCRIPTION in edit else edit
        worked = (_worked_ticket(db, workspace_id, params.get("task_id"))
                  if isinstance(brief, str) and brief.strip() else None)
        if worked is None:
            return await handler(db, workspace_id, edit)
        rest = {k: v for k, v in edit.items() if k != DESCRIPTION}
        out = await handler(db, workspace_id, rest) if any(rest.get(f) for f in OTHER_FIELDS) else {}
        if out and out.get("success") is not True:
            return out
        return _rebrief(db, workspace_id, params, brief.strip(), out.get("updated") or {})
    return wrapped


async def _sent_back(handler: Handler, db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """The board's Reject: any other fields first, then the card goes back to its agent
    with the owner's words, through platform_update_task_status as the board's own move."""
    from modules.tools.discovery.handlers_board_task_done import update_board_task_status
    from services.ticket_numbers import ticket_number

    note, brief = str(params.get(NOTE) or "").strip(), str(params.get(DESCRIPTION) or "").strip()
    if note and brief:
        return {"success": False, "error": BOTH_GIVEN}
    if not (note or brief):
        return {"success": False, "error": NEEDS_THE_WORDS}
    task = _ticket(db, workspace_id, params.get("task_id"))
    if task is None:
        return {"success": False, "error": f"Task {params.get('task_id')} not found in this workspace"}
    rest = {k: v for k, v in params.items() if k not in (SEND_BACK, DESCRIPTION, NOTE)}
    out = await handler(db, workspace_id, rest) if any(rest.get(f) for f in OTHER_FIELDS) else {}
    if out and out.get("success") is not True:
        return out
    move = {"task_id": ticket_number(db, task) or task.id, "status": SEND_BACK_STATUS, NOTE: note or brief}
    move.update({key: params[key] for key in ("_user_id", CLICKED_BY) if params.get(key)})
    return await update_board_task_status(db, workspace_id, move)


def _ticket(db: Session, workspace_id: Any, task_id: Any) -> Any:
    from core.models.core import BoardTask

    said = str(task_id).strip()
    if not said.isdigit():
        return None
    return db.query(BoardTask).filter(BoardTask.id == int(said), BoardTask.workspace_id == workspace_id).first()


def _worked_ticket(db: Session, workspace_id: Any, task_id: Any) -> Any:
    """The card when its agent has worked on it and it is not a mission's own card
    (whose plan changes on the mission's page), else None."""
    from api.board_tasks import MISSION_CARD_SOURCE

    task = _ticket(db, workspace_id, task_id)
    if task is None or task.status not in WORKED or task.source_type == MISSION_CARD_SOURCE:
        return None
    return task


def _rebrief(db: Session, workspace_id: Any, params: Dict[str, Any], brief: str,
             updated: Dict[str, Any]) -> Dict[str, Any]:
    """The board's Re-brief (``rebrief``) of the ticket, or the refusal that left it as it was."""
    from fastapi import HTTPException

    from api.board_task_rebrief import rebrief
    from services.board_consent import actor_from_user_id
    from services.ticket_numbers import ticket_label
    from services.ticket_redo import MAX_BRIEF_CHARS

    task, refusal = _ticket_to_rebrief(db, workspace_id, params.get("task_id"))
    if refusal:
        return {"success": False, "error": refusal}
    driver = params.get("_user_id")
    ctx = SimpleNamespace(workspace_id=workspace_id, user_id=driver)
    try:
        rebrief(db, ctx, task, brief[:MAX_BRIEF_CHARS], by=actor_from_user_id(driver) if driver else BY_AN_AGENT)
    except HTTPException as refused:
        db.rollback()
        return {"success": False, "error": str(refused.detail)}
    _note_it(db, workspace_id, task.id)
    return {"success": True, "task_id": task.id, "status": task.status,
            "updated": {**updated, DESCRIPTION: "the new brief", "back_to_its_agent": True},
            "message": REBRIEFED.format(label=ticket_label(task, capital=True))}


def _ticket_to_rebrief(db: Session, workspace_id: Any, task_id: Any):
    """The ticket, or why it is not re-briefed: not this workspace's, a mission's own
    card (its plan is changed on the mission's page), or closed (F241)."""
    from api.board_tasks import MISSION_CARD_SOURCE, MISSION_CARD_VERDICT
    from core.models.core import BoardTask
    from modules.tools.discovery.ticket_changes import CLOSED, CLOSED_REFUSAL
    from services.ticket_numbers import ticket_label

    said = str(task_id).strip()
    task = (db.query(BoardTask).filter(BoardTask.id == int(said), BoardTask.workspace_id == workspace_id).first()
            if said.isdigit() else None)
    if task is None:
        return None, f"Task {task_id} not found in this workspace"
    if task.source_type == MISSION_CARD_SOURCE:
        return None, MISSION_CARD_VERDICT
    if task.status == CLOSED:
        return None, CLOSED_REFUSAL.format(labels=f"{ticket_label(task, capital=True)} ('{task.title}')", verb="is")
    return task, None


def _note_it(db: Session, workspace_id: Any, task_id: int) -> None:
    """The re-brief on the ticket's notes, as Auto's other changes are (F241)."""
    from modules.tools.discovery.ticket_changes import note_change

    note_change(db, workspace_id, task_id, SENT_BACK)


__all__ = ["BY_THE_PERSON", "briefs_like_the_board", "notes_say_who_asked"]
