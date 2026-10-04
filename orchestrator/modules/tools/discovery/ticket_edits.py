"""``platform_update_task`` in a chat a person drives (F241, night 7b).

- The owner's note: "Approve #0198 with this note: Right, 204 bags a sack" ended with
  the owner's note on #0198 credited to "an agent". Auto writes a note in a chat
  because the person behind the chat said it, so it is theirs: the board says "you",
  as it does for a note the owner types on the card. A note from an agent's own lane
  (a heartbeat, a ticket, a playbook step), where no person drives the call, still
  says "an agent".
- The re-brief: "Update #0199 with that brief and send it back" was refused twice
  ("platform_update_task doesn't allow me to change the brief"), and once Auto moved
  the card with no brief at all, so it re-ran the old one. With ``send_back``, the new
  description is the ticket's agreed brief and the ticket goes back to its agent on
  the same card, through the board's own "Update ticket and re-queue"
  (``api.board_task_rebrief.rebrief``): the old brief and the last draft stay on record.

``_user_id`` is the server-injected driver (platform executor, OPERATOR_CONSENT_ACTIONS),
never a model argument.
"""
from __future__ import annotations

import functools
import logging
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict

from sqlalchemy.orm import Session

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
NEEDS_A_BRIEF = ("send_back sends the ticket back with a new brief, so the brief goes in description. Nothing was "
                 "done. To send it back with the owner's words instead, call platform_update_task_status with "
                 "status \"assigned\" and the words in note.")
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


def rebriefs_on_send_back(handler: Handler) -> Handler:
    """With ``send_back``: the other fields are the handler's, then the description is
    the ticket's new brief and the ticket goes back to its agent. Without it, the call
    is the handler's."""
    @functools.wraps(handler)
    async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        from modules.tools.discovery.new_card_checks import without_status_orders

        brief, _ = without_status_orders((params or {}).get(DESCRIPTION))   # F265: the board moves the card
        if not (params or {}).get(SEND_BACK):
            edit = {k: v for k, v in (params or {}).items() if k != SEND_BACK}
            return await handler(db, workspace_id, {**edit, DESCRIPTION: brief} if DESCRIPTION in edit else edit)
        if not isinstance(brief, str) or not brief.strip():
            return {"success": False, "error": NEEDS_A_BRIEF}
        rest = {k: v for k, v in params.items() if k not in (SEND_BACK, DESCRIPTION)}
        out = await handler(db, workspace_id, rest) if any(rest.get(f) for f in OTHER_FIELDS) else {}
        if out and out.get("success") is not True:
            return out
        return _send_back(db, workspace_id, params, brief.strip(), out.get("updated") or {})
    return wrapped


def _send_back(db: Session, workspace_id: Any, params: Dict[str, Any], brief: str,
               updated: Dict[str, Any]) -> Dict[str, Any]:
    """The board's re-brief (``rebrief``) of the ticket, or the refusal that left it as it was."""
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
            "updated": {**updated, DESCRIPTION: "the new brief", SEND_BACK: True},
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


__all__ = ["BY_THE_PERSON", "notes_say_who_asked", "rebriefs_on_send_back"]
