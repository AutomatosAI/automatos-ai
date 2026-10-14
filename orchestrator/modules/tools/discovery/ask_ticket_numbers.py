"""P256-FIX-N15-9 (night 15, F429): a question parks a ticket named by its board number, and says it.

Night 15: one card went by three numbers: the board's #2894, Auto's '1607' and '2603'.
``platform_ask_human`` took only the ticket's row id (a '#2894' was refused: 'board_task subject_id
must be an integer id') and answered 'Parked board_task 2603 and asked the human (ask #1607)': a row
id and a question's own id written as a card number, both read back as a card's.

Now its ticket is read the way the ticket tools read one (``services.ticket_refs``): '#2894' (or
'#0051.3' for a mission step) is the card numbered so, bare digits are the board's number and the
row id only when no card has that number, and the platform's own ``TicketId`` is the id. Its answer
names the ticket by number (``subject_id`` is the number; the row id stays on the question it filed)
and the question by no number at all. Each read is a plain equality on the board, one at a time.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable, Dict, Optional, Tuple

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

Handler = Callable[[Session, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

BOARD_TASK = "board_task"
PARKED = "Parked {ticket} and asked the owner; their answer resumes it. Move on to other work."
ASKED = "Asked the owner about {ticket}; the work carries on meanwhile."


def ticket_asked_about(db: Session, workspace_id: Any, ref: Any) -> Tuple[Any, Optional[str]]:
    """The row id of the ticket ``ref`` names (as a string), ``ref`` as given when it names none
    (the handler then says it was not found), or why a number names no ticket."""
    from core.models.core import BoardTask
    from services.ticket_numbers import TicketId, is_bare_ref, is_number_ref, resolve_ticket_ref
    from services.ticket_refs import NO_SUCH_NUMBER

    if isinstance(ref, TicketId):
        return str(int(ref)), None
    if is_number_ref(ref):
        found = resolve_ticket_ref(db, workspace_id, ref)
        return (str(found), None) if found is not None else (ref, NO_SUCH_NUMBER.format(ref=str(ref).strip()))
    if not is_bare_ref(ref):
        return ref, None
    n = int(str(ref).strip())
    numbered = db.query(BoardTask).filter(BoardTask.workspace_id == workspace_id, BoardTask.workspace_seq == n).first()
    return str(numbered.id if numbered is not None else n), None


def _said(db: Session, workspace_id: Any, answer: Dict[str, Any], task_id: str) -> Dict[str, Any]:
    """The answer naming its ticket by number, and its question by none (a new dict)."""
    from core.models.core import BoardTask
    from services.ticket_numbers import ticket_label, ticket_number

    task = db.query(BoardTask).filter(BoardTask.id == int(task_id), BoardTask.workspace_id == workspace_id).first()
    if task is None:
        return answer
    number = ticket_number(db, task)
    said = (PARKED if answer.get("parked") else ASKED).format(ticket=ticket_label(task, number))
    return {**answer, "subject_id": number or answer.get("subject_id"), "message": said}


def asks_by_ticket_number(handler: Handler) -> Handler:
    """Wrap ``ask_human``: a ``board_task`` subject is taken by its board number, and the answer says it."""
    @functools.wraps(handler)
    async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        params = params or {}
        ref = params.get("subject_id")
        if str(params.get("subject_type") or "").strip() != BOARD_TASK or ref in (None, ""):
            return await handler(db, workspace_id, params)
        task_id, error = ticket_asked_about(db, workspace_id, ref)
        if error:
            return {"success": False, "error": error}
        answer = await handler(db, workspace_id, {**params, "subject_id": task_id})
        if not (isinstance(answer, dict) and answer.get("success") and str(task_id).isdigit()):
            return answer
        try:
            return _said(db, workspace_id, answer, task_id)
        except Exception:  # noqa: BLE001 — logged; the question is filed, and its own answer stands
            logger.exception("[N15-9] the question on ticket %s is filed but not named by its number", task_id)
            return answer
    return wrapped


__all__ = ["asks_by_ticket_number", "ticket_asked_about"]
