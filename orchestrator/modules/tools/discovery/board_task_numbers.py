"""P256-FIX-N15-9 (night 15, F429): a tool that names a ticket by ``board_task_id`` names its number too.

``platform_get_schedule`` lists a ticket due by its row id (``board_task_id``) and
``platform_notify_owner`` answers with the row id of the ticket it filed; neither carried the
ticket's board number, so the model read a row id as the card's number (night 15: one card went by
the board's #2894 and by '2603'). Each such ticket now carries its ``number`` beside the row id, and
the model reads the number alone (``formatting.tickets_by_number`` leaves a numbered ticket's row
ids out). The platform's own code still reads ``board_task_id``. Never on a public widget turn (F155:
a number would say how many tickets the business has).
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable, Dict, List

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

Handler = Callable[[Session, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

KEY = "board_task_id"


def _named(answer: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The answer and the dicts its lists hold that name a ticket by ``board_task_id``."""
    rows = [answer, *(item for value in answer.values() if isinstance(value, list) for item in value)]
    return [row for row in rows if isinstance(row, dict) and isinstance(row.get(KEY), int)
            and not isinstance(row.get(KEY), bool)]


def _numbered(row: Dict[str, Any], numbers: Dict[int, Any]) -> Dict[str, Any]:
    number = numbers.get(row.get(KEY)) if isinstance(row, dict) else None
    return {**row, "number": number} if number else row


def with_board_task_numbers(db: Session, workspace_id: Any, answer: Any) -> Any:
    """``answer`` with each ticket it names by ``board_task_id`` carrying its number (a new dict)."""
    from core.models.core import BoardTask
    from core.security.surface import widget_turn
    from services.ticket_numbers import ticket_numbers

    if not isinstance(answer, dict) or widget_turn():
        return answer
    ids = {row[KEY] for row in _named(answer)}
    if not ids:
        return answer
    tickets = db.query(BoardTask).filter(BoardTask.id.in_(ids), BoardTask.workspace_id == workspace_id).all()
    numbers = ticket_numbers(db, workspace_id, tickets)
    out = _numbered(answer, numbers)
    return {key: [_numbered(item, numbers) for item in value] if isinstance(value, list) else value
            for key, value in out.items()}


def numbers_its_tickets(handler: Handler) -> Handler:
    """Wrap a handler whose answer names tickets by ``board_task_id``: each carries its number. The
    numbers are a reading aid: a failed read is logged and the answer stands as it was."""
    @functools.wraps(handler)
    async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        answer = await handler(db, workspace_id, params)
        try:
            return with_board_task_numbers(db, workspace_id, answer)
        except Exception:  # noqa: BLE001 — logged; the tool's own answer still stands
            logger.exception("[N15-9] could not number the tickets in %s's answer", handler.__name__)
            return answer
    return wrapped


__all__ = ["numbers_its_tickets", "with_board_task_numbers"]
