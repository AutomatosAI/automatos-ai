"""F263 (night 8): "What's waiting for me?" answers with what the board's Needs you shows.

Asked "What's waiting for me?", Auto said "5 failed tasks" four times when nothing
had failed, named last week's failures by title, and gave totals ("2 in review")
without saying which. The board summary it reads counted a card as failed when it
had ever carried an error, over all time, so cards that failed once and were then
redone, approved or cancelled still read as failures; and it never said which
cards were the owner's to deal with.

- ``failed_cards``: the cards whose status is failed now, by number and title.
  A mission step's failure is its mission's to handle, as the board counts it.
- ``with_whats_waiting``: the summary and the snapshot carry ``waiting_for_you``,
  read from the board's own Needs-you service (services.needs_you): its total and
  each row's kind, number and title. Auto's answer is the board's count and the
  same cards, whatever Needs you comes to count.

A public widget turn gets neither (F155: a visitor sees counts, never the owner's).
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable, Dict, Iterable, List

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

Handler = Callable[[Session, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]
FAILED = "failed"
MISSION_STEP = "orchestration_task"
LISTED = 5
ERROR_CHARS = 200
TITLE_CHARS = 120


def failed_cards(db: Session, workspace_id: Any, tasks: Iterable[Any]) -> List[Dict[str, Any]]:
    """The cards failed now (not a mission's steps), newest first, by number."""
    from services.ticket_numbers import ticket_numbers

    failed = sorted((t for t in tasks if t.status == FAILED and t.source_type != MISSION_STEP),
                    key=lambda t: t.id, reverse=True)[:LISTED]
    numbers = ticket_numbers(db, workspace_id, failed) if failed else {}
    return [{"id": t.id, "number": numbers.get(t.id), "title": (t.title or "")[:TITLE_CHARS],
             "error": (t.error_message or "")[:ERROR_CHARS]} for t in failed]


def whats_waiting(db: Session, workspace_id: Any) -> Dict[str, Any]:
    """What waits for the owner, as the board's Needs you counts and lists it."""
    from services.needs_you import needs_you

    found = needs_you(db, workspace_id)
    cards = [{"kind": kind, "number": row.get("number"), "title": (row.get("title") or "")[:TITLE_CHARS]}
             for kind, rows in found["rows"].items() for row in rows]
    return {"total": found["total"], "by_kind": found["counts"], "cards": cards}


def with_whats_waiting(handler: Handler) -> Handler:
    """``handler``'s board answer with ``waiting_for_you`` (never on a widget turn)."""
    @functools.wraps(handler)
    async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        from core.security.surface import widget_turn

        result = await handler(db, workspace_id, params)
        if widget_turn() or not (isinstance(result, dict) and result.get("success")):
            return result
        try:
            return {**result, "waiting_for_you": whats_waiting(db, workspace_id)}
        except Exception:
            # The board's answer stands; the owner's list is read on the board.
            logger.exception("[F263] Needs you unreadable for %s's answer", handler.__name__)
            return result
    return wrapped


__all__ = ["failed_cards", "whats_waiting", "with_whats_waiting"]
