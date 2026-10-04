"""A card sent back runs its redo with the owner's correction, whoever starts it (F249, night 8).

A card the owner sends back waits in Assigned with their words (``review_feedback``)
until the board's dispatcher claims it; the dispatcher runs it with its last draft
and every correction (``services.ticket_redo.redo_block``). Auto's platform_update_task_status
to In progress launched the card itself, from the bare brief: the redo lost the
owner's words, and the next draft repeated the mistake they had just sent back.

So a plain card with a correction waiting is never launched by Auto's move: it is
left to the dispatcher, which starts it within seconds with the correction, and
Auto is told so. A playbook's card or a mission's step starts its own redo
(``run_redo``), and any other card runs as before.
"""
from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable, Dict

from sqlalchemy.orm import Session

Handler = Callable[[Session, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

IN_PROGRESS = "in_progress"
WAITS_FOR_ITS_AGENT = ("assigned",)
LEFT_TO_THE_BOARD = ("{label} has the owner's correction waiting, so it was left with its agent the board's way: "
                     "the redo starts within seconds from its last draft and every correction on the card. "
                     "Nothing else was changed.")


def redo_keeps_the_correction(handler: Handler) -> Handler:
    """Wrap update_board_task_status: In progress on a plain card whose redo waits is
    left to the board's dispatcher (see the module)."""
    @functools.wraps(handler)
    async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        task = _waiting_redo(db, workspace_id, params or {})
        if task is None:
            return await handler(db, workspace_id, params)
        from services.ticket_numbers import ticket_label

        return {"success": True, "task_id": task.id, "status": task.status, "triggered_execution": False,
                "message": LEFT_TO_THE_BOARD.format(label=ticket_label(task, capital=True))}
    return wrapped


def _waiting_redo(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Any:
    """The card when the call moves one plain card, waiting for its redo, to In progress."""
    from core.models.core import BoardTask
    from services.run_redo import takes_its_own_redo

    said = str(params.get("task_id") or "").strip()
    if str(params.get("status") or "").strip().lower() != IN_PROGRESS or params.get("task_ids") or not said.isdigit():
        return None
    task = db.query(BoardTask).filter(BoardTask.id == int(said), BoardTask.workspace_id == workspace_id).first()
    if task is None or takes_its_own_redo(task) or getattr(task, "status", None) not in WAITS_FOR_ITS_AGENT:
        return None
    return task if getattr(task, "review_feedback", None) and getattr(task, "assigned_agent_id", None) else None


__all__ = ["LEFT_TO_THE_BOARD", "redo_keeps_the_correction"]
