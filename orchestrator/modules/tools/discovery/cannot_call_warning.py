"""PRD-256 F430: a card given to a CLI agent says what the agent's sessions cannot call.

Socials day run (14 Oct): Auto filed "make a quote card post" for the Social Media
Director, a CLI agent whose sessions had no Socials tool, and nothing said so: the
run refused three times, and asked why, Auto guessed. platform_create_task and
platform_assign_task only checked the agent's model (F141).

Now a card that went to an agent carries ``cannot_call`` (the platform tools its title
and brief name that the agent's sessions cannot call) and ``warning`` (who, what, and
the fix), read by ``services.session_capability_gaps.card_assignment_warning`` from the
card as it was saved. The assignment always stands: the warning is said beside it.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable, Dict

logger = logging.getLogger(__name__)

Handler = Callable[[Any, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

# What Auto does with the warning: the owner hears it now, not from a refused run.
TELL_THE_OWNER = " Tell the owner in one line."
CANNOT_CALL_KEYS = ("cannot_call", "warning")


def _rolled_back(db: Any) -> None:
    """After a failed read on Auto's session: its next tool call must not inherit the aborted read."""
    try:
        db.rollback()
    except Exception:  # noqa: BLE001 — logged; the card is already saved
        logger.exception("[F430] rollback after the card read failed")


def _for_the_card(db: Any, workspace_id: Any, task_id: Any) -> Dict[str, Any]:
    """The card's warning, for Auto; ``{}`` when there is none or the card can't be read."""
    from core.models.core import BoardTask
    from services.session_capability_gaps import card_assignment_warning

    try:
        task = db.query(BoardTask).filter(BoardTask.id == task_id, BoardTask.workspace_id == workspace_id).first()
    except Exception:  # noqa: BLE001 — logged; the card stands without the warning
        logger.exception("[F430] could not read card %s for its agent's tools", task_id)
        _rolled_back(db)
        return {}
    said = card_assignment_warning(db, task) if task is not None else {}
    return {**said, "warning": said["warning"] + TELL_THE_OWNER} if said else {}


def says_what_it_cannot_call(handler: Handler) -> Handler:
    """Wrap platform_create_task or platform_assign_task (inside the ticket-number wrapper,
    where the answer's ``task_id`` is the card's id): a card given to an agent carries the
    warning when its sessions cannot call what the card names."""
    @functools.wraps(handler)
    async def wrapped(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        out = await handler(db, workspace_id, params)
        if not isinstance(out, dict) or out.get("success") is not True or not out.get("task_id"):
            return out
        warning = _for_the_card(db, workspace_id, out["task_id"])
        return {**out, **warning} if warning else out
    return wrapped


__all__ = ["CANNOT_CALL_KEYS", "TELL_THE_OWNER", "says_what_it_cannot_call"]
