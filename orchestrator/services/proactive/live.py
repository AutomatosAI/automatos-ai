"""PRD-268 §6.5.2, US-C08 — ``suggestion_changed``: every open client hears that a suggestion changed.

A suggestion's change rides the live lane the board and the chats already use
(``services.board_events``, the ``board_events`` channel): ``suggestion_changed`` with the
workspace, the suggestion's id, its addressee and its new status, never its words. A client
refetches its suggestions on it, and ``user_id`` lets it drop another person's.

It is told after the change has committed, for each of four changes:

* **made**: a suggestion posted in its addressee's Auto thread (``services.proactive.delivery``),
  a run's suggestions and a failing streak's first card alike;
* **started** and **hidden**: a person's Start or Hide (``services.proactive.decisions``);
* **expired**: a failing streak's card when the agent works again
  (``services.proactive.failing_streaks``), and an open suggestion past its 7 days, lapsed
  where Start, Hide, the list and every check-in lapse them (``services.proactive.expiry``).

Best effort, as the lane's other events: Postgres sends a NOTIFY when its transaction commits,
so the NOTIFY gets a commit of its own, and a NOTIFY or a commit that fails is logged and changes
nothing that was decided.
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

__all__ = ["suggestion_changed"]


def suggestion_changed(db: Session, *, workspace_id: Any, suggestion_id: Any, user_id: int, status: str) -> None:
    """Tell the live lane the suggestion is now ``status``, once its change has committed; never raises."""
    from services.board_events import notify_suggestion_event

    try:
        notify_suggestion_event(db, workspace_id=workspace_id, suggestion_id=suggestion_id, user_id=user_id,
                                status=status)
        db.commit()  # the NOTIFY is sent when this commits
    except Exception:  # noqa: BLE001 — logged; the change it tells of has committed and stands
        db.rollback()
        logger.exception("[proactive] the live event of suggestion %s could not be sent", suggestion_id)
