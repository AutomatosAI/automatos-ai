"""PRD-268 §6.2.4, §6.9 — open suggestions lapse after their 7 days.

A suggestion is open until the person starts or hides it, or it expires: to make room past
``max_open_per_person`` (the suggestion tool), when a failing streak ends
(``services.proactive.failing_streaks``), or when its ``expires_at`` has passed. A
suggestion's ``expires_at`` is 7 days after it was made, and a streak's card has it pushed
on at each failure, so a live streak never lapses.

``expire_lapsed`` is the lapse: one workspace-scoped UPDATE, run before anything reads the
open suggestions. Every check-in runs it at its start, before its caps and the dedupe read
(``services.proactive.runner``), and so do Start, Hide and the list
(``services.proactive.decisions``, ``services.proactive.listing``). A lapsed suggestion
then counts toward no cap, blocks no repeat, lists as ``expired``, and Start or Hide of it
is refused by its state. Each one that lapsed is told on the live lane (``suggestion_changed``,
``services.proactive.live``) once the lapse has committed.
"""
from __future__ import annotations

import logging
from datetime import datetime
from uuid import UUID

from sqlalchemy import update
from sqlalchemy.orm import Session

from core.models.proactive import SUGGESTION_EXPIRED, SUGGESTION_OPEN, ProactiveSuggestion
from services.proactive.live import suggestion_changed

logger = logging.getLogger(__name__)

__all__ = ["expire_lapsed"]


def expire_lapsed(db: Session, workspace_id: UUID, now: datetime) -> int:
    """Expire the workspace's open suggestions whose ``expires_at`` is at or before ``now``,
    in one UPDATE, commit when any lapsed and tell each on the live lane; how many did. A
    suggestion with no ``expires_at`` never lapses, and no other workspace's row is touched."""
    table = ProactiveSuggestion.__table__
    lapsed = db.execute(
        update(table)
        .where(table.c.workspace_id == workspace_id, table.c.status == SUGGESTION_OPEN, table.c.expires_at <= now)
        .values(status=SUGGESTION_EXPIRED)
        .returning(table.c.id, table.c.user_id)
    ).all()
    if lapsed:
        db.commit()
        logger.info("[proactive] %d open suggestion(s) lapsed in workspace %s", len(lapsed), workspace_id)
    for row in lapsed:
        suggestion_changed(db, workspace_id=workspace_id, suggestion_id=row.id, user_id=row.user_id,
                           status=SUGGESTION_EXPIRED)
    return len(lapsed)
