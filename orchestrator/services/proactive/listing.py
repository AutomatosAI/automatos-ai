"""PRD-268 §6.2.4, FR-14 — what the suggestion routes list: suggestions by state, and the runs.

* ``suggestions``: one workspace's suggestions in one state, newest first. ``everyone`` (an
  owner or admin) lists every person's; anyone else lists only the ones addressed to them
  (``user_id``), and nothing without one. A person who paused suggestions (§6.9,
  ``services.proactive.switches.person_paused``: a pause that cannot be read is a pause)
  lists none of their own: nothing, or, as an owner or admin, every other person's. The
  workspace's open suggestions past their 7 days lapse first (``services.proactive.expiry``):
  they list as ``expired``, never ``open``.
* ``runs``: one workspace's check-in runs since a time, newest first: the audit an owner
  compares with the tickets the heartbeats filed over the shadow week (§10). Without a time,
  the last ``RUNS_WINDOW``.

Both read only the caller's workspace, and each list is bounded by its ``limit``.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import List, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.proactive import ProactiveRun, ProactiveSuggestion
from services.proactive.expiry import expire_lapsed
from services.proactive.switches import person_paused

__all__ = [
    "DEFAULT_RUNS",
    "DEFAULT_SUGGESTIONS",
    "MAX_RUNS",
    "MAX_SUGGESTIONS",
    "RUNS_WINDOW",
    "runs",
    "suggestions",
]

DEFAULT_SUGGESTIONS = 50
MAX_SUGGESTIONS = 100
DEFAULT_RUNS = 200
MAX_RUNS = 500
# The shadow week (§10): the runs an owner compares with the heartbeats' tickets.
RUNS_WINDOW = timedelta(days=7)


def suggestions(db: Session, workspace_id: UUID, *, status: str, limit: int, everyone: bool,
                user_id: Optional[int]) -> List[ProactiveSuggestion]:
    """The workspace's suggestions in ``status``, newest first: every person's when
    ``everyone``, else ``user_id``'s alone (none without one). While ``user_id`` has paused
    suggestions (§6.9; a pause that cannot be read is a pause) none of theirs are listed:
    nothing, or every other person's when ``everyone``. Open ones that have lapsed are
    expired before the read."""
    paused = user_id is not None and person_paused(db, workspace_id, user_id)
    if not everyone and (user_id is None or paused):
        return []
    expire_lapsed(db, workspace_id, datetime.now(timezone.utc))
    query = db.query(ProactiveSuggestion).filter(
        ProactiveSuggestion.workspace_id == workspace_id, ProactiveSuggestion.status == status)
    if not everyone:
        query = query.filter(ProactiveSuggestion.user_id == user_id)
    elif paused:
        query = query.filter(ProactiveSuggestion.user_id != user_id)
    return query.order_by(ProactiveSuggestion.created_at.desc(), ProactiveSuggestion.id).limit(limit).all()


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)


def runs(db: Session, workspace_id: UUID, *, since: Optional[datetime], limit: int,
         now: Optional[datetime] = None) -> List[ProactiveRun]:
    """The workspace's runs created at or after ``since`` (a time without a zone is UTC),
    else in the last ``RUNS_WINDOW``, newest first."""
    start = _aware(since) if since is not None else (now or datetime.now(timezone.utc)) - RUNS_WINDOW
    return (
        db.query(ProactiveRun)
        .filter(ProactiveRun.workspace_id == workspace_id, ProactiveRun.created_at >= start)
        .order_by(ProactiveRun.created_at.desc(), ProactiveRun.id)
        .limit(limit)
        .all()
    )
