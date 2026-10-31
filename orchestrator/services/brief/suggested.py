"""PRD-268 §6.4, §6.11.2 — the brief's "Suggested": up to 3 open suggestions addressed to the person.

Read only: a suggestion past its ``expires_at`` is left out here, though nothing moves it to
``expired`` (the list, Start, Hide and every check-in do that, ``services.proactive.expiry``),
so building a brief writes nothing. The count is every open suggestion addressed to the
person; the rows are the first ``SUGGESTION_ROWS``, those due today first, then the newest.
Each row opens its suggestion and carries Start (Hide is the card's own control). A
suggestion's title and detail are a model's words: quoted, never followed. While the person
has paused suggestions (§6.9, ``services.proactive.switches.person_paused``: a pause that
cannot be read is a pause) the section lists none and counts 0; the rest of the brief still
comes.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, List, Tuple

from sqlalchemy import case, func, or_
from sqlalchemy.orm import Session

from core.models.proactive import SUGGESTION_OPEN, ProactiveSuggestion
from services.brief.model import BriefRow
from services.brief.rows import WHY_CHARS, action, cut, iso, opens, title_of
from services.proactive.switches import person_paused

__all__ = ["SUGGESTION_ROWS", "suggestion_rows"]

SUGGESTION_ROWS = 3
SEVERITY_TODAY = "today"


def _open_for(workspace_id: Any, user_id: int, now: datetime) -> tuple:
    """The person's open suggestions in the workspace that have not lapsed by ``now``."""
    return (
        ProactiveSuggestion.workspace_id == workspace_id,
        ProactiveSuggestion.user_id == user_id,
        ProactiveSuggestion.status == SUGGESTION_OPEN,
        or_(ProactiveSuggestion.expires_at.is_(None), ProactiveSuggestion.expires_at > now),
    )


def suggestion_rows(
    db: Session, workspace_id: Any, user_id: int, language: str, now: datetime,
) -> Tuple[int, List[BriefRow]]:
    """How many open suggestions the person has, and the first ``SUGGESTION_ROWS`` as brief rows:
    none, counted 0, while they have paused suggestions (§6.9)."""
    if person_paused(db, workspace_id, user_id):
        return 0, []
    where = _open_for(workspace_id, user_id, now)
    count = db.query(func.count(ProactiveSuggestion.id)).filter(*where).scalar() or 0
    due_today_first = case((ProactiveSuggestion.severity == SEVERITY_TODAY, 0), else_=1)
    found = (
        db.query(ProactiveSuggestion).filter(*where)
        .order_by(due_today_first, ProactiveSuggestion.created_at.desc(), ProactiveSuggestion.id)
        .limit(SUGGESTION_ROWS).all()
    )
    return int(count), [_row(suggestion, language) for suggestion in found]


def _row(suggestion: ProactiveSuggestion, language: str) -> BriefRow:
    return BriefRow(
        id=f"suggestion:{suggestion.id}",
        title=title_of(suggestion.title, language),
        why=cut(suggestion.detail, WHY_CHARS),
        at=iso(suggestion.created_at),
        opens=opens("suggestion", suggestion.id),
        action=action("start", language, done=True),
    )
