"""
Watch notification seam -- PRD-204 S6
=====================================

Single owner of watch-flavoured NotificationDispatcher calls. Extracted from
``WatchTicker._dispatch_watch_event`` (stage 1) so the S6 verdict path, the
S8 action/escalation paths, and the S10 decision step all dispatch through
ONE tested seam instead of three re-implementations.

Contract:
- getattr-defensive on the watch (tests pass SimpleNamespace stand-ins);
- resolves ``watch.created_by`` (Clerk id) to an internal user id so the
  creator is targeted; falls back to workspace-wide when unresolvable;
- NEVER raises into the caller (returns False on any failure) -- watcher
  notifications are best-effort, the registry rows are the truth;
- scores are 0-1 internal everywhere; ONLY this display edge formats x10
  (PRD-204 Section 8 Q8).
"""

from __future__ import annotations

import logging
from typing import Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def format_score_display(score: Optional[float]) -> str:
    """0-1 internal score -> the x10 display string ('0.83' -> '8.3/10')."""
    if score is None:
        return "unscored"
    return f"{score * 10:.1f}/10"


async def dispatch_watch_notification(
    db: Session,
    watch,
    *,
    event_type: str,
    title: str,
    message: Optional[str],
    status: str = "ok",
) -> bool:
    """Fire a watch-related event through NotificationDispatcher.

    Joins the caller's transaction. Returns True iff the dispatch call
    completed; False (logged) on any failure -- never raises.
    """
    try:
        from core.models.core import User
        from core.services.notification_dispatcher import NotificationDispatcher

        workspace_id = getattr(watch, "workspace_id", None)
        if workspace_id is None:
            return False

        user_id: Optional[int] = None
        created_by = getattr(watch, "created_by", None)
        if created_by:
            user_row = (
                db.query(User.id)
                .filter(User.clerk_user_id == created_by)
                .first()
            )
            if user_row:
                user_id = user_row[0]

        dispatcher = NotificationDispatcher(db, str(workspace_id))
        await dispatcher.dispatch(
            event_type=event_type,
            title=title,
            message=message,
            link_type="watch",
            link_id=str(getattr(watch, "id", "")),
            status=status,
            user_id=user_id,
        )
        return True
    except Exception:
        logger.error(
            "[WatchNotifications] %s dispatch failed for watch %s",
            event_type,
            getattr(watch, "id", "?"),
            exc_info=True,
        )
        return False


def build_verdict_message(
    watch,
    *,
    score: Optional[float],
    explanation: Optional[str],
    terminal_state: Optional[str] = None,
) -> str:
    """One-paragraph verdict body: score (displayed x10) + explanation."""
    threshold = getattr(watch, "quality_threshold", None)
    parts = []
    if score is not None:
        line = f"Run scored {format_score_display(score)}"
        if threshold is not None:
            line += f" against a bar of {format_score_display(threshold)}"
        parts.append(line + ".")
    elif terminal_state:
        parts.append(f"The watched work reached terminal state '{terminal_state}'.")
    if explanation:
        parts.append(explanation.strip())
    return " ".join(p for p in parts if p)


async def notify_watch_verdict(
    db: Session,
    watch,
    *,
    score: Optional[float],
    explanation: Optional[str],
    passed: bool,
    terminal_state: Optional[str] = None,
) -> bool:
    """The S6 verdict notification: score + one-paragraph explanation."""
    title_bits = getattr(watch, "title", "") or "watched work"
    verdict_word = "passed" if passed else "needs a look"
    title = f"Watch verdict ({verdict_word}): {title_bits[:100]}"
    message = build_verdict_message(
        watch, score=score, explanation=explanation, terminal_state=terminal_state
    )
    return await dispatch_watch_notification(
        db,
        watch,
        event_type="watch_verdict",
        title=title,
        message=message,
        status="ok" if passed else "warning",
    )
