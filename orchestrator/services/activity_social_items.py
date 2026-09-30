"""PRD-251 D10 (US-307): scheduled social posts in the Command Center calendar.

The sixth source of ``ActivityService.get_schedule`` (``services/activity_service.py``,
over 800 lines, only adds it to its sources). Each scheduled post of the caller's
workspace whose slot falls in the window is one item, ``social-<post_id>``: its
title, its slot (UTC ISO), the post's id and its own timezone (``social_posts.timezone``:
there is no workspace timezone), so the calendar shows the time where the post was
scheduled, opens the post and reschedules it (``POST /api/socials/posts/{id}/schedule``:
the slot moves, the approval stands). Nothing shows while Socials is off for the
workspace (D1).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.socials import SocialPost
from core.models.workspaces import Workspace
from modules.socials.service import SCHEDULED
from modules.socials.settings import socials_off_reason

ITEM_TYPE = "social"
NAME_MAX_CHARS = 80
DEFAULT_TIMEZONE = "UTC"


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _item(row: Any) -> Dict[str, Any]:
    tz = row.timezone or DEFAULT_TIMEZONE
    return {
        "id": f"social-{row.id}",
        "post_id": str(row.id),
        "type": ITEM_TYPE,
        "name": (row.title or "Social post").strip()[:NAME_MAX_CHARS],
        "next_run_at": _utc(row.scheduled_for).isoformat(),
        "timezone": tz,
        "frequency": "One-off",
        "agent_name": None,
        "agent_id": None,
        "recurrence": {"cron_expression": None, "interval_minutes": None, "timezone": tz, "active_hours": None},
    }


def social_post_items(db: Session, workspace_id: UUID, now: datetime, horizon: datetime) -> List[Dict[str, Any]]:
    """The workspace's scheduled posts with a slot in ``[now, horizon]``, while
    Socials is on for it."""
    if socials_off_reason(db.get(Workspace, workspace_id)) is not None:
        return []
    rows = (
        db.query(SocialPost.id, SocialPost.title, SocialPost.scheduled_for, SocialPost.timezone)
        .filter(
            SocialPost.workspace_id == workspace_id,
            SocialPost.status == SCHEDULED,
            SocialPost.scheduled_for >= now,
            SocialPost.scheduled_for <= horizon,
        )
        .order_by(SocialPost.scheduled_for)
        .all()
    )
    return [_item(row) for row in rows]
