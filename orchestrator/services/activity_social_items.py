"""PRD-251 D10 (US-307): the workspace's coming social posts in the Command Center calendar.

The sixth source of ``ActivityService.get_schedule`` (``services/activity_service.py``,
over 800 lines, only adds it to its sources). Each post of the caller's workspace that is
still to go out and whose slot falls in the window is one item, ``social-<post_id>``: its
title, its slot (UTC ISO), its status, the post's id and its own timezone
(``social_posts.timezone``: there is no workspace timezone), so the calendar shows the time
where the post was planned, says where it stands, opens it and moves it
(``PUT /api/socials/posts/{id}/slot``: the slot moves, an approval stands).

16 Nov 2026: it was scheduled posts only, so a post waiting for approval or planned for
Friday was on the Socials calendar and missing from this one. A scheduled post sits at the
time it publishes (``scheduled_for``); a draft, a post waiting for approval, one sent back
and an approved one sit at the slot they are planned for (``planned_for``). A post that
went out, failed, was missed or is being made is the Socials calendar's own.
Nothing shows while Socials is off for the workspace (D1).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import UUID

from sqlalchemy import and_, or_
from sqlalchemy.orm import Session

from core.models.socials import SocialPost
from core.models.workspaces import Workspace
from modules.socials.service import APPROVED, CHANGES_REQUESTED, DRAFT, NEEDS_APPROVAL, SCHEDULED
from modules.socials.settings import socials_off_reason

ITEM_TYPE = "social"
NAME_MAX_CHARS = 80
DEFAULT_TIMEZONE = "UTC"
# Not scheduled yet: these sit at the slot they are planned for.
PLANNED_STATUSES = (DRAFT, NEEDS_APPROVAL, CHANGES_REQUESTED, APPROVED)


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _slot(row: Any) -> datetime:
    """Where the post sits: the time it publishes once scheduled, else the slot it is planned for."""
    return row.scheduled_for if row.status == SCHEDULED else row.planned_for


def _item(row: Any) -> Dict[str, Any]:
    tz = row.timezone or DEFAULT_TIMEZONE
    return {
        "id": f"social-{row.id}",
        "post_id": str(row.id),
        "type": ITEM_TYPE,
        "name": (row.title or "Social post").strip()[:NAME_MAX_CHARS],
        "next_run_at": _utc(_slot(row)).isoformat(),
        "status": row.status,
        "timezone": tz,
        "frequency": "One-off",
        "agent_name": None,
        "agent_id": None,
        "recurrence": {"cron_expression": None, "interval_minutes": None, "timezone": tz, "active_hours": None},
    }


def social_post_items(db: Session, workspace_id: UUID, now: datetime, horizon: datetime) -> List[Dict[str, Any]]:
    """The workspace's posts still to go out with a slot in ``[now, horizon]``, soonest
    first, while Socials is on for it."""
    if socials_off_reason(db.get(Workspace, workspace_id)) is not None:
        return []
    scheduled = and_(SocialPost.status == SCHEDULED, SocialPost.scheduled_for >= now, SocialPost.scheduled_for <= horizon)
    planned = and_(SocialPost.status.in_(PLANNED_STATUSES), SocialPost.planned_for >= now, SocialPost.planned_for <= horizon)
    rows = (
        db.query(SocialPost.id, SocialPost.title, SocialPost.status, SocialPost.scheduled_for, SocialPost.planned_for,
                 SocialPost.timezone)
        .filter(SocialPost.workspace_id == workspace_id, or_(scheduled, planned))
        .all()
    )
    return [_item(row) for row in sorted(rows, key=_slot)]
