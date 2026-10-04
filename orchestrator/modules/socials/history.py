"""PRD-251C Wave 1 (C5; US-C103): what the workspace posted, so nothing repeats by accident.

History is every post of the workspace that went out, is approved or scheduled, or waits for
a person (``HISTORY_STATUSES``), across every plan and the posts people made by hand. It
survives a plan's deletion: the plan's posts stay, unlinked, and each keeps its brief, whose
first line is the topic a plan made it from.

Each item says what research and the content bank compare a new idea with: the post's title;
its topic and angle (the bank's topic it was made from, while its plan lives; else the
brief's first line, with no angle); its format and channels; the first line of its copy; its
date (when it went out, else when it is scheduled or planned, else when it was made), its
state, and its numbers and engagement once read (PRD-251C US-C405: research prefers topics like
the best performers). Newest first, ``days`` back (a post scheduled ahead counts too) and at most ``limit``:
``SOCIALS_HISTORY_DAYS`` and ``SOCIALS_HISTORY_LIMIT`` unless the caller asks for others,
never more than ``MAX_DAYS`` and ``MAX_LIMIT``.

Read by research (``platform_get_social_history``, and the ``history`` in
``platform_get_social_plan``'s answer, so a workspace copy of the research playbook made before
PRD-251C sees it without a prompt change), and by ``GET /api/socials/history``. The composer
gets the opening lines of the last ``SOCIALS_COMPOSE_RECENT_OPENINGS`` posts (``recent_openings``,
US-C105), so a new post does not reuse a hook.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional
from uuid import UUID

from sqlalchemy import func

from config import config
from core.models.socials import SocialPost, SocialTopic
from modules.socials import results

HISTORY_STATUSES = (
    "rendering", "needs_approval", "changes_requested",  # waiting for a person
    "approved", "scheduled",
    "publishing", "published", "partially_published",  # went out, or going
)
# The most a caller may ask for (the agent tool's schema states the same numbers).
MAX_DAYS = 365
MAX_LIMIT = 200
LINE_MAX_CHARS = 200


def first_line(text: Any) -> Optional[str]:
    """The first line of ``text`` that has words, trimmed and cut at ``LINE_MAX_CHARS``."""
    for line in str(text or "").splitlines():
        if line.strip():
            return line.strip()[:LINE_MAX_CHARS]
    return None


def opening(post: SocialPost) -> Optional[str]:
    """The first line of the post's copy: its base text, else its first channel's own."""
    copy = post.copy if isinstance(post.copy, dict) else {}
    channels = copy.get("channels") if isinstance(copy.get("channels"), dict) else {}
    lines = (first_line(text) for text in (copy.get("base"), *channels.values()))
    return next((line for line in lines if line), None)


def post_date(post: SocialPost) -> Optional[datetime]:
    """When the post went out (its first channel), else when it is scheduled or planned, else made."""
    out = [target.published_at for target in post.targets or [] if target.published_at is not None]
    return min(out) if out else (post.scheduled_for or post.planned_for or post.created_at)


def _bounded(asked: Optional[int], default: int, most: int) -> int:
    return max(1, min(asked if asked is not None else default, most))


def topics_by_post(db: Any, workspace_id: UUID, post_ids: Iterable[UUID]) -> Dict[UUID, SocialTopic]:
    ids = list(post_ids)
    if not ids:
        return {}
    rows = db.query(SocialTopic).filter(SocialTopic.workspace_id == workspace_id, SocialTopic.used_post_id.in_(ids)).all()
    return {row.used_post_id: row for row in rows}


def _item(post: SocialPost, topic: Optional[SocialTopic], numbers: Optional[Any] = None) -> Dict[str, Any]:
    moment = post_date(post)
    return {
        "id": str(post.id),
        "title": post.title,
        "topic": topic.title if topic is not None else first_line(post.brief),
        "angle": topic.angle if topic is not None else None,
        "format": post.format,
        "channels": sorted({target.toolkit for target in post.targets or []}),
        "opening": opening(post),
        "date": moment.isoformat() if moment is not None else None,
        "state": post.status,
        "plan_id": str(post.campaign_id) if post.campaign_id else None,
        # PRD-251C (US-C405): what the post did, once read: research prefers topics like the best.
        "numbers": dict(numbers.numbers) if numbers is not None else None,
        "engagement": numbers.engagement if numbers is not None else None,
    }


def _posts(db: Any, workspace_id: UUID, limit: int, since: Optional[datetime] = None) -> List[SocialPost]:
    """The workspace's posts in its history, newest first (scheduled, else planned, else made)."""
    moment = func.coalesce(SocialPost.scheduled_for, SocialPost.planned_for, SocialPost.created_at)
    query = db.query(SocialPost).filter(SocialPost.workspace_id == workspace_id, SocialPost.status.in_(HISTORY_STATUSES))
    if since is not None:
        query = query.filter(moment >= since)
    return query.order_by(moment.desc(), SocialPost.id).limit(limit).all()


def history(
    db: Any, workspace_id: UUID, *, days: Optional[int] = None, limit: Optional[int] = None,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    """The workspace's history (the module docstring), newest first. Another workspace's posts
    are never read."""
    days = _bounded(days, config.SOCIALS_HISTORY_DAYS, MAX_DAYS)
    limit = _bounded(limit, config.SOCIALS_HISTORY_LIMIT, MAX_LIMIT)
    since = (now or datetime.now(timezone.utc)) - timedelta(days=days)
    posts = _posts(db, workspace_id, limit, since)
    topics = topics_by_post(db, workspace_id, [post.id for post in posts])
    numbers = results.post_numbers(db, workspace_id, [post.id for post in posts])
    return [_item(post, topics.get(post.id), numbers.get(post.id)) for post in posts]


def recent_openings(db: Any, workspace_id: UUID, limit: Optional[int] = None) -> List[str]:
    """How the workspace's last posts began, newest first: the composer opens a new one differently."""
    count = _bounded(limit, config.SOCIALS_COMPOSE_RECENT_OPENINGS, MAX_LIMIT)
    return [line for line in (opening(post) for post in _posts(db, workspace_id, count)) if line]
