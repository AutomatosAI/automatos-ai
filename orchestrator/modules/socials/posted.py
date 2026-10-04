"""PRD-251C (C7, US-C408): Posted, what went out, newest first.

Every post of the workspace that went out on at least one channel (``published`` or
``partially_published``), ordered by when its last channel went out. Each says what the
Posted view shows: its title, format and plan, its topic (the bank's topic it was made from,
else its brief's first line), when it went out, each channel's receipt (the link and the id
the platform gave) and its numbers (``results.post_numbers``: empty until the first read).
Filtered by plan, channel and format; at most ``MAX_POSTED`` a page. Another workspace's
posts are never read.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from uuid import UUID

from sqlalchemy import func, select

from core.models.socials import SocialCampaign, SocialPost, SocialPostTarget
from modules.socials import history, results

POSTED_STATUSES = ("published", "partially_published")
DEFAULT_POSTED = 50
MAX_POSTED = 200
RECEIPT_KEYS = ("toolkit", "post_kind", "permalink", "remote_id", "published_at")


def _receipts(post: SocialPost) -> List[Dict[str, Any]]:
    went = [target.to_dict() for target in post.targets or [] if target.published_at is not None]
    return [{key: target[key] for key in RECEIPT_KEYS} for target in sorted(went, key=lambda t: t["published_at"] or "")]


def _plan_names(db: Any, workspace_id: UUID, posts: List[SocialPost]) -> Dict[UUID, str]:
    ids = {post.campaign_id for post in posts if post.campaign_id}
    if not ids:
        return {}
    rows = db.query(SocialCampaign.id, SocialCampaign.name).filter(SocialCampaign.workspace_id == workspace_id, SocialCampaign.id.in_(ids))
    return {row.id: row.name for row in rows}


def _query(db: Any, workspace_id: UUID, plan_id: Optional[UUID], channel: Optional[str], fmt: Optional[str]) -> Any:
    went_out = func.max(SocialPostTarget.published_at)
    query = (
        db.query(SocialPost, went_out.label("went_out"))
        .join(SocialPostTarget, SocialPostTarget.post_id == SocialPost.id)
        .filter(SocialPost.workspace_id == workspace_id, SocialPost.status.in_(POSTED_STATUSES), SocialPostTarget.published_at.isnot(None))
    )
    if plan_id is not None:
        query = query.filter(SocialPost.campaign_id == plan_id)
    if fmt:
        query = query.filter(SocialPost.format == fmt)
    if channel:
        on_channel = select(SocialPostTarget.post_id).where(SocialPostTarget.toolkit == channel, SocialPostTarget.published_at.isnot(None))
        query = query.filter(SocialPost.id.in_(on_channel))
    return query.group_by(SocialPost.id).order_by(went_out.desc(), SocialPost.id)


def posted(
    db: Any, workspace_id: UUID, *, plan_id: Optional[UUID] = None, channel: Optional[str] = None, fmt: Optional[str] = None,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """What went out (the module docstring), newest first."""
    count = max(1, min(limit or DEFAULT_POSTED, MAX_POSTED))
    rows = _query(db, workspace_id, plan_id, channel, fmt).limit(count).all()
    posts = [post for post, _ in rows]
    topics = history.topics_by_post(db, workspace_id, [post.id for post in posts])
    numbers = results.post_numbers(db, workspace_id, [post.id for post in posts])
    plans = _plan_names(db, workspace_id, posts)
    return [
        {
            "id": str(post.id), "title": post.title, "format": post.format,
            "plan_id": str(post.campaign_id) if post.campaign_id else None, "plan_name": plans.get(post.campaign_id),
            "topic": topics[post.id].title if post.id in topics else history.first_line(post.brief),
            "went_out_at": went_out.isoformat() if went_out else None,
            "receipts": _receipts(post),
            "numbers": numbers[post.id].to_dict() if post.id in numbers else None,
        }
        for post, went_out in rows
    ]
