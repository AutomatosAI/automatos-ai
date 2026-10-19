"""PRD-256 P256-FIX-N17-11 (F443, night 16): a publish card shows the post it asks about.

Night 16's blog card (#2705, 19:59) read "publish a blog post title='Guji Uraga: our Ethiopian
lot'" and nothing more, for a call that saved a draft (``publish_immediately`` false): the
owner could not see what the post said, nor that nothing would go live. The social card
(#2710, 20:30) read "submit a social post to publish" with no post at all: its only subject
is a ``post_id``, which the card's subject line does not show (F091 keeps ids off it).

Both actions now have card lines (``card_question.READERS``):
- a blog post: whether it goes live (the handler's own reading, ``handlers_blog.goes_live``),
  its excerpt, category, tags, first line and length;
- a social post sent for approval: the post this workspace holds under that id (its title,
  channels, format, caption and when it is planned or scheduled), the note for the reviewer,
  and that nothing posts until it is approved in Socials. An id that is not a UUID, or a post
  of another workspace, gives no lines: the card asks on its act alone.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import UUID

from modules.tools.discovery.card_question_text import said_line, value_line

GOES_LIVE = "goes live: yes, on this workspace's blog"
STAYS_A_DRAFT = "goes live: no, saved as a draft; nothing is published"
NO_EXCERPT = "excerpt: none given; the post's opening is used"
LENGTH = "length: {words} words"
POST = "post: '{title}'"
AFTER_THE_CLICK = "after the click: it waits in Socials for approval; nothing posts until it is approved there"
WHEN = "{moment:%H:%M} on {moment.day} {moment:%b} UTC"
HEADING = "#"


def blog_lines(db: Any, workspace_id: Any, action: str, params: Dict[str, Any]) -> List[str]:
    """Whether the post goes live, then its excerpt, category, tags, first line and length."""
    from modules.tools.discovery.handlers_blog import _normalize_tags, goes_live

    content = str(params.get("content") or "")
    excerpt = params.get("excerpt")
    return [
        value_line(GOES_LIVE if goes_live(params) else STAYS_A_DRAFT),
        said_line("excerpt", excerpt) if excerpt not in (None, "") else value_line(NO_EXCERPT),
        said_line("category", params.get("category")),
        said_line("tags", _normalize_tags(params.get("tags"))),
        said_line("first line", first_line(content)),
        value_line(LENGTH.format(words=len(content.split()))),
    ]


def social_submit_lines(db: Any, workspace_id: Any, action: str, params: Dict[str, Any]) -> List[str]:
    """The post this workspace holds under ``post_id``: its title, channels, format, caption and
    when it is planned or scheduled; the reviewer's note; what the click does. [] when the id
    is not a post of this workspace."""
    from modules.socials.service import get_post

    try:
        key = UUID(str(params.get("post_id")))
    except (TypeError, ValueError, AttributeError):
        return []
    post = get_post(db, workspace_id, key)
    if post is None:
        return []
    copy = post.copy if isinstance(post.copy, dict) else {}
    return [
        value_line(POST.format(title=post.title)),
        said_line("channels", _channels(post, copy)),
        said_line("format", post.format),
        said_line("caption", first_line(str(copy.get("base") or ""))),
        *_when_lines(post),
        said_line("note", params.get("note")),
        value_line(AFTER_THE_CLICK),
    ]


def first_line(text: str) -> str:
    """The first line with words on it that is not a heading."""
    return next((line.strip() for line in text.splitlines()
                 if line.strip() and not line.strip().startswith(HEADING)), "")


def _channels(post: Any, copy: Dict[str, Any]) -> List[str]:
    """The post's target channels, else the channels its copy is written for, in name order."""
    targets = sorted({target.toolkit for target in (post.targets or []) if getattr(target, "toolkit", None)})
    channels = copy.get("channels")
    return targets or (sorted(channels) if isinstance(channels, dict) else [])


def _when_lines(post: Any) -> List[str]:
    """When the post is scheduled, else planned for; that no time is set when neither is."""
    if post.scheduled_for is not None:
        return [said_line("scheduled for", _utc(post.scheduled_for))]
    if post.planned_for is not None:
        return [said_line("planned for", _utc(post.planned_for))]
    return [said_line("planned for", None)]


def _utc(moment: datetime) -> str:
    """'09:00 on 20 Oct UTC' (a naive time is the database's own, UTC)."""
    moment = moment.replace(tzinfo=timezone.utc) if moment.tzinfo is None else moment.astimezone(timezone.utc)
    return WHEN.format(moment=moment)


__all__ = ["blog_lines", "first_line", "social_submit_lines"]
