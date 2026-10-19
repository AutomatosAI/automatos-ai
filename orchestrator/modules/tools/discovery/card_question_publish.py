"""PRD-256 P256-FIX-N17-11 (F443, night 16): a publish card shows the post it asks about.

Night 16's blog card (#2705, 19:59) read "publish a blog post title='Guji Uraga: our Ethiopian
lot'" and nothing more, for a call that saved a draft (``publish_immediately`` false): the
owner could not see what the post said, nor that nothing would go live. The social card
(#2710, 20:30) read "submit a social post to publish" with no post at all: its only subject
is a ``post_id``, which the card's subject line does not show (F091 keeps ids off it).

Both actions now have card lines (``card_question.READERS``):
- a blog post: whether it goes live (the handler's own reading, ``handlers_blog.goes_live``),
  its excerpt, category, tags, first line and length;
- a social post sent for approval: the post this workspace holds under that id (its title and
  id, its status as the click moves it, its channels, format, caption, rendered files and when
  it is planned or scheduled), the note for the reviewer, and that it publishes only after a
  person approves it in Socials. An id that is not a post of this workspace gives no lines: the
  card asks on its act alone.

P256-FIX-SOC-4 (F431-b/c, card #2648): the channels are the post's targets alone. Its copy's
per-channel text is not where it goes (Auto wrote copy.channels.instagram on a post with no
channel), so a post with none says it would publish nowhere.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List

from modules.tools.discovery.card_question_text import change_line, said_line, value_line

GOES_LIVE = "goes live: yes, on this workspace's blog"
STAYS_A_DRAFT = "goes live: no, saved as a draft; nothing is published"
NO_EXCERPT = "excerpt: none given; the post's opening is used"
LENGTH = "length: {words} words"
POST = "post: '{title}' ({id})"
AFTER_THE_CLICK = ("the click sends it for approval; it publishes only after a person approves it in the "
                   "Socials tab")
NO_CHANNELS = "none: it would publish nowhere"
RENDERED = "rendered files: {count}"
WAITING = "needs approval"
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
    """The post this workspace holds under ``post_id`` (``social_post_ref``, the tool's own reader):
    its title and id, status, channels, format, caption, rendered files and when it is planned or
    scheduled; the reviewer's note; what the click does. [] when the id names no post of this workspace."""
    from modules.tools.discovery.social_post_ref import post_named

    post = post_named(db, workspace_id, params.get("post_id"))[0]
    if post is None:
        return []
    copy = post.copy if isinstance(post.copy, dict) else {}
    return [
        value_line(POST.format(title=post.title, id=post.id)),
        change_line("status", str(post.status).replace("_", " "), WAITING),
        said_line("channels", _channels(post)),
        said_line("format", post.format),
        said_line("caption", first_line(str(copy.get("base") or ""))),
        value_line(RENDERED.format(count=_rendered_files(post))),
        *_when_lines(post),
        said_line("note", params.get("note")),
        value_line(AFTER_THE_CLICK),
    ]


def first_line(text: str) -> str:
    """The first line with words on it that is not a heading."""
    return next((line.strip() for line in text.splitlines()
                 if line.strip() and not line.strip().startswith(HEADING)), "")


def _channels(post: Any) -> Any:
    """Where the post publishes: each target as 'instagram carousel', in name order, or that it has none."""
    targets = sorted(f"{target.toolkit} {target.post_kind}" for target in (post.targets or [])
                     if getattr(target, "toolkit", None))
    return targets or NO_CHANNELS


def _rendered_files(post: Any) -> int:
    """How many files the post's render made (``media``: aspect → its files)."""
    media = post.media if isinstance(post.media, dict) else {}
    return sum(len(files) for files in media.values() if isinstance(files, list))


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
