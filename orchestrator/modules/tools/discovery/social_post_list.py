"""F379 (night 11, 7 Oct): Auto lists every post it is asked about, and what each still needs.

Night 11 (B20, chat b88ce13d): asked what was waiting for approval from tonight, Auto listed 2 of
about 35 posts, called a carousel with an empty caption "good to post", and turned the placeholder
card down for its weekday rather than for its missing photo. ``platform_list_social_posts`` gave
each post's whole content (``post.to_dict()``: every variable, the history), and the chat keeps
about 2,000 tokens of a tool's answer, so two posts fitted; there was no count of the rest.

So each post is one compact row: its id, title, status, format, template, date, whether it has
rendered files and a caption, and what it still needs in plain words (the owner's approval; a
render that failed, and why; a render not made yet; a caption). The answer gives the total that
match before the rows, and ``queue`` picks the posts by what they need: ``awaiting_approval``
(waiting for the owner), ``not_rendered`` (failed renders and drafts with nothing rendered yet)
or ``open`` (everything not yet approved or out).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

AWAITING_APPROVAL, NOT_RENDERED, OPEN = "awaiting_approval", "not_rendered", "open"
QUEUES = (AWAITING_APPROVAL, NOT_RENDERED, OPEN)
FAILED, DRAFT, CHANGES, WAITING, RENDERING = "failed", "draft", "changes_requested", "needs_approval", "rendering"
RENDER_FAILED, REQUEST_CHANGES = "render_failed", "request_changes"   # modules/socials/service.py ACTION_*
REASON_CHARS = 200

NEEDS_APPROVAL = "the owner's approval in the Socials tab"
NEEDS_RENDER = "a render: nothing is rendered yet"
NEEDS_FIX = "its render failed: {why}"
NEEDS_CHANGES = "the changes a reviewer asked for: {why}"
NEEDS_CAPTION = "a caption: its copy is empty"
IS_RENDERING = "rendering now"
UNKNOWN_QUEUE = "queue must be one of {queues}."


def _rendered(post: Mapping[str, Any]) -> bool:
    media = post.get("media") if isinstance(post.get("media"), dict) else {}
    return any(media.get(aspect) for aspect in media)


def _has_copy(post: Mapping[str, Any]) -> bool:
    copy = post.get("copy") if isinstance(post.get("copy"), dict) else {}
    channels = copy.get("channels") if isinstance(copy.get("channels"), dict) else {}
    return bool(str(copy.get("base") or "").strip() or any(str(text or "").strip() for text in channels.values()))


def _last_comment(post: Mapping[str, Any], *actions: str) -> str:
    """The latest history comment of one of ``actions`` (why a render failed, what a reviewer asked)."""
    for entry in reversed(post.get("review_log") or []):
        if isinstance(entry, dict) and entry.get("action") in actions and entry.get("comment"):
            return str(entry["comment"])[:REASON_CHARS]
    return "see platform_get_social_post"


def what_it_needs(post: Mapping[str, Any]) -> List[str]:
    """What the post still needs before it can go out, in plain words; [] when nothing."""
    status = post.get("status")
    needs: List[str] = []
    if status == FAILED:
        needs.append(NEEDS_FIX.format(why=_last_comment(post, RENDER_FAILED)))
    elif status == RENDERING:
        needs.append(IS_RENDERING)
    elif status in (DRAFT, CHANGES) and not _rendered(post):
        needs.append(NEEDS_RENDER)
    if status == CHANGES:
        needs.append(NEEDS_CHANGES.format(why=_last_comment(post, REQUEST_CHANGES)))
    if status == WAITING:
        needs.append(NEEDS_APPROVAL)
    if post.get("format") != "text" and status in (DRAFT, CHANGES, WAITING, FAILED) and not _has_copy(post):
        needs.append(NEEDS_CAPTION)
    return needs


def post_row(post: Mapping[str, Any]) -> Dict[str, Any]:
    """One post as a list row: who it is and what it still needs, never its whole content."""
    return {
        "id": post.get("id"), "title": post.get("title"), "status": post.get("status"), "format": post.get("format"),
        "template_id": post.get("template_id"), "created_at": post.get("created_at"),
        "scheduled_for": post.get("scheduled_for"), "rendered": _rendered(post), "has_caption": _has_copy(post),
        "needs": what_it_needs(post),
    }


def _not_rendered(post: Mapping[str, Any]) -> bool:
    return post.get("status") == FAILED or (post.get("status") in (DRAFT, CHANGES) and not _rendered(post))


# Each queue: the statuses it reads, and the posts among them it keeps.
_QUEUES: Dict[str, Tuple[Tuple[str, ...], Callable[[Mapping[str, Any]], bool]]] = {
    AWAITING_APPROVAL: ((WAITING,), lambda post: True),
    NOT_RENDERED: ((FAILED, DRAFT, CHANGES), _not_rendered),
    OPEN: ((DRAFT, RENDERING, WAITING, CHANGES, FAILED), lambda post: True),
}


def queue_of(value: Any) -> Tuple[Optional[Tuple[str, ...]], Callable[[Mapping[str, Any]], bool], Optional[str]]:
    """The statuses and the keep-test ``queue`` names, and None; or why it is refused. No queue keeps all."""
    if value in (None, ""):
        return None, lambda post: True, None
    if value not in _QUEUES:
        return None, lambda post: True, UNKNOWN_QUEUE.format(queues=", ".join(QUEUES))
    statuses, keep = _QUEUES[value]
    return statuses, keep, None


def listed(posts: List[Mapping[str, Any]], keep: Callable[[Mapping[str, Any]], bool], limit: int) -> Dict[str, Any]:
    """The answer: how many match, then the newest ``limit`` of them as rows."""
    matching = [post for post in posts if keep(post)]
    rows = [post_row(post) for post in matching[:limit]]
    return {"success": True, "total": len(matching), "count": len(rows), "limit": limit, "posts": rows}


__all__ = ["AWAITING_APPROVAL", "NOT_RENDERED", "OPEN", "QUEUES", "listed", "post_row", "queue_of", "what_it_needs"]
