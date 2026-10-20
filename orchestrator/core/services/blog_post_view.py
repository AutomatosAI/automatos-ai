"""A blog post as the blog API returns it, marked when an agent wrote it (PRD-264 EUAI-09, EUAI-08b).

Article 50(2) of the AI Act asks that generated text be marked machine-readable as
AI-generated, and Article 50(4) ¶2 lifts the duty to disclose once a person has
reviewed the text and holds editorial responsibility for it. Both blog APIs, the
dashboard's (``api/blog.py``) and the public widget's (``api/widgets/blog.py``),
answer with :func:`read_post_view` (or :func:`post_view` when they already hold the
post's text): the post's own fields (``BlogPost.to_dict``), ``ai_generated`` and
``reviewed_at``.

A review covers the text that was reviewed, and the text lives in a workspace file
an agent can rewrite (``core.services.blog_review.review_covers``). So the view of a
reviewed post needs its current text: ``ai_generated`` is true when an agent wrote
the post (``author_agent_id`` is set) and no review covers its current text, and
``reviewed_at`` is shown only while the review does. Who reviewed it
(``reviewed_by_user_id``) and the reviewed text's hash stay out of the view: the
widget's answer is public.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from core.services.blog_review import review_covers

AI_GENERATED_FIELD = "ai_generated"
REVIEWED_AT_FIELD = "reviewed_at"


def needs_current_text(post: Any) -> bool:
    """True when the view of ``post`` depends on its current text: someone marked it reviewed."""
    return getattr(post, "reviewed_at", None) is not None


def is_ai_generated(post: Any, current_text: Optional[str] = None) -> bool:
    """True when an agent wrote ``post`` and no review covers ``current_text``, its text as it is now."""
    agent_wrote_it = getattr(post, "author_agent_id", None) is not None
    return agent_wrote_it and not review_covers(post, current_text)


def _reviewed_at(post: Any, current_text: Optional[str]) -> Optional[str]:
    if not review_covers(post, current_text):
        return None
    return post.reviewed_at.isoformat()


def post_view(post: Any, include_content: bool = False, current_text: Optional[str] = None) -> Dict[str, Any]:
    """``post.to_dict(include_content)`` with ``ai_generated`` and ``reviewed_at``: a new dict, the post untouched.

    ``current_text`` is the post's text as it is now; without it no review counts,
    so a reviewed agent post reads ``ai_generated: true``.
    """
    return {
        **post.to_dict(include_content=include_content),
        AI_GENERATED_FIELD: is_ai_generated(post, current_text),
        REVIEWED_AT_FIELD: _reviewed_at(post, current_text),
    }


async def read_post_view(service: Any, post: Any, include_content: bool = False) -> Dict[str, Any]:
    """:func:`post_view`, reading the post's current text (``service.get_content``) only when a review needs it."""
    current_text = await service.get_content(post) if needs_current_text(post) else None
    return post_view(post, include_content=include_content, current_text=current_text)


__all__ = ["AI_GENERATED_FIELD", "REVIEWED_AT_FIELD", "is_ai_generated", "needs_current_text", "post_view",
           "read_post_view"]
