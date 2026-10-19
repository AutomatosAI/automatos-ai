"""A blog post as the blog API returns it, marked when an agent wrote it (PRD-264 EUAI-09, EUAI-08b).

Article 50(2) of the AI Act asks that generated text be marked machine-readable as
AI-generated, and Article 50(4) ¶2 lifts the duty to disclose once a person has
reviewed the text and holds editorial responsibility for it. Both blog APIs, the
dashboard's (``api/blog.py``) and the public widget's (``api/widgets/blog.py``),
answer with :func:`post_view`: the post's own fields (``BlogPost.to_dict``),
``ai_generated`` and ``reviewed_at``.

``ai_generated`` is true when an agent wrote the post (``author_agent_id`` is set)
and no person has marked it reviewed (``reviewed_at`` is empty). Who reviewed it
(``reviewed_by_user_id``) stays out of the view: the widget's answer is public.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

AI_GENERATED_FIELD = "ai_generated"
REVIEWED_AT_FIELD = "reviewed_at"


def is_ai_generated(post: Any) -> bool:
    """True when an agent wrote ``post`` and no person has marked it reviewed."""
    agent_wrote_it = getattr(post, "author_agent_id", None) is not None
    return agent_wrote_it and getattr(post, "reviewed_at", None) is None


def _reviewed_at(post: Any) -> Optional[str]:
    reviewed_at = getattr(post, "reviewed_at", None)
    return reviewed_at.isoformat() if reviewed_at else None


def post_view(post: Any, include_content: bool = False) -> Dict[str, Any]:
    """``post.to_dict(include_content)`` with ``ai_generated`` and ``reviewed_at``: a new dict, the post untouched."""
    return {
        **post.to_dict(include_content=include_content),
        AI_GENERATED_FIELD: is_ai_generated(post),
        REVIEWED_AT_FIELD: _reviewed_at(post),
    }


__all__ = ["AI_GENERATED_FIELD", "REVIEWED_AT_FIELD", "is_ai_generated", "post_view"]
