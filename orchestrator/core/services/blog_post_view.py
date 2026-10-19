"""A blog post as the blog API returns it, marked when an agent wrote it (PRD-264 EUAI-09).

Article 50(2) of the AI Act asks that generated text be marked machine-readable as
AI-generated. Both blog APIs, the dashboard's (``api/blog.py``) and the public widget's
(``api/widgets/blog.py``), answer with :func:`post_view`: the post's own fields
(``BlogPost.to_dict``) and ``ai_generated``.

``ai_generated`` is true when an agent wrote the post (``author_agent_id`` is set).
EUAI-08b adds "and no person has marked it reviewed" here, when its columns land.
"""
from __future__ import annotations

from typing import Any, Dict

AI_GENERATED_FIELD = "ai_generated"


def is_ai_generated(post: Any) -> bool:
    """True when an agent wrote ``post`` (its ``author_agent_id`` is set)."""
    return getattr(post, "author_agent_id", None) is not None


def post_view(post: Any, include_content: bool = False) -> Dict[str, Any]:
    """``post.to_dict(include_content)`` with ``ai_generated``: a new dict, the post untouched."""
    return {**post.to_dict(include_content=include_content), AI_GENERATED_FIELD: is_ai_generated(post)}


__all__ = ["AI_GENERATED_FIELD", "is_ai_generated", "post_view"]
