"""Editorial review of a blog post, and the edits that undo it (PRD-264 EUAI-08b).

Article 50(4) ¶2 of the AI Act: AI-written text published to inform the public is
disclosed as AI-generated unless a person reviewed it and holds editorial
responsibility. A workspace member marks a post reviewed (``BlogService.mark_reviewed``
records who and when). The review covers the text as it was reviewed, so an edit
that changes the post afterwards clears it and the post is labelled again
(``BlogService.update_post``).

An edit is a change to any field but ``status``: publishing or unpublishing does
not change what the reviewer read. Saving the editor without changing anything is
not an edit either, so a reviewed post can be saved and then published without
losing its review. Text is compared without its surrounding whitespace, the way
the editor trims what it sends.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Optional

# The values ``update_post`` writes when an edit clears the review.
REVIEW_CLEARED: Mapping[str, None] = MappingProxyType({"reviewed_by_user_id": None, "reviewed_at": None})

# Fields an update may carry that do not change what the reviewer read.
NOT_AN_EDIT = frozenset({"status"})


def _same(old: Any, new: Any) -> bool:
    if isinstance(old, str) or isinstance(new, str):
        return (old or "").strip() == (new or "").strip()
    return (old or None) == (new or None)


def edits_change_the_post(post: Any, updates: Mapping[str, Any], current_content: Optional[str]) -> bool:
    """True when ``updates`` change any field of ``post`` but its status.

    ``current_content`` is the post's full text (it lives in a workspace file, not
    on the row); it is only read when ``updates`` carries ``content``.
    """
    def old(key: str) -> Any:
        return current_content if key == "content" else getattr(post, key, None)

    return any(not _same(old(key), new) for key, new in updates.items() if key not in NOT_AN_EDIT)


__all__ = ["NOT_AN_EDIT", "REVIEW_CLEARED", "edits_change_the_post"]
