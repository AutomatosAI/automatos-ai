"""Editorial review of a blog post, what it covers, and the edits that undo it (PRD-264 EUAI-08b).

Article 50(4) ¶2 of the AI Act: AI-written text published to inform the public is
disclosed as AI-generated unless a person reviewed it and holds editorial
responsibility. A workspace member marks a post reviewed (``BlogService.mark_reviewed``
records who, when, and the SHA-256 of the text they read). The review covers that
text only:

* An edit through ``BlogService.update_post`` that changes the post clears it.
* The text lives in a workspace file (``content/blog/<slug>.md``) that an agent's
  file tools can rewrite without ``update_post``. So a review counts only while the
  post's current text still hashes to the reviewed value (:func:`review_covers`).

An edit is a change to any field but ``status``: publishing or unpublishing does
not change what the reviewer read. Saving the editor without changing anything is
not an edit either, so a reviewed post can be saved and then published without
losing its review. Text is compared, and hashed, without its surrounding
whitespace, the way the editor trims what it sends.
"""
from __future__ import annotations

import hashlib
from types import MappingProxyType
from typing import Any, Mapping, Optional

# The values ``update_post`` writes when an edit clears the review.
REVIEW_CLEARED: Mapping[str, None] = MappingProxyType(
    {"reviewed_by_user_id": None, "reviewed_at": None, "reviewed_content_sha256": None}
)

# Fields an update may carry that do not change what the reviewer read.
NOT_AN_EDIT = frozenset({"status"})

TEXT_ENCODING = "utf-8"


def content_sha256(text: Optional[str]) -> str:
    """The SHA-256 (lowercase hex) of a post's text, without its surrounding whitespace."""
    return hashlib.sha256((text or "").strip().encode(TEXT_ENCODING)).hexdigest()


def review_covers(post: Any, current_text: Optional[str]) -> bool:
    """True when ``post`` is marked reviewed and ``current_text`` is the text that was reviewed.

    ``current_text`` is the post's text as it is now (``BlogService.get_content``).
    ``None`` means it was not read, so no review can be shown to cover it.
    """
    reviewed_hash = getattr(post, "reviewed_content_sha256", None)
    if getattr(post, "reviewed_at", None) is None or not reviewed_hash or current_text is None:
        return False
    return content_sha256(current_text) == reviewed_hash


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


__all__ = ["NOT_AN_EDIT", "REVIEW_CLEARED", "content_sha256", "edits_change_the_post", "review_covers"]
