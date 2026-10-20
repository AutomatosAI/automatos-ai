"""The editorial-review columns of ``blog_posts`` (PRD-264 EUAI-08b).

Article 50(4) ¶2 of the AI Act: AI-written text published to inform the public is
disclosed as AI-generated unless a person reviewed it and holds editorial
responsibility. ``BlogPost`` takes these columns from :class:`BlogPostReviewColumns`
(``core.models.core`` is over the file-size limit, so they live here):

* ``reviewed_by_user_id`` and ``reviewed_at``: who marked the post reviewed, and when.
* ``reviewed_content_sha256``: the SHA-256 of the text they reviewed. A post's text
  lives in a workspace file (``content/blog/<slug>.md``) that an agent's file tools
  can rewrite without going through ``BlogService.update_post``, so the review holds
  only while the current text still hashes to this value
  (``core.services.blog_review.review_covers``).

All three are nullable with no default, and the revision ``prd264_blog_review`` adds
them with ``IF NOT EXISTS``.
"""
from __future__ import annotations

from sqlalchemy import Column, DateTime, String

# A SHA-256 digest as lowercase hex.
SHA256_HEX_LENGTH = 64


class BlogPostReviewColumns:
    """Declarative mixin: who reviewed a blog post, when, and the hash of the text they read."""

    reviewed_by_user_id = Column(String(255), nullable=True)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)
    reviewed_content_sha256 = Column(String(SHA256_HEX_LENGTH), nullable=True)


__all__ = ["SHA256_HEX_LENGTH", "BlogPostReviewColumns"]
