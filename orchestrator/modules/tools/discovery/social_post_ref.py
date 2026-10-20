"""The post a Socials tool call names by ``post_id``: one reader for the tools and the owner's card.

P256-FIX-SOC-4 (row 16, 14 Oct): Auto sent ``post_id`` '7e7c402c', the id as the Socials tab
and its own answers shorten it, and the tool refused it. A post is named by its whole id, or by
a prefix of at least ``PREFIX_MIN`` of its hex characters that starts the id of exactly one post
of the workspace. The handlers (``handlers_socials._post``) and the submit card
(``card_question_socials``) read through :func:`post_named`, so the card shows the post the click
runs on, and the card binds the call to the whole id before its grant. Another workspace's post
is never read.
"""
from __future__ import annotations

import re
from typing import Any, Optional, Tuple
from uuid import UUID

MISSING = "Missing required parameter: post_id"
NOT_AN_ID = "post_id {ref!r} is not a post's id: platform_list_social_posts lists them."
NOT_FOUND = "Post not found in this workspace: platform_list_social_posts lists them."
AMBIGUOUS = ("post_id {ref!r} starts the id of more than one post: send the whole id "
             "(platform_list_social_posts lists them).")
PREFIX_MIN = 8
# Hex first (a prefix's first 8 characters never hold a dash), then hex and dashes, shorter than a whole id.
_PREFIX = re.compile(rf"^[0-9a-f]{{{PREFIX_MIN}}}[0-9a-f-]{{0,27}}$")
_MATCHES_READ = 2  # one is the post; a second says the prefix is not unique


def post_named(db: Any, workspace_id: Any, ref: Any) -> Tuple[Optional[Any], Optional[str]]:
    """(the workspace's post ``ref`` names, None), or (None, why not, in the tools' own words)."""
    from modules.socials import service

    if ref in (None, ""):
        return None, MISSING
    try:
        key = UUID(str(ref))
    except (TypeError, ValueError, AttributeError):
        return _by_prefix(db, workspace_id, ref)
    post = service.get_post(db, UUID(str(workspace_id)), key)
    return (post, None) if post is not None else (None, NOT_FOUND)


def _by_prefix(db: Any, workspace_id: Any, ref: Any) -> Tuple[Optional[Any], Optional[str]]:
    """The one post of the workspace whose id starts with ``ref`` (hex, checked: no wildcard reaches the query)."""
    from sqlalchemy import String, cast

    from core.models.socials import SocialPost

    prefix = str(ref).strip().lower()
    if not _PREFIX.match(prefix):
        return None, NOT_AN_ID.format(ref=ref)
    rows = (db.query(SocialPost)
            .filter(SocialPost.workspace_id == UUID(str(workspace_id)), cast(SocialPost.id, String).like(f"{prefix}%"))
            .limit(_MATCHES_READ).all())
    if not rows:
        return None, NOT_FOUND
    return (rows[0], None) if len(rows) == 1 else (None, AMBIGUOUS.format(ref=ref))


__all__ = ["AMBIGUOUS", "MISSING", "NOT_AN_ID", "NOT_FOUND", "PREFIX_MIN", "post_named"]
