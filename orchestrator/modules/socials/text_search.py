"""PRD-251: what a person types into a Socials search, matched with LIKE.

The text is matched literally. LIKE's wildcards (``%``, ``_``) and its escape
character are escaped, so "100%" finds "100%" and not every title that starts
with "100". Matching is case-insensitive: the text is folded here and the
column with ``lower()`` in the query, alike on PostgreSQL and SQLite. The source
picker (S1.4, ``sources.py``) and the posts list's ``q`` (US-205, global search)
both search this way.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import func

LIKE_ESCAPE = "\\"
# The longest ``q`` GET /api/socials/posts takes.
QUERY_MAX_CHARS = 200


def escape_like(text: str) -> str:
    """``text`` with LIKE's wildcards and its escape character made literal."""
    return text.replace(LIKE_ESCAPE, LIKE_ESCAPE * 2).replace("%", LIKE_ESCAPE + "%").replace("_", LIKE_ESCAPE + "_")


def contains_pattern(text: str) -> str:
    """A LIKE pattern matching ``text`` anywhere, case folded, wildcards literal."""
    return f"%{escape_like(text.lower())}%"


def holds(column: Any, text: str) -> Any:
    """A filter: ``column`` contains ``text``, case-insensitively; NULL holds nothing."""
    return func.lower(func.coalesce(column, "")).like(contains_pattern(text), escape=LIKE_ESCAPE)
