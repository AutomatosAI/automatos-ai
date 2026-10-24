"""One filename-safe slug for the modules that name workspace files.

A title becomes lowercase, every run of non-alphanumeric characters collapses to
a single ``-``, the result is cut to ``max_len`` and trimmed again, and an empty
result falls back to whatever the caller uses to keep the name unique (``post``,
``report``, ``card-<id>``). One implementation means the filenames a feature
writes stay consistent with the others.
"""

from __future__ import annotations

import re
from typing import Any

_SEPARATORS = re.compile(r"[^a-z0-9]+")
DEFAULT_MAX_LEN = 60


def slugify(title: Any, *, max_len: int = DEFAULT_MAX_LEN, fallback: str) -> str:
    """Return ``title`` as a slug of at most ``max_len`` characters.

    Non-ASCII letters are separators, which is the behaviour both callers had
    before this helper existed: a title that is entirely non-Latin collapses to
    ``fallback`` rather than to an empty filename.
    """
    slug = _SEPARATORS.sub("-", str(title or "").lower()).strip("-")
    return slug[:max_len].strip("-") or fallback
