"""PRD-251 S2.2a (US-207): how much copy each channel takes.

The composer fits its proposal to these, and ``GET /api/socials/channels`` hands
them to the composer's live character counts (US-209), so the two never differ:

* X (``twitter``) 280 characters;
* LinkedIn 3000;
* Instagram 2200, with at most 30 hashtags;
* TikTok 2200;
* YouTube: the title 100, the description (the channel's copy) 5000.

A channel the registry knows only generically has no limit here. Text over its
limit is trimmed at a word boundary, never mid-word; extra Instagram hashtags are
dropped from the end. Each fix says what it did, so the composer can warn.
"""
from __future__ import annotations

import re
from typing import Dict, List, Mapping, Optional, Tuple

TEXT = "text"
TITLE = "title"
HASHTAGS = "hashtags"

COPY_LIMITS: Mapping[str, Mapping[str, int]] = {
    "twitter": {TEXT: 280},
    "linkedin": {TEXT: 3000},
    "instagram": {TEXT: 2200, HASHTAGS: 30},
    "tiktok": {TEXT: 2200},
    "youtube": {TEXT: 5000, TITLE: 100},
}

_HASHTAG = re.compile(r"(?<![\w#])#\w+")
_SPACES = re.compile(r"[ \t]{2,}")


def limits_for(toolkit: str) -> Optional[Dict[str, int]]:
    """The channel's limits (``text``, and ``title`` / ``hashtags`` where it has
    them), or ``None`` for a channel with none known."""
    limits = COPY_LIMITS.get(toolkit)
    return dict(limits) if limits is not None else None


def trim_at_word(text: str, limit: int) -> str:
    """``text`` cut to at most ``limit`` characters at a word boundary. A first
    word longer than the limit is dropped whole rather than split."""
    if len(text) <= limit:
        return text
    head = text[: limit + 1]
    cut = max(head.rfind(" "), head.rfind("\n"), head.rfind("\t"))
    return head[:cut].rstrip() if cut > 0 else ""


def drop_extra_hashtags(text: str, most: int) -> Tuple[str, int]:
    """``text`` keeping its first ``most`` hashtags, and how many were dropped."""
    tags = list(_HASHTAG.finditer(text))
    if len(tags) <= most:
        return text, 0
    parts: List[str] = []
    last = 0
    for match in tags[most:]:
        parts.append(text[last:match.start()])
        last = match.end()
    parts.append(text[last:])
    return _SPACES.sub(" ", "".join(parts)).strip(), len(tags) - most


def fit_copy(toolkit: str, text: str) -> Tuple[str, List[str]]:
    """``text`` fitted to the channel's limits, and a warning for each fix."""
    limits = COPY_LIMITS.get(toolkit)
    if not limits:
        return text, []
    warnings: List[str] = []
    most = limits.get(HASHTAGS)
    if most is not None:
        text, dropped = drop_extra_hashtags(text, most)
        if dropped:
            warnings.append(f"{toolkit}: {dropped} hashtags over the limit of {most} were dropped")
    limit = limits[TEXT]
    if len(text) > limit:
        text = trim_at_word(text, limit)
        warnings.append(f"{toolkit}: the copy was trimmed to {limit} characters at a word boundary")
    return text, warnings


def fit_title(toolkits: List[str], title: str) -> Tuple[str, List[str]]:
    """``title`` fitted to the tightest title limit of ``toolkits`` (YouTube's)."""
    limits = [COPY_LIMITS[t][TITLE] for t in toolkits if TITLE in COPY_LIMITS.get(t, {})]
    if not limits or len(title) <= min(limits):
        return title, []
    limit = min(limits)
    return trim_at_word(title, limit), [f"the title was trimmed to {limit} characters at a word boundary"]
