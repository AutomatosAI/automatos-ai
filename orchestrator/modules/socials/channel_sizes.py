"""Each channel its own shape (3 Oct 2026, Gerard: "different socials have different styles
and sizes, so when we create images we need multiple for the different formats").

A post renders each template size its channels need: for every channel, the template's
size closest in shape to what that channel shows best (``KIND_ASPECT``: X 16:9, LinkedIn
and Instagram images 1:1, carousels 4:5, reels, stories, shorts and TikTok 9:16). A
template without that exact shape gives its closest one, so X takes a 1600x900 text card
or a 1200x628 photo card, and Instagram a 1080x1080 photo card or a 4:5 text card. A post
with no channel renders the template's default (first) size, as it always has.

Publishing then gives each channel only its own size's files (``publish_sources``), and the
editor's preview shows the same file (``frontend .../studio/editor-model.ts`` holds the same
table for its labels). Shape is compared by the log of the width-to-height ratio, so a
portrait and a landscape twice as far from square count as equally far.
"""
from __future__ import annotations

import math
import re
from typing import Iterable, List, Optional, Sequence

KIND_ASPECT = {
    ("twitter", "image"): "16:9", ("twitter", "video"): "16:9",
    ("linkedin", "image"): "1:1", ("linkedin", "video"): "16:9", ("linkedin", "carousel"): "4:5",
    ("instagram", "image"): "1:1", ("instagram", "carousel"): "4:5",
    ("instagram", "reel"): "9:16", ("instagram", "story"): "9:16",
    ("tiktok", "video"): "9:16", ("youtube", "short"): "9:16", ("youtube", "video"): "16:9",
}
DEFAULT_ASPECT = {"video": "16:9", "reel": "9:16", "short": "9:16", "story": "9:16", "carousel": "4:5", "image": "1:1"}
_RATIO = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*[:x]\s*(\d+(?:\.\d+)?)\s*$")


def aspect_of(toolkit: str, kind: str) -> Optional[str]:
    """The shape a channel shows a post of ``kind`` best in, or ``None`` (a text post)."""
    return KIND_ASPECT.get((str(toolkit), str(kind))) or DEFAULT_ASPECT.get(str(kind))


def ratio(value: str) -> Optional[float]:
    """``"4:5"``, ``"300:157"`` or ``"1080x1350"`` as width over height; ``None`` when not a shape."""
    match = _RATIO.match(str(value or ""))
    if not match or not float(match.group(2)) or not float(match.group(1)):
        return None
    return float(match.group(1)) / float(match.group(2))


def closest(candidates: Iterable[str], aspect: str) -> Optional[str]:
    """The candidate (a size or an aspect) closest in shape to ``aspect``; the first of a tie."""
    wanted = ratio(aspect)
    best, best_gap = None, math.inf
    for candidate in candidates:
        shape = ratio(candidate)
        if wanted is None or shape is None:
            continue
        gap = abs(math.log(shape / wanted))
        if gap < best_gap:
            best, best_gap = candidate, gap
    return best


def render_sizes(template_sizes: Sequence[str], targets: Iterable[object]) -> List[str]:
    """The template sizes the post renders: each channel's closest, in channel order, once
    each; the template's default (first) size when no channel needs one."""
    chosen = []
    for target in targets:
        aspect = aspect_of(getattr(target, "toolkit", ""), getattr(target, "post_kind", ""))
        size = closest(template_sizes, aspect) if aspect else None
        if size and size not in chosen:
            chosen.append(size)
    return chosen or list(template_sizes[:1])


def aspect_for_target(available: Sequence[str], toolkit: str, kind: str) -> Optional[str]:
    """Which of the post's rendered aspects a channel publishes: the closest to its own;
    ``None`` when there is no choice to make (one aspect, or a kind with no shape)."""
    unique = list(dict.fromkeys(available))
    aspect = aspect_of(toolkit, kind)
    if len(unique) < 2 or not aspect:
        return None
    return closest(unique, aspect)
