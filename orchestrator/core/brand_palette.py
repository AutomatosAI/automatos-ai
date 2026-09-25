"""The dark stage a social video reads, and the paper a social image reads, derived from the brand kit (PRD-251 D4, D5).

The reference videos are dark stages: a near-black ink, light text on it, and
the brand colour for the accents (docs/PRDS/prd251-reference/). A brand kit is
written for documents, dark text on white paper, and its defaults are all dark
(``modules/documents/brand_kit.py``). Its colours cannot be a video's stage as
they are: dark text on a dark stage fails the WCAG pass of `hyperframes check`,
and media-render refuses the render.

:func:`stage_palette` derives the stage from the kit and keeps each colour's hue:

* ``ink``: the kit's secondary colour, darkened until light text reads on it;
* ``on-ink``: the kit's text colour, lightened until it reads on the ink;
* ``on-ink-muted`` and ``on-ink-dim``: two quieter text tones between the two,
  each still readable on the ink and on the cards a template lays over it;
* ``primary-on-ink`` and ``accent-on-ink``: the brand colours, lightened only as
  far as they must be to read on the ink (display words, shapes, glows);
* ``primary-light``: the primary a little lighter again, for small accent text
  on tinted pills (the reference's own fix: #E96235 became #F07A50 there).

A kit colour that already meets its target is used exactly as it is, so the
Automatos Studio Dark kit reproduces the reference palette.

:func:`paper_palette` derives the light page the social image families read
(US-107: automatos-social's cream paper, ink and brand-orange display words),
again keeping each colour's hue. Every text tone is measured against the card,
the darker of the two surfaces, so it reads on both:

* ``paper``: the kit's text colour, lightened until it is a page (a light kit
  colour, like the Studio Dark cream, is the page as it is);
* ``paper-card``: the paper a shade darker, for the cards laid on it;
* ``on-paper``: the kit's secondary colour, darkened until it reads on the card;
* ``on-paper-muted`` and ``on-paper-dim``: two quieter text tones;
* ``primary-on-paper`` and ``accent-on-paper``: the brand colours, darkened only
  as far as small text in them must be (WCAG AA 4.5:1, with a margin);
* ``primary-on-paper-large``: the primary darkened only as far as LARGE text
  must be (AA 3:1 for 24 px, or 19 px bold, with a margin): display words and
  big numbers.

Every token is a 6-digit hex: the contrast pass reads computed ``rgb()``
colours, so it checks each of them. Pure: no IO, no database.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

RGB = Tuple[int, int, int]

BLACK: RGB = (0, 0, 0)
WHITE: RGB = (255, 255, 255)

# The stage's targets (WCAG 2 relative luminance and contrast ratios).
INK_MAX_LUMINANCE = 0.025
ON_INK_MIN_CONTRAST = 10.0
# The reference's muted text (#B0A38D) sits at 6.9:1 on its ink; its dim labels
# (#938876) at 5.0:1, which leaves 4.6:1 on the cards above the ink. The dim tone
# keeps a margin over that, so small labels on any kit's cards stay over 4.5:1.
MUTED_CONTRAST = 6.9
DIM_CONTRAST = 5.3
ACCENT_MIN_CONTRAST = 4.5
ACCENT_LIGHT_MIN_CONTRAST = 6.0
SEARCH_STEPS = 24

INK, ON_INK, ON_INK_MUTED, ON_INK_DIM = "ink", "on-ink", "on-ink-muted", "on-ink-dim"
PRIMARY_ON_INK, PRIMARY_LIGHT, ACCENT_ON_INK = "primary-on-ink", "primary-light", "accent-on-ink"
STAGE_TOKENS = (INK, ON_INK, ON_INK_MUTED, ON_INK_DIM, PRIMARY_ON_INK, PRIMARY_LIGHT, ACCENT_ON_INK)

# The paper's targets. automatos-social's cream (#f1e9dd) is 0.82 luminance, and its
# card (#e3d9c8) sits 1.16:1 below it; the muted and dim tones reuse the stage's
# ratios. WCAG AA is 4.5:1 for text and 3:1 for large text: the brand colours keep
# a margin over each, because the check measures rendered pixels and rounds.
PAPER_MIN_LUMINANCE = 0.80
CARD_CONTRAST = 1.16
ON_PAPER_MIN_CONTRAST = 10.0
PAPER_TEXT_MIN_CONTRAST = 4.7
LARGE_TEXT_MIN_CONTRAST = 3.2

PAPER, PAPER_CARD, ON_PAPER = "paper", "paper-card", "on-paper"
ON_PAPER_MUTED, ON_PAPER_DIM = "on-paper-muted", "on-paper-dim"
PRIMARY_ON_PAPER, PRIMARY_ON_PAPER_LARGE, ACCENT_ON_PAPER = "primary-on-paper", "primary-on-paper-large", "accent-on-paper"
PAPER_TOKENS = (
    PAPER, PAPER_CARD, ON_PAPER, ON_PAPER_MUTED, ON_PAPER_DIM, PRIMARY_ON_PAPER, PRIMARY_ON_PAPER_LARGE, ACCENT_ON_PAPER,
)

_HEX = re.compile(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def parse_hex(value: Any) -> Optional[RGB]:
    """``#abc`` or ``#aabbcc`` as ``(r, g, b)``; ``None`` for anything else."""
    match = _HEX.match(value.strip()) if isinstance(value, str) else None
    if match is None:
        return None
    digits = match.group(1)
    if len(digits) == 3:
        digits = "".join(ch * 2 for ch in digits)
    return int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16)


def to_hex(rgb: RGB) -> str:
    return "#" + "".join(f"{channel:02x}" for channel in rgb)


def _linear(channel: int) -> float:
    c = channel / 255
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(rgb: RGB) -> float:
    """WCAG 2 relative luminance."""
    r, g, b = (_linear(channel) for channel in rgb)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a: RGB, b: RGB) -> float:
    """WCAG 2 contrast ratio, 1 to 21."""
    high, low = sorted((luminance(a), luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def mix(a: RGB, b: RGB, t: float) -> RGB:
    """``a`` moved ``t`` of the way to ``b`` in sRGB (0 is ``a``, 1 is ``b``)."""
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))  # type: ignore[return-value]


def _least(base: RGB, towards: RGB, meets: Callable[[RGB], bool]) -> RGB:
    """``base`` moved the least distance towards ``towards`` that ``meets`` the target.

    ``meets`` must hold at ``towards`` and grow monotonically along the way.
    """
    if meets(base):
        return base
    low, high = 0.0, 1.0
    for _ in range(SEARCH_STEPS):
        middle = (low + high) / 2
        if meets(mix(base, towards, middle)):
            high = middle
        else:
            low = middle
    return mix(base, towards, high)


def _most(base: RGB, towards: RGB, meets: Callable[[RGB], bool]) -> RGB:
    """``base`` moved the most distance towards ``towards`` that still ``meets`` the target.

    ``meets`` must hold at ``base`` and fail monotonically along the way.
    """
    if not meets(base):
        return base
    low, high = 0.0, 1.0
    for _ in range(SEARCH_STEPS):
        middle = (low + high) / 2
        if meets(mix(base, towards, middle)):
            low = middle
        else:
            high = middle
    return mix(base, towards, low)


def stage_palette(kit: Mapping[str, Any]) -> Dict[str, str]:
    """The stage tokens for ``kit`` (a brand kit dict); ``{}`` when it has no usable secondary colour.

    A role whose kit colour is missing or not a hex colour is left out, and the
    template's own ``var()`` fallback applies to it.
    """
    secondary = parse_hex(kit.get("secondary_color"))
    if secondary is None:
        return {}
    ink = _least(secondary, BLACK, lambda c: luminance(c) <= INK_MAX_LUMINANCE)
    palette = {INK: ink}
    text = parse_hex(kit.get("text_color"))
    if text is not None:
        on_ink = _least(text, WHITE, lambda c: contrast(c, ink) >= ON_INK_MIN_CONTRAST)
        palette[ON_INK] = on_ink
        palette[ON_INK_MUTED] = _most(on_ink, ink, lambda c: contrast(c, ink) >= MUTED_CONTRAST)
        palette[ON_INK_DIM] = _most(on_ink, ink, lambda c: contrast(c, ink) >= DIM_CONTRAST)
    primary = parse_hex(kit.get("primary_color"))
    if primary is not None:
        palette[PRIMARY_ON_INK] = _least(primary, WHITE, lambda c: contrast(c, ink) >= ACCENT_MIN_CONTRAST)
        palette[PRIMARY_LIGHT] = _least(primary, WHITE, lambda c: contrast(c, ink) >= ACCENT_LIGHT_MIN_CONTRAST)
    accent = parse_hex(kit.get("accent_color"))
    if accent is not None:
        palette[ACCENT_ON_INK] = _least(accent, WHITE, lambda c: contrast(c, ink) >= ACCENT_MIN_CONTRAST)
    return {name: to_hex(rgb) for name, rgb in palette.items()}


def paper_palette(kit: Mapping[str, Any]) -> Dict[str, str]:
    """The paper tokens for ``kit`` (a brand kit dict); ``{}`` when it has no usable text colour.

    A role whose kit colour is missing or not a hex colour is left out, and the
    template's own ``var()`` fallback applies to it.
    """
    text = parse_hex(kit.get("text_color"))
    if text is None:
        return {}
    paper = _least(text, WHITE, lambda c: luminance(c) >= PAPER_MIN_LUMINANCE)
    card = _most(paper, BLACK, lambda c: contrast(c, paper) <= CARD_CONTRAST)
    palette = {PAPER: paper, PAPER_CARD: card}
    secondary = parse_hex(kit.get("secondary_color"))
    if secondary is not None:
        on_paper = _least(secondary, BLACK, lambda c: contrast(c, card) >= ON_PAPER_MIN_CONTRAST)
        palette[ON_PAPER] = on_paper
        palette[ON_PAPER_MUTED] = _most(on_paper, card, lambda c: contrast(c, card) >= MUTED_CONTRAST)
        palette[ON_PAPER_DIM] = _most(on_paper, card, lambda c: contrast(c, card) >= DIM_CONTRAST)
    primary = parse_hex(kit.get("primary_color"))
    if primary is not None:
        palette[PRIMARY_ON_PAPER] = _least(primary, BLACK, lambda c: contrast(c, card) >= PAPER_TEXT_MIN_CONTRAST)
        palette[PRIMARY_ON_PAPER_LARGE] = _least(primary, BLACK, lambda c: contrast(c, card) >= LARGE_TEXT_MIN_CONTRAST)
    accent = parse_hex(kit.get("accent_color"))
    if accent is not None:
        palette[ACCENT_ON_PAPER] = _least(accent, BLACK, lambda c: contrast(c, card) >= PAPER_TEXT_MIN_CONTRAST)
    return {name: to_hex(rgb) for name, rgb in palette.items()}


__all__ = [
    "PAPER_TOKENS",
    "STAGE_TOKENS",
    "contrast",
    "luminance",
    "mix",
    "paper_palette",
    "parse_hex",
    "stage_palette",
    "to_hex",
]
