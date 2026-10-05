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
colours, so it checks each of them.

:func:`derive_palette` derives the kit's colour roles (PRD-255 FR-1..FR-3), the
single source of colour for documents, spreadsheets and socials. A v1 kit has
four colours and no roles; every role is derived from them at read time, so no
stored kit is migrated, and a role the kit stores (``kit['palette']``) always
wins. Every text role is measured against the darker of ``paper`` and
``surface_2``, so it reads on both:

* ``paper``: the kit's lightest colour when it is light enough to be a page,
  else white; ``surface`` and ``surface_2``: the paper tinted 4% and 8% toward
  the secondary (cards, zebra rows, table header fills);
* ``ink``: the kit's text colour, darkened until it reads at 10:1;
* ``heading``: the text colour moved to near-black, never the primary;
* ``accent``: the primary, darkened only as far as AA text needs; ``accent_2``:
  the secondary, the same way, when its hue differs from the primary's;
* ``muted`` (secondary text) and ``rule`` (hairlines): the ink, lightened toward
  the paper as far as each may go.

:func:`effective_palette` adds which roles are ``set`` and which ``derived``.
Pure: no IO, no database.
"""
from __future__ import annotations

import colorsys
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

# The kit's colour roles (PRD-255 FR-1). ``accent_2`` is optional: it is derived
# only when the secondary is a second hue.
ROLE_INK, ROLE_HEADING, ROLE_PAPER = "ink", "heading", "paper"
ROLE_SURFACE, ROLE_SURFACE_2, ROLE_ACCENT, ROLE_ACCENT_2 = "surface", "surface_2", "accent", "accent_2"
ROLE_MUTED, ROLE_RULE = "muted", "rule"
PALETTE_ROLES = (
    ROLE_INK, ROLE_HEADING, ROLE_PAPER, ROLE_SURFACE, ROLE_SURFACE_2, ROLE_ACCENT, ROLE_ACCENT_2, ROLE_MUTED, ROLE_RULE,
)
TEXT_ROLES = (ROLE_INK, ROLE_HEADING, ROLE_ACCENT, ROLE_ACCENT_2, ROLE_MUTED)
ROLE_SET, ROLE_DERIVED = "set", "derived"

# The roles' targets. The page reuses the paper's luminance floor; the surfaces
# are the paper tinted toward the secondary. Body ink reads at 10:1, headings are
# near-black (at most the stage ink's luminance), the accents read as AA text
# (with the paper's margin), muted text keeps the stage's muted ratio, and a
# hairline is the ink lightened to a faint line on the paper.
SURFACE_TINT = 0.04
SURFACE_2_TINT = 0.08
INK_ON_PAPER_MIN_CONTRAST = 10.0
HEADING_MAX_LUMINANCE = INK_MAX_LUMINANCE
ACCENT_TEXT_MIN_CONTRAST = PAPER_TEXT_MIN_CONTRAST
MUTED_TEXT_MIN_CONTRAST = MUTED_CONTRAST
RULE_MIN_CONTRAST = 1.35
# accent_2: the secondary is a second accent only when it has a hue (saturation
# over the floor) and that hue is this far round the wheel from the primary's.
ACCENT_2_MIN_HUE_DEGREES = 30.0
ACCENT_2_MIN_SATURATION = 0.15
HUE_CIRCLE_DEGREES = 360.0
KIT_COLOUR_FIELDS = ("primary_color", "secondary_color", "accent_color", "text_color")

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


def _kit_colour(kit: Mapping[str, Any], *fields: str) -> Optional[RGB]:
    """The first of ``fields`` the kit has as a hex colour; ``None`` when none is."""
    for field in fields:
        rgb = parse_hex(kit.get(field))
        if rgb is not None:
            return rgb
    return None


def _stored_roles(kit: Mapping[str, Any]) -> Dict[str, RGB]:
    """The roles ``kit['palette']`` stores as valid hex colours; anything else is left to derivation."""
    stored = kit.get("palette")
    if not isinstance(stored, Mapping):
        return {}
    roles = {role: parse_hex(stored.get(role)) for role in PALETTE_ROLES}
    return {role: rgb for role, rgb in roles.items() if rgb is not None}


def _derived_paper(kit: Mapping[str, Any]) -> RGB:
    """The kit's lightest colour when it is light enough to be a page; else white."""
    colours = [rgb for rgb in (parse_hex(kit.get(field)) for field in KIT_COLOUR_FIELDS) if rgb is not None]
    lightest = max(colours, key=luminance, default=WHITE)
    return lightest if luminance(lightest) >= PAPER_MIN_LUMINANCE else WHITE


def _hue_saturation(rgb: RGB) -> Tuple[float, float]:
    hue, _lightness, saturation = colorsys.rgb_to_hls(*(channel / 255 for channel in rgb))
    return hue * HUE_CIRCLE_DEGREES, saturation


def _second_hue(primary: Optional[RGB], secondary: RGB) -> bool:
    """Whether ``secondary`` is a hue of its own: chromatic, and far enough round the wheel from ``primary``."""
    hue, saturation = _hue_saturation(secondary)
    if saturation < ACCENT_2_MIN_SATURATION:
        return False
    if primary is None:
        return True
    primary_hue, primary_saturation = _hue_saturation(primary)
    if primary_saturation < ACCENT_2_MIN_SATURATION:
        return True
    apart = abs(hue - primary_hue) % HUE_CIRCLE_DEGREES
    return min(apart, HUE_CIRCLE_DEGREES - apart) >= ACCENT_2_MIN_HUE_DEGREES


def _grounds(kit: Mapping[str, Any], stored: Mapping[str, RGB]) -> Dict[str, RGB]:
    """``paper``, ``surface`` and ``surface_2``: stored ones as they are, the rest derived."""
    paper = stored.get(ROLE_PAPER) or _derived_paper(kit)
    tint = _kit_colour(kit, "secondary_color", "text_color") or BLACK
    return {
        ROLE_PAPER: paper,
        ROLE_SURFACE: stored.get(ROLE_SURFACE) or mix(paper, tint, SURFACE_TINT),
        ROLE_SURFACE_2: stored.get(ROLE_SURFACE_2) or mix(paper, tint, SURFACE_2_TINT),
    }


def _text_roles(kit: Mapping[str, Any], stored: Mapping[str, RGB], grounds: Mapping[str, RGB]) -> Dict[str, RGB]:
    """The text roles, each reading on both ``paper`` and ``surface_2`` (a stored role as it is)."""
    paper, surface_2 = grounds[ROLE_PAPER], grounds[ROLE_SURFACE_2]

    def reads(target: float) -> Callable[[RGB], bool]:
        return lambda c: min(contrast(c, paper), contrast(c, surface_2)) >= target

    text = _kit_colour(kit, "text_color", "secondary_color") or BLACK
    ink = stored.get(ROLE_INK) or _least(text, BLACK, reads(INK_ON_PAPER_MIN_CONTRAST))
    heading_max = min(luminance(ink), HEADING_MAX_LUMINANCE)
    roles = {
        ROLE_INK: ink,
        ROLE_HEADING: stored.get(ROLE_HEADING) or _least(text, BLACK, lambda c: luminance(c) <= heading_max),
        ROLE_MUTED: stored.get(ROLE_MUTED) or _most(ink, paper, reads(MUTED_TEXT_MIN_CONTRAST)),
        ROLE_RULE: stored.get(ROLE_RULE) or _most(ink, paper, lambda c: contrast(c, paper) >= RULE_MIN_CONTRAST),
    }
    primary = _kit_colour(kit, "primary_color", "accent_color")
    roles[ROLE_ACCENT] = stored.get(ROLE_ACCENT) or _least(primary or ink, BLACK, reads(ACCENT_TEXT_MIN_CONTRAST))
    secondary = _kit_colour(kit, "secondary_color")
    if ROLE_ACCENT_2 in stored:
        roles[ROLE_ACCENT_2] = stored[ROLE_ACCENT_2]
    elif secondary is not None and _second_hue(primary, secondary):
        roles[ROLE_ACCENT_2] = _least(secondary, BLACK, reads(ACCENT_TEXT_MIN_CONTRAST))
    return roles


def derive_palette(kit: Mapping[str, Any]) -> Dict[str, str]:
    """The colour roles for ``kit`` (a brand kit dict, v1 or v2), each a 6-digit hex.

    A role ``kit['palette']`` stores as a hex colour is used as it is, and the
    roles derived after it are measured against it (a stored paper moves the
    derived ink). Every other role is derived from the kit's four colours, so a
    v1 kit reads as a v2 kit. ``accent_2`` is present only when stored or when
    the secondary is a second hue.
    """
    stored = _stored_roles(kit)
    grounds = _grounds(kit, stored)
    roles = {**grounds, **_text_roles(kit, stored, grounds)}
    return {role: to_hex(roles[role]) for role in PALETTE_ROLES if role in roles}


def effective_palette(kit: Mapping[str, Any]) -> Tuple[Dict[str, str], Dict[str, str]]:
    """``(roles, sources)``: :func:`derive_palette`, and each role marked ``set`` (stored) or ``derived``."""
    roles = derive_palette(kit)
    stored = _stored_roles(kit)
    sources = {role: ROLE_SET if role in stored else ROLE_DERIVED for role in roles}
    return roles, sources


__all__ = [
    "PALETTE_ROLES",
    "PAPER_TOKENS",
    "STAGE_TOKENS",
    "TEXT_ROLES",
    "contrast",
    "derive_palette",
    "effective_palette",
    "luminance",
    "mix",
    "paper_palette",
    "parse_hex",
    "stage_palette",
    "to_hex",
]
