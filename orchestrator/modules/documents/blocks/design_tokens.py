"""The one design system every document prints with: type, spacing and colour roles (F356).

F356 (5 Oct): after F350's pass the owner said the documents "look more
professional now… I need them to be as professional as we can make them". Each
renderer (the PDF stylesheet, the Word writer, the legacy Jinja starters) picked
its own sizes, gaps and colours. They now all read this module:

* a type scale of six steps (title, H2, H3, body, small, caption) in two weights;
* a spacing scale on a 4 pt grid;
* one colour rule, every colour from the brand kit: the kit's primary (the
  owner's orange) for the title and table headers, its accent (the owner's
  navy) for section headings and rules, its text colour for body text. Tints
  (zebra rows, hairlines, muted print) are those colours mixed towards white, so
  a kit with other colours gets the same structure in its own colours.

Pure: the kit in, values out.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from core.brand_palette import WHITE, contrast, mix, parse_hex, to_hex

DEFAULT_PRIMARY = "#1a1a2e"
DEFAULT_SECONDARY = "#16213e"
DEFAULT_ACCENT = "#0f3460"
DEFAULT_TEXT = "#1a1a2e"

# Type scale, in points.
TITLE_PT = 22
H2_PT = 13
H3_PT = 10.5
BODY_PT = 10
SMALL_PT = 8.5
CAPTION_PT = 7.5
# The two weights.
REGULAR = 400
BOLD = 700
# Body line height, as a multiple of the size.
LEADING = 1.45

# Spacing scale, in points: a 4 pt grid.
SPACE_1 = 4
SPACE_2 = 8
SPACE_3 = 12
SPACE_4 = 16
SPACE_5 = 24
SPACE_6 = 32

# Rule weights, in points.
HAIRLINE_PT = 0.5
RULE_PT = 1

# How far towards white each tint is mixed (0 is the colour, 1 is white).
ZEBRA_MIX = 0.94
HAIRLINE_MIX = 0.72
MUTED_MIX = 0.38
PANEL_MIX = 0.92
# Text on a filled header: white when it reads (WCAG AA), else the kit's text colour.
HEADER_TEXT_MIN_CONTRAST = 4.5
WHITE_HEX = "#ffffff"


@dataclass(frozen=True)
class Palette:
    """The colour roles a document prints with, each a ``#rrggbb`` from the kit."""

    title: str
    heading: str
    rule: str
    text: str
    muted: str
    hairline: str
    zebra: str
    panel: str
    header_fill: str
    header_text: str
    secondary: str


def _hex(value: Any, default: str) -> str:
    """``value`` as ``#rrggbb`` when it is a hex colour, else ``default``."""
    rgb = parse_hex(value)
    return to_hex(rgb) if rgb is not None else default


def _towards_white(colour: str, amount: float) -> str:
    return to_hex(mix(parse_hex(colour), WHITE, amount))


def header_text(fill: str, text: str) -> str:
    """White on ``fill`` when it reads, else the kit's text colour."""
    return WHITE_HEX if contrast(parse_hex(fill), WHITE) >= HEADER_TEXT_MIN_CONTRAST else text


def palette(kit: Mapping[str, Any]) -> Palette:
    """The colour roles for ``kit`` (a brand kit dict; empty means the defaults). Pure."""
    bk = kit or {}
    primary = _hex(bk.get("primary_color"), DEFAULT_PRIMARY)
    accent = _hex(bk.get("accent_color"), DEFAULT_ACCENT)
    text = _hex(bk.get("text_color"), DEFAULT_TEXT)
    return Palette(
        title=primary,
        heading=accent,
        rule=accent,
        text=text,
        muted=_towards_white(text, MUTED_MIX),
        hairline=_towards_white(accent, HAIRLINE_MIX),
        zebra=_towards_white(accent, ZEBRA_MIX),
        panel=_towards_white(accent, PANEL_MIX),
        header_fill=primary,
        header_text=header_text(primary, text),
        secondary=_hex(bk.get("secondary_color"), DEFAULT_SECONDARY),
    )


__all__ = [
    "BODY_PT", "BOLD", "CAPTION_PT", "H2_PT", "H3_PT", "HAIRLINE_PT", "LEADING", "Palette", "REGULAR", "RULE_PT",
    "SMALL_PT", "SPACE_1", "SPACE_2", "SPACE_3", "SPACE_4", "SPACE_5", "SPACE_6", "TITLE_PT", "header_text", "palette",
]
