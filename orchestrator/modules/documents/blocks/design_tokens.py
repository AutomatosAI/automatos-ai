"""The one design system every document prints with, read from the brand kit (F356; PRD-255 US-004).

F356 (5 Oct) gave every renderer (the PDF stylesheet, the Word writer, the legacy
Jinja starters, the branded spreadsheet) one type scale, one spacing grid and one
set of colour roles, but its own constants, and the kit's primary on the title and
table headers: "a lot of orange in there". PRD-255 makes the kit the design
system, so this module only reads it:

* the colour roles are the kit's (``core.brand_palette.derive_palette``: a stored
  role as it is, the rest derived from the kit's four colours, FR-3). Headings are
  ``heading`` (near-black), body text ``ink``, zebra rows ``surface``, hairlines
  ``rule``. Under ``accent_use: sparing`` (the default for every kit, FR-6) the
  accent is a highlight only: the title's rule, key numbers, links, one element
  per section; the table header is ``surface_2`` with ``heading`` text. Under
  ``bold`` the table header is filled with the accent, its text white where white
  reads. Small text in an accent below AA on the paper prints in ``heading``;
* the type scale is the kit's ``type_scale`` (one default scale, Decision Q5);
* gaps are multiples of the kit's ``spacing_unit_pt``, the page margins its
  ``page_margin_mm``, the letterhead logo its ``logo_rules``.

Read leniently, like ``brand_kit.get_brand_kit``: a value that does not validate
takes its default, so a render never fails on the kit. Pure: the kit in, values out.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Mapping

from pydantic import ValidationError

from core.brand_palette import (
    ROLE_ACCENT, ROLE_HEADING, ROLE_INK, ROLE_MUTED, ROLE_PAPER, ROLE_RULE, ROLE_SURFACE, ROLE_SURFACE_2, WHITE,
    contrast, derive_palette, parse_hex,
)

from ..brand_system import (
    ACCENT_BOLD,
    DEFAULT_PAGE_MARGIN_MM,
    DEFAULT_SPACING_UNIT_PT,
    MAX_PAGE_MARGIN_MM,
    MAX_SPACING_UNIT_PT,
    MIN_PAGE_MARGIN_MM,
    MIN_SPACING_UNIT_PT,
    TYPE_STEPS,
    LogoRules,
    TypeScale,
)

logger = logging.getLogger(__name__)

# The gaps: multiples of the kit's spacing unit (s1..s6; 4, 8, 12, 16, 24, 32 pt on the default 4 pt grid).
SPACE_STEPS = (1, 2, 3, 4, 6, 8)
# Strong text (``<strong>``, labels) and the least weight a Word style prints bold at.
BOLD = 700
WORD_BOLD_FROM = 600
# Rule weights, in points: hairlines between rows, rules under a letterhead or over a total,
# and the accent rule under the title.
HAIRLINE_PT = 0.5
RULE_PT = 1
TITLE_RULE_PT = 2
# The title's accent rule is this many spacing units long (24 pt on the default grid).
TITLE_RULE_UNITS = 6
# The footer prints in the bottom margin: it gets this much more than the other margins.
FOOTER_BAND_MM = 6
# AA for text: white on a filled header, and an accent printed as small text on the paper.
TEXT_MIN_CONTRAST = 4.5
WHITE_HEX = "#ffffff"


@dataclass(frozen=True)
class Palette:
    """The colour roles a document prints with, each a ``#rrggbb`` (FR-3).

    ``accent_text`` is the accent where small text needs AA on the paper, else
    ``heading``; ``header_fill`` / ``header_text`` are a table header's (FR-6)."""

    ink: str
    heading: str
    paper: str
    surface: str
    surface_2: str
    accent: str
    accent_text: str
    muted: str
    rule: str
    header_fill: str
    header_text: str


@dataclass(frozen=True)
class Step:
    """One step of the kit's type scale: size and line height in points, and weight."""

    size_pt: float
    line_pt: float
    weight: int

    @property
    def bold(self) -> bool:
        """Whether Word prints this step bold (Word has no semibold)."""
        return self.weight >= WORD_BOLD_FROM


@dataclass(frozen=True)
class Design:
    """Everything a renderer reads from the kit: colours, type, spacing, margins and the logo."""

    palette: Palette
    type: Mapping[str, Step]
    spacing_unit_pt: float
    page_margin_mm: float
    logo_mm: float
    logo_clear_mm: float

    def space(self, step: int) -> float:
        """Gap ``step`` (1-6) in points."""
        return self.spacing_unit_pt * SPACE_STEPS[step - 1]

    @property
    def bottom_margin_mm(self) -> float:
        """The bottom margin, with room for the footer."""
        return self.page_margin_mm + FOOTER_BAND_MM


def header_text(fill: str, text: str) -> str:
    """White on ``fill`` when it reads (AA), else ``text``."""
    return WHITE_HEX if contrast(parse_hex(fill), WHITE) >= TEXT_MIN_CONTRAST else text


def palette(kit: Mapping[str, Any]) -> Palette:
    """The colour roles for ``kit`` (a brand kit dict, v1 or v2; empty means the defaults). Pure."""
    bk = kit or {}
    roles = derive_palette(bk)
    accent, paper, heading = roles[ROLE_ACCENT], roles[ROLE_PAPER], roles[ROLE_HEADING]
    fill = accent if bk.get("accent_use") == ACCENT_BOLD else roles[ROLE_SURFACE_2]
    reads = contrast(parse_hex(accent), parse_hex(paper)) >= TEXT_MIN_CONTRAST
    return Palette(
        ink=roles[ROLE_INK],
        heading=heading,
        paper=paper,
        surface=roles[ROLE_SURFACE],
        surface_2=roles[ROLE_SURFACE_2],
        accent=accent,
        accent_text=accent if reads else heading,
        muted=roles[ROLE_MUTED],
        rule=roles[ROLE_RULE],
        header_fill=fill,
        header_text=header_text(fill, heading),
    )


def _record(model: Any, value: Any, what: str) -> Any:
    """``value`` validated as ``model``; the model's defaults when it does not validate."""
    try:
        return model.model_validate(value if isinstance(value, Mapping) else {})
    except ValidationError:
        logger.warning("[DesignTokens] the kit's %s does not validate; its defaults are used", what)
        return model()


def type_scale(kit: Mapping[str, Any]) -> Dict[str, Step]:
    """The kit's type scale, every step (a step it leaves out at its default). Pure."""
    scale = _record(TypeScale, (kit or {}).get("type_scale"), "type_scale")
    return {name: Step(**getattr(scale, name).model_dump()) for name in TYPE_STEPS}


def _bounded(value: Any, low: float, high: float, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not low <= value <= high:
        return default
    return float(value)


def design(kit: Mapping[str, Any]) -> Design:
    """The whole design system for ``kit``. Pure."""
    bk = kit or {}
    logo = _record(LogoRules, bk.get("logo_rules"), "logo_rules")
    return Design(
        palette=palette(bk),
        type=type_scale(bk),
        spacing_unit_pt=_bounded(bk.get("spacing_unit_pt"), MIN_SPACING_UNIT_PT, MAX_SPACING_UNIT_PT,
                                 DEFAULT_SPACING_UNIT_PT),
        page_margin_mm=_bounded(bk.get("page_margin_mm"), MIN_PAGE_MARGIN_MM, MAX_PAGE_MARGIN_MM,
                                DEFAULT_PAGE_MARGIN_MM),
        logo_mm=logo.letterhead_mm,
        logo_clear_mm=logo.letterhead_mm * logo.clear_space,
    )


__all__ = [
    "BOLD", "Design", "FOOTER_BAND_MM", "HAIRLINE_PT", "Palette", "RULE_PT", "SPACE_STEPS", "Step", "TITLE_RULE_PT",
    "TITLE_RULE_UNITS", "design", "header_text", "palette", "type_scale",
]
