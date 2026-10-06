"""What each part of the brand board shows, read from the kit (PRD-255 US-009, FR-9, FR-10).

The brand board is the kit on one page: the logo large and its variants on light
and dark, the colour roles as swatches with their hex codes, the type scale with
samples, the spacing grid and the logo's clear space, the tone words with their
meanings, and three miniature applications. The board's ``brand`` blocks
(``schema.BrandBlock``) hold no chip and no data: both renderers
(``brand_board_html``, ``brand_board_docx``) print what this module reads from the
render-ready kit, so the board always shows the kit as it is saved.

FR-9: a logo variant is shown only when the owner uploaded it; one that is not set
is shown as its fallback (the logo on a light chip on a dark ground; the logo
itself for one colour), with a line saying so. Nothing is generated.

Pure: the kit in, values out.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Tuple

from core.brand_palette import PALETTE_ROLES, effective_palette

from ..brand_system import ACCENT_BOLD, ACCENT_SPARING, ACCENT_USES, DEFAULT_ACCENT_USE, ROLE_JOBS, TYPE_STEPS, tone_words
from . import design_tokens as t

# The render-ready kit's inlined logo variants (``brand_fonts.INLINED_LOGOS``).
LOGO_FIELD, LOGO_DARK_FIELD, LOGO_MONO_FIELD = "logo_url", "logo_dark_url", "logo_mono_url"
NO_LOGO_NOTE = "No logo yet: upload one on the Brand kit page."
NO_DARK_LOGO_NOTE = "Not uploaded: the logo on a light chip."
NO_MONO_LOGO_NOTE = "Not uploaded: the logo itself."
BOARD_TITLE = "Brand board"
# The type samples print this line at each step of the scale.
TYPE_SAMPLE = "The quick brown fox jumps over the lazy dog"
TYPE_STEP_LABELS = {"display": "Display", "h1": "H1", "h2": "H2", "h3": "H3", "body": "Body", "small": "Small",
                    "caption": "Caption"}
NO_TONE_NOTE = "No tone words yet: add three to five on the Brand kit page."
# The three miniature applications, in the order the board shows them.
APPLICATION_INVOICE, APPLICATION_LETTER, APPLICATION_SOCIAL = "invoice", "letter", "social"
APPLICATIONS = ((APPLICATION_INVOICE, "Invoice"), (APPLICATION_LETTER, "Letter"), (APPLICATION_SOCIAL, "Social card"))
SOCIAL_SAMPLE_HEADLINE = "Your headline, in your type"
# What each accent use lets the accent do, in one line (``brand_system.ACCENT_USE_RULES`` says it in full).
ACCENT_USE_SHORT = {ACCENT_SPARING: "highlights only", ACCENT_BOLD: "highlights and table header fills"}


@dataclass(frozen=True)
class Swatch:
    """One colour role: its name, its job, its hex (upper case) and whether it is ``set`` or ``derived``."""

    role: str
    label: str
    job: str
    hex: str
    source: str


@dataclass(frozen=True)
class Variant:
    """One logo tile: the image (``""`` for none), the ground it sits on, whether a light chip
    goes behind it, and the line that says a variant is not set (``""`` when it is)."""

    label: str
    src: str
    ground: str
    chip: bool
    note: str


@dataclass(frozen=True)
class TypeSample:
    """One step of the type scale, with the label it prints beside its sample."""

    step: str
    label: str
    size_pt: float
    line_pt: float
    weight: int


@dataclass(frozen=True)
class Spacing:
    """The spacing grid (each gap in points), the page margin and the logo's clear space."""

    unit_pt: float
    gaps_pt: Tuple[float, ...]
    margin_mm: float
    logo_mm: float
    clear_space: float
    clear_mm: float


def brand_name(kit: Mapping[str, Any]) -> str:
    """The name the board prints: the kit's, else its company's."""
    company = kit.get("company") if isinstance(kit.get("company"), Mapping) else {}
    return str(kit.get("name") or company.get("name") or "").strip()


def tagline(kit: Mapping[str, Any]) -> str:
    return str(kit.get("tagline") or "").strip()


def logo(kit: Mapping[str, Any]) -> str:
    """The logo as the renderers load it (an inlined upload or a public URL); ``""`` without one."""
    return str(kit.get(LOGO_FIELD) or "")


def _label(role: str) -> str:
    return role.replace("_", " ").capitalize()


def swatches(kit: Mapping[str, Any]) -> List[Swatch]:
    """Every colour role of the kit's effective palette, in the palette's order."""
    roles, sources = effective_palette(kit)
    return [Swatch(role, _label(role), ROLE_JOBS[role], roles[role].upper(), sources[role]) for role in PALETTE_ROLES]


def accent_rule(kit: Mapping[str, Any]) -> str:
    """How the kit uses its accent, in words (sparing, every kit's default, Decision Q1)."""
    use = kit.get("accent_use") if kit.get("accent_use") in ACCENT_USES else DEFAULT_ACCENT_USE
    return f"Accent use: {use} ({ACCENT_USE_SHORT[use]})."


def variants(kit: Mapping[str, Any]) -> List[Variant]:
    """The logo on light, on dark and in one colour (FR-9: an unset variant is its fallback, never invented)."""
    roles = t.palette(kit)
    main, dark, mono = logo(kit), str(kit.get(LOGO_DARK_FIELD) or ""), str(kit.get(LOGO_MONO_FIELD) or "")
    return [
        Variant("On light", main, roles.paper, False, "" if main else NO_LOGO_NOTE),
        Variant("On dark", dark or main, roles.heading, not dark and bool(main), "" if dark else NO_DARK_LOGO_NOTE),
        Variant("One colour", mono or main, roles.paper, False, "" if mono else NO_MONO_LOGO_NOTE),
    ]


def type_samples(kit: Mapping[str, Any]) -> List[TypeSample]:
    """Each step of the kit's type scale, largest first."""
    scale = t.type_scale(kit)
    samples = []
    for step in TYPE_STEPS:
        found = scale[step]
        label = f"{TYPE_STEP_LABELS[step]} {found.size_pt:g}/{found.line_pt:g} pt, {found.weight}"
        samples.append(TypeSample(step, label, found.size_pt, found.line_pt, found.weight))
    return samples


def spacing(kit: Mapping[str, Any]) -> Spacing:
    """The kit's spacing grid, page margin and logo clear space."""
    design = t.design(kit)
    gaps = tuple(design.space(step) for step in range(1, len(t.SPACE_STEPS) + 1))
    clear_space = design.logo_clear_mm / design.logo_mm if design.logo_mm else 0.0
    return Spacing(design.spacing_unit_pt, gaps, design.page_margin_mm, design.logo_mm, clear_space,
                   design.logo_clear_mm)


def voice(kit: Mapping[str, Any]) -> Tuple[List[Mapping[str, str]], str]:
    """``(tone words with meanings, sign-off)``; the sign-off is ``""`` when the kit sets none."""
    raw = kit.get("voice") if isinstance(kit.get("voice"), Mapping) else {}
    return tone_words(kit), str(raw.get("sign_off") or "").strip()


def social_colours(kit: Mapping[str, Any]) -> Tuple[str, str, str]:
    """``(paper, ink, accent)``: the colours the social card miniature is drawn in."""
    roles = t.palette(kit)
    return roles.paper, roles.heading, roles.accent


def miniature(miniatures: Optional[Mapping[str, str]], key: str) -> str:
    """The rendered page-1 PNG (a data: URI) for ``key``; ``""`` when it could not be drawn."""
    return str((miniatures or {}).get(key) or "")


__all__ = [
    "ACCENT_USE_SHORT", "APPLICATIONS", "BOARD_TITLE", "APPLICATION_INVOICE", "APPLICATION_LETTER", "APPLICATION_SOCIAL", "SOCIAL_SAMPLE_HEADLINE",
    "Spacing", "Swatch", "TYPE_SAMPLE", "TypeSample", "Variant", "accent_rule", "brand_name", "logo", "miniature",
    "social_colours", "spacing", "swatches", "tagline", "type_samples", "variants", "voice",
]
