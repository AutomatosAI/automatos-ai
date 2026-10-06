"""The kit's design system, in the rules block every drafting run is given (PRD-255 US-008).

F332 gave agents the kit's four colours, its fonts, the contact details and where the
logo is. Four colours with no roles is what made every renderer paint the primary on
everything (nights 10 and 10b); an agent told only the four colours does the same. The
rules block (:func:`services.brand_rules.rules_for_kit`) now also carries, from here:

* the colour roles, each with its hex, its job and whether the owner set it or it is
  derived, and how far the accent goes (``accent_use``);
* a one-line type scale summary, the spacing unit and page margin, and the logo's size;
* the logo variants the owner uploaded, by the file name a session's folder holds them
  under (the F332 pattern, :mod:`services.session_brand_files`);
* the currency, only when the kit has one (FR-7), and the date style.

The roles, type and spacing lines appear only when the owner set some part of the look:
a kit of neutral defaults says nothing of colour (F332: the defaults are not the brand's).
Every line is lenient, as the block reads a stored kit as it is. Pure: no IO.
"""
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Mapping, Optional, Tuple

from core.brand_palette import ROLE_SET, effective_palette
from core.brand_type import DEFAULT_TYPE_SCALE, TYPE_SCALE_FIELD, step_size_pt
from modules.documents.brand_system import (
    ACCENT_BOLD,
    ACCENT_USE_RULES,
    DEFAULT_ACCENT_USE,
    DEFAULT_CLEAR_SPACE,
    DEFAULT_LETTERHEAD_MM,
    DEFAULT_LOGO_MIN_MM,
    DEFAULT_PAGE_MARGIN_MM,
    DEFAULT_SPACING_UNIT_PT,
    LOGO_RULES_FIELD,
    ROLE_JOBS,
)
from modules.documents.locale_text import currency_of, currency_prefix, date_style_of, long_date
from services.session_brand_files import LOGO_FILES, SESSION_BRAND_FOLDER, session_file_name

# The date the date-style line writes, as an example of the style.
EXAMPLE_DATE = date(2026, 10, 5)
TYPE_STEP_JOIN = " · "
# The uploaded logo variants the block names: the kit field, and what each is for.
LOGO_VARIANTS = {"logo_dark_path": "for dark backgrounds", "logo_mono_path": "one colour"}
LOGO_FIELDS = ("logo_path", "logo_url", "logo_mark_path", "logo_mark_url")
SPACING_FIELD, MARGIN_FIELD, ACCENT_USE_FIELD = "spacing_unit_pt", "page_margin_mm", "accent_use"


def _number(value: Any, default: float) -> float:
    """``value`` when it is a positive number; else ``default``."""
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    return default


def _type_steps(kit: Mapping[str, Any]) -> List[Tuple[str, float, float]]:
    """Each step of the kit's type scale as ``(step, size_pt, line_pt)``; a step it leaves out is the default's."""
    scale = kit.get(TYPE_SCALE_FIELD)
    steps = []
    for step, (_size, line_pt, _weight) in DEFAULT_TYPE_SCALE.items():
        given = scale.get(step) if isinstance(scale, Mapping) else None
        line = _number(given.get("line_pt") if isinstance(given, Mapping) else None, line_pt)
        steps.append((step, step_size_pt(kit, step), line))
    return steps


def type_scale_line(kit: Mapping[str, Any]) -> str:
    """The type scale on one line: "- Type scale (pt, size/line): display 32/38 · h1 22/28 · …"."""
    steps = TYPE_STEP_JOIN.join(f"{step} {size:g}/{line:g}" for step, size, line in _type_steps(kit))
    return f"- Type scale (pt, size/line): {steps}."


def _roles_line(roles: Dict[str, str], sources: Dict[str, str]) -> str:
    listed = "; ".join(f"{role} {roles[role]} ({sources[role]}): {ROLE_JOBS[role]}" for role in roles)
    return f"- Colour roles (hex, set by the owner or derived from the colours): {listed}."


def _accent_line(kit: Mapping[str, Any]) -> str:
    use = kit.get(ACCENT_USE_FIELD)
    use = use if use in ACCENT_USE_RULES else DEFAULT_ACCENT_USE
    return f"- Accent use: {use}: {ACCENT_USE_RULES[use]}."


def _spacing_line(kit: Mapping[str, Any]) -> str:
    unit = _number(kit.get(SPACING_FIELD), DEFAULT_SPACING_UNIT_PT)
    margin = _number(kit.get(MARGIN_FIELD), DEFAULT_PAGE_MARGIN_MM)
    return f"- Spacing: a {unit:g} pt grid (every space a multiple of {unit:g} pt); page margins {margin:g} mm."


def _owner_set_a_look(kit: Mapping[str, Any], brand_colours: bool, sources: Dict[str, str]) -> bool:
    """Whether the owner set any part of the look: a colour, a role, the accent's use, the type or the spacing."""
    return (brand_colours or ROLE_SET in sources.values() or kit.get(ACCENT_USE_FIELD) == ACCENT_BOLD
            or type_scale_line(kit) != type_scale_line({})
            or _spacing_line(kit) != _spacing_line({}))


def look_system_lines(kit: Mapping[str, Any], brand_colours: bool) -> List[str]:
    """The roles, the accent's use, the type scale and the spacing; none when the owner set no part of the look.

    ``brand_colours``: whether the kit's four colours are the owner's (not the platform's defaults)."""
    roles, sources = effective_palette(kit)
    if not _owner_set_a_look(kit, brand_colours, sources):
        return []
    return [_roles_line(roles, sources), _accent_line(kit), type_scale_line(kit), _spacing_line(kit)]


def logo_size_line(kit: Mapping[str, Any]) -> str:
    """How big the logo prints (the kit's logo rules); empty when the kit has no logo."""
    if not any(isinstance(kit.get(field), str) and kit[field].strip() for field in LOGO_FIELDS):
        return ""
    rules = kit.get(LOGO_RULES_FIELD) if isinstance(kit.get(LOGO_RULES_FIELD), Mapping) else {}
    height = _number(rules.get("letterhead_mm"), DEFAULT_LETTERHEAD_MM)
    least = _number(rules.get("min_mm"), DEFAULT_LOGO_MIN_MM)
    clear = rules.get("clear_space")
    clear = clear if isinstance(clear, (int, float)) and not isinstance(clear, bool) and clear >= 0 else DEFAULT_CLEAR_SPACE
    return (f"- Logo size: {height:g} mm high on a letterhead, never under {least:g} mm, "
            f"with clear space of {clear:g} times its height all round.")


def logo_variants_line(kit: Mapping[str, Any]) -> str:
    """The logo variants the owner uploaded, by the name a session's folder holds them under; empty for none."""
    stems = dict(LOGO_FILES)
    found = [f"{SESSION_BRAND_FOLDER}/{session_file_name(kit[field], stems[field])} ({what})"
             for field, what in LOGO_VARIANTS.items() if isinstance(kit.get(field), str) and kit[field].strip()]
    if not found:
        return ""
    return (f"- Logo variants, uploaded to the brand kit: {', '.join(found)}. In a session, use the copies "
            "its ticket file lists, in the folder it saves into. Never draw, recolour or invent a logo variant.")


def currency_line(kit: Mapping[str, Any]) -> str:
    """The kit's currency; empty when it has none, and then no amount gets one (FR-7)."""
    code = currency_of(kit)
    if not code:
        return ""
    symbol = currency_prefix(code).strip()
    named = f"{code} ({symbol})" if symbol != code else code
    return f"- Currency: {named}. Print every amount in it, never another currency."


def date_line(kit: Optional[Mapping[str, Any]]) -> str:
    """How dates are written: an example in the kit's date style."""
    style = date_style_of(kit)
    return f"- Dates: written as {long_date(EXAMPLE_DATE, style)}."


__all__ = [
    "currency_line", "date_line", "logo_size_line", "logo_variants_line", "look_system_lines", "type_scale_line",
]
