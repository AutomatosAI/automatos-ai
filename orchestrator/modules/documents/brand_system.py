"""The brand kit's design system (PRD-255 Brand Kit v2): the colour roles and how the accent is used.

A v1 kit is four colours with no roles, so every renderer painted ``primary`` on
everything. v2 gives each colour a job (FR-1). The roles are stored, sparse, on
``kit['palette']``: a role the owner sets is kept, and every other role is derived
at read time from the kit's four colours (``core.brand_palette.derive_palette``,
FR-2), so no stored kit is migrated.

* :class:`BrandPalette`: the nine roles, each optional; an empty or missing role is
  derived. A set role is stored as a 6-digit hex.
* :data:`ACCENT_USES`: ``sparing`` (the default for every kit, Decision Q1) keeps
  the accent to highlights; ``bold`` lets it fill headers.
* :func:`require_readable_palette`: the contrast check on save (FR-5). Each text
  role is measured on the effective paper and surface_2 (stored, else derived):
  ink, heading and muted need 4.5:1; the accents need 3:1 (large text, rules and
  fills; renderers print small text in an accent only at 4.5:1).
* :func:`brand_kit_view`: the kit as GET answers it, with the effective roles and,
  per role, whether it is ``set`` or ``derived``.

The kit also holds the rest of one designer's rules (US-002), each with a default so
a v1 kit reads complete:

* :class:`TypeScale`: seven steps, each :class:`TypeStep` ``{size_pt, line_pt, weight}``;
  ONE default professional document scale (Decision Q5: no presets).
* ``spacing_unit_pt`` and ``page_margin_mm`` (:data:`DEFAULT_SPACING_UNIT_PT`,
  :data:`DEFAULT_PAGE_MARGIN_MM`), and :class:`LogoRules` (the letterhead logo's
  height, its clear space and its least size).
* Locale: :func:`currency_code` (ISO 4217 shape; empty, the default, prints no
  currency: FR-7) and :data:`DATE_STYLES`.
* :class:`ToneWord`: a tone word and the one line that says what it means. A plain
  string reads as a word with no meaning; :func:`tone_words` is every reader's view.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Literal, Mapping, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_serializer, model_validator
from pydantic_core import InitErrorDetails, PydanticCustomError

from core.brand_palette import (
    PALETTE_ROLES,
    ROLE_ACCENT,
    ROLE_ACCENT_2,
    ROLE_HEADING,
    ROLE_INK,
    ROLE_MUTED,
    ROLE_PAPER,
    ROLE_RULE,
    ROLE_SET,
    ROLE_SURFACE,
    ROLE_SURFACE_2,
    contrast,
    effective_palette,
    parse_hex,
    to_hex,
)
from core.brand_type import DEFAULT_TYPE_SCALE, REGULAR_WEIGHT

PALETTE_FIELD = "palette"
PALETTE_SOURCE_FIELD = "palette_source"

ACCENT_SPARING, ACCENT_BOLD = "sparing", "bold"
ACCENT_USES = (ACCENT_SPARING, ACCENT_BOLD)
DEFAULT_ACCENT_USE = ACCENT_SPARING
# Each role's job, in plain words: the update tool's schema and the agents' rules block say it.
ROLE_JOBS = {
    ROLE_INK: "body text",
    ROLE_HEADING: "headings: near-black, not the accent",
    ROLE_PAPER: "the page background",
    ROLE_SURFACE: "cards and zebra rows",
    ROLE_SURFACE_2: "table header fills",
    ROLE_ACCENT: "highlights: the title rule, key numbers, links",
    ROLE_ACCENT_2: "an optional second accent",
    ROLE_MUTED: "secondary text",
    ROLE_RULE: "hairlines",
}
# What each accent use lets the accent do.
ACCENT_USE_RULES = {
    ACCENT_SPARING: ("the accent only for highlights (the title rule, key numbers, links); "
                     "headings and table headers never filled with it"),
    ACCENT_BOLD: "the accent for highlights and table header fills; headings stay in the heading colour",
}

# WCAG AA, on save: text 4.5:1; large text (and rules and fills) 3:1.
SAVE_TEXT_MIN_CONTRAST = 4.5
SAVE_LARGE_TEXT_MIN_CONTRAST = 3.0
# Each text role and the least contrast it is saved with.
SAVE_ROLE_MIN_CONTRAST = {
    ROLE_INK: SAVE_TEXT_MIN_CONTRAST,
    ROLE_HEADING: SAVE_TEXT_MIN_CONTRAST,
    ROLE_MUTED: SAVE_TEXT_MIN_CONTRAST,
    ROLE_ACCENT: SAVE_LARGE_TEXT_MIN_CONTRAST,
    ROLE_ACCENT_2: SAVE_LARGE_TEXT_MIN_CONTRAST,
}
# The grounds text is printed on.
TEXT_GROUNDS = (ROLE_PAPER, ROLE_SURFACE_2)
# A ratio is reported rounded down, so a near miss never reads as the target.
RATIO_DECIMALS = 1
CONTRAST_ERROR = "palette_contrast"
HEX_RULE = "must be a hex colour such as #1a1a2e or #abc"


class BrandPalette(BaseModel):
    """The kit's colour roles (PRD-255 FR-1). Each is optional: a role left out, or empty, is derived."""

    model_config = ConfigDict(extra="forbid")

    ink: Optional[str] = None
    heading: Optional[str] = None
    paper: Optional[str] = None
    surface: Optional[str] = None
    surface_2: Optional[str] = None
    accent: Optional[str] = None
    accent_2: Optional[str] = None
    muted: Optional[str] = None
    rule: Optional[str] = None

    @field_validator(*PALETTE_ROLES, mode="before")
    @classmethod
    def _six_digit_hex(cls, value: Any) -> Optional[str]:
        if value is None or (isinstance(value, str) and not value.strip()):
            return None
        rgb = parse_hex(value)
        if rgb is None:
            raise ValueError(HEX_RULE)
        return to_hex(rgb)

    @model_serializer(mode="plain")
    def _set_roles_only(self) -> Dict[str, str]:
        """Stored sparse: only the roles that are set."""
        return {role: getattr(self, role) for role in PALETTE_ROLES if getattr(self, role)}


def _rounded_down(ratio: float) -> str:
    scale = 10**RATIO_DECIMALS
    return f"{math.floor(ratio * scale) / scale:.{RATIO_DECIMALS}f}"


def _contrast_message(role: str, ground: str, ratio: float, derived: bool) -> str:
    need = SAVE_ROLE_MIN_CONTRAST[role]
    rule = f"text needs {SAVE_TEXT_MIN_CONTRAST:g}:1"
    if need != SAVE_TEXT_MIN_CONTRAST:
        rule = f"{rule} (large text {need:g}:1)"
    hint = f"; {role} is derived from the kit's colours: set it, or choose a lighter {ground}" if derived else ""
    return f"{role} on {ground} is {_rounded_down(ratio)}:1; {rule}{hint}"


def palette_contrast_errors(kit: Mapping[str, Any]) -> List[InitErrorDetails]:
    """Each text role of ``kit``'s effective palette that does not read on its grounds, as a validation error."""
    roles, sources = effective_palette(kit)
    errors: List[InitErrorDetails] = []
    for role, need in SAVE_ROLE_MIN_CONTRAST.items():
        colour = parse_hex(roles.get(role))
        if colour is None:
            continue
        ratio, ground = min((contrast(colour, parse_hex(roles[g])), g) for g in TEXT_GROUNDS)
        if ratio < need:
            message = _contrast_message(role, ground, ratio, sources.get(role) != ROLE_SET)
            errors.append(InitErrorDetails(
                type=PydanticCustomError(CONTRAST_ERROR, message),
                loc=(PALETTE_FIELD, role),
                input=roles[role],
            ))
    return errors


def require_readable_palette(kit: Mapping[str, Any]) -> None:
    """Raise ``pydantic.ValidationError`` (a 422 on the PUT) naming each role and its failing ratio."""
    errors = palette_contrast_errors(kit)
    if errors:
        raise ValidationError.from_exception_data("BrandKit", errors)


def brand_kit_view(kit: Mapping[str, Any]) -> Dict[str, Any]:
    """``kit`` as GET answers it: ``palette`` is every effective role, ``palette_source`` each one's source."""
    roles, sources = effective_palette(kit)
    return {**kit, PALETTE_FIELD: roles, PALETTE_SOURCE_FIELD: sources}


# ---------------------------------------------------------------------------
# One line of text (the voice's words, meanings and sign-off)
# ---------------------------------------------------------------------------


def one_line_text(value: str, what: str, max_chars: int) -> str:
    """``value`` trimmed; ``ValueError`` when it is longer than ``max_chars`` or not one line."""
    text = value.strip()
    if len(text) > max_chars:
        raise ValueError(f"each {what} is at most {max_chars} characters")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise ValueError(f"each {what} is one line of text")
    return text


# ---------------------------------------------------------------------------
# The type scale (FR-1; Decision Q5: one default scale, no presets)
# ---------------------------------------------------------------------------

# The default scale itself (DEFAULT_TYPE_SCALE) is core/brand_type.py's: a social render reads it too.
TYPE_STEPS = tuple(DEFAULT_TYPE_SCALE)
MIN_TYPE_PT, MAX_TYPE_PT = 5.0, 96.0
# A line is at least its size, and at most this many times it.
MAX_LINE_RATIO = 3.0
MIN_WEIGHT, MAX_WEIGHT, WEIGHT_STEP = 100, 900, 100


class TypeStep(BaseModel):
    """One step of the type scale: its size and line height in points, and its weight."""

    model_config = ConfigDict(extra="forbid")

    size_pt: float
    line_pt: float
    weight: int = REGULAR_WEIGHT

    @field_validator("size_pt")
    @classmethod
    def _size(cls, value: float) -> float:
        if not MIN_TYPE_PT <= value <= MAX_TYPE_PT:
            raise ValueError(f"size_pt must be {MIN_TYPE_PT:g} to {MAX_TYPE_PT:g} pt (got {value:g})")
        return value

    @field_validator("weight")
    @classmethod
    def _weight(cls, value: int) -> int:
        if not MIN_WEIGHT <= value <= MAX_WEIGHT or value % WEIGHT_STEP:
            raise ValueError(f"weight must be {MIN_WEIGHT} to {MAX_WEIGHT}, in hundreds (got {value})")
        return value

    @model_validator(mode="after")
    def _line_fits_size(self) -> "TypeStep":
        if not self.size_pt <= self.line_pt <= self.size_pt * MAX_LINE_RATIO:
            raise ValueError(
                f"line_pt must be from size_pt to {MAX_LINE_RATIO:g} times it "
                f"(got {self.line_pt:g} for {self.size_pt:g} pt)"
            )
        return self


def default_type_step(step: str) -> Dict[str, Any]:
    """The default ``step`` of the scale, as stored."""
    size_pt, line_pt, weight = DEFAULT_TYPE_SCALE[step]
    return {"size_pt": size_pt, "line_pt": line_pt, "weight": weight}


def _step_field(step: str) -> Any:
    return Field(default_factory=lambda: TypeStep(**default_type_step(step)))


class TypeScale(BaseModel):
    """The kit's seven type steps. A step left out takes its default, and so does a field a step leaves out."""

    model_config = ConfigDict(extra="forbid")

    display: TypeStep = _step_field("display")
    h1: TypeStep = _step_field("h1")
    h2: TypeStep = _step_field("h2")
    h3: TypeStep = _step_field("h3")
    body: TypeStep = _step_field("body")
    small: TypeStep = _step_field("small")
    caption: TypeStep = _step_field("caption")

    @model_validator(mode="before")
    @classmethod
    def _steps_over_defaults(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        return {
            step: {**default_type_step(step), **given} if isinstance(given, Mapping) and step in TYPE_STEPS else given
            for step, given in value.items()
        }


# ---------------------------------------------------------------------------
# Spacing, margins and the logo's rules
# ---------------------------------------------------------------------------

DEFAULT_SPACING_UNIT_PT = 4.0
MIN_SPACING_UNIT_PT, MAX_SPACING_UNIT_PT = 2.0, 12.0
DEFAULT_PAGE_MARGIN_MM = 18.0
MIN_PAGE_MARGIN_MM, MAX_PAGE_MARGIN_MM = 6.0, 50.0

LOGO_RULES_FIELD = "logo_rules"
DEFAULT_LETTERHEAD_MM = 16.0
MIN_LETTERHEAD_MM, MAX_LETTERHEAD_MM = 6.0, 60.0
# Clear space round the logo, in logo heights.
DEFAULT_CLEAR_SPACE = 0.5
MIN_CLEAR_SPACE, MAX_CLEAR_SPACE = 0.0, 2.0
DEFAULT_LOGO_MIN_MM = 8.0
MIN_LOGO_MIN_MM, MAX_LOGO_MIN_MM = 4.0, 40.0


def within(value: float, low: float, high: float, what: str, unit: str = "") -> float:
    """``value`` when it is from ``low`` to ``high``; ``ValueError`` naming ``what`` and the bounds if not."""
    if not low <= value <= high:
        raise ValueError(f"{what} must be {low:g} to {high:g}{unit} (got {value:g})")
    return value


class LogoRules(BaseModel):
    """How the logo is placed: its letterhead height, its clear space and the least size it prints at."""

    model_config = ConfigDict(extra="forbid")

    letterhead_mm: float = DEFAULT_LETTERHEAD_MM
    clear_space: float = DEFAULT_CLEAR_SPACE
    min_mm: float = DEFAULT_LOGO_MIN_MM

    @field_validator("letterhead_mm")
    @classmethod
    def _letterhead(cls, value: float) -> float:
        return within(value, MIN_LETTERHEAD_MM, MAX_LETTERHEAD_MM, "letterhead_mm", " mm")

    @field_validator("clear_space")
    @classmethod
    def _clear_space(cls, value: float) -> float:
        return within(value, MIN_CLEAR_SPACE, MAX_CLEAR_SPACE, "clear_space", " logo heights")

    @field_validator("min_mm")
    @classmethod
    def _min(cls, value: float) -> float:
        return within(value, MIN_LOGO_MIN_MM, MAX_LOGO_MIN_MM, "min_mm", " mm")

    @model_validator(mode="after")
    def _letterhead_at_least_the_least_size(self) -> "LogoRules":
        if self.letterhead_mm < self.min_mm:
            raise ValueError(
                f"letterhead_mm ({self.letterhead_mm:g} mm) must be at least min_mm ({self.min_mm:g} mm)"
            )
        return self


# ---------------------------------------------------------------------------
# Locale: currency and date style (FR-7, FR-8)
# ---------------------------------------------------------------------------

# Empty: the kit has no currency, and no renderer adds one (FR-7).
DEFAULT_CURRENCY = ""
CURRENCY_CODE = re.compile(r"^[A-Z]{3}$")
DATE_STYLE_DAY_FIRST, DATE_STYLE_MONTH_FIRST = "d MMMM yyyy", "MMMM d, yyyy"
DATE_STYLES = (DATE_STYLE_DAY_FIRST, DATE_STYLE_MONTH_FIRST)
DateStyle = Literal["d MMMM yyyy", "MMMM d, yyyy"]
DEFAULT_DATE_STYLE = DATE_STYLE_DAY_FIRST


def currency_code(value: str) -> str:
    """``value`` as an ISO 4217 code (three letters, upper case), or empty for none."""
    code = value.strip().upper()
    if code and not CURRENCY_CODE.match(code):
        raise ValueError(f"currency must be a three-letter ISO 4217 code such as GBP, or empty (got {value.strip()!r})")
    return code


# ---------------------------------------------------------------------------
# Tone words with meanings
# ---------------------------------------------------------------------------

MAX_TONE_WORD_CHARS = 32
MAX_TONE_MEANING_CHARS = 120


class ToneWord(BaseModel):
    """A tone word and, optionally, the one line that says what it means for this brand."""

    model_config = ConfigDict(extra="forbid")

    word: str
    meaning: str = ""

    @model_validator(mode="before")
    @classmethod
    def _plain_word(cls, value: Any) -> Any:
        return {"word": value} if isinstance(value, str) else value

    @field_validator("word")
    @classmethod
    def _word(cls, value: str) -> str:
        text = one_line_text(value, "tone word", MAX_TONE_WORD_CHARS)
        if text and not any(ch.isalpha() for ch in text):
            raise ValueError("each tone word needs a letter")
        return text

    @field_validator("meaning")
    @classmethod
    def _meaning(cls, value: str) -> str:
        return one_line_text(value, "tone word's meaning", MAX_TONE_MEANING_CHARS)


def tone_words(kit: Optional[Mapping[str, Any]]) -> List[Dict[str, str]]:
    """The kit's tone words as ``[{word, meaning}]``, read leniently: a plain string is a word with no meaning."""
    voice = (kit or {}).get("voice")
    entries = voice.get("tone") if isinstance(voice, Mapping) else None
    found: List[Dict[str, str]] = []
    for entry in entries if isinstance(entries, list) else []:
        raw = {"word": entry} if isinstance(entry, str) else entry
        if not isinstance(raw, Mapping):
            continue
        word, meaning = str(raw.get("word") or "").strip(), str(raw.get("meaning") or "").strip()
        if word:
            found.append({"word": word, "meaning": meaning})
    return found


__all__ = [
    "ACCENT_USES",
    "ACCENT_USE_RULES",
    "BrandPalette",
    "DATE_STYLES",
    "DEFAULT_ACCENT_USE",
    "DEFAULT_CURRENCY",
    "DEFAULT_DATE_STYLE",
    "DEFAULT_PAGE_MARGIN_MM",
    "DEFAULT_SPACING_UNIT_PT",
    "DEFAULT_TYPE_SCALE",
    "DateStyle",
    "LogoRules",
    "MAX_TONE_MEANING_CHARS",
    "MAX_TONE_WORD_CHARS",
    "PALETTE_FIELD",
    "PALETTE_SOURCE_FIELD",
    "ROLE_JOBS",
    "SAVE_ROLE_MIN_CONTRAST",
    "TYPE_STEPS",
    "ToneWord",
    "TypeScale",
    "TypeStep",
    "brand_kit_view",
    "currency_code",
    "default_type_step",
    "one_line_text",
    "palette_contrast_errors",
    "require_readable_palette",
    "tone_words",
    "within",
]
