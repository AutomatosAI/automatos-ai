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
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_serializer
from pydantic_core import InitErrorDetails, PydanticCustomError

from core.brand_palette import (
    PALETTE_ROLES,
    ROLE_ACCENT,
    ROLE_ACCENT_2,
    ROLE_HEADING,
    ROLE_INK,
    ROLE_MUTED,
    ROLE_PAPER,
    ROLE_SET,
    ROLE_SURFACE_2,
    contrast,
    effective_palette,
    parse_hex,
    to_hex,
)

PALETTE_FIELD = "palette"
PALETTE_SOURCE_FIELD = "palette_source"

ACCENT_SPARING, ACCENT_BOLD = "sparing", "bold"
ACCENT_USES = (ACCENT_SPARING, ACCENT_BOLD)
DEFAULT_ACCENT_USE = ACCENT_SPARING

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


__all__ = [
    "ACCENT_USES",
    "BrandPalette",
    "DEFAULT_ACCENT_USE",
    "PALETTE_FIELD",
    "PALETTE_SOURCE_FIELD",
    "SAVE_ROLE_MIN_CONTRAST",
    "brand_kit_view",
    "palette_contrast_errors",
    "require_readable_palette",
]
