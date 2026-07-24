"""Workspace brand kit (PRD-167 S4).

The brand kit is stored on ``workspace.settings['brand_kit']`` — no new table, per the
repo rule ("no new table when an existing one fits"). It supplies ``{{brand.*}}``
variables and the palette/fonts/logo applied to rendered PDFs and DOCX, replacing the
hardcoded Automatos branding (the ``#ff6b35`` orange) that used to live in the
renderers.

Defaults are a neutral professional palette — an unconfigured workspace renders cleanly
(and *not* in Automatos orange).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

logger = logging.getLogger(__name__)

BRAND_KIT_SETTINGS_KEY = "brand_kit"

_HEX_RE = re.compile(r"^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")

# Neutral professional defaults (NOT Automatos orange — that was the hardcoded brand
# this PRD removes from the render paths).
DEFAULT_PRIMARY = "#1a1a2e"
DEFAULT_SECONDARY = "#16213e"
DEFAULT_ACCENT = "#0f3460"
DEFAULT_TEXT = "#1a1a2e"
DEFAULT_FONT = "Inter, 'Segoe UI', system-ui, sans-serif"


class CompanyContact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = ""
    address: str = ""
    email: str = ""
    phone: str = ""
    website: str = ""


class BrandKit(BaseModel):
    """Validated brand kit. Stored as a plain dict; this model gates writes and
    supplies defaults on read."""

    model_config = ConfigDict(extra="forbid")

    name: str = ""
    tagline: str = ""
    logo_url: str = ""
    primary_color: str = DEFAULT_PRIMARY
    secondary_color: str = DEFAULT_SECONDARY
    accent_color: str = DEFAULT_ACCENT
    text_color: str = DEFAULT_TEXT
    font_family: str = DEFAULT_FONT
    company: CompanyContact = Field(default_factory=CompanyContact)

    @field_validator("primary_color", "secondary_color", "accent_color", "text_color")
    @classmethod
    def _validate_hex(cls, v: str) -> str:
        if v and not _HEX_RE.match(v):
            raise ValueError("must be a hex color such as #1a1a2e or #abc")
        return v


def get_brand_kit(settings: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the workspace brand kit as a fully-populated dict (defaults merged in).

    Lenient on read: malformed stored data falls back to defaults so a render never
    crashes on a bad brand kit. Writes go through :func:`validate_brand_kit` which is
    strict.
    """
    raw = (settings or {}).get(BRAND_KIT_SETTINGS_KEY) or {}
    try:
        return BrandKit.model_validate(raw).model_dump()
    except Exception:  # noqa: BLE001 — read path must not raise
        logger.warning("[BrandKit] stored brand kit failed validation; using defaults")
        return BrandKit().model_dump()


def validate_brand_kit(patch: Dict[str, Any], existing: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Validate + merge a brand-kit patch over the existing kit, returning the new dict.

    Raises ``pydantic.ValidationError`` (surfaced as 422 by the API) on bad input.
    """
    base = get_brand_kit({BRAND_KIT_SETTINGS_KEY: existing} if existing else None)
    merged = {**base, **{k: v for k, v in patch.items() if v is not None}}
    if "company" in patch and patch["company"] is not None:
        merged["company"] = {**base.get("company", {}), **patch["company"]}
    return BrandKit.model_validate(merged).model_dump()


__all__ = [
    "BRAND_KIT_SETTINGS_KEY",
    "BrandKit",
    "CompanyContact",
    "get_brand_kit",
    "validate_brand_kit",
    "DEFAULT_FONT",
]
