"""PRD-259 §3 — the skin format, automatos-skin/v1 (backend copy).

Studio owns the schema in ``frontend/lib/skins/schema.ts``; these lists are the
same, in the same order, and ``tests/fixtures/skins/cases.json`` holds the
corpus both validators run. A skin is data: token values, shipped font keys,
numbers and catalogue names. Never CSS, script or a URL.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, NamedTuple, Tuple

SKIN_SCHEMA_ID = "automatos-skin/v1"

ALLOWED_TOKENS: Tuple[str, ...] = (
    "background", "foreground", "card", "card-foreground", "popover", "popover-foreground",
    "primary", "primary-foreground", "secondary", "secondary-foreground", "muted", "muted-foreground",
    "accent", "accent-foreground", "destructive", "destructive-foreground", "border", "input", "ring",
    "success", "success-foreground", "warning", "warning-foreground", "info", "info-foreground",
    "agent", "agent-foreground",
    "chart-1", "chart-2", "chart-3", "chart-4", "chart-5",
    "stage-1", "stage-2", "stage-3", "stage-4", "stage-5", "stage-6", "stage-7", "stage-8", "stage-9",
    "olive", "warn-ink", "on-strong", "border-2", "accent-2", "navy",
)

DERIVED_ALIASES: Tuple[str, ...] = ("paper", "bg-1", "bg-2", "fg", "fg-2", "fg-3")

AA_TEXT = 4.5
AA_NON_TEXT = 3


class ContrastPair(NamedTuple):
    """A foreground token checked against a background token at ``min``:1."""

    foreground: str
    background: str
    min: float


CONTRAST_PAIRS: Tuple[ContrastPair, ...] = (
    ContrastPair("foreground", "background", AA_TEXT),
    ContrastPair("card-foreground", "card", AA_TEXT),
    ContrastPair("popover-foreground", "popover", AA_TEXT),
    ContrastPair("primary-foreground", "primary", AA_TEXT),
    ContrastPair("secondary-foreground", "secondary", AA_TEXT),
    ContrastPair("muted-foreground", "background", AA_TEXT),
    ContrastPair("muted-foreground", "card", AA_TEXT),
    ContrastPair("accent-foreground", "accent", AA_TEXT),
    ContrastPair("destructive-foreground", "destructive", AA_TEXT),
    ContrastPair("success-foreground", "success", AA_TEXT),
    ContrastPair("warning-foreground", "warning", AA_TEXT),
    ContrastPair("info-foreground", "info", AA_TEXT),
    ContrastPair("agent-foreground", "agent", AA_TEXT),
    ContrastPair("ring", "background", AA_NON_TEXT),
)

SHIPPED_FONTS: Tuple[str, ...] = ("geist", "geist-mono", "newsreader")
FONT_SLOTS: Tuple[str, ...] = ("sans", "serif", "mono")
FONT_FILE_PATTERN = re.compile(r"assets/fonts/[A-Za-z0-9_-]{1,64}\.woff2")

MASCOT_MODES: Tuple[str, ...] = ("off", "auto", "custom")
MASCOT_STATES: Tuple[str, ...] = (
    "hello", "idle", "listening", "thinking", "working", "searching",
    "reading", "celebrate", "oops", "sleeping", "lost", "loading",
)
MASCOT_SHELLS: Tuple[str, ...] = ("bronze", "gold", "silver", "cobalt")
MASCOT_LIGHTS: Tuple[str, ...] = ("mint", "amber", "ice", "rose")

SKIN_BASES: Tuple[str, ...] = ("studio", "classic")
SKIN_TONES: Tuple[str, ...] = ("light", "dark")

RADIUS_MIN, RADIUS_MAX = 0, 24
GLOW_MIN, GLOW_MAX = 0, 40

SLUG_MAX = 64
SLUG_PATTERN = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
VERSION_MAX = 64
VERSION_PATTERN = re.compile(
    r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)"
    r"(-[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?(\+[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?"
)
PREVIEW_PATH = "assets/preview.png"

# Free-text limits in code points; min 1 means the field can't be empty.
TEXT_LIMITS: Dict[str, Tuple[int, int]] = {
    "name": (1, 40),
    "author": (0, 80),
    "description": (0, 280),
    "licence": (0, 80),
}
# Refused anywhere in free text, compared case-insensitively; control characters too.
TEXT_FORBIDDEN: Tuple[str, ...] = ("<", ">", "://", "url(", "javascript:")

SKIN_ID_PATTERN = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")

SKIN_FIELDS: Tuple[str, ...] = (
    "schema", "slug", "name", "version", "author", "description", "licence",
    "base", "tones", "tokens", "fonts", "shape", "mascot", "preview",
)
REQUIRED_FIELDS: Tuple[str, ...] = ("schema", "slug", "name", "version", "base", "tones", "tokens")

SKIN_ERROR_CODES: Tuple[str, ...] = (
    "SCHEMA", "UNKNOWN_FIELD", "UNKNOWN_TOKEN", "DERIVED_TOKEN", "BAD_VALUE", "TONES_MISMATCH",
    "PAIR_INCOMPLETE", "CONTRAST", "BAD_FONT", "BAD_SHAPE", "BAD_MASCOT", "BAD_TEXT",
)


def contrast_pairs_document() -> List[Dict[str, Any]]:
    """The contrast pairs as plain JSON objects (the corpus and the route's shape)."""
    return [{"foreground": p.foreground, "background": p.background, "min": p.min} for p in CONTRAST_PAIRS]


def schema_document() -> Dict[str, Any]:
    """The rule set as JSON, served at GET /api/skins/schema for the Skin Kit to sync from."""
    return {
        "schema": SKIN_SCHEMA_ID,
        "allowedTokens": list(ALLOWED_TOKENS),
        "derivedAliases": list(DERIVED_ALIASES),
        "contrastPairs": contrast_pairs_document(),
        "shippedFonts": list(SHIPPED_FONTS),
        "mascot": {
            "modes": list(MASCOT_MODES),
            "states": list(MASCOT_STATES),
            "shells": list(MASCOT_SHELLS),
            "lights": list(MASCOT_LIGHTS),
        },
        "shape": {"radius": {"min": RADIUS_MIN, "max": RADIUS_MAX}, "glow": {"min": GLOW_MIN, "max": GLOW_MAX}},
        "text": {
            "limits": {field: {"min": lo, "max": hi} for field, (lo, hi) in TEXT_LIMITS.items()},
            "forbidden": list(TEXT_FORBIDDEN),
        },
    }
