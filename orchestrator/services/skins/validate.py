"""PRD-259 §3 — the one gate for a skin.json (mirror of ``frontend/lib/skins/validate.ts``).

Runs at import and upload, server-side. Same rules, same codes as the
frontend; ``tests/fixtures/skins/cases.json`` is the corpus both run.
Returns ``{"ok": True, "skin": {...}}`` with token values normalised to
triplets, or ``{"ok": False, "errors": [{code, path, message}]}``.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from services.skins.fields import SkinError, is_record, is_whole_number, one_of, skin_error, unknown_fields
from services.skins.schema import (
    FONT_FILE_PATTERN, FONT_SLOTS, GLOW_MAX, GLOW_MIN, MASCOT_LIGHTS, MASCOT_MODES, MASCOT_SHELLS,
    PREVIEW_PATH, RADIUS_MAX, RADIUS_MIN, REQUIRED_FIELDS, SHIPPED_FONTS, SKIN_BASES, SKIN_FIELDS,
    SKIN_SCHEMA_ID, SKIN_TONES, SLUG_MAX, SLUG_PATTERN, TEXT_FORBIDDEN, TEXT_LIMITS, VERSION_MAX,
    VERSION_PATTERN,
)
from services.skins.tokens import TokensResult, check_tokens

MASCOT_FIELDS = ("mode", "shell", "light")
SHAPE_LIMITS: Dict[str, Tuple[int, int]] = {"radius": (RADIUS_MIN, RADIUS_MAX), "glow": (GLOW_MIN, GLOW_MAX)}
OPTIONAL_COPIED = ("author", "description", "licence", "preview")
SECTIONS_COPIED = ("fonts", "shape", "mascot")


def _matches(pattern: Any, limit: int) -> Callable[[Any], bool]:
    return lambda v: isinstance(v, str) and len(v) <= limit and pattern.fullmatch(v) is not None


# Structural fields: present means it must pass, else SCHEMA.
_HEADER_CHECKS: Tuple[Tuple[str, Callable[[Any], bool], str], ...] = (
    ("schema", lambda v: v == SKIN_SCHEMA_ID, f'"schema" must be "{SKIN_SCHEMA_ID}".'),
    ("slug", _matches(SLUG_PATTERN, SLUG_MAX), f'"slug" must be kebab-case, 1 to {SLUG_MAX} characters.'),
    ("version", _matches(VERSION_PATTERN, VERSION_MAX), '"version" must be a semantic version like "1.0.0".'),
    ("base", lambda v: one_of(SKIN_BASES, v), f'"base" must be one of: {", ".join(SKIN_BASES)}.'),
    ("preview", lambda v: v == PREVIEW_PATH, f'"preview" must be "{PREVIEW_PATH}".'),
)


def _header_errors(data: Dict[str, Any]) -> List[SkinError]:
    missing = [skin_error("SCHEMA", f, f'"{f}" is required.') for f in REQUIRED_FIELDS if f not in data]
    bad = [skin_error("SCHEMA", f, msg) for f, ok, msg in _HEADER_CHECKS if f in data and not ok(data[f])]
    return missing + bad


def _has_control(text: str) -> bool:
    return any(ord(ch) < 0x20 or 0x7F <= ord(ch) <= 0x9F for ch in text)


def _text_error(field: str, value: Any) -> Optional[SkinError]:
    if not isinstance(value, str):
        return skin_error("SCHEMA", field, f'"{field}" must be text.')
    low, high = TEXT_LIMITS[field]
    if not low <= len(value) <= high:
        return skin_error("BAD_TEXT", field, f'"{field}" must be {low} to {high} characters.')
    lower = value.lower()
    if _has_control(value) or any(bad in lower for bad in TEXT_FORBIDDEN):
        return skin_error("BAD_TEXT", field, f'"{field}" must be one line of plain text: no markup, links or control characters.')
    return None


def _text_errors(data: Dict[str, Any]) -> List[SkinError]:
    found = (_text_error(field, data[field]) for field in TEXT_LIMITS if field in data)
    return [e for e in found if e is not None]


def _parse_tones(value: Any) -> Optional[List[str]]:
    """The skin's tones, or None when the field is invalid."""
    if not isinstance(value, list) or not 1 <= len(value) <= len(SKIN_TONES):
        return None
    if not all(one_of(SKIN_TONES, tone) for tone in value):
        return None
    return list(value) if len(set(value)) == len(value) else None


def _font_errors(raw: Any) -> List[SkinError]:
    if not is_record(raw):
        return [skin_error("BAD_FONT", "fonts", '"fonts" must be an object with "sans", "serif" and/or "mono".')]

    def ok(value: Any) -> bool:
        return one_of(SHIPPED_FONTS, value) or (isinstance(value, str) and FONT_FILE_PATTERN.fullmatch(value) is not None)

    bad = [
        skin_error("BAD_FONT", f"fonts.{slot}", f'"fonts.{slot}" must be one of {", ".join(SHIPPED_FONTS)} or a woff2 file in assets/fonts/.')
        for slot in FONT_SLOTS if slot in raw and not ok(raw[slot])
    ]
    return unknown_fields(raw, FONT_SLOTS, "fonts") + bad


def _shape_errors(raw: Any) -> List[SkinError]:
    if not is_record(raw):
        return [skin_error("BAD_SHAPE", "shape", '"shape" must be an object with "radius" and/or "glow".')]
    bad = [
        skin_error("BAD_SHAPE", f"shape.{key}", f'"shape.{key}" must be a whole number from {low} to {high}.')
        for key, (low, high) in SHAPE_LIMITS.items()
        if key in raw and not (is_whole_number(raw[key]) and low <= raw[key] <= high)
    ]
    return unknown_fields(raw, SHAPE_LIMITS, "shape") + bad


def _mascot_errors(raw: Any) -> List[SkinError]:
    if not is_record(raw):
        return [skin_error("BAD_MASCOT", "mascot", '"mascot" must be an object with a "mode".')]
    checks = (("mode", MASCOT_MODES, True), ("shell", MASCOT_SHELLS, False), ("light", MASCOT_LIGHTS, False))
    bad = [
        skin_error("BAD_MASCOT", f"mascot.{key}", f'"mascot.{key}" must be one of: {", ".join(choices)}.')
        for key, choices, required in checks
        if (not one_of(choices, raw[key]) if key in raw else required)
    ]
    return unknown_fields(raw, MASCOT_FIELDS, "mascot") + bad


def _section_errors(data: Dict[str, Any]) -> List[SkinError]:
    checks = (("fonts", _font_errors), ("shape", _shape_errors), ("mascot", _mascot_errors))
    return [e for key, check in checks if key in data for e in check(data[key])]


def _to_skin(data: Dict[str, Any], tones: Sequence[str], tokens: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """A fresh skin with only the known fields; called once every check has passed."""
    optional = {key: data[key] for key in OPTIONAL_COPIED if key in data}
    sections = {key: dict(data[key]) for key in SECTIONS_COPIED if key in data}
    return {
        "schema": SKIN_SCHEMA_ID,
        "slug": data["slug"],
        "name": data["name"],
        "version": data["version"],
        "base": data["base"],
        "tones": list(tones),
        "tokens": tokens,
        **optional,
        **sections,
    }


def validate_skin(data: Any) -> Dict[str, Any]:
    """Validate a parsed skin.json; see the module docstring for the result's shape."""
    if not is_record(data):
        return {"ok": False, "errors": [skin_error("SCHEMA", "", "A skin must be a JSON object.")]}
    tones = _parse_tones(data["tones"]) if "tones" in data else None
    tones_error = [
        skin_error("SCHEMA", "tones", '"tones" must be ["light", "dark"], ["light"] or ["dark"].')
    ] if "tones" in data and tones is None else []
    tokens = check_tokens(data["tokens"], tones) if "tokens" in data else TokensResult([], {})
    errors = (
        unknown_fields(data, SKIN_FIELDS, "") + _header_errors(data) + _text_errors(data)
        + tones_error + tokens.errors + _section_errors(data)
    )
    if errors or tones is None:
        return {"ok": False, "errors": errors}
    return {"ok": True, "skin": _to_skin(data, tones, tokens.tokens)}
