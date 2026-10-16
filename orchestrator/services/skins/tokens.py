"""PRD-259 §3 — a skin's tokens (mirror of ``frontend/lib/skins/tokens.ts``).

Allowed names only, colour values only, the tones they cover, and AA contrast
on the skin's own values (decision K5: setting one side of a pair requires the
other, because a check against base values would fail every Classic skin).
"""

from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional, Sequence

from services.skins.color import normalise_color
from services.skins.contrast import contrast_ratio
from services.skins.fields import SkinError, echo, is_record, one_of, skin_error, unknown_fields
from services.skins.schema import ALLOWED_TOKENS, CONTRAST_PAIRS, DERIVED_ALIASES, SKIN_TONES, ContrastPair

ToneValues = Dict[str, str]


class TokensResult(NamedTuple):
    """The token errors and the normalised values per tone."""

    errors: List[SkinError]
    tokens: Dict[str, ToneValues]


def _entry_error(tone: str, key: str, value: Any) -> Optional[SkinError]:
    path = f"tokens.{tone}.{key}"
    if key in DERIVED_ALIASES:
        return skin_error("DERIVED_TOKEN", path, f'"{key}" is derived by Studio from the core tokens; a skin can\'t set it.')
    if not one_of(ALLOWED_TOKENS, key):
        return skin_error("UNKNOWN_TOKEN", path, f'"{echo(key)}" isn\'t a token a skin can set.')
    if normalise_color(value) is None:
        return skin_error("BAD_VALUE", path, f'"{key}" must be an HSL triplet like "16 100% 50%" or a #rrggbb colour.')
    return None


def _normalised_values(raw: Dict[str, Any]) -> ToneValues:
    values: ToneValues = {}
    for key, value in raw.items():
        normalised = normalise_color(value) if one_of(ALLOWED_TOKENS, key) else None
        if normalised is not None:
            values = {**values, key: normalised}
    return values


def _pair_error(tone: str, raw: Dict[str, Any], values: ToneValues, pair: ContrastPair) -> Optional[SkinError]:
    fg_set, bg_set = pair.foreground in raw, pair.background in raw
    if not fg_set and not bg_set:
        return None
    if fg_set != bg_set:
        missing, given = (pair.background, pair.foreground) if fg_set else (pair.foreground, pair.background)
        return skin_error(
            "PAIR_INCOMPLETE", f"tokens.{tone}.{missing}",
            f'Setting "{given}" in {tone} needs "{missing}" too: they are checked as a pair.',
        )
    fg, bg = values.get(pair.foreground), values.get(pair.background)
    if fg is None or bg is None:
        return None
    ratio = contrast_ratio(fg, bg)
    if ratio >= pair.min:
        return None
    return skin_error(
        "CONTRAST", f"tokens.{tone}.{pair.foreground}",
        f'"{pair.foreground}" on "{pair.background}" in {tone} is {ratio:.2f}:1; it must be at least {pair.min:g}:1.',
    )


def _check_tone(tone: str, raw: Any) -> TokensResult:
    if not is_record(raw):
        return TokensResult([skin_error("SCHEMA", f"tokens.{tone}", f'"tokens.{tone}" must be an object of token values.')], {})
    entry_errors = [e for e in (_entry_error(tone, key, value) for key, value in raw.items()) if e is not None]
    values = _normalised_values(raw)
    pair_errors = [e for e in (_pair_error(tone, raw, values, pair) for pair in CONTRAST_PAIRS) if e is not None]
    return TokensResult(entry_errors + pair_errors, {tone: values})


def _tone_mismatch(tone: str, present: bool, tones: Optional[Sequence[str]]) -> List[SkinError]:
    if tones is None or present == (tone in tones):
        return []
    message = (
        f'"tokens.{tone}" is set but "tones" doesn\'t include "{tone}".' if present
        else f'"tones" includes "{tone}", so "tokens.{tone}" is required.'
    )
    return [skin_error("TONES_MISMATCH", f"tokens.{tone}", message)]


def check_tokens(raw: Any, tones: Optional[Sequence[str]]) -> TokensResult:
    """Check ``tokens``; ``tones`` is None when the skin's own ``tones`` field was invalid."""
    if not is_record(raw):
        return TokensResult([skin_error("SCHEMA", "tokens", '"tokens" must be an object with "light" and/or "dark".')], {})
    errors = unknown_fields(raw, SKIN_TONES, "tokens")
    tokens: Dict[str, ToneValues] = {}
    for tone in SKIN_TONES:
        present = tone in raw
        checked = _check_tone(tone, raw[tone]) if present else TokensResult([], {})
        errors = errors + _tone_mismatch(tone, present, tones) + checked.errors
        tokens = {**tokens, **checked.tokens}
    return TokensResult(errors, tokens)
