"""PRD-259 §3 — a skin's colour values (backend copy of ``frontend/lib/skins/color.ts``).

Studio's tokens are HSL triplets ('30 12% 9%') read as ``hsl(var(--x))``; a
skin may also write '#rrggbb'. Both normalise to one triplet form, with the
same sums in the same order as the frontend so the two always agree.
"""

from __future__ import annotations

import math
import re
from typing import NamedTuple, Optional, Tuple

HUE_MAX = 360
PERCENT_MAX = 100
CHANNEL_MAX = 255
DECIMALS_SCALE = 100
_NUMBER = r"([0-9]{1,3}(?:\.[0-9]{1,2})?)"
_TRIPLET = re.compile(rf"{_NUMBER} {_NUMBER}% {_NUMBER}%")
_HEX = re.compile(r"#([0-9a-fA-F]{6})")


class Hsl(NamedTuple):
    """Hue 0-360, saturation and lightness 0-100."""

    hue: float
    saturation: float
    lightness: float


def _js_round(value: float) -> float:
    """JavaScript's Math.round: nearest integer, halves toward +infinity."""
    floor = math.floor(value)
    return float(floor + 1 if value - floor >= 0.5 else floor)


def round2(value: float) -> float:
    """Rounded to at most 2 decimals, half up (as the frontend's round2)."""
    return _js_round(value * DECIMALS_SCALE) / DECIMALS_SCALE


def _number_text(value: float) -> str:
    """The number as JavaScript's String() writes it for these magnitudes."""
    rounded = round2(value)
    return str(int(rounded)) if rounded == int(rounded) else repr(rounded)


def format_triplet(hue: float, saturation: float, lightness: float) -> str:
    """The one written form of a token value: 'H S% L%', at most 2 decimals each."""
    return f"{_number_text(hue)} {_number_text(saturation)}% {_number_text(lightness)}%"


def _parse_triplet(value: str) -> Optional[Hsl]:
    match = _TRIPLET.fullmatch(value)
    if not match:
        return None
    hue, saturation, lightness = (float(match.group(i)) for i in (1, 2, 3))
    if hue > HUE_MAX or saturation > PERCENT_MAX or lightness > PERCENT_MAX:
        return None
    return Hsl(hue, saturation, lightness)


def _hue_of(r: float, g: float, b: float, high: float, delta: float) -> float:
    if high == r:
        return ((g - b) / delta + (6 if g < b else 0)) * 60
    if high == g:
        return ((b - r) / delta + 2) * 60
    return ((r - g) / delta + 4) * 60


def _parse_hex(value: str) -> Optional[Hsl]:
    match = _HEX.fullmatch(value)
    if not match:
        return None
    digits = match.group(1)
    r, g, b = (int(digits[at:at + 2], 16) / CHANNEL_MAX for at in (0, 2, 4))
    high, low = max(r, g, b), min(r, g, b)
    light = (high + low) / 2
    if high == low:
        return Hsl(0, 0, round2(light * PERCENT_MAX))
    delta = high - low
    sat = delta / (2 - high - low) if light > 0.5 else delta / (high + low)
    return Hsl(round2(_hue_of(r, g, b, high, delta)), round2(sat * PERCENT_MAX), round2(light * PERCENT_MAX))


def parse_color(value: object) -> Optional[Hsl]:
    """A token value as HSL, or None when it is neither form (BAD_VALUE)."""
    if not isinstance(value, str):
        return None
    return _parse_triplet(value) or _parse_hex(value)


def normalise_color(value: object) -> Optional[str]:
    """A token value in its one written form, or None when it isn't a colour."""
    hsl = parse_color(value)
    return None if hsl is None else format_triplet(hsl.hue, hsl.saturation, hsl.lightness)


def hsl_to_rgb(hsl: Hsl) -> Tuple[float, float, float]:
    """sRGB channels in 0..1 (CSS Color 4's hsl-to-rgb)."""
    sat = hsl.saturation / PERCENT_MAX
    light = hsl.lightness / PERCENT_MAX
    a = sat * min(light, 1 - light)

    def channel(n: int) -> float:
        k = (n + hsl.hue / 30) % 12
        return light - a * max(-1, min(k - 3, 9 - k, 1))

    return channel(0), channel(8), channel(4)
