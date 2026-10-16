"""PRD-259 §3 — WCAG 2.x relative luminance and contrast ratio for skin tokens.

Ported from Play's ``lib/looks/contrast.ts`` (read-only reference) and the
mirror of ``frontend/lib/skins/contrast.ts``: a value goes HSL → sRGB first,
because Studio's tokens are HSL triplets.
"""

from __future__ import annotations

from services.skins.color import hsl_to_rgb, parse_color

LUMINANCE_WEIGHTS = (0.2126, 0.7152, 0.0722)
FLARE = 0.05


def _linear(channel: float) -> float:
    return channel / 12.92 if channel <= 0.04045 else ((channel + 0.055) / 1.055) ** 2.4


def relative_luminance(value: str) -> float:
    """Relative luminance of a token value ('H S% L%' or '#rrggbb'); ValueError on anything else."""
    hsl = parse_color(value)
    if hsl is None:
        raise ValueError("not a skin colour value")
    r, g, b = hsl_to_rgb(hsl)
    return LUMINANCE_WEIGHTS[0] * _linear(r) + LUMINANCE_WEIGHTS[1] * _linear(g) + LUMINANCE_WEIGHTS[2] * _linear(b)


def contrast_ratio(foreground: str, background: str) -> float:
    """The WCAG contrast ratio of two token values, 1 to 21."""
    a = relative_luminance(foreground)
    b = relative_luminance(background)
    return (max(a, b) + FLARE) / (min(a, b) + FLARE)
