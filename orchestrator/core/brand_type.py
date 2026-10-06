"""The brand kit's one default type scale (PRD-255 FR-4, Decision Q5: no presets).

Every renderer reads its sizes from the kit's ``type_scale``, and a step the kit
leaves out takes its default here. The documents validate the kit's scale
(``modules/documents/brand_system.TypeScale``); a social render reads it in core
(``core/media_render_bundle.py``), which the media-render CI job runs with the
standard library only, so the defaults live here, once, for both.

:func:`step_size_pt` is the size of one step as a render reads it: the kit's own
when it is a usable number, else the default. Pure: no IO.
"""
from __future__ import annotations

from typing import Any, Mapping

REGULAR_WEIGHT, SEMIBOLD_WEIGHT = 400, 600
DISPLAY_STEP, BODY_STEP = "display", "body"
# step -> (size_pt, line_pt, weight): a professional document scale.
DEFAULT_TYPE_SCALE = {
    DISPLAY_STEP: (32.0, 38.0, SEMIBOLD_WEIGHT),
    "h1": (22.0, 28.0, SEMIBOLD_WEIGHT),
    "h2": (15.0, 20.0, SEMIBOLD_WEIGHT),
    "h3": (12.0, 16.0, SEMIBOLD_WEIGHT),
    BODY_STEP: (10.0, 15.0, REGULAR_WEIGHT),
    "small": (8.5, 12.0, REGULAR_WEIGHT),
    "caption": (7.5, 10.0, REGULAR_WEIGHT),
}
TYPE_SCALE_FIELD = "type_scale"


def step_size_pt(kit: Mapping[str, Any], step: str) -> float:
    """The kit's ``size_pt`` for ``step`` when it is a positive number; else the default."""
    scale = kit.get(TYPE_SCALE_FIELD)
    given = scale.get(step) if isinstance(scale, Mapping) else None
    size = given.get("size_pt") if isinstance(given, Mapping) else None
    if isinstance(size, (int, float)) and not isinstance(size, bool) and size > 0:
        return float(size)
    return DEFAULT_TYPE_SCALE[step][0]


__all__ = [
    "BODY_STEP",
    "DEFAULT_TYPE_SCALE",
    "DISPLAY_STEP",
    "REGULAR_WEIGHT",
    "SEMIBOLD_WEIGHT",
    "TYPE_SCALE_FIELD",
    "step_size_pt",
]
