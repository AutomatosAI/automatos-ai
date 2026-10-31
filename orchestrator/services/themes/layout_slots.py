"""PRD-271 C01, C02 (D-1) — a layout slot's kinds, and a slot's value from theme.json or from a look.

A slot in a theme's ``mobile.slots`` is a literal of its kind, or ``{"look": …}``:

* ``colour``: ``#rrggbb`` or Studio's ``H S% L%`` triplet (``mobile_tokens.parse_colour``, the phone's
  reader of a skin token, so both forms parse there already);
* ``tint``: a colour, ``rgba(r,g,b,a)`` (each channel 0 to 255, the alpha 0 to 1 with up to four
  decimals, no spaces, as look-kit's ``rgba()`` writes it) or ``transparent`` (Candy's prepaint paints
  its aura blobs ``transparent`` off Vivid);
* ``number``: a whole number in the slot's ``min``..``max``;
* ``choice``: one of the slot's ``choices``.

``{"look": "--name"}`` takes a colour or tint slot from a variable the theme's ``boot/prepaint.json``
writes, named as the browser's first paint keeps one (``mobile_tokens.BOOT_VAR_NAME``). ``{"look":
"knob"}`` takes a number or choice slot from a look knob, named as the first paint keeps one
(``css_value.is_knob_name``). A number slot takes a knob's number, or a choice key that writes one
(``"28"``: Candy's ``roundness`` is a choice since kit 1.0.1).

Every value here is data the phone reads as a colour, a number or a named choice: no ``url(``, no
``var(``, no text (PRD-271 §4.7). ``frontend/lib/theme-host/mobile-layout.ts`` is the TypeScript twin.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

from services.themes.contract import is_whole_number, js_number
from services.themes.css_value import is_knob_name
from services.themes.layout_contracts import KIND_COLOUR, KIND_NUMBER, KIND_TINT
from services.themes.mobile_tokens import BOOT_VAR_NAME, parse_colour

LOOK_KEY = "look"
TRANSPARENT = "transparent"
RGBA_PATTERN = re.compile(r"rgba\(([0-9]{1,3}),([0-9]{1,3}),([0-9]{1,3}),(?:0|1|0?\.[0-9]{1,4}|1\.0{1,4})\)")
CHANNEL_MAX = 255
# A choice key that writes a whole number (Candy's roundness keys '16' to '40').
NUMBER_KEY = re.compile(r"-?[0-9]{1,6}")
VARIABLE_KINDS = frozenset({KIND_COLOUR, KIND_TINT})


def is_tint(value: Any) -> bool:
    """A colour, an ``rgba(r,g,b,a)`` whose channels are 0 to 255, or ``transparent``."""
    if not isinstance(value, str):
        return False
    if value == TRANSPARENT or parse_colour(value) is not None:
        return True
    match = RGBA_PATTERN.fullmatch(value)
    return match is not None and all(int(channel) <= CHANNEL_MAX for channel in match.groups())


def _in_range(spec: Mapping[str, Any], number: float) -> bool:
    return spec["min"] <= number <= spec["max"]


def slot_value(spec: Mapping[str, Any], value: Any) -> Optional[Any]:
    """``value`` as the phone gets it when it fits the slot's kind (a number as a whole ``int``), else None."""
    kind = spec["kind"]
    if kind == KIND_COLOUR:
        return value if parse_colour(value) is not None else None
    if kind == KIND_TINT:
        return value if is_tint(value) else None
    if kind == KIND_NUMBER:
        return int(value) if is_whole_number(value) and _in_range(spec, value) else None
    return value if isinstance(value, str) and value in spec["choices"] else None


def knob_number(value: Any) -> Optional[int]:
    """A look knob's value as a whole number: a number, or a choice key that writes one; else None."""
    if isinstance(value, str):
        return int(value) if NUMBER_KEY.fullmatch(value) else None
    number = js_number(value)
    return int(number) if number is not None and number.is_integer() else None


def knob_slot_value(spec: Mapping[str, Any], value: Any) -> Optional[Any]:
    """A number or choice slot's value from a look knob's value, else None (it doesn't fit the slot)."""
    if spec["kind"] != KIND_NUMBER:
        return slot_value(spec, value)
    number = knob_number(value)
    return number if number is not None and _in_range(spec, number) else None


def look_ref(value: Any) -> Optional[str]:
    """The name in ``{"look": "<name>"}`` (that one key, its value text), else None."""
    if not isinstance(value, dict) or set(value) != {LOOK_KEY}:
        return None
    name = value[LOOK_KEY]
    return name if isinstance(name, str) else None


def takes_variable(spec: Mapping[str, Any]) -> bool:
    """A colour or tint slot comes from a prepaint variable; a number or choice slot from a look knob."""
    return spec["kind"] in VARIABLE_KINDS


def is_look_source(spec: Mapping[str, Any], name: Any) -> bool:
    """``name`` is a variable (colour and tint slots) or a knob (number and choice slots) the first paint keeps."""
    if takes_variable(spec):
        return isinstance(name, str) and BOOT_VAR_NAME.fullmatch(name) is not None
    return is_knob_name(name)


__all__ = [
    "LOOK_KEY", "TRANSPARENT", "is_look_source", "is_tint", "knob_number", "knob_slot_value", "look_ref",
    "slot_value", "takes_variable",
]
