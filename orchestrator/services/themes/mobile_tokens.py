"""PRD-261 M-01 (host 1.1.0, plan §2.2.11) — a theme's colours as a phone reads them, and a look projected to them.

* :func:`parse_colour` reads one ``mobile`` token value: Studio's ``H S% L%`` triplet (H from 0 up to,
  not including, 360; S and L 0 to 100; at most two decimals, the form ``hexToTriplet`` writes) or
  ``#rrggbb``. It answers ``#RRGGBB``, the hex ``frontend/lib/look-kit/hsl.ts`` ``tripletToHex`` gives
  for a triplet (and ``contrast.ts`` ``toHex(parseHex(...))`` for a hex), or None. The manifest check
  (``mobile_manifest.py``) refuses exactly the values this returns None for, so a checked block always
  parses, here and on the phone, which vendors ``hsl.ts``.
* :func:`project_look` paints a look through a theme's ``boot/prepaint.json`` table the way the web
  boot script walks it (``frontend/lib/theme-host/boot.ts``): every knob table's entry for the look's
  value, then every pair table's entry for the look's two values, merged in table order; numeric rules
  are skipped (they paint glows, not tokens). A value the table lacks takes the table's ``defaults``
  value. Only ``MOBILE_TOKEN_NAMES`` whose value parses are kept, as the table writes them.
* :func:`look_tone` is the look's tone: the table's tone rule (A-19c) for the knob value the projection
  painted, when the rule gives it one, else theme.json's fixed ``light`` or ``dark``, else ``light``.

Standard library only; nothing here raises on a table or look of the wrong shape.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Mapping, Optional

from services.themes.contract import MOBILE_TOKEN_NAMES, TONE_RULE_KEY, TONE_RULE_TONES

# hsl.ts TRIPLET and contrast.ts HEX, with ASCII digits ([0-9]: Python's \d also takes other scripts' digits).
TRIPLET_PATTERN = re.compile(r"([0-9]{1,3}(?:\.[0-9]{1,2})?) ([0-9]{1,3}(?:\.[0-9]{1,2})?)% ([0-9]{1,3}(?:\.[0-9]{1,2})?)%")
HEX_PATTERN = re.compile(r"#[0-9A-Fa-f]{6}")
FULL_TURN = 360.0
PERCENT = 100.0
CHANNEL_MAX = 255
HUE_STEPS = 12
# hslToHex's channel offsets, in red, green, blue order.
CHANNEL_OFFSETS = (0, 8, 4)
ROUND_HALF = 0.5
# A look with no tone rule and a theme.json tone other than light or dark is drawn light.
MOBILE_DEFAULT_TONE = "light"


def _js_round(value: float) -> int:
    """JavaScript's ``Math.round``: the nearest whole number, a half rounding up (Python's ``round`` goes to even)."""
    whole = math.floor(value)
    return whole + 1 if value - whole >= ROUND_HALF else whole


def _hsl_to_hex(hue: float, saturation: float, lightness: float) -> str:
    """hsl.ts ``hslToHex``, step for step, so both give the same channel for the same triplet."""
    sat, light = saturation / PERCENT, lightness / PERCENT
    chroma = (1 - abs(2 * light - 1)) * sat

    def channel(offset: int) -> int:
        k = (offset + hue / (FULL_TURN / HUE_STEPS)) % HUE_STEPS
        return _js_round((light - chroma / 2 * max(-1, min(k - 3, 9 - k, 1))) * CHANNEL_MAX)

    return "#" + "".join(f"{channel(offset):02X}" for offset in CHANNEL_OFFSETS)


def parse_colour(value: Any) -> Optional[str]:
    """``value`` as ``#RRGGBB`` when it is a triplet or a ``#rrggbb``, else None."""
    if not isinstance(value, str):
        return None
    if HEX_PATTERN.fullmatch(value):
        return value.upper()
    match = TRIPLET_PATTERN.fullmatch(value)
    if match is None:
        return None
    hue, saturation, lightness = (float(part) for part in match.groups())
    if hue >= FULL_TURN or saturation > PERCENT or lightness > PERCENT:
        return None
    return _hsl_to_hex(hue, saturation, lightness)


def _object(value: Any) -> Mapping[str, Any]:
    """``value`` when it is a JSON object, else an empty one."""
    return value if isinstance(value, dict) else {}


def _choice(look: Mapping[str, Any], defaults: Mapping[str, Any], knob: Any, options: Mapping[str, Any]) -> Optional[str]:
    """The look's value for ``knob`` when ``options`` has it, else the defaults' value when it does, else None."""
    for source in (look, defaults):
        value = source.get(knob) if isinstance(knob, str) else None
        if isinstance(value, str) and value in options:
            return value
    return None


def _kept(entry: Any) -> Dict[str, str]:
    """The entry's mobile tokens: names in ``MOBILE_TOKEN_NAMES`` whose value parses, as written."""
    return {
        name: value for name, value in _object(entry).items()
        if name in MOBILE_TOKEN_NAMES and parse_colour(value) is not None
    }


def _knob_entries(table: Mapping[str, Any], look: Mapping[str, Any], defaults: Mapping[str, Any]) -> Dict[str, str]:
    tokens: Dict[str, str] = {}
    for knob, raw_options in _object(table.get("knobs")).items():
        options = _object(raw_options)
        value = _choice(look, defaults, knob, options)
        tokens = {**tokens, **(_kept(options[value]) if value is not None else {})}
    return tokens


def _pair_entries(table: Mapping[str, Any], look: Mapping[str, Any], defaults: Mapping[str, Any]) -> Dict[str, str]:
    tokens: Dict[str, str] = {}
    pairs = table.get("pairs")
    for raw_pair in pairs if isinstance(pairs, list) else []:
        pair = _object(raw_pair)
        rows = _object(pair.get("table"))
        first = _choice(look, defaults, pair.get("a"), rows)
        cells = _object(rows[first]) if first is not None else {}
        second = _choice(look, defaults, pair.get("b"), cells)
        tokens = {**tokens, **(_kept(cells[second]) if second is not None else {})}
    return tokens


def project_look(table: Any, look: Any) -> Dict[str, str]:
    """``look`` painted through the prepaint ``table`` as ``{mobile token: value}`` (module docstring)."""
    table, look = _object(table), _object(look)
    defaults = _object(table.get("defaults"))
    return {**_knob_entries(table, look, defaults), **_pair_entries(table, look, defaults)}


def look_tone(table: Any, look: Any, manifest_tone: Any) -> str:
    """The look's tone: the table's tone rule, else theme.json's ``light`` or ``dark``, else ``light``.

    The rule's knob takes the value the projection painted (its knob table's, else the defaults'), so
    the tone always belongs to the tokens ``project_look`` gives.
    """
    table, look = _object(table), _object(look)
    rule = _object(table.get(TONE_RULE_KEY))
    knob = rule.get("knob")
    options = _object(_object(table.get("knobs")).get(knob)) if isinstance(knob, str) else {}
    choice = _choice(look, _object(table.get("defaults")), knob, options)
    tone = _object(rule.get("values")).get(choice) if choice is not None else None
    if tone in TONE_RULE_TONES:
        return tone
    return manifest_tone if manifest_tone in TONE_RULE_TONES else MOBILE_DEFAULT_TONE
