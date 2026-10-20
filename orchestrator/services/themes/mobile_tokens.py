"""PRD-261 M-01 (host 1.1.0, plan §2.2.11) — a theme's colours as a phone reads them, and a look painted for one.

The phone shows exactly what the web paints first (W-M1's seam rule), so a look is painted here the
way the web's first paint paints it: ``frontend/lib/theme-host/boot-sanitize.ts`` ``sanitizePrepaint``
(every installed boot passes it, ``boot.ts`` :129) and then ``THEME_BOOT_SCRIPT`` (``boot.ts`` :156-:167).

* :func:`parse_colour` reads one ``mobile`` token value: Studio's ``H S% L%`` triplet (H from 0 up to,
  not including, 360; S and L 0 to 100; at most two decimals, as ``hexToTriplet`` writes) or ``#rrggbb``,
  as the ``#RRGGBB`` look-kit's ``hsl.ts`` ``tripletToHex`` (and ``contrast.ts`` for a hex) gives, or None.
  ``mobile_manifest.py`` refuses exactly what this returns None for, so a checked block always parses,
  here and on the phone, which vendors ``hsl.ts``.
* :func:`is_studio_triplet`: a value the web paints a Studio token with. ``frontend/tailwind.config.ts``
  and ``globals.css`` read every ``MOBILE_TOKEN_NAMES`` token as ``hsl(var(--x))``, so a ``#rrggbb``
  paints nothing there; ``mobile.tokens``, which only the phone reads, keeps it (``parse_colour``).
* :func:`relative_luminance` is ``contrast.ts`` ``relativeLuminance`` (WCAG 2.x), step for step.
* :func:`boot_table` is a ``boot/prepaint.json`` table as the boot script gets it (``sanitizePrepaint``):
  knobs named like a look field (``KNOB_NAME``), option keys of 1 to 64 UTF-16 units, each option's
  variables those whose name and value pass the boot's gate (``sanitizeVars``); pair tables whose two
  knobs are look fields; numeric rules that carry every field (``template`` ``rgba-alpha``, a bool
  ``integer``, ``min`` <= ``max``, ``scale`` > 0, ``capTable`` null or two look fields, a colour knob),
  other rules dropped; ``defaults`` with text or finite number values; the tone rule kept only when
  every choice of its knob has a tone. None when the defaults don't fit what is left (boot-sanitize.ts
  :231): the web then paints nothing from the table. A number is the double ``JSON.parse`` gives the
  web (``contract.js_number``): a 310-digit literal is Infinity, so not finite.
* :func:`look_fits` is the boot script's ``fits`` (boot.ts :158-:159; ``lookFitsPrepaint``): the look
  names an option of every knob table and a cell of every pair table, and for every numeric rule its
  knob is a finite number in range (whole when ``integer``), its colour knob a key of the rule's
  colours and, with a cap table, its two knobs a cell of the caps.
* :func:`paint_look` is what the web paints for a look, as the phone takes it: the look when it fits,
  else the table's ``defaults`` WHOLE (boot.ts :164; never value by value); every knob table's entry,
  then every pair table's, a later one overwriting (:165-:166); a numeric rule's variable painted last
  with ``rgba()`` (:167), no phone colour, so not painted for the phone. Kept: the ``MOBILE_TOKEN_NAMES``
  whose final value ``is_studio_triplet``, as the table writes them (a ``#rrggbb`` is not painted).
* :func:`missing_tokens` names the ``MOBILE_REQUIRED_TOKENS`` a painting or scheme lacks: the one rule
  the reader (M-02), the review (M-03) and the answer (M-04) share.
* :func:`look_tone`: theme.json's fixed ``light`` or ``dark`` wins (the boot's tone value is the
  policy's, boot.ts :57-:60 and :69); under ``from-look`` the table's tone rule at the painted look's
  value; under ``from-look`` with no rule, and under ``user`` (the phone has no tone cookie), the
  painted ground's: ``dark`` when ``--background`` is darker than ``--foreground`` (look-kit
  ``css.ts`` ``isDarkGround``), else ``light``, and ``light`` when either is missing.

Standard library only; nothing here raises on a table or look of the wrong shape.
"""

from __future__ import annotations

import math
import re
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Tuple

from services.themes.contract import (
    MOBILE_REQUIRED_TOKENS, MOBILE_TOKEN_NAMES, TONE_RULE_KEY, TONE_RULE_TONES, is_finite_number, js_number,
)

# hsl.ts TRIPLET and contrast.ts HEX, with ASCII digits ([0-9]: Python's \d also takes other scripts' digits).
TRIPLET_PATTERN = re.compile(r"([0-9]{1,3}(?:\.[0-9]{1,2})?) ([0-9]{1,3}(?:\.[0-9]{1,2})?)% ([0-9]{1,3}(?:\.[0-9]{1,2})?)%")
HEX_PATTERN = re.compile(r"#[0-9A-Fa-f]{6}")
FULL_TURN, PERCENT = 360.0, 100.0
CHANNEL_MAX, HUE_STEPS = 255, 12
# hslToHex's channel offsets, in red, green, blue order.
CHANNEL_OFFSETS = (0, 8, 4)
ROUND_HALF = 0.5
# contrast.ts relativeLuminance (WCAG 2.x), with its constants.
HEX_BASE = 16
HEX_CHANNEL_STARTS: Tuple[int, ...] = (1, 3, 5)
LINEAR_KNEE, LINEAR_SLOPE = 0.04045, 12.92
GAMMA_OFFSET, GAMMA_SCALE, GAMMA = 0.055, 1.055, 2.4
RED_WEIGHT, GREEN_WEIGHT, BLUE_WEIGHT = 0.2126, 0.7152, 0.0722

# boot-sanitize.ts: KNOB_NAME, OPTION_KEY_MAX_LENGTH, CSS_VAR_NAME_PATTERN (boot.ts V), the CSS-value
# gate isCssValue (CSS_VALUE_MAX_LENGTH, CSS_VALUE_CHARS, CSS_FUNCTION_NAME, CSS_VALUE_FUNCTIONS),
# RGB_LIST, CAP_MIN and CAP_MAX. A JavaScript string's length counts UTF-16 units.
KNOB_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,31}")
OPTION_KEY_MAX_LENGTH = 64
BOOT_VAR_NAME = re.compile(r"--[a-z0-9-]{1,64}")
BOOT_VALUE_MAX_LENGTH = 200
BOOT_VALUE_CHARS = re.compile(r"[A-Za-z0-9#%.,()/ +-]+")
BOOT_FUNCTION_NAME = re.compile(r"([A-Za-z-]+)\(")
BOOT_VALUE_FUNCTIONS = frozenset({
    "rgb", "rgba", "hsl", "hsla", "var", "calc", "min", "max", "clamp",
    "hue-rotate", "saturate", "brightness", "grayscale", "contrast", "sepia", "invert",
})
RGB_LIST = re.compile(r"[0-9]{1,3},[0-9]{1,3},[0-9]{1,3}")
CAP_MIN, CAP_MAX = 0, 1
NUMERIC_TEMPLATE = "rgba-alpha"
UTF16_UNIT_BYTES = 2

# The tones, the policy that takes the tone from the look, and the two tokens a ground is read from.
TONE_LIGHT, TONE_DARK = TONE_RULE_TONES
TONE_FROM_LOOK = "from-look"
GROUND_TOKEN, TEXT_TOKEN = "--background", "--foreground"
# A painting with no ground or no text is drawn light.
MOBILE_DEFAULT_TONE = TONE_LIGHT


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


def is_studio_triplet(value: Any) -> bool:
    """``value`` is an ``H S% L%`` triplet the phone reads: the one form the web paints a Studio token in (module docstring)."""
    return isinstance(value, str) and TRIPLET_PATTERN.fullmatch(value) is not None and parse_colour(value) is not None


def _linear(channel: int) -> float:
    srgb = channel / CHANNEL_MAX
    return srgb / LINEAR_SLOPE if srgb <= LINEAR_KNEE else ((srgb + GAMMA_OFFSET) / GAMMA_SCALE) ** GAMMA


def relative_luminance(hex_colour: str) -> float:
    """contrast.ts ``relativeLuminance`` of a ``#rrggbb`` (either case); ValueError for anything else."""
    if not isinstance(hex_colour, str) or HEX_PATTERN.fullmatch(hex_colour) is None:
        raise ValueError(f"{hex_colour!r} isn't a #rrggbb colour")
    red, green, blue = (int(hex_colour[start:start + 2], HEX_BASE) for start in HEX_CHANNEL_STARTS)
    return RED_WEIGHT * _linear(red) + GREEN_WEIGHT * _linear(green) + BLUE_WEIGHT * _linear(blue)


# ---------------------------------------------------------------------------
# The table the boot script gets (boot-sanitize.ts sanitizePrepaint)
# ---------------------------------------------------------------------------


class BootPair(NamedTuple):
    a: str
    b: str
    table: Mapping[str, Mapping[str, Mapping[str, str]]]


class BootCaps(NamedTuple):
    knobs: Optional[Tuple[str, str]]
    cells: Mapping[str, Mapping[str, float]]


class BootRule(NamedTuple):
    var: str
    knob: str
    low: float
    high: float
    integer: bool
    color_knob: str
    rgb: Mapping[str, str]
    caps: BootCaps


class BootTone(NamedTuple):
    knob: str
    values: Mapping[str, str]


class BootTable(NamedTuple):
    knobs: Mapping[str, Mapping[str, Mapping[str, str]]]
    pairs: Tuple[BootPair, ...]
    numeric: Tuple[BootRule, ...]
    defaults: Mapping[str, Any]
    tone: Optional[BootTone]


NO_CAPS = BootCaps(None, {})


def _object(value: Any) -> Mapping[str, Any]:
    """``value`` when it is a JSON object, else an empty one."""
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> List[Any]:
    return value if isinstance(value, list) else []


def _js_length(text: str) -> int:
    return len(text.encode("utf-16-le", "surrogatepass")) // UTF16_UNIT_BYTES


def _is_knob(name: Any) -> bool:
    return isinstance(name, str) and KNOB_NAME.fullmatch(name) is not None


def _boot_value(value: Any) -> bool:
    """boot-sanitize.ts ``isCssValue``: the boot's own gate, smaller than css_value.py's grammar."""
    if not isinstance(value, str) or not 0 < _js_length(value) <= BOOT_VALUE_MAX_LENGTH:
        return False
    if BOOT_VALUE_CHARS.fullmatch(value) is None or value.count("(") != value.count(")"):
        return False
    return all(name.lower() in BOOT_VALUE_FUNCTIONS for name in BOOT_FUNCTION_NAME.findall(value))


def _boot_vars(entry: Any) -> Dict[str, str]:
    """``sanitizeVars``: the entry's variables whose name and value the boot passes; never None."""
    return {name: value for name, value in _object(entry).items() if BOOT_VAR_NAME.fullmatch(name) and _boot_value(value)}


def _option_table(table: Any, clean: Callable[[Any], Any]) -> Dict[str, Any]:
    """``sanitizeTable``: keys of 1 to 64 units, each with ``clean``'s value; a None from ``clean`` drops the key."""
    cleaned = {key: clean(value) for key, value in _object(table).items() if 0 < _js_length(key) <= OPTION_KEY_MAX_LENGTH}
    return {key: value for key, value in cleaned.items() if value is not None}


def _pair_table(table: Any, clean: Callable[[Any], Any]) -> Dict[str, Dict[str, Any]]:
    """``sanitizePairTable``: rows of cells, each through ``_option_table``."""
    return _option_table(table, lambda row: _option_table(row, clean))


def _boot_pair(raw: Any) -> Optional[BootPair]:
    """``sanitizePair``: both knobs look fields, the cells' variables gated; None drops the pair."""
    pair = _object(raw)
    first, second = pair.get("a"), pair.get("b")
    if not (_is_knob(first) and _is_knob(second)):
        return None
    return BootPair(first, second, _pair_table(pair.get("table"), _boot_vars))


def _cap(value: Any) -> Optional[float]:
    number = js_number(value)
    return number if number is not None and CAP_MIN <= number <= CAP_MAX else None


def _boot_caps(rule: Mapping[str, Any]) -> Optional[BootCaps]:
    """``sanitizeCapTable``: ``capTable`` null (no caps) or two look fields with their caps; None drops the rule."""
    if "capTable" in rule and rule["capTable"] is None:
        return NO_CAPS
    knobs = rule.get("capTable")
    if not isinstance(knobs, list) or len(knobs) != 2 or not all(_is_knob(knob) for knob in knobs):
        return None
    return BootCaps((knobs[0], knobs[1]), _pair_table(rule.get("caps"), _cap))


def _rgb(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and RGB_LIST.fullmatch(value) else None


def _boot_rule(raw: Any) -> Optional[BootRule]:
    """``sanitizeNumericRule``: a rule with every field the boot script reads; None drops it."""
    rule = _object(raw)
    var, knob, integer = rule.get("var"), rule.get("knob"), rule.get("integer")
    if rule.get("template") != NUMERIC_TEMPLATE or not isinstance(integer, bool):
        return None
    if not isinstance(var, str) or BOOT_VAR_NAME.fullmatch(var) is None or not _is_knob(knob):
        return None
    low, high, scale = (js_number(rule.get(field)) for field in ("min", "max", "scale"))
    if low is None or high is None or scale is None or low > high or scale <= 0:
        return None
    caps, color = _boot_caps(rule), _object(rule.get("color"))
    if caps is None or not _is_knob(color.get("knob")):
        return None
    return BootRule(var, knob, low, high, integer, color["knob"], _option_table(color.get("rgb"), _rgb), caps)


def _boot_defaults(defaults: Any) -> Dict[str, Any]:
    """``sanitizeDefaults``: look fields whose value is text or a number JavaScript reads as finite."""
    kept = _object(defaults).items()
    return {knob: value for knob, value in kept if _is_knob(knob) and (isinstance(value, str) or is_finite_number(value))}


def _boot_tone(raw: Any, table: BootTable) -> Optional[BootTone]:
    """``sanitizeToneRule``: a choice knob of what is left, every one of its choices given a tone; else None."""
    rule = _object(raw)
    knob, values = rule.get("knob"), rule.get("values")
    if not isinstance(knob, str) or not isinstance(values, dict) or knob not in table.knobs:
        return None
    if any(numeric.knob == knob for numeric in table.numeric):
        return None
    choices = list(table.knobs[knob])
    mapped = {choice: values[choice] for choice in choices if choice in values and values[choice] in TONE_RULE_TONES}
    return BootTone(knob, mapped) if choices and len(mapped) == len(choices) else None


def boot_table(table: Any) -> Optional[BootTable]:
    """The prepaint ``table`` as the boot script gets it, or None when the web paints nothing from it (module docstring)."""
    if not isinstance(table, dict):
        return None
    knobs = {knob: _option_table(options, _boot_vars) for knob, options in _object(table.get("knobs")).items() if _is_knob(knob)}
    pairs = tuple(pair for pair in map(_boot_pair, _list(table.get("pairs"))) if pair is not None)
    numeric = tuple(rule for rule in map(_boot_rule, _list(table.get("numeric"))) if rule is not None)
    kept = BootTable(knobs, pairs, numeric, _boot_defaults(table.get("defaults")), None)
    if not look_fits(kept, kept.defaults):
        return None
    return kept._replace(tone=_boot_tone(table.get(TONE_RULE_KEY), kept))


# ---------------------------------------------------------------------------
# The boot script: fits, then the walk (boot.ts :158-:167)
# ---------------------------------------------------------------------------


def _has(table: Any, key: Any) -> bool:
    """boot.ts ``has`` and boot-sanitize.ts ``hasKey``: ``key`` is text and an own key of the object ``table``."""
    return isinstance(table, dict) and isinstance(key, str) and key in table


def _rule_fits(rule: BootRule, look: Mapping[str, Any]) -> bool:
    """boot.ts ``num`` (``numericFits``)."""
    value = js_number(look.get(rule.knob))
    if value is None or not rule.low <= value <= rule.high or (rule.integer and value % 1 != 0):
        return False
    if not _has(rule.rgb, look.get(rule.color_knob)):
        return False
    if rule.caps.knobs is None:
        return True
    row, column = rule.caps.knobs
    return _has(rule.caps.cells, look.get(row)) and _has(rule.caps.cells[look[row]], look.get(column))


def look_fits(table: BootTable, look: Any) -> bool:
    """boot.ts ``fits``: the look names an option of every table and fits every numeric rule."""
    if not isinstance(look, dict):
        return False
    knobs = all(_has(options, look.get(knob)) for knob, options in table.knobs.items())
    pairs = all(
        _has(pair.table, look.get(pair.a)) and _has(pair.table[look[pair.a]], look.get(pair.b)) for pair in table.pairs
    )
    return knobs and pairs and all(_rule_fits(rule, look) for rule in table.numeric)


def _painted(table: Any, look: Any) -> Tuple[Optional[BootTable], Mapping[str, Any]]:
    """The table as the boot gets it, and the look it paints: ``look`` when it fits, else the defaults (boot.ts :164)."""
    kept = boot_table(table)
    if kept is None:
        return None, {}
    return kept, (look if look_fits(kept, look) else kept.defaults)


def _phone_tokens(table: BootTable, look: Mapping[str, Any]) -> Dict[str, str]:
    """boot.ts :165-:167 for a look that fits, mobile tokens only (module docstring)."""
    painted: Dict[str, str] = {}
    for knob, options in table.knobs.items():
        painted = {**painted, **options[look[knob]]}
    for pair in table.pairs:
        painted = {**painted, **pair.table[look[pair.a]][look[pair.b]]}
    glows = {rule.var for rule in table.numeric}
    return {
        name: value for name, value in painted.items()
        if name in MOBILE_TOKEN_NAMES and name not in glows and is_studio_triplet(value)
    }


def paint_look(table: Any, look: Any) -> Dict[str, str]:
    """What the web paints for ``look`` through the prepaint ``table``, as ``{mobile token: value}`` (module docstring).

    ``look`` None (or any look that doesn't fit) paints the table's defaults; a table the web drops paints nothing.
    """
    kept, painted_look = _painted(table, look)
    return {} if kept is None else _phone_tokens(kept, painted_look)


def missing_tokens(tokens: Any) -> List[str]:
    """The ``MOBILE_REQUIRED_TOKENS`` that ``tokens`` doesn't give a colour: a phone can't draw it without them."""
    given = _object(tokens)
    return [name for name in MOBILE_REQUIRED_TOKENS if parse_colour(given.get(name)) is None]


def ground_tone(tokens: Any) -> str:
    """look-kit ``isDarkGround``: ``dark`` when the ground is darker than its text, else ``light`` (also with either missing)."""
    given = _object(tokens)
    ground, text = parse_colour(given.get(GROUND_TOKEN)), parse_colour(given.get(TEXT_TOKEN))
    if ground is None or text is None:
        return MOBILE_DEFAULT_TONE
    return TONE_DARK if relative_luminance(ground) < relative_luminance(text) else TONE_LIGHT


def look_tone(table: Any, look: Any, manifest_tone: Any) -> str:
    """The tone ``paint_look(table, look)`` is drawn in, by theme.json's tone policy (module docstring)."""
    if manifest_tone in TONE_RULE_TONES:
        return manifest_tone
    kept, painted_look = _painted(table, look)
    if kept is None:
        return MOBILE_DEFAULT_TONE
    if manifest_tone == TONE_FROM_LOOK and kept.tone is not None:
        return kept.tone.values[painted_look[kept.tone.knob]]
    return ground_tone(_phone_tokens(kept, painted_look))
