"""PRD-261 — the CSS-value grammar for a theme's first-paint values (``boot/prepaint.json``).

The root layout's one constant script applies these values with ``style.setProperty`` before
React runs (A-03), so a value must never carry a URL, a quote, markup or anything that could
end a declaration. A value passes when it is, end to end, a run of:

* numbers with an optional unit from ``UNITS`` (``12px``, ``-.03em``, ``40``, ``50%``);
* ``#`` colours of 3, 4, 6 or 8 hex digits;
* identifiers (``serif``, ``none``, ``italic``, ``--pl-glow``);
* calls to the functions in ``FUNCTIONS``, nested at most ``MAX_FUNCTION_DEPTH`` deep, whose
  first ``var()`` argument is a custom property name;
* separated by spaces, commas or ``/``.

Every other character, ``url(``, ``expression(`` and any unknown function are refused. The
TypeScript twin (``frontend/lib/theme-host/css-value.ts``, A-06b) is held to the same corpus
(``tests/fixtures/themes/cases.json``, ``cssValues``).

PRD-261 A-19c (host 1.1.0): a table may carry a tone rule, ``tone: {knob, values}``
(``contract.TONE_RULE_KEY``). The knob must be a choice knob of the table (a key of ``knobs``,
not a ``numeric`` rule's knob) and ``values`` must give every one of its choices ``light`` or
``dark``, nothing else; otherwise the table is refused with ``BAD_TONE_RULE`` and a plain
sentence, the same one the TypeScript twin gives (the corpus pins a few).

The browser's first paint has its own, smaller gate (``frontend/lib/theme-host/boot-sanitize.ts``
``isCssValue``): only the functions in ``BOOT_VALUE_FUNCTIONS``, so no gradient and no ``color-mix()``,
only the characters in ``BOOT_VALUE_CHARS``, at most ``PREPAINT_VALUE_MAX`` UTF-16 units. A value it
drops vanishes without an error, and so does every value built on it with ``var()``, so the reader
refuses one (:func:`prepaint_drop_errors`, ``BAD_CSS_VALUE``). The gate lives here alone:
``mobile_tokens.py`` paints a look through it, and ``css-value.ts`` is the one TypeScript copy, which
``boot-sanitize.ts`` imports (the corpus's ``prepaintDrops`` cases hold the two to the same verdicts).
The boot read (``serve.theme_prepaint``) keeps to ``prepaint_errors``, so a package imported before
the gate was checked still boots, painting what the browser keeps.
"""

from __future__ import annotations

import re
from typing import Any, Dict, FrozenSet, List, Optional, Tuple

from services.themes.contract import (
    PREPAINT_VALUE_MAX, TONE_RULE_FIELDS, TONE_RULE_KEY, TONE_RULE_TONES, ThemeError, echo, is_finite_number,
    theme_error,
)

PREPAINT_ERROR_CODES = ("BAD_PREPAINT", "BAD_CSS_VALUE", "BAD_TONE_RULE")
TONE_RULE_CODE = "BAD_TONE_RULE"
CSS_VALUE_MAX = PREPAINT_VALUE_MAX
MAX_FUNCTION_DEPTH = 4
CUSTOM_PROPERTY_PATTERN = re.compile(r"--[a-z][a-z0-9-]{1,63}")
# What a prepaint table may name besides custom properties: knob keys, table names, templates.
PREPAINT_KEY_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,64}")
PREPAINT_MAX_ERRORS = 20
PREPAINT_PATH = "boot/prepaint.json"
TONE_RULE_PATH = f"{PREPAINT_PATH}#{TONE_RULE_KEY}"

FUNCTIONS = frozenset({
    "rgb", "rgba", "hsl", "hsla", "var", "color-mix", "linear-gradient", "radial-gradient",
    "hue-rotate", "saturate", "brightness", "grayscale", "contrast", "invert", "sepia", "opacity",
    "blur", "drop-shadow",
})
UNITS = frozenset({
    "", "%", "px", "rem", "em", "ch", "ex", "vh", "vw", "vmin", "vmax", "dvh", "svh", "lvh",
    "deg", "rad", "turn", "s", "ms", "fr",
})
HEX_LENGTHS = frozenset({3, 4, 6, 8})

# One token of a value; anything the pattern can't match is refused.
TOKEN = re.compile(
    r"(?P<space>[ ]+)"
    r"|(?P<function>-{0,2}[A-Za-z_][A-Za-z0-9_-]*)\("
    r"|(?P<number>[+-]?(?:[0-9]+(?:\.[0-9]+)?|\.[0-9]+))(?P<unit>%|[a-z]+)?"
    r"|(?P<hash>#[0-9A-Fa-f]+)"
    r"|(?P<ident>-{0,2}[A-Za-z_][A-Za-z0-9_-]*)"
    r"|(?P<close>\))"
    r"|(?P<separator>[,/])"
)


def _token_problem(match: "re.Match[str]") -> Optional[str]:
    """Why one matched token is refused, or None."""
    if match.group("unit") is not None and match.group("unit") not in UNITS:
        return f'the unit "{match.group("unit")}" is not one a theme may use'
    hex_run = match.group("hash")
    if hex_run is not None and len(hex_run) - 1 not in HEX_LENGTHS:
        return "a # colour has 3, 4, 6 or 8 hex digits"
    function = match.group("function")
    if function is not None and function.lower() not in FUNCTIONS:
        return f'"{function}()" is not a function a theme may use'
    return None


def _var_problem(value: str, at: int) -> Optional[str]:
    """var() must start with a custom property name."""
    name = re.match(r"[ ]*(--[A-Za-z0-9_-]*)", value[at:])
    if name is None or CUSTOM_PROPERTY_PATTERN.fullmatch(name.group(1)) is None:
        return "var() must name a custom property such as --theme-accent"
    return None


def _depth_after(match: "re.Match[str]", value: str, depth: int) -> Tuple[int, Optional[str]]:
    """The nesting depth after this token, or why it is refused."""
    if match.group("function") is not None:
        if depth >= MAX_FUNCTION_DEPTH:
            return depth, f"functions nest at most {MAX_FUNCTION_DEPTH} deep"
        is_var = match.group("function").lower() == "var"
        return depth + 1, _var_problem(value, match.end()) if is_var else None
    if match.group("close") is not None:
        return depth - 1, None if depth > 0 else "a ) closes nothing"
    return depth, None


def css_value_problem(value: Any) -> Optional[str]:
    """Why ``value`` is not a CSS value a theme may set before first paint, or None."""
    if not isinstance(value, str) or not value:
        return "a CSS value must be non-empty text"
    if len(value) > CSS_VALUE_MAX:
        return f"a CSS value must be at most {CSS_VALUE_MAX} characters"
    at, depth = 0, 0
    while at < len(value):
        match = TOKEN.match(value, at)
        if match is None:
            return f'"{echo(value[at])}" can\'t appear in a CSS value here'
        problem = _token_problem(match)
        if problem is None:
            depth, problem = _depth_after(match, value, depth)
        if problem is not None:
            return problem
        at = match.end()
    return None if depth == 0 else "a ( is never closed"


def is_css_value(value: Any) -> bool:
    """True when ``value`` passes the grammar above."""
    return css_value_problem(value) is None


def _leaf_error(path: str, key: str, value: Any) -> Optional[ThemeError]:
    """The error for one key and its value, or None."""
    where = f"{PREPAINT_PATH}#{echo(path)}"
    if key.startswith("--"):
        problem = None if CUSTOM_PROPERTY_PATTERN.fullmatch(key) else f'"{echo(key)}" is not a custom property name'
        problem = problem or css_value_problem(value)
        return theme_error("BAD_CSS_VALUE", where, f"{where}: {problem}.") if problem else None
    if PREPAINT_KEY_PATTERN.fullmatch(key) is None:
        return theme_error("BAD_PREPAINT", where, f'{where}: "{echo(key)}" is not a name a prepaint table may use.')
    return _scalar_error(where, value)


def _scalar_error(where: str, value: Any) -> Optional[ThemeError]:
    """A non-variable leaf: a finite number, a bool, null, or text that passes the grammar."""
    if isinstance(value, (dict, list)) or value is None or isinstance(value, bool) or is_finite_number(value):
        return None
    problem = css_value_problem(value) if isinstance(value, str) else "only text, numbers, true, false and null"
    return theme_error("BAD_CSS_VALUE", where, f"{where}: {problem}.") if problem else None


def _children(node: Any, path: str) -> List[Tuple[str, Optional[str], Any]]:
    """``(path, key, value)`` for each child of an object, or ``(path, None, value)`` of a list."""
    if isinstance(node, dict):
        return [(f"{path}.{key}" if path else key, key, child) for key, child in node.items()]
    if isinstance(node, list):
        return [(f"{path}[{index}]", None, child) for index, child in enumerate(node)]
    return []


def _walk_errors(document: Any) -> List[ThemeError]:
    """Every refused name or value the walk of the table finds, at most ``PREPAINT_MAX_ERRORS``."""
    errors: List[ThemeError] = []
    stack = list(reversed(_children(document, "")))
    while stack and len(errors) < PREPAINT_MAX_ERRORS:
        path, key, value = stack.pop()
        error = _scalar_error(f"{PREPAINT_PATH}#{echo(path)}", value) if key is None else _leaf_error(path, key, value)
        if error is not None:
            errors.append(error)
        stack.extend(reversed(_children(value, path)))
    return errors


def prepaint_errors(document: Any) -> List[ThemeError]:
    """Every refused name or value in a prepaint table, and its tone rule's, at most ``PREPAINT_MAX_ERRORS``.

    The table's layout is the boot contract's (A-03); this check holds only what reaches
    ``style.setProperty``: each ``--name`` key is a custom property and its value passes the
    grammar, and every other text in the table passes it too. A tone rule (A-19c) is checked
    against the table's own knobs (``tone_rule_errors``).
    """
    if not isinstance(document, dict):
        return [theme_error("BAD_PREPAINT", PREPAINT_PATH, f"{PREPAINT_PATH} must be a JSON object.")]
    return (_walk_errors(document) + tone_rule_errors(document))[:PREPAINT_MAX_ERRORS]


# ---------------------------------------------------------------------------
# The tone rule (A-19c, host 1.1.0)
# ---------------------------------------------------------------------------


def _tone_error(field: str, problem: str) -> ThemeError:
    """One refusal of the tone rule, at ``tone<field>``."""
    where = f"{TONE_RULE_PATH}{field}"
    return theme_error(TONE_RULE_CODE, where, f"{where}: {problem}.")


def _choice_knobs(document: Dict[str, Any]) -> Dict[str, List[str]]:
    """``{knob: its choices}`` for every knob of the table whose options are an object."""
    knobs = document.get("knobs")
    if not isinstance(knobs, dict):
        return {}
    return {name: list(options) for name, options in knobs.items() if isinstance(options, dict)}


def _number_knobs(document: Dict[str, Any]) -> FrozenSet[str]:
    """The knob of every numeric rule of the table."""
    rules = document.get("numeric")
    if not isinstance(rules, list):
        return frozenset()
    return frozenset(rule["knob"] for rule in rules if isinstance(rule, dict) and isinstance(rule.get("knob"), str))


def _tone_knob_problem(document: Dict[str, Any], knob: Any) -> Optional[str]:
    """Why ``knob`` can't carry the tone rule, or None."""
    if not isinstance(knob, str):
        return '"knob" must name a choice knob of this table'
    if knob in _number_knobs(document):
        return f'"{echo(knob)}" is a number knob; the tone rule needs a choice knob'
    if knob not in _choice_knobs(document):
        return f'"{echo(knob)}" is not a knob of this table'
    return None


def _tone_value_errors(knob: str, choices: List[str], values: Dict[str, Any]) -> List[ThemeError]:
    """Every choice of ``knob`` mapped to light or dark, and nothing else mapped."""
    errors = [
        _tone_error(f".values.{echo(choice)}", f'the choice "{echo(choice)}" of "{echo(knob)}" has no tone; map it to "light" or "dark"')
        for choice in choices if choice not in values
    ]
    for choice, tone in values.items():
        if choice not in choices:
            errors.append(_tone_error(f".values.{echo(choice)}", f'"{echo(choice)}" is not a choice of "{echo(knob)}"'))
        elif tone not in TONE_RULE_TONES:
            errors.append(_tone_error(f".values.{echo(choice)}", 'a tone is "light" or "dark"'))
    return errors


def tone_rule_errors(document: Any) -> List[ThemeError]:
    """Why the table's tone rule (``tone``) is refused; empty when it has none or it is valid.

    The rule names one choice knob of the table and gives each of its choices a tone, so a look
    runtime can set the tone from the look alone (A-19d2) and every look has one.
    """
    if not isinstance(document, dict) or TONE_RULE_KEY not in document:
        return []
    rule = document[TONE_RULE_KEY]
    if not isinstance(rule, dict):
        return [_tone_error("", 'the tone rule must be an object with "knob" and "values"')]
    errors = [
        _tone_error(f".{echo(key)}", f'"{echo(key)}" isn\'t part of the tone rule. Remove it')
        for key in rule if key not in TONE_RULE_FIELDS
    ]
    knob, values = rule.get("knob"), rule.get("values")
    problem = _tone_knob_problem(document, knob)
    if problem is not None:
        return errors + [_tone_error(".knob", problem)]
    if not isinstance(values, dict):
        return errors + [_tone_error(".values", f'"values" must map every choice of "{echo(knob)}" to "light" or "dark"')]
    return errors + _tone_value_errors(knob, _choice_knobs(document)[knob], values)


# ---------------------------------------------------------------------------
# The browser's first-paint gate (boot-sanitize.ts isCssValue) and what it drops
# ---------------------------------------------------------------------------

# boot-sanitize.ts's gate, smaller than the grammar above: these characters, balanced parentheses and
# these functions only. css-value.ts holds the same three as BOOT_VALUE_CHARS, BOOT_FUNCTION_NAME and
# BOOT_VALUE_FUNCTIONS (validator-parity.test.ts holds the functions equal, in this order, and the corpus's
# prepaintDrops cases the verdicts). A JavaScript string's length counts UTF-16 units.
BOOT_VALUE_CHARS = re.compile(r"[A-Za-z0-9#%.,()/ +-]+")
BOOT_FUNCTION_NAME = re.compile(r"([A-Za-z-]+)\(")
BOOT_VALUE_FUNCTIONS: Tuple[str, ...] = (
    "rgb", "rgba", "hsl", "hsla", "var", "calc", "min", "max", "clamp",
    "hue-rotate", "saturate", "brightness", "grayscale", "contrast", "sepia", "invert",
)
# boot-sanitize.ts KNOB_NAME: a look field's name, which the gate takes a knob table, a pair's knobs and a
# numeric rule's knobs by (css-value.ts BOOT_KNOB_NAME; mobile_tokens.py keeps the table by it).
KNOB_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,31}")
KNOB_NAME_RULE = "a knob's name here is a letter, then up to 31 letters, digits or _"
GRADIENT_SUFFIX = "-gradient"
UTF16_UNIT_BYTES = 2
DROPPED = "the browser's first paint drops it, and every value built on it"


def utf16_length(text: str) -> int:
    """``text``'s length as JavaScript counts it, in UTF-16 units."""
    return len(text.encode("utf-16-le", "surrogatepass")) // UTF16_UNIT_BYTES


def _function_problem(value: str) -> Optional[str]:
    """The first function the gate doesn't run, said as what it is; None when it runs every one."""
    for name in BOOT_FUNCTION_NAME.findall(value):
        if name.lower() not in BOOT_VALUE_FUNCTIONS:
            if name.lower().endswith(GRADIENT_SUFFIX):
                return f"a gradient, {echo(name)}()"
            return f"a call to {echo(name)}(), a function the first paint never runs"
    return None


def boot_value_problem(value: Any) -> Optional[str]:
    """What ``value`` is that boot-sanitize.ts ``isCssValue`` drops (its checks in its order), or None when it keeps it."""
    if not isinstance(value, str) or not value:
        return "not text"
    if utf16_length(value) > PREPAINT_VALUE_MAX:
        return f"over {PREPAINT_VALUE_MAX} characters"
    refused = next((char for char in value if BOOT_VALUE_CHARS.fullmatch(char) is None), None)
    if refused is not None:
        return f'text with "{echo(refused)}", a character the first paint refuses'
    if value.count("(") != value.count(")"):
        return "text whose brackets don't pair"
    return _function_problem(value)


def is_boot_value(value: Any) -> bool:
    """True when the browser's first paint keeps ``value`` (boot-sanitize.ts ``isCssValue``)."""
    return boot_value_problem(value) is None


# One cell of the table: its dotted path, the look it paints for and its variables.
Cell = Tuple[str, str, Any]
# What the gate drops with a knob name it refuses.
DROPS_TABLE = "the knob's table and everything it paints"
DROPS_PAIR = "the pair table and everything it paints"
DROPS_RULE = "the rule and what it paints"


def is_knob_name(value: Any) -> bool:
    """A name boot-sanitize.ts takes for a knob (``KNOB_NAME``): a letter, then up to 31 letters, digits or ``_``."""
    return isinstance(value, str) and KNOB_NAME.fullmatch(value) is not None


def _name_error(path: str, drops: str) -> ThemeError:
    """A knob's name the gate refuses at ``path``, and what it drops with it."""
    where = f"{PREPAINT_PATH}#{echo(path)}"
    return theme_error("BAD_PREPAINT", where, f"{where}: {KNOB_NAME_RULE}, so the browser's first paint drops {drops}.")


def _cell_drops(cell: Cell) -> List[ThemeError]:
    """A refusal for each variable of one cell whose value the gate drops."""
    path, look, variables = cell
    named = variables.items() if isinstance(variables, dict) else ()
    errors: List[ThemeError] = []
    for name, value in named:
        problem = boot_value_problem(value) if name.startswith("--") else None
        if problem is not None:
            where = f"{PREPAINT_PATH}#{echo(f'{path}.{name}')}"
            errors.append(theme_error("BAD_CSS_VALUE", where, f"{where}: {echo(name)} for {look} is {problem}: {DROPPED}."))
    return errors


def _knob_errors(knob: str, options: Any) -> List[ThemeError]:
    """One table under ``knobs``: its name when the gate refuses it, else each of its cells' dropped values."""
    if not is_knob_name(knob):
        return [_name_error(f"knobs.{knob}", DROPS_TABLE)]
    cells = options.items() if isinstance(options, dict) else ()
    return [error for option, variables in cells for error in _cell_drops((f"knobs.{knob}.{option}", f'{echo(knob)} "{echo(option)}"', variables))]


def _pair_errors(index: int, pair: Any) -> List[ThemeError]:
    """One pair table: each of its two knob names the gate refuses, else each of its cells' dropped values."""
    if not isinstance(pair, dict):
        return []
    names = [_name_error(f"pairs[{index}].{side}", DROPS_PAIR) for side in ("a", "b") if not is_knob_name(pair.get(side))]
    if names:
        return names
    table = pair.get("table")
    rows = table.items() if isinstance(table, dict) else ()
    cells = [
        (f"pairs[{index}].table.{row}.{column}", f'{echo(pair["a"])} "{echo(row)}" and {echo(pair["b"])} "{echo(column)}"', variables)
        for row, columns in rows if isinstance(columns, dict)
        for column, variables in columns.items()
    ]
    return [error for cell in cells for error in _cell_drops(cell)]


def _rule_errors(index: int, rule: Any) -> List[ThemeError]:
    """One numeric rule: each knob it names (its own, its colour's, its cap table's) that the gate refuses."""
    if not isinstance(rule, dict):
        return []
    color, caps = rule.get("color"), rule.get("capTable")
    named = [("knob", rule.get("knob"))]
    named += [("color.knob", color.get("knob"))] if isinstance(color, dict) else []
    named += [(f"capTable[{at}]", name) for at, name in enumerate(caps)] if isinstance(caps, list) else []
    return [_name_error(f"numeric[{index}].{field}", DROPS_RULE) for field, name in named if not is_knob_name(name)]


def _listed(document: Dict[str, Any], key: str) -> List[Any]:
    value = document.get(key)
    return value if isinstance(value, list) else []


def prepaint_drop_errors(document: Any) -> List[ThemeError]:
    """What the browser's first paint drops of the table, at most ``PREPAINT_MAX_ERRORS``.

    Its gate (boot-sanitize.ts ``sanitizePrepaint``) drops a table under ``knobs``, a pair table or a
    numeric rule whose knob name it refuses (``BAD_PREPAINT``), and each value of a kept knob or pair
    cell it refuses (``BAD_CSS_VALUE``), in the table's order. What ``prepaint_errors`` refuses is its
    alone; the reader runs this on a table that passed it.
    """
    if not isinstance(document, dict):
        return []
    knobs = document.get("knobs")
    tables = knobs.items() if isinstance(knobs, dict) else ()
    errors = [error for knob, options in tables for error in _knob_errors(knob, options)]
    errors += [error for index, pair in enumerate(_listed(document, "pairs")) for error in _pair_errors(index, pair)]
    errors += [error for index, rule in enumerate(_listed(document, "numeric")) for error in _rule_errors(index, rule)]
    return errors[:PREPAINT_MAX_ERRORS]
