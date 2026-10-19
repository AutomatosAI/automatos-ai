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
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

from services.themes.contract import ThemeError, echo, is_finite_number, theme_error

PREPAINT_ERROR_CODES = ("BAD_PREPAINT", "BAD_CSS_VALUE")
CSS_VALUE_MAX = 200
MAX_FUNCTION_DEPTH = 4
CUSTOM_PROPERTY_PATTERN = re.compile(r"--[a-z][a-z0-9-]{1,63}")
# What a prepaint table may name besides custom properties: knob keys, table names, templates.
PREPAINT_KEY_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,64}")
PREPAINT_MAX_ERRORS = 20
PREPAINT_PATH = "boot/prepaint.json"

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
        return "var() must name a custom property such as --pl-accent"
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


def prepaint_errors(document: Any) -> List[ThemeError]:
    """Every refused name or value in a prepaint table, at most ``PREPAINT_MAX_ERRORS``.

    The table's layout is the boot contract's (A-03); this check holds only what reaches
    ``style.setProperty``: each ``--name`` key is a custom property and its value passes the
    grammar, and every other text in the table passes it too.
    """
    if not isinstance(document, dict):
        return [theme_error("BAD_PREPAINT", PREPAINT_PATH, f"{PREPAINT_PATH} must be a JSON object.")]
    errors: List[ThemeError] = []
    stack = list(reversed(_children(document, "")))
    while stack and len(errors) < PREPAINT_MAX_ERRORS:
        path, key, value = stack.pop()
        error = _scalar_error(f"{PREPAINT_PATH}#{echo(path)}", value) if key is None else _leaf_error(path, key, value)
        if error is not None:
            errors.append(error)
        stack.extend(reversed(_children(value, path)))
    return errors
