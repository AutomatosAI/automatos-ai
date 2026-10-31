"""PRD-261 M-01 (host 1.1.0, plan §2.2.11) — theme.json's optional ``mobile`` block, checked.

A phone draws a theme from data alone: it never loads ``dist/entry.js`` or ``dist/theme.css``. The
block holds exactly one source of colours and, optionally, fonts and a mascot mode::

    "mobile": {
      "tokens": {"light": {"--background": "0 0% 100%", "--foreground": "#14181c"}, "dark": {...}},
      // or, instead of "tokens":
      "fromLooks": true,
      "fonts": {"sans": "assets/fonts/mobile/inter-regular.ttf", "serif": "...", "mono": "..."},
      "mascot": "clips"
    }

* exactly one of ``tokens`` and ``fromLooks``, and ``fromLooks`` is only ever ``true`` (the member's
  active look is projected through ``boot/prepaint.json``; ``services/themes/mobile_tokens.py``);
* ``tokens.light`` is required and ``tokens.dark`` optional; each names only ``MOBILE_TOKEN_NAMES``,
  includes ``MOBILE_REQUIRED_TOKENS``, and every value parses (``mobile_tokens.parse_colour``);
* ``fonts`` gives ``MOBILE_FONT_ROLES`` only, each a different ``assets/fonts/mobile/<NAME>.ttf`` or
  ``.otf`` (``manifest.referenced_files`` lists them, so a named file the package lacks is
  ``MISSING_FILE``);
* ``mascot`` is one of ``MOBILE_MASCOT_MODES`` (absent means ``none``); ``clips`` and ``stills`` need
  the top-level ``mascot`` with its ``states``;
* ``engines.automatos`` admits no host below ``MOBILE_MIN_HOST``: an older host refuses the field;
* PRD-271 (host 1.2.0): ``layout`` and ``slots``, one of the phone's layouts and its contract's slots, are
  ``mobile_layout.layout_errors``'s, and a block with ``layout`` admits no host below ``LAYOUT_MIN_HOST``.

Each refusal is ``BAD_MOBILE`` naming the field, or ``UNKNOWN_FIELD`` for a key outside the block.
The TypeScript twin is ``frontend/lib/theme-host/mobile-manifest.ts``; the shared corpus
(``tests/fixtures/themes/cases.json``, ``manifests``) holds both to the same verdicts.
"""

from __future__ import annotations

import re
from typing import Any, List, Mapping, Tuple

from services.themes.contract import (
    LAYOUT_MIN_HOST, MOBILE_FONT_ROLES, MOBILE_MASCOT_MODES, MOBILE_MIN_HOST, MOBILE_REQUIRED_TOKENS, MOBILE_TOKEN_NAMES,
    ThemeError, echo, theme_error,
)
from services.themes.layout_paint import LAYOUT_FIELD, LAYOUT_FIELDS
from services.themes.mobile_layout import MOBILE_CODE, layout_errors
from services.themes.mobile_tokens import parse_colour

MOBILE_FIELD = "mobile"
MOBILE_FIELDS: Tuple[str, ...] = ("tokens", "fromLooks", "fonts", "mascot", *LAYOUT_FIELDS)
MOBILE_SCHEMES: Tuple[str, ...] = ("light", "dark")
MOBILE_REQUIRED_SCHEME = "light"
MASCOT_NONE = "none"
# A package file name, as manifest.py ASSET_NAME and package.py NAME spell it.
FONT_NAME = r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}"
MOBILE_FONT_PATTERN = re.compile(rf"assets/fonts/mobile/{FONT_NAME}\.(?:ttf|otf)")
# An engines range (manifest.py RANGE_PATTERN): the lowest host it admits is its own version.
HOST_RANGE_PATTERN = re.compile(r"\^?([0-9]{1,6})\.([0-9]{1,6})\.([0-9]{1,6})")


def _bad(path: str, message: str) -> List[ThemeError]:
    return [theme_error(MOBILE_CODE, path, message)]


def _unknown_errors(block: Mapping[str, Any]) -> List[ThemeError]:
    """UNKNOWN_FIELD for every key outside the block's shape (manifest.py's sentence)."""
    return [
        theme_error("UNKNOWN_FIELD", f"{MOBILE_FIELD}.{echo(key)}", f'"{echo(key)}" isn\'t part of {MOBILE_FIELD}. Remove it.')
        for key in block if key not in MOBILE_FIELDS
    ]


def _scheme_errors(scheme: str, values: Any) -> List[ThemeError]:
    where = f"mobile.tokens.{scheme}"
    if not isinstance(values, dict):
        return _bad(where, f'"{where}" must be an object of Studio token names and their colours.')
    errors: List[ThemeError] = []
    for name, value in values.items():
        if name not in MOBILE_TOKEN_NAMES:
            errors += _bad(f"{where}.{echo(name)}", (
                f'"{echo(name)}" is not one of the {len(MOBILE_TOKEN_NAMES)} Studio tokens a phone draws.'
            ))
        elif parse_colour(value) is None:
            errors += _bad(f"{where}.{name}", f'"{where}.{name}" must be "H S% L%" (such as "0 0% 100%") or "#rrggbb".')
    return errors + [
        theme_error(MOBILE_CODE, where, f'"{where}" needs "{name}".') for name in MOBILE_REQUIRED_TOKENS if name not in values
    ]


def _tokens_errors(tokens: Any) -> List[ThemeError]:
    if not isinstance(tokens, dict):
        return _bad("mobile.tokens", '"mobile.tokens" must be {"light": {...}, "dark": {...}}, with "dark" optional.')
    errors = [
        theme_error(MOBILE_CODE, f"mobile.tokens.{echo(key)}", f'"{echo(key)}" is not a scheme: "mobile.tokens" holds "light" and "dark".')
        for key in tokens if key not in MOBILE_SCHEMES
    ]
    if MOBILE_REQUIRED_SCHEME not in tokens:
        errors += _bad(f"mobile.tokens.{MOBILE_REQUIRED_SCHEME}", f'"mobile.tokens" needs "{MOBILE_REQUIRED_SCHEME}"; "dark" is optional.')
    return errors + [error for scheme in MOBILE_SCHEMES if scheme in tokens for error in _scheme_errors(scheme, tokens[scheme])]


def _source_errors(block: Mapping[str, Any]) -> List[ThemeError]:
    """Exactly one of ``tokens`` and ``fromLooks``, and that one well formed."""
    has_tokens, has_looks = "tokens" in block, "fromLooks" in block
    if has_tokens == has_looks:
        return _bad(MOBILE_FIELD, '"mobile" needs exactly one of "tokens" and "fromLooks".')
    if has_looks:
        return [] if block["fromLooks"] is True else _bad("mobile.fromLooks", (
            '"mobile.fromLooks" is only ever true: leave it out and give "tokens" instead.'
        ))
    return _tokens_errors(block["tokens"])


def _role_errors(role: str, path: Any) -> List[ThemeError]:
    where = f"mobile.fonts.{echo(role)}"
    if role not in MOBILE_FONT_ROLES:
        return _bad(where, f'"{echo(role)}" is not a font role: "sans", "serif" or "mono".')
    if not isinstance(path, str) or MOBILE_FONT_PATTERN.fullmatch(path) is None:
        return _bad(where, f'"{where}" must be a .ttf or .otf file in assets/fonts/mobile/.')
    return []


def _font_errors(block: Mapping[str, Any]) -> List[ThemeError]:
    if "fonts" not in block:
        return []
    fonts = block["fonts"]
    if not isinstance(fonts, dict):
        return _bad("mobile.fonts", '"mobile.fonts" must be {"sans", "serif", "mono"}, each a .ttf or .otf in assets/fonts/mobile/.')
    errors = [error for role, path in fonts.items() for error in _role_errors(role, path)]
    paths = [path for path in fonts.values() if isinstance(path, str)]
    if len(set(paths)) != len(paths):
        errors += _bad("mobile.fonts", 'Each role in "mobile.fonts" needs its own file.')
    return errors


def _has_no_states(mascot: Any) -> bool:
    """True when the top-level ``mascot`` is absent, null or ``{"states": []}``."""
    if mascot is None:
        return True
    states = mascot.get("states") if isinstance(mascot, dict) else None
    return isinstance(states, list) and not states


def _mascot_errors(block: Mapping[str, Any], mascot: Any) -> List[ThemeError]:
    """The mode is known; clips and stills need the top-level ``mascot`` (a malformed one is BAD_MASCOT's)."""
    if "mascot" not in block:
        return []
    mode = block["mascot"]
    if not isinstance(mode, str) or mode not in MOBILE_MASCOT_MODES:
        return _bad("mobile.mascot", '"mobile.mascot" must be "clips", "stills" or "none".')
    if mode == MASCOT_NONE or not _has_no_states(mascot):
        return []
    return _bad("mobile.mascot", f'"mobile.mascot" "{mode}" needs the top-level "mascot" with its "states".')


def _engines_errors(engines: Any, block: Mapping[str, Any]) -> List[ThemeError]:
    """The block is host 1.1.0's, and 1.2.0's with ``layout``: a range that admits an older host is refused
    (BAD_ENGINES covers a malformed one)."""
    minimum, needs = (LAYOUT_MIN_HOST, '"mobile.layout"') if LAYOUT_FIELD in block else (MOBILE_MIN_HOST, '"mobile"')
    host = engines.get("automatos") if isinstance(engines, dict) else None
    match = HOST_RANGE_PATTERN.fullmatch(host) if isinstance(host, str) else None
    if match is None or tuple(int(part) for part in match.groups()) >= tuple(int(part) for part in minimum.split(".")):
        return []
    return _bad("engines.automatos", (
        f'A theme with {needs} needs host {minimum} or later: "engines.automatos" must not admit an earlier one '
        f'(use ^{minimum}).'
    ))


def mobile_errors(document: Mapping[str, Any]) -> List[ThemeError]:
    """Every reason theme.json's ``mobile`` block is refused; empty when it is valid."""
    block = document.get(MOBILE_FIELD)
    if not isinstance(block, dict):
        return _bad(MOBILE_FIELD, '"mobile" must be an object with "tokens" or "fromLooks".')
    return (
        _unknown_errors(block) + _source_errors(block) + _font_errors(block) + _mascot_errors(block, document.get("mascot"))
        + layout_errors(block) + _engines_errors(document.get("engines"), block)
    )


def mobile_font_files(manifest: Mapping[str, Any]) -> List[str]:
    """The font files a checked manifest's ``mobile`` block names, in role order as written."""
    block = manifest.get(MOBILE_FIELD)
    fonts = block.get("fonts") if isinstance(block, dict) else None
    return [path for path in fonts.values() if isinstance(path, str)] if isinstance(fonts, dict) else []
