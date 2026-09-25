"""Social templates as data (PRD-251 S1.2, D4).

A ``document_templates`` row whose format is ``social_image`` or ``social_video``
carries a composition in ``blocks``, which media-render renders:

    {
      "html": "<!doctype html>…",           a full Hyperframes composition
      "css": "…",                            optional, injected after the brand styles
      "variables_schema": {                  every {{ name }} the html and css use
        "headline": {"type": "text", "label": "Headline", "max_chars": 60},
        "stat": {"type": "number", "claim": true},
        "subtitle": {"type": "text", "default": ""}
      },
      "sizes": ["1080x1920", "1080x1350"],   WIDTHxHEIGHT, the first is the default
      "audio_plan": {"voice": …, "music": …, "sfx": […]}   social_video only
    }

* **Variables.** A variable is ``text``, a ``number`` or a ``boolean``, and it
  fills in text, never markup: media-render escapes every value (D4, every word
  on screen is template text). A variable with a ``default`` is optional; one
  without must be supplied. ``claim: true`` marks a fact that needs a source (D7).
  The render bundle also sets ``brand.*`` and ``size.*`` for every template
  (``core/media_render_bundle.py``), so the html may use them undeclared.
* **The brand comes from the brand kit (D4).** Colours, fonts and the logo reach
  a composition as ``--brand-*`` CSS variables and ``{{ brand.logo }}``.
  :func:`brand_literals` finds a hex colour, a named font family or a logo baked
  into the template outside a ``var()`` fallback, and a template carrying one is
  refused on save.

Pure: no database, no IO. Used by the template service (validation on save),
the documents generator and a Socials post's render (variables and sizes).
"""
from __future__ import annotations

import copy
import math
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from core.models.core import SOCIAL_TEMPLATE_FORMATS

SOCIAL_IMAGE, SOCIAL_VIDEO = SOCIAL_TEMPLATE_FORMATS

BLOCK_KEYS = ("html", "css", "variables_schema", "sizes", "audio_plan")
REQUIRED_BLOCK_KEYS = ("html", "variables_schema", "sizes")
AUDIO_PLAN_KEYS = ("voice", "music", "sfx")

TEXT, NUMBER, BOOLEAN = "text", "number", "boolean"
VARIABLE_TYPES = (TEXT, NUMBER, BOOLEAN)
COMMON_SPEC_KEYS = ("type", "label", "description", "default", "claim")
SPEC_KEYS_BY_TYPE = {
    TEXT: COMMON_SPEC_KEYS + ("max_chars",),
    NUMBER: COMMON_SPEC_KEYS + ("min", "max"),
    BOOLEAN: COMMON_SPEC_KEYS,
}
# A template's own variable names are flat; the bundle's are dotted (brand.*, size.*).
VARIABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,63}$")
BUNDLE_VARIABLE_PREFIXES = ("brand.", "size.")

# Bounds. The media-render service enforces its own on every bundle; these keep
# a stored template far inside them.
MAX_COMPOSITION_CHARS = 1_000_000
MAX_VARIABLES = 100
MAX_TEXT_CHARS = 2000
MAX_LABEL_CHARS = 200
MAX_SIZES = 8
MIN_DIMENSION, MAX_DIMENSION = 16, 4096
MAX_REPORTED_ERRORS = 50

PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*)\s*\}\}")
_LEFTOVER = re.compile(r"\{\{[^{}]*\}\}")
_SIZE = re.compile(r"^(\d{2,4})x(\d{2,4})$")
_HEAD_CLOSE = re.compile(r"</head\s*>", re.IGNORECASE)
_ROOT = re.compile(r"""data-composition-id\s*=\s*["']main["']""", re.IGNORECASE)


class SocialTemplateError(ValueError):
    """A social template that breaks the contract; ``errors`` names each problem."""

    def __init__(self, errors: Sequence[Mapping[str, str]]) -> None:
        self.errors = [dict(e) for e in errors][:MAX_REPORTED_ERRORS]
        super().__init__("; ".join(f"{e['field']}: {e['message']}" for e in self.errors))


class InvalidVariableValues(SocialTemplateError):
    """Values supplied for a render that do not fit their variables (the input, not the template)."""


def is_social_format(fmt: Optional[str]) -> bool:
    return fmt in SOCIAL_TEMPLATE_FORMATS


def _error(field: str, message: str) -> Dict[str, str]:
    return {"field": field, "message": message}


# ── sizes ───────────────────────────────────────────────────────────────────
def parse_size(value: Any) -> Tuple[int, int]:
    """``"1080x1920"`` → ``(1080, 1920)``; ValueError when it is not a size a render can take."""
    match = _SIZE.match(value) if isinstance(value, str) else None
    if match is None:
        raise ValueError(f"{value!r} is not WIDTHxHEIGHT, e.g. 1080x1920")
    width, height = int(match.group(1)), int(match.group(2))
    if not (MIN_DIMENSION <= width <= MAX_DIMENSION and MIN_DIMENSION <= height <= MAX_DIMENSION):
        raise ValueError(f"{value!r}: each side must be {MIN_DIMENSION}-{MAX_DIMENSION} px")
    return width, height


def _size_errors(sizes: Any) -> List[Dict[str, str]]:
    if not isinstance(sizes, list) or not sizes:
        return [_error("sizes", "must be a non-empty list of WIDTHxHEIGHT sizes, e.g. [\"1080x1920\"]")]
    if len(sizes) > MAX_SIZES:
        return [_error("sizes", f"at most {MAX_SIZES} sizes")]
    errors = []
    for i, size in enumerate(sizes):
        try:
            parse_size(size)
        except ValueError as exc:
            errors.append(_error(f"sizes[{i}]", str(exc)))
    if len(set(map(str, sizes))) != len(sizes):
        errors.append(_error("sizes", "lists a size twice"))
    return errors


# ── variables ───────────────────────────────────────────────────────────────
def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _value_problem(spec: Mapping[str, Any], value: Any) -> Optional[str]:
    """Why ``value`` does not fit the variable, or ``None``."""
    kind = spec.get("type")
    if kind == TEXT:
        if not isinstance(value, str):
            return "must be text"
        limit = spec.get("max_chars", MAX_TEXT_CHARS)
        return f"is longer than {limit} characters" if len(value) > limit else None
    if kind == NUMBER:
        if not _is_number(value):
            return "must be a number"
        if spec.get("min") is not None and value < spec["min"]:
            return f"must be at least {spec['min']:g}"
        if spec.get("max") is not None and value > spec["max"]:
            return f"must be at most {spec['max']:g}"
        return None
    return None if isinstance(value, bool) else "must be true or false"


def _spec_errors(name: str, spec: Any) -> List[Dict[str, str]]:
    where = f"variables_schema.{name}"
    if not VARIABLE_NAME.match(name):
        return [_error(where, "a variable name is letters, digits and _ (64 at most), not starting with a digit")]
    if not isinstance(spec, dict):
        return [_error(where, 'must be an object such as {"type": "text"}')]
    kind = spec.get("type")
    if kind not in VARIABLE_TYPES:
        return [_error(f"{where}.type", f"must be one of {list(VARIABLE_TYPES)}")]
    errors = [
        _error(f"{where}.{key}", f"is not a {kind} variable setting ({', '.join(SPEC_KEYS_BY_TYPE[kind])})")
        for key in spec
        if key not in SPEC_KEYS_BY_TYPE[kind]
    ]
    for key in ("label", "description"):
        if key in spec and (not isinstance(spec[key], str) or len(spec[key]) > MAX_LABEL_CHARS):
            errors.append(_error(f"{where}.{key}", f"must be text of at most {MAX_LABEL_CHARS} characters"))
    if "claim" in spec and not isinstance(spec["claim"], bool):
        errors.append(_error(f"{where}.claim", "must be true or false"))
    if "max_chars" in spec:
        limit = spec["max_chars"]
        if not isinstance(limit, int) or isinstance(limit, bool) or not 0 < limit <= MAX_TEXT_CHARS:
            errors.append(_error(f"{where}.max_chars", f"must be a whole number from 1 to {MAX_TEXT_CHARS}"))
    for key in ("min", "max"):
        if key in spec and not _is_number(spec[key]):
            errors.append(_error(f"{where}.{key}", "must be a number"))
    if not errors and _is_number(spec.get("min")) and _is_number(spec.get("max")) and spec["min"] > spec["max"]:
        errors.append(_error(where, "min is greater than max"))
    if not errors and "default" in spec:
        problem = _value_problem(spec, spec["default"])
        if problem:
            errors.append(_error(f"{where}.default", problem))
    return errors


def _schema_errors(schema: Any) -> List[Dict[str, str]]:
    if not isinstance(schema, dict):
        return [_error("variables_schema", 'must be an object of variables, e.g. {"headline": {"type": "text"}}')]
    if len(schema) > MAX_VARIABLES:
        return [_error("variables_schema", f"at most {MAX_VARIABLES} variables")]
    return [error for name, spec in schema.items() for error in _spec_errors(str(name), spec)]


def is_bundle_variable(name: str) -> bool:
    return name.startswith(BUNDLE_VARIABLE_PREFIXES)


def placeholders(text: str) -> List[str]:
    """The ``{{ name }}`` names ``text`` uses, in order, each once."""
    return list(dict.fromkeys(PLACEHOLDER.findall(text or "")))


def _placeholder_errors(field: str, text: str, schema: Mapping[str, Any]) -> List[Dict[str, str]]:
    errors = [
        _error(field, f"uses {{{{ {name} }}}}, which variables_schema does not declare")
        for name in placeholders(text)
        if not is_bundle_variable(name) and name not in schema
    ]
    leftover = _LEFTOVER.search(PLACEHOLDER.sub("", text or ""))
    if leftover:
        errors.append(_error(field, f"has a placeholder that is not a variable name: {leftover.group(0)[:60]}"))
    return errors


@dataclass(frozen=True)
class ResolvedVariables:
    """What a render fills in: the values, and what stops it."""

    values: Dict[str, Any]
    missing: List[str]
    invalid: List[str]


def resolve_variables(schema: Mapping[str, Any], supplied: Mapping[str, Any]) -> ResolvedVariables:
    """Each declared variable's value: the one supplied, else its default.

    ``missing`` names the variables with neither; ``invalid`` says which values
    do not fit their variable. Undeclared names in ``supplied`` are ignored. A
    number supplied for a text variable is taken as its text (an agent's 42).
    """
    values: Dict[str, Any] = {}
    missing: List[str] = []
    invalid: List[str] = []
    for name, spec in schema.items():
        value = supplied.get(name) if isinstance(supplied, Mapping) else None
        if spec.get("type") == TEXT and _is_number(value):
            value = str(value)
        if value is None:
            value = spec.get("default")
        if value is None:
            missing.append(name)
            continue
        problem = _value_problem(spec, value)
        if problem:
            invalid.append(f"{name} {problem}")
            continue
        values[name] = value
    return ResolvedVariables(values=values, missing=missing, invalid=invalid)


def claim_names(schema: Mapping[str, Any]) -> List[str]:
    """The variables marked ``claim: true`` (D7: each needs a source)."""
    return [name for name, spec in schema.items() if isinstance(spec, dict) and spec.get("claim") is True]


# ── the whole template ──────────────────────────────────────────────────────
def _text_errors(field: str, value: Any, *, required: bool) -> List[Dict[str, str]]:
    if value is None and not required:
        return []
    if not isinstance(value, str) or (required and not value.strip()):
        return [_error(field, "must be text" + (" and not empty" if required else ""))]
    if len(value) > MAX_COMPOSITION_CHARS:
        return [_error(field, f"is longer than {MAX_COMPOSITION_CHARS} characters")]
    return []


def _document_errors(html: str) -> List[Dict[str, str]]:
    errors = []
    if len(_HEAD_CLOSE.findall(html)) != 1:
        errors.append(_error("html", "must be a full document with exactly one </head>"))
    if len(_ROOT.findall(html)) != 1:
        errors.append(_error("html", 'needs exactly one root element with data-composition-id="main"'))
    return errors


def _audio_errors(plan: Any, fmt: str) -> List[Dict[str, str]]:
    if plan is None or plan == {}:
        return []
    if fmt == SOCIAL_IMAGE:
        return [_error("audio_plan", "an image has no audio: leave audio_plan out")]
    if not isinstance(plan, dict):
        return [_error("audio_plan", f"must be an object with {', '.join(AUDIO_PLAN_KEYS)}")]
    return [
        _error(f"audio_plan.{key}", f"is not part of an audio plan ({', '.join(AUDIO_PLAN_KEYS)})")
        for key in plan
        if key not in AUDIO_PLAN_KEYS
    ]


def validate_social_blocks(blocks: Any, fmt: str) -> Dict[str, Any]:
    """Check a social template's ``blocks`` for ``fmt``; a normalised copy, or :class:`SocialTemplateError`.

    Every problem is reported at once, the brand rule (D4) included.
    """
    if not isinstance(blocks, dict):
        raise SocialTemplateError([_error("blocks", f"a {fmt} template needs blocks {{{', '.join(BLOCK_KEYS)}}}")])
    errors = [_error(key, "is required") for key in REQUIRED_BLOCK_KEYS if key not in blocks]
    errors += [_error(key, f"is not part of a social template ({', '.join(BLOCK_KEYS)})") for key in blocks if key not in BLOCK_KEYS]
    html, css = blocks.get("html"), blocks.get("css")
    errors += _text_errors("html", html, required=True) if "html" in blocks else []
    errors += _text_errors("css", css, required=False)
    if "variables_schema" in blocks:
        errors += _schema_errors(blocks["variables_schema"])
    if "sizes" in blocks:
        errors += _size_errors(blocks["sizes"])
    errors += _audio_errors(blocks.get("audio_plan"), fmt)
    if not errors:
        schema = blocks["variables_schema"]
        errors += _document_errors(html)
        errors += _placeholder_errors("html", html, schema) + _placeholder_errors("css", css or "", schema)
        errors += [_error(field, message) for field, message in brand_literals(html, css or "")]
    if errors:
        raise SocialTemplateError(errors)
    return {key: copy.deepcopy(blocks[key]) for key in BLOCK_KEYS if key in blocks}


# ── the brand rule (D4) ─────────────────────────────────────────────────────
HEX_COLOUR = re.compile(r"#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{3,4})(?![0-9A-Za-z_-])")
_WHOLE_HEX = re.compile(r"^\s*#(?:[0-9a-fA-F]{3,4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})\s*$")
_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_INNERMOST_BLOCK = re.compile(r"\{([^{}]*)\}")
_CSS_URL = re.compile(r"url\(\s*(?:\"([^\"]*)\"|'([^']*)'|([^)\s]*))\s*\)", re.IGNORECASE)
_QUOTED = re.compile(r"\"[^\"]*\"|'[^']*'")
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
# A script's colour or font is a value: after ':' or '=' ("color: '#fff'", "el.style.color = '#fff'").
_SCRIPT_HEX = re.compile(r"""[:=]\s*(["'`])(#[0-9a-fA-F]{3,8})\1""")
_SCRIPT_FONT = re.compile(r"""fontFamily\s*[:=]\s*(["'`])(.*?)\1""")
# A font shorthand's family list follows its size: "500 42px/1.2 Geist, sans-serif".
_FONT_SIZE_THEN_FAMILY = re.compile(
    r"(?:^|\s)(?:\d*\.?\d+(?:px|em|rem|%|pt|pc|vh|vw|vmin|vmax|ch|ex|cm|mm|in|q|lh|rlh)"
    r"|xx-small|x-small|small|medium|large|x-large|xx-large|xxx-large|larger|smaller)"
    r"(?:\s*/\s*\S+)?\s+(.+)$",
    re.IGNORECASE,
)
GENERIC_FAMILIES = frozenset(
    {
        "serif", "sans-serif", "monospace", "cursive", "fantasy", "system-ui", "ui-serif",
        "ui-sans-serif", "ui-monospace", "ui-rounded", "emoji", "math", "fangsong",
        "inherit", "initial", "unset", "revert", "revert-layer",
    }
)
# Attributes that load something (media-render's list); a logo would sit in one.
REFERENCE_ATTRIBUTES = frozenset(
    {"src", "href", "xlink:href", "poster", "srcset", "data", "background", "data-composition-src"}
)
BRAND_ASSET_DIR = "assets/brand/"


def _call_end(text: str, start: int) -> int:
    """The index just past the ``)`` that closes the call opened at ``text[start] == '('``."""
    depth, quote, i = 0, None, start
    while i < len(text):
        ch = text[i]
        if quote:
            quote = None if ch == quote else quote
        elif ch in "\"'":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return len(text)


def strip_var_calls(value: str) -> str:
    """``value`` without its ``var(...)`` calls, fallbacks and nesting included."""
    out, i = [], 0
    lowered = value.lower()
    while i < len(value):
        if lowered.startswith("var(", i) and (i == 0 or not (value[i - 1].isalnum() or value[i - 1] in "-_")):
            i = _call_end(value, i + 3)
            continue
        out.append(value[i])
        i += 1
    return "".join(out)


def _split_top(text: str, separator: str) -> List[str]:
    """Split ``text`` on ``separator`` outside quotes and parentheses."""
    parts, buf, depth, quote = [], [], 0, None
    for ch in text:
        if quote:
            quote = None if ch == quote else quote
        elif ch in "\"'":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        elif ch == separator and depth == 0:
            parts.append("".join(buf))
            buf = []
            continue
        buf.append(ch)
    parts.append("".join(buf))
    return parts


def _declarations(block: str) -> Iterable[Tuple[str, str]]:
    """``(property, value)`` for each declaration in a declaration list."""
    for part in _split_top(block, ";"):
        head, *rest = _split_top(part, ":")
        if rest and head.strip():
            yield head.strip().lower(), ":".join(rest).strip()


def _named_families(family_list: str) -> List[str]:
    names = []
    for entry in _split_top(family_list, ","):
        name = entry.strip().strip("\"'").strip()
        if name and name.lower() not in GENERIC_FAMILIES:
            names.append(name)
    return names


def _font_findings(prop: str, value: str) -> List[str]:
    bare = strip_var_calls(value).replace("!important", "").strip()
    if prop == "font-family":
        names = _named_families(bare)
    elif prop == "font":
        match = _FONT_SIZE_THEN_FAMILY.search(bare)
        names = _named_families(match.group(1)) if match else [q.strip("\"'") for q in _QUOTED.findall(bare)]
    else:
        return []
    return [f"font family {name!r} outside a var() fallback; use var(--brand-body-font) or var(--brand-heading-font)" for name in names]


def _reference_finding(value: str) -> Optional[str]:
    """Why a referenced file bakes a brand asset into the template, or ``None``."""
    ref = value.strip()
    if not ref or "{{" in ref or ref.startswith("#") or ref.lower().startswith("data:"):
        return None
    if _SCHEME.match(ref) or ref.startswith("//"):
        return f"{ref[:80]!r} is a URL; a template carries no URLs (the brand kit supplies the logo)"
    path = ref.split("#", 1)[0].split("?", 1)[0].lower()
    path = path[2:] if path.startswith("./") else path
    if "logo" in path or path.startswith(BRAND_ASSET_DIR):
        return f"{ref[:80]!r} bakes a logo into the template; use {{{{ brand.logo }}}}"
    return None


def _declaration_findings(prop: str, value: str) -> List[str]:
    findings = _font_findings(prop, value)
    bare = strip_var_calls(value)
    for match in _CSS_URL.finditer(bare):
        finding = _reference_finding(next(g for g in match.groups() if g is not None))
        if finding:
            findings.append(finding)
    colours = _QUOTED.sub("", _CSS_URL.sub("", bare))
    findings += [
        f"hex colour {colour} outside a var() fallback in '{prop}'; use var(--brand-…, {colour})"
        for colour in HEX_COLOUR.findall(colours)
    ]
    return findings


def _stylesheet_findings(css: str) -> List[str]:
    # A placeholder's braces would hide its rule from the innermost-block scan.
    text = PLACEHOLDER.sub("0", _CSS_COMMENT.sub("", css or ""))
    return [
        finding
        for block in _INNERMOST_BLOCK.findall(text)
        for prop, value in _declarations(block)
        for finding in _declaration_findings(prop, value)
    ]


class _Scan(HTMLParser):
    """Attributes, <style> bodies and <script> bodies of a composition."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.attributes: List[Tuple[str, str, str]] = []
        self.styles: List[str] = []
        self.scripts: List[str] = []
        self._open: Optional[str] = None
        self._buffer: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        self.attributes += [(tag, name.lower(), value) for name, value in attrs if value is not None]
        if tag in ("style", "script"):
            self._open, self._buffer = tag, []

    def handle_endtag(self, tag: str) -> None:
        if tag == self._open:
            (self.styles if tag == "style" else self.scripts).append("".join(self._buffer))
            self._open = None

    def handle_data(self, data: str) -> None:
        if self._open:
            self._buffer.append(data)


def _attribute_findings(tag: str, name: str, value: str) -> List[str]:
    if name == "style":
        return [f for prop, val in _declarations(value) for f in _declaration_findings(prop, val)]
    if name in REFERENCE_ATTRIBUTES:
        refs = [p.split()[0] for p in value.split(",") if p.split()] if name == "srcset" else [value]
        return [f"<{tag} {name}>: {f}" for f in map(_reference_finding, refs) if f]
    if name == "font-family" or (tag == "font" and name == "face"):
        return [f"<{tag} {name}>: {f}" for f in _font_findings("font-family", value)]
    if _WHOLE_HEX.match(value):
        return [f"<{tag} {name}>: hex colour {value.strip()}; use a --brand-* variable in a style"]
    return []


def _script_findings(script: str) -> List[str]:
    findings = [f"script: hex colour {m.group(2)} as a value; read a --brand-* variable" for m in _SCRIPT_HEX.finditer(script)]
    findings += [f"script: {f}" for m in _SCRIPT_FONT.finditer(script) for f in _font_findings("font-family", m.group(2))]
    return findings


def brand_literals(html: str, css: str = "") -> List[Tuple[str, str]]:
    """``(field, finding)`` for every colour, font or logo the template hardcodes (D4).

    A hex colour, a named font family or a logo file counts only OUTSIDE a
    ``var()`` fallback: ``color: var(--brand-primary, #1a1a2e)`` is the rule,
    ``color: #1a1a2e`` breaks it. Generic families (``sans-serif``) and CSS-wide
    keywords are fine; so are ``{{ brand.logo }}`` and data: URIs (textures).
    """
    scan = _Scan()
    scan.feed(html or "")
    scan.close()
    found: List[Tuple[str, str]] = []
    for style in scan.styles:
        found += [("html", f"<style>: {f}") for f in _stylesheet_findings(style)]
    for tag, name, value in scan.attributes:
        found += [("html", f) for f in _attribute_findings(tag, name, value)]
    for script in scan.scripts:
        found += [("html", f) for f in _script_findings(script)]
    found += [("css", f) for f in _stylesheet_findings(css or "")]
    return list(dict.fromkeys(found))


__all__ = [
    "BLOCK_KEYS",
    "InvalidVariableValues",
    "ResolvedVariables",
    "SOCIAL_IMAGE",
    "SOCIAL_TEMPLATE_FORMATS",
    "SOCIAL_VIDEO",
    "SocialTemplateError",
    "brand_literals",
    "claim_names",
    "is_bundle_variable",
    "is_social_format",
    "parse_size",
    "placeholders",
    "resolve_variables",
    "strip_var_calls",
    "validate_social_blocks",
]
