"""The composition document: a template's HTML/CSS filled in and checked.

A render bundle carries the template's HTML (a full document) and optional CSS,
the variables, and the brand tokens and fonts. This module:

- fills ``{{ name }}`` placeholders with the variables, HTML-escaped: every word
  on screen is template text, and a variable can only ever be text (D4). Inside
  a ``<script>`` or an ``on*`` event handler a variable must be a number or
  true/false (a scene time, say): escaping keeps text inert in markup, not in
  code, where the browser runs what it decodes;
- injects the brand tokens as CSS custom properties (``--brand-<name>``), the
  brand fonts as ``@font-face`` rules pointing at their staged files, and the
  template CSS, just before ``</head>``;
- reads the root (``data-composition-id="main"``): duration, width, height;
- refuses markup and CSS that point anywhere but a file the bundle provides
  (checked after the fill, so no variable can smuggle a URL in). A render only
  ever reads files staged beside it; media arrive through the storage allowlist
  (media_urls.py), never as a URL in the markup. The template's own scripts are
  the template's: they are data we seed (D4), not request input.

The staged layout it relies on: ``assets/vendor/gsap.min.js`` (GSAP, staged
from the image) and ``assets/audio/mix.wav`` (the mix, the document's single
``<audio>``).
"""

from __future__ import annotations

import html
import math
import posixpath
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import AbstractSet, Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .validate import BundleError

ROOT_ID = "main"
GSAP_PATH = "assets/vendor/gsap.min.js"
MIX_PATH = "assets/audio/mix.wav"
STAGED_PATHS = frozenset({GSAP_PATH, MIX_PATH})

VARIABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*$")
_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*)\s*\}\}")
_LEFTOVER = re.compile(r"\{\{[^{}]*\}\}")
_HEAD_CLOSE = re.compile(r"</head\s*>", re.IGNORECASE)
_STYLE_CLOSE = re.compile(r"</style", re.IGNORECASE)
_SCHEME = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*):")
_STRIPPED = re.compile(r"[\x00-\x20\x7f]")
# A quoted url() consumes its whole string, so the url(%23n) inside an inline
# SVG data: URI (the references' grain texture) is never read as a reference.
_CSS_URL = re.compile(r"url\(\s*(?:\"([^\"]*)\"|'([^']*)'|([^)\s]*))\s*\)", re.IGNORECASE)
_CSS_IMPORT = re.compile(r"@import\s+(?:url\(\s*)?[\"']?([^\"')\s;]+)", re.IGNORECASE)
_CSS_IMAGE_SET = re.compile(r"image-set\(([^)]*)\)", re.IGNORECASE)
_CSS_STRING = re.compile(r"\"([^\"]*)\"|'([^']*)'")
# Where a filled-in value would be run rather than shown: a <script> body, or an
# on* attribute inside a tag (the browser decodes the attribute's entities first).
_SCRIPT_BODY = re.compile(r"<script\b[^>]*>(.*?)</script\s*>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[A-Za-z][^<>]*>")
_EVENT_HANDLER = re.compile(r"""\son[a-z]+\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)""", re.IGNORECASE)

# Attributes that make a browser load something. srcset holds a list.
URL_ATTRIBUTES = frozenset(
    {
        "src",
        "href",
        "xlink:href",
        "poster",
        "srcset",
        "data",
        "action",
        "formaction",
        "background",
        "ping",
        "manifest",
        "lowsrc",
        "longdesc",
        "codebase",
        "cite",
        "icon",
        "data-composition-src",
    }
)
# Elements that embed or navigate to other documents: nothing a video needs.
FORBIDDEN_ELEMENTS = frozenset({"iframe", "frame", "frameset", "object", "embed", "base", "form", "portal"})

DIMENSION_RANGE = (16, 4096)


@dataclass(frozen=True)
class FontFace:
    family: str
    weight: str
    style: str
    path: str


@dataclass(frozen=True)
class Composition:
    html: str
    duration: float
    width: int
    height: int

    @property
    def aspect(self) -> str:
        divisor = math.gcd(self.width, self.height)
        return f"{self.width // divisor}:{self.height // divisor}"


def _as_text(name: str, value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    raise BundleError(f"variables.{name} must be text, a number or true/false")


def _code_spans(text: str) -> List[Tuple[int, int]]:
    """The (start, end) offsets of every script body and event-handler value in ``text``."""
    spans = [match.span(1) for match in _SCRIPT_BODY.finditer(text)]
    for tag in _TAG.finditer(text):
        spans.extend(
            (tag.start() + handler.start(1), tag.start() + handler.end(1))
            for handler in _EVENT_HANDLER.finditer(tag.group(0))
        )
    return spans


def fill_placeholders(text: str, variables: Mapping[str, Any], where: str) -> str:
    """Replace every ``{{ name }}`` with its variable, HTML-escaped; refuse any left unfilled.

    Text never lands in code: inside a script or an event handler only a
    number or true/false is filled in.
    """
    missing: List[str] = []
    in_code: List[str] = []
    code = _code_spans(text)

    def replace(match: "re.Match[str]") -> str:
        name = match.group(1)
        if name not in variables:
            missing.append(name)
            return match.group(0)
        value = variables[name]
        if isinstance(value, str) and any(start <= match.start() < end for start, end in code):
            in_code.append(name)
            return match.group(0)
        return html.escape(_as_text(name, value), quote=True)

    filled = _PLACEHOLDER.sub(replace, text)
    if missing:
        raise BundleError(f"{where} uses variable(s) the bundle does not set: {', '.join(sorted(set(missing)))}")
    if in_code:
        names = ", ".join(sorted(set(in_code)))
        raise BundleError(
            f"{where} puts text variable(s) {names} inside a <script> or an event handler; only numbers and true/false go there"
        )
    leftover = _LEFTOVER.search(filled)
    if leftover:
        raise BundleError(f"{where} has a placeholder that is not a valid variable name: {leftover.group(0)[:60]}")
    return filled


def brand_stylesheet(tokens: Mapping[str, str], fonts: Sequence[FontFace]) -> str:
    """The brand kit as CSS: tokens as --brand-* custom properties, fonts as @font-face."""
    rules: List[str] = []
    if tokens:
        rules.append(":root{" + "".join(f"--brand-{name}:{value};" for name, value in sorted(tokens.items())) + "}")
    for font in fonts:
        extension = font.path.rsplit(".", 1)[-1].lower()
        font_format = {"woff2": "woff2", "woff": "woff", "ttf": "truetype", "otf": "opentype"}[extension]
        rules.append(
            f'@font-face{{font-family:"{font.family}";src:url("{font.path}") format("{font_format}");'
            f"font-weight:{font.weight};font-style:{font.style};font-display:block}}"
        )
    return "".join(rules)


def build_document(document: str, css: str, variables: Mapping[str, Any], brand_css: str) -> str:
    """Fill the placeholders, then inject the brand and template styles before ``</head>``."""
    filled = fill_placeholders(document, variables, "composition.html")
    styles = [brand_css, fill_placeholders(css, variables, "composition.css")]
    for style in styles:
        if _STYLE_CLOSE.search(style):
            raise BundleError("composition.css must not close its <style> element")
    closes = list(_HEAD_CLOSE.finditer(filled))
    if len(closes) != 1:
        raise BundleError("composition.html must be a full document with exactly one </head>")
    injected = "".join(f"<style>{style}</style>" for style in styles if style.strip())
    at = closes[0].start()
    return filled[:at] + injected + filled[at:]


class _Scanner(HTMLParser):
    """One pass over the document: the root, the audio elements, every reference."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.roots: List[Dict[str, Optional[str]]] = []
        self.audio_sources: List[Optional[str]] = []
        self.references: List[Tuple[str, str]] = []
        self.stylesheets: List[str] = []
        self.forbidden: List[str] = []
        self._style_depth = 0
        self._style_text: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        attributes = dict(attrs)
        if tag in FORBIDDEN_ELEMENTS:
            self.forbidden.append(f"<{tag}>")
        if tag == "meta" and (attributes.get("http-equiv") or "").strip().lower() == "refresh":
            self.forbidden.append('<meta http-equiv="refresh">')
        if attributes.get("data-composition-id") == ROOT_ID:
            self.roots.append(attributes)
        if tag == "audio":
            self.audio_sources.append(attributes.get("src"))
        for name, value in attrs:
            if value is None:
                continue
            if name in URL_ATTRIBUTES:
                where = f"<{tag} {name}>"
                candidates = [part.split()[0] for part in value.split(",") if part.split()] if name == "srcset" else [value]
                self.references.extend((where, candidate) for candidate in candidates)
            elif name == "style":
                self.stylesheets.append(value)
        if tag == "style":
            self._style_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "style" and self._style_depth:
            self._style_depth -= 1
            self.stylesheets.append("".join(self._style_text))
            self._style_text = []

    def handle_data(self, data: str) -> None:
        if self._style_depth:
            self._style_text.append(data)


def _reference_problem(where: str, value: str, known: AbstractSet[str]) -> Optional[str]:
    compact = _STRIPPED.sub("", value)
    if not compact or compact.startswith("#"):
        return None
    if "\\" in compact:
        return f"{where} {value!r} has a backslash"
    scheme = _SCHEME.match(compact)
    if scheme and scheme.group(1).lower() == "data":
        return None
    if scheme or compact.startswith("//"):
        return f"{where} {value!r} points outside the composition; media come through the bundle's media list"
    path = compact.split("#", 1)[0].split("?", 1)[0]
    if path.startswith("./"):
        path = path[2:]
    if path.startswith("/") or ".." in path.split("/"):
        return f"{where} {value!r} must be a path relative to the composition"
    if posixpath.normpath(path) not in known:
        return f"{where} {value!r} names a file the bundle does not provide"
    return None


def _stylesheet_problems(css: str, known: AbstractSet[str]) -> List[str]:
    references = [("CSS url()", next(g for g in m.groups() if g is not None)) for m in _CSS_URL.finditer(css)]
    for image_set in _CSS_IMAGE_SET.finditer(_CSS_URL.sub("", css)):
        references.extend(
            ("CSS image-set()", next(g for g in m.groups() if g is not None))
            for m in _CSS_STRING.finditer(image_set.group(1))
        )
    problems = [problem for where, value in references for problem in [_reference_problem(where, value, known)] if problem]
    problems.extend(
        f"CSS @import {match.group(1)!r}: a composition carries its styles inline" for match in _CSS_IMPORT.finditer(css)
    )
    return problems


def _root_number(root: Mapping[str, Optional[str]], name: str) -> float:
    raw = (root.get(name) or "").strip()
    try:
        value = float(raw)
    except ValueError:
        raise BundleError(f'the root element needs a numeric {name} (found "{raw}")') from None
    if not math.isfinite(value):
        raise BundleError(f'the root element needs a numeric {name} (found "{raw}")')
    return value


def inspect_document(document: str, *, known_paths: AbstractSet[str], max_duration: float) -> Composition:
    """Read the root and refuse anything a render must not do; returns the composition."""
    scanner = _Scanner()
    scanner.feed(document)
    scanner.close()

    if scanner.forbidden:
        raise BundleError(f"the composition contains {', '.join(sorted(set(scanner.forbidden)))}")
    if len(scanner.roots) != 1:
        raise BundleError(f'the composition needs exactly one root with data-composition-id="{ROOT_ID}"')
    root = scanner.roots[0]
    duration = _root_number(root, "data-duration")
    if not 0 < duration <= max_duration:
        raise BundleError(f"the composition lasts {duration:g} s; it must be over 0 and at most {max_duration:g} s")
    width, height = (_root_number(root, name) for name in ("data-width", "data-height"))
    low, high = DIMENSION_RANGE
    if not (width.is_integer() and height.is_integer() and low <= width <= high and low <= height <= high):
        raise BundleError(f"the composition's data-width and data-height must be whole numbers from {low} to {high}")
    if scanner.audio_sources != [MIX_PATH]:
        raise BundleError(f'the composition needs exactly one <audio>, with src="{MIX_PATH}" (the mix)')

    known = frozenset(known_paths) | STAGED_PATHS
    problems = [
        problem
        for where, value in scanner.references
        for problem in [_reference_problem(where, value, known)]
        if problem
    ]
    for css in scanner.stylesheets:
        problems.extend(_stylesheet_problems(css, known))
    if problems:
        raise BundleError("; ".join(dict.fromkeys(problems)))
    return Composition(html=document, duration=duration, width=int(width), height=int(height))
