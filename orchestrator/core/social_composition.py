"""The renderer's page assembly, held in the orchestrator (PRD-251D US-D201, E8).

The instant preview of an image post (``modules/socials/instant_preview.py``) is the
template's own page, filled here and shown in a sandboxed frame in the browser. The
renderer's own code lives in another container
(``services/media-render/media_render/composition.py``) and cannot be imported by the
orchestrator, so the rules a preview needs are restated here, rule for rule:

* :func:`fill_placeholders` fills each ``{{ name }}`` with its variable, HTML-escaped, and
  refuses a text variable inside a ``<script>`` body or an ``on*`` attribute, where the
  browser runs what it decodes: only a number or true/false goes there;
* :func:`build_document` fills the document and its css, then puts the brand styles and
  the template css just before ``</head>``;
* :func:`token_stylesheet` writes the brand tokens as ``:root{--brand-<name>:value}`` and
  :func:`font_face_rule` one ``@font-face`` rule, as the renderer's ``brand_stylesheet``
  writes them;
* :func:`forbidden_elements` names what a composition may not hold: an element that embeds
  or navigates to another document, a base URL, a form, a refresh.

``tests/test_prd251d_instant_preview_parity.py`` imports the renderer's module and holds
each of these equal to its own, so the two cannot drift. Pure: the standard library only.
"""
from __future__ import annotations

import html
import re
from html.parser import HTMLParser
from typing import Any, List, Mapping, Optional, Tuple

# The renderer's staged files: the timeline library and the mix (composition.py).
GSAP_PATH = "assets/vendor/gsap.min.js"
MIX_PATH = "assets/audio/mix.wav"
HTML_WHERE = "composition.html"
CSS_WHERE = "composition.css"

_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z0-9_]+)*)\s*\}\}")
_LEFTOVER = re.compile(r"\{\{[^{}]*\}\}")
_HEAD_CLOSE = re.compile(r"</head\s*>", re.IGNORECASE)
_STYLE_CLOSE = re.compile(r"</style", re.IGNORECASE)
# Where a filled-in value would be run rather than shown: a <script> body, or an on*
# attribute inside a tag. The end tag's name ends only at whitespace, '/' or '>', and the
# rest of it stops at the next '<' (the renderer's reasons, kept with its pattern).
_SCRIPT_BODY = re.compile(r"<script\b[^>]*>(.*?)</script(?=[\t\n\f\r />])[^<>]*>", re.IGNORECASE | re.DOTALL)
_TAG = re.compile(r"<[A-Za-z][^<>]*>")
_EVENT_HANDLER = re.compile(r"""\son[a-z]+\s*=\s*("[^"]*"|'[^']*'|[^\s>]+)""", re.IGNORECASE)

# Elements that embed or navigate to other documents: nothing a post needs.
FORBIDDEN_ELEMENTS = frozenset({"iframe", "frame", "frameset", "object", "embed", "base", "form", "portal"})
REFRESH = "refresh"
# A font file's extension → the format() an @font-face names.
FONT_FORMATS = {"woff2": "woff2", "woff": "woff", "ttf": "truetype", "otf": "opentype"}


class CompositionError(ValueError):
    """The composition cannot be filled, and why (the renderer's ``BundleError``, word for word)."""


def _as_text(name: str, value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    raise CompositionError(f"variables.{name} must be text, a number or true/false")


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

    Text never lands in code: inside a script or an event handler only a number or
    true/false is filled in. :class:`CompositionError` names what is wrong.
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
        raise CompositionError(f"{where} uses variable(s) the bundle does not set: {', '.join(sorted(set(missing)))}")
    if in_code:
        names = ", ".join(sorted(set(in_code)))
        raise CompositionError(
            f"{where} puts text variable(s) {names} inside a <script> or an event handler; only numbers and true/false go there"
        )
    leftover = _LEFTOVER.search(filled)
    if leftover:
        raise CompositionError(f"{where} has a placeholder that is not a valid variable name: {leftover.group(0)[:60]}")
    return filled


def token_stylesheet(tokens: Mapping[str, str]) -> str:
    """The brand tokens as one ``:root`` rule of ``--brand-<name>`` custom properties, sorted ("" for none)."""
    if not tokens:
        return ""
    return ":root{" + "".join(f"--brand-{name}:{value};" for name, value in sorted(tokens.items())) + "}"


def font_face_rule(family: str, weight: str, style: str, url: str, extension: str) -> str:
    """One ``@font-face`` rule as the renderer writes it; ``url`` is the face's file and
    ``extension`` its file's (woff2, woff, ttf or otf)."""
    font_format = FONT_FORMATS[extension.lower()]
    return (
        f'@font-face{{font-family:"{family}";src:url("{url}") format("{font_format}");'
        f"font-weight:{weight};font-style:{style};font-display:block}}"
    )


def build_document(document: str, css: str, variables: Mapping[str, Any], brand_css: str) -> str:
    """Fill the placeholders, then inject the brand and template styles before ``</head>``."""
    filled = fill_placeholders(document, variables, HTML_WHERE)
    styles = [brand_css, fill_placeholders(css, variables, CSS_WHERE)]
    for style in styles:
        if _STYLE_CLOSE.search(style):
            raise CompositionError("composition.css must not close its <style> element")
    closes = list(_HEAD_CLOSE.finditer(filled))
    if len(closes) != 1:
        raise CompositionError("composition.html must be a full document with exactly one </head>")
    injected = "".join(f"<style>{style}</style>" for style in styles if style.strip())
    at = closes[0].start()
    return filled[:at] + injected + filled[at:]


class _ForbiddenScan(HTMLParser):
    """One pass over a document, naming each element a composition may not hold."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.found: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag in FORBIDDEN_ELEMENTS:
            self.found.append(f"<{tag}>")
        if tag == "meta" and (dict(attrs).get("http-equiv") or "").strip().lower() == REFRESH:
            self.found.append('<meta http-equiv="refresh">')


def forbidden_elements(document: str) -> List[str]:
    """What ``document`` holds that a composition may not, sorted, each once ([] for nothing)."""
    scan = _ForbiddenScan()
    scan.feed(document)
    scan.close()
    return sorted(set(scan.found))


__all__ = [
    "CSS_WHERE",
    "FORBIDDEN_ELEMENTS",
    "GSAP_PATH",
    "HTML_WHERE",
    "MIX_PATH",
    "CompositionError",
    "build_document",
    "fill_placeholders",
    "font_face_rule",
    "forbidden_elements",
    "token_stylesheet",
]
