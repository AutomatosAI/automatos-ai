"""The brand rule for social templates (PRD-251 D4): the brand comes from the brand kit.

A social template's colours, fonts and logo reach its composition from the
workspace's brand kit: ``--brand-*`` CSS variables, ``{{ brand.logo }}`` and
``{{ brand.logo_mark }}`` (``core/media_render_bundle.py``). :func:`brand_literals`
finds what a template hardcodes instead. A template carrying one is refused on
save (``core.social_templates.validate_social_blocks``), and CI holds every seeded
starter to it.

What counts, outside a ``var()`` fallback (``color: var(--brand-primary, #1a1a2e)``
is the rule, ``color: #1a1a2e`` breaks it):

* **A colour.** A hex colour anywhere in a style. An ``rgb()``, ``rgba()``,
  ``hsl()`` or ``hsla()`` whose channels are written out. A named colour
  (``orange``) in the value of a colour property: ``color``, ``background``,
  ``fill``, ``stroke``, ``border``, any ``*-color``, a shadow, a filter, or the
  template's own ``--`` token. A named colour as the whole value of an SVG
  ``fill``, ``stroke`` or ``stop-color``. And in a script, a colour set as a
  value: a string that IS one of these, after ``:`` or ``=``
  (``backgroundColor: "rgb(233, 98, 53)"``, ``el.style.color = "orange"``).
* **A font.** A named font family. Generic families and CSS-wide keywords are fine.
* **A logo.** A file under ``assets/brand/`` or named like a logo, or any URL.

These pass, because none of them is a brand colour:

* ``transparent`` and ``currentColor``;
* black or white mixed into a colour by ``color-mix()``: a shade, a tint, or with
  ``transparent`` a shadow (``color-mix(in srgb, black 45%, transparent)``);
* a colour a script computes at runtime from a ``--brand-*`` variable
  (``rgba(${ACCENT}, 0.3)``);
* a named colour in an alpha mask (``mask-image: linear-gradient(black, transparent)``),
  whose colour never shows;
* data: URIs (textures) and ``{{ brand.logo }}``.

Pure: the standard library only, so the media-render CI job checks the seeded
templates with this very code.
"""
from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Iterable, Iterator, List, Optional, Tuple

HEX_COLOUR = re.compile(r"#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{3,4})(?![0-9A-Za-z_-])")
_WHOLE_HEX = re.compile(r"^\s*#(?:[0-9a-fA-F]{3,4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})\s*$")
# rgb()/rgba()/hsl()/hsla() with its first three channels written out: numbers,
# percentages or a hue's angle, separated by commas or spaces.
_CHANNEL = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?(?:%|deg|grad|rad|turn)?"
COLOUR_FUNCTION = re.compile(
    rf"(?<![\w$.-])(rgba?|hsla?)\(\s*{_CHANNEL}(?:\s*,\s*|\s+){_CHANNEL}(?:\s*,\s*|\s+){_CHANNEL}",
    re.IGNORECASE,
)
# CSS's named colours (CSS Color 4). transparent and currentColor are keywords, not among them.
NAMED_COLOURS = frozenset(
    """
    aliceblue antiquewhite aqua aquamarine azure beige bisque black blanchedalmond blue blueviolet
    brown burlywood cadetblue chartreuse chocolate coral cornflowerblue cornsilk crimson cyan
    darkblue darkcyan darkgoldenrod darkgray darkgreen darkgrey darkkhaki darkmagenta
    darkolivegreen darkorange darkorchid darkred darksalmon darkseagreen darkslateblue
    darkslategray darkslategrey darkturquoise darkviolet deeppink deepskyblue dimgray dimgrey
    dodgerblue firebrick floralwhite forestgreen fuchsia gainsboro ghostwhite gold goldenrod gray
    green greenyellow grey honeydew hotpink indianred indigo ivory khaki lavender lavenderblush
    lawngreen lemonchiffon lightblue lightcoral lightcyan lightgoldenrodyellow lightgray lightgreen
    lightgrey lightpink lightsalmon lightseagreen lightskyblue lightslategray lightslategrey
    lightsteelblue lightyellow lime limegreen linen magenta maroon mediumaquamarine mediumblue
    mediumorchid mediumpurple mediumseagreen mediumslateblue mediumspringgreen mediumturquoise
    mediumvioletred midnightblue mintcream mistyrose moccasin navajowhite navy oldlace olive
    olivedrab orange orangered orchid palegoldenrod palegreen paleturquoise palevioletred
    papayawhip peachpuff peru pink plum powderblue purple rebeccapurple red rosybrown royalblue
    saddlebrown salmon sandybrown seagreen seashell sienna silver skyblue slateblue slategray
    slategrey snow springgreen steelblue tan teal thistle tomato turquoise violet wheat white
    whitesmoke yellow yellowgreen
    """.split()
)
# A word in a CSS value: not part of a hex colour, a number's unit, a longer name or a function name.
_WORD = re.compile(r"(?<![\w#.$-])[A-Za-z]+(?![\w(-])")
# What color-mix() may mix a colour with and stay brand-true: black shades it, white tints it.
_MIX_NEUTRAL = re.compile(r"(?<![\w#.$-])(?:black|white)(?![\w(-])", re.IGNORECASE)
_COLOUR_MIX = re.compile(r"(?<![\w-])color-mix\(", re.IGNORECASE)
# The properties whose value is or carries a colour, besides ``color``, every ``*-color``
# and the template's own ``--`` tokens. A mask is not one: its colour never shows.
COLOUR_SHORTHANDS = frozenset(
    {
        "background", "background-image", "fill", "stroke", "border", "border-top", "border-right",
        "border-bottom", "border-left", "border-block", "border-block-start", "border-block-end",
        "border-inline", "border-inline-start", "border-inline-end", "border-image", "outline",
        "column-rule", "text-decoration", "text-emphasis", "box-shadow", "text-shadow", "filter",
        "-webkit-text-stroke",
    }
)
# The attributes that paint (SVG's, and HTML's old colour attributes).
COLOUR_ATTRIBUTES = frozenset({"fill", "stroke", "stop-color", "flood-color", "lighting-color", "color", "bgcolor"})

_CSS_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_INNERMOST_BLOCK = re.compile(r"\{([^{}]*)\}")
# A {{ … }} is filled at render time (its grammar is core.social_templates'); blanked,
# its braces cannot hide its rule from the innermost-block scan.
_PLACEHOLDER = re.compile(r"\{\{[^{}]*\}\}")
_CSS_URL = re.compile(r"url\(\s*(?:\"([^\"]*)\"|'([^']*)'|([^)\s]*))\s*\)", re.IGNORECASE)
_QUOTED = re.compile(r"\"[^\"]*\"|'[^']*'")
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
# A script's colour or font is a value: after ':' or '=' ("color: '#fff'", "el.style.color = '#fff'").
_SCRIPT_HEX = re.compile(r"""[:=]\s*(["'`])(#[0-9a-fA-F]{3,8})\1""")
_SCRIPT_STRING = re.compile(r"""[:=]\s*(["'`])([^"'`\n]{1,80})\1""")
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


def _unquoted_depths(text: str, start: int = 0) -> Iterator[Tuple[int, str, int]]:
    """``(index, char, depth)`` for each character from ``start`` outside quoted strings;
    ``depth`` is how many parentheses are open once the character is read."""
    depth, quote = 0, None
    for i in range(start, len(text)):
        ch = text[i]
        if quote or ch in "\"'":
            quote = None if ch == quote else (quote or ch)
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        yield i, ch, depth


def _call_end(text: str, start: int) -> int:
    """The index just past the ``)`` that closes the call opened at ``text[start] == '('``."""
    return next((i + 1 for i, ch, depth in _unquoted_depths(text, start) if ch == ")" and depth == 0), len(text))


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
    cuts = [i for i, ch, depth in _unquoted_depths(text) if ch == separator and depth == 0]
    return [text[a + 1 : b] for a, b in zip([-1] + cuts, cuts + [len(text)])]


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


def _colour_calls(text: str) -> List[str]:
    """Each rgb()/rgba()/hsl()/hsla() in ``text`` whose channels are written out, whole."""
    return [text[match.start() : _call_end(text, match.end(1))] for match in COLOUR_FUNCTION.finditer(text)]


def _is_colour_call(text: str) -> bool:
    """Whether the whole of ``text`` is an rgb()/rgba()/hsl()/hsla() with its channels written out."""
    match = COLOUR_FUNCTION.match(text)
    return bool(match) and _call_end(text, match.end(1)) == len(text)


def _without_mix_neutrals(value: str) -> str:
    """``value`` with the black and white that a color-mix() mixes in blanked out."""
    out, last = [], 0
    for match in _COLOUR_MIX.finditer(value):
        if match.start() < last:
            continue  # a mix inside a mix already blanked
        end = _call_end(value, match.end() - 1)
        out += [value[last : match.start()], _MIX_NEUTRAL.sub(" ", value[match.start() : end])]
        last = end
    return "".join(out + [value[last:]])


def _is_colour_property(prop: str) -> bool:
    return prop == "color" or prop.endswith("-color") or prop.startswith("--") or prop in COLOUR_SHORTHANDS


def _colour_findings(prop: str, colours: str) -> List[str]:
    """The colours a declaration's value hardcodes; its var() calls, URLs and strings are already out."""
    where = f"outside a var() fallback in '{prop}'; use var(--brand-…,"
    findings = [f"hex colour {colour} {where} {colour})" for colour in HEX_COLOUR.findall(colours)]
    findings += [f"colour {colour} {where} {colour})" for colour in _colour_calls(colours)]
    if _is_colour_property(prop):
        named = [word for word in _WORD.findall(_without_mix_neutrals(colours)) if word.lower() in NAMED_COLOURS]
        findings += [f"named colour {name!r} {where} {name})" for name in named]
    return findings


def _declaration_findings(prop: str, value: str) -> List[str]:
    findings = _font_findings(prop, value)
    bare = strip_var_calls(value)
    for match in _CSS_URL.finditer(bare):
        finding = _reference_finding(next(g for g in match.groups() if g is not None))
        if finding:
            findings.append(finding)
    return findings + _colour_findings(prop, _QUOTED.sub("", _CSS_URL.sub("", bare)))


def _stylesheet_findings(css: str) -> List[str]:
    text = _PLACEHOLDER.sub("0", _CSS_COMMENT.sub("", css or ""))
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


def _whole_colour(value: str, *, named: bool) -> Optional[str]:
    """What ``value`` hardcodes when the whole of it is a colour, or ``None``: a hex colour,
    an rgb()/hsl() with its channels written out or, when ``named`` counts, a named colour."""
    text = value.strip()
    if _WHOLE_HEX.match(text):
        return f"hex colour {text}"
    if _is_colour_call(text):
        return f"colour {text}"
    if named and text.lower() in NAMED_COLOURS:
        return f"named colour {text!r}"
    return None


def _attribute_findings(tag: str, name: str, value: str) -> List[str]:
    if name == "style":
        return [f for prop, val in _declarations(value) for f in _declaration_findings(prop, val)]
    if name in REFERENCE_ATTRIBUTES:
        refs = [p.split()[0] for p in value.split(",") if p.split()] if name == "srcset" else [value]
        return [f"<{tag} {name}>: {f}" for f in map(_reference_finding, refs) if f]
    if name == "font-family" or (tag == "font" and name == "face"):
        return [f"<{tag} {name}>: {f}" for f in _font_findings("font-family", value)]
    colour = _whole_colour(value, named=name in COLOUR_ATTRIBUTES)
    return [f"<{tag} {name}>: {colour}; use a --brand-* variable in a style"] if colour else []


def _script_colours(script: str) -> List[str]:
    """The rgb()/hsl() and named colours a script sets as a value (its hex colours are _SCRIPT_HEX's)."""
    found = []
    for match in _SCRIPT_STRING.finditer(script):
        text = match.group(2).strip()
        if _is_colour_call(text):
            found.append(f"colour {text}")
        elif text.lower() in NAMED_COLOURS:
            found.append(f"named colour {text!r}")
    return found


def _script_findings(script: str) -> List[str]:
    findings = [f"script: hex colour {m.group(2)} as a value; read a --brand-* variable" for m in _SCRIPT_HEX.finditer(script)]
    findings += [f"script: {colour} as a value; read a --brand-* variable" for colour in _script_colours(script)]
    findings += [f"script: {f}" for m in _SCRIPT_FONT.finditer(script) for f in _font_findings("font-family", m.group(2))]
    return findings


def brand_literals(html: str, css: str = "") -> List[Tuple[str, str]]:
    """``(field, finding)`` for every colour, font or logo the template hardcodes (D4).

    A colour, a named font family or a logo file counts only OUTSIDE a ``var()``
    fallback: ``color: var(--brand-primary, #1a1a2e)`` is the rule, ``color:
    #1a1a2e`` or ``color: orange`` breaks it. What passes is in the module's
    docstring: generic families, CSS-wide keywords, ``transparent``,
    ``currentColor``, black or white mixed in by ``color-mix()``, a colour a
    script computes from a ``--brand-*`` variable, ``{{ brand.logo }}`` and data:
    URIs (textures).
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
    "COLOUR_FUNCTION",
    "HEX_COLOUR",
    "NAMED_COLOURS",
    "brand_literals",
    "strip_var_calls",
]
