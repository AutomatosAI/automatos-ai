"""What a website shows of its brand, read from its HTML and stylesheets ('Brand kit from your website').

``platform_web_fetch`` reduces a page to readable text: it drops ``<style>``, ``<svg>``, the header and
the footer. That is right for reading an article and wrong for a brand, whose colours, fonts, logo and
social links live in exactly what it drops, so the Brand designer was left to guess them. With
``brand: true`` the fetch also returns :func:`site_brand`'s evidence: every value names where on the
site it was found, so a proposal is built from what the site shows, never from a guess.

Pure: HTML and stylesheet text in, evidence out. The page and its stylesheets are read as served, so a
site drawn by JavaScript shows less; the ``notes`` say when the page looks like one.
"""
from __future__ import annotations

import colorsys
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qs, urljoin, urlparse

from bs4 import BeautifulSoup

from core.brand_palette import RGB, contrast, parse_hex, to_hex

MAX_COLOURS = 12
MAX_FONTS = 8
MAX_LOGOS = 5
MAX_MARKS = 4
# Below this chroma (the spread between its strongest and weakest channel, 0 to 255) a colour is a neutral:
# text, page or rule, not the brand. Chroma, not HSL saturation, which calls a warm near-black (#1a1714) vivid.
NEUTRAL_CHROMA = 20
# A colour this transparent is a tint or a shadow, not a brand colour.
MIN_ALPHA = 0.5
# Fewer visible characters than this on a page that has scripts: probably drawn by JavaScript.
THIN_PAGE_CHARS = 200
WHITE: RGB = (255, 255, 255)
BLACK: RGB = (0, 0, 0)

# A custom property or rule whose name says what the colour is for.
BRAND_WORDS = re.compile(r"brand|primary|secondary|accent|main|highlight|cta|button|btn|link|theme", re.I)
_RULE = re.compile(r"([^{}@;]+)\{([^{}]*)\}")
_DECLARATION = re.compile(r"([\w-]+)\s*:\s*([^;]+)")
_CUSTOM_PROPERTY = re.compile(r"(--[\w-]+)\s*:\s*([^;{}]+)")
_VAR = re.compile(r"var\(\s*(--[\w-]+)\s*(?:,\s*([^)]+))?\)")
_HEX_COLOUR = re.compile(r"#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{3,4})\b")
_FUNC_COLOUR = re.compile(r"\b(rgba?|hsla?)\(\s*([^)]*)\)", re.I)
# A size, a weight, a var() or a number: what a font-ish custom property holds when it is no family.
_NOT_A_FAMILY = re.compile(r"^[\d.]|\d(?:px|rem|em|%)\b|^var\(|^(?:normal|bold|bolder|lighter|italic)$", re.I)
_FONT_FACE = re.compile(r"@font-face\s*\{([^{}]*)\}", re.I)
_GENERIC_FAMILIES = frozenset({"serif", "sans-serif", "monospace", "cursive", "fantasy", "system-ui", "ui-sans-serif",
                               "ui-serif", "ui-monospace", "inherit", "initial", "unset", "-apple-system",
                               "blinkmacsystemfont", "emoji", "math"})
_COLOUR_PROPERTIES = frozenset({"color", "background", "background-color", "border-color", "fill", "stroke",
                                "outline-color", "text-decoration-color"})
# Which elements a rule styles, by its selector: the role its colour and font play on the site.
_ROLES: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("buttons", re.compile(r"\bbutton\b|\.btn|\.button|\bcta\b|\[type=.?submit", re.I)),
    ("links", re.compile(r"(?:^|[\s,>])a(?:[:\s.,]|$)", re.I)),
    ("headings", re.compile(r"(?:^|[\s,>])h[1-3]\b", re.I)),
    ("body", re.compile(r"(?:^|[\s,])(?:body|html|:root)\b", re.I)),
)
_LOGO_WORD = re.compile(r"logo", re.I)
_COMMENT = re.compile(r"/\*.*?\*/", re.S)
# A dark theme's override (`.dark`, `[data-theme=dark]`, a prefers-color-scheme: dark block) is not the brand's
# own colour: the site's light default is.
_DARK_SELECTOR = re.compile(r"\.(?:theme-)?dark\b|\[data-[\w-]*=[\"']?dark\b", re.I)
_DARK_MEDIA = re.compile(r"@media[^{]*prefers-color-scheme\s*:\s*dark[^{]*\{", re.I)
_SOCIAL_HOSTS: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("linkedin", re.compile(r"(?:^|\.)linkedin\.com$")),
    ("twitter", re.compile(r"(?:^|\.)(?:twitter|x)\.com$")),
    ("instagram", re.compile(r"(?:^|\.)instagram\.com$")),
    ("tiktok", re.compile(r"(?:^|\.)tiktok\.com$")),
    ("youtube", re.compile(r"(?:^|\.)youtube\.com$")),
    ("facebook", re.compile(r"(?:^|\.)facebook\.com$")),
    ("threads", re.compile(r"(?:^|\.)threads\.net$")),
    ("pinterest", re.compile(r"(?:^|\.)pinterest\.[a-z.]+$")),
)
# A path that is a share or intent link, a login or a search, never the brand's own profile.
_NOT_A_PROFILE = frozenset({"share", "sharer", "sharer.php", "intent", "home", "login", "search", "hashtag",
                            "watch", "p", "reel", "explore", "shareArticle", "feed", "dialog", "status", "pages",
                            "groups", "profile.php", "events"})


# ── colours ────────────────────────────────────────────────────────────────────────────────────

def _alpha_ok(value: str) -> bool:
    """Whether an alpha channel (``0.4`` or ``40%``) leaves the colour solid enough to count."""
    text = value.strip()
    try:
        alpha = float(text[:-1]) / 100 if text.endswith("%") else float(text)
    except ValueError:
        return True
    return alpha >= MIN_ALPHA


def _channels(args: str) -> List[str]:
    return [part for part in re.split(r"[\s,/]+", args.strip()) if part]


def _rgb_function(name: str, args: str) -> Optional[RGB]:
    """``rgb()``/``rgba()``/``hsl()``/``hsla()`` as ``(r, g, b)``; ``None`` when unreadable or mostly transparent."""
    parts = _channels(args)
    if len(parts) < 3 or (len(parts) > 3 and not _alpha_ok(parts[3])):
        return None
    try:
        if name.lower().startswith("rgb"):
            values = [float(p[:-1]) * 2.55 if p.endswith("%") else float(p) for p in parts[:3]]
            return tuple(max(0, min(255, round(v))) for v in values)  # type: ignore[return-value]
        hue = float(re.sub(r"deg$", "", parts[0])) / 360.0
        sat, light = (float(p.rstrip("%")) / 100.0 for p in parts[1:3])
        r, g, b = colorsys.hls_to_rgb(hue % 1.0, light, sat)
        return round(r * 255), round(g * 255), round(b * 255)
    except ValueError:
        return None


def _hex_colour(token: str) -> Optional[RGB]:
    digits = token[1:]
    if len(digits) in (4, 8):
        alpha = int(digits[3] * 2 if len(digits) == 4 else digits[6:8], 16) / 255
        if alpha < MIN_ALPHA:
            return None
        digits = digits[:3] if len(digits) == 4 else digits[:6]
    return parse_hex("#" + digits)


def colours_in(value: str) -> List[RGB]:
    """Every readable colour in a CSS value, in order."""
    found: List[Tuple[int, RGB]] = []
    for match in _HEX_COLOUR.finditer(value):
        rgb = _hex_colour(match.group(0))
        if rgb is not None:
            found.append((match.start(), rgb))
    for match in _FUNC_COLOUR.finditer(value):
        rgb = _rgb_function(match.group(1), match.group(2))
        if rgb is not None:
            found.append((match.start(), rgb))
    return [rgb for _, rgb in sorted(found)]


def is_neutral(rgb: RGB) -> bool:
    """Grey, off-white or near black: the page and its text, not the brand's colour."""
    return max(rgb) - min(rgb) < NEUTRAL_CHROMA


def without_dark_and_comments(css: str) -> str:
    """``css`` with its comments and its ``prefers-color-scheme: dark`` blocks left out (braces balanced)."""
    css = _COMMENT.sub("", css)
    kept: List[str] = []
    position = 0
    for match in _DARK_MEDIA.finditer(css):
        if match.start() < position:
            continue  # inside a block already skipped
        kept.append(css[position:match.start()])
        depth, index = 1, match.end()
        while index < len(css) and depth:
            depth += {"{": 1, "}": -1}.get(css[index], 0)
            index += 1
        position = index
    kept.append(css[position:])
    return "".join(kept)


def custom_properties(css: str) -> Dict[str, str]:
    """``--name: value`` declarations outside a dark theme, the FIRST one winning: a site declares its default
    first and overrides it later (a breakpoint, a theme), and the default is the brand."""
    props: Dict[str, str] = {}
    for selector, body in _RULE.findall(css):
        if _DARK_SELECTOR.search(selector):
            continue
        for name, value in _CUSTOM_PROPERTY.findall(body):
            props.setdefault(name, value.strip())
    return props


def resolve_vars(value: str, props: Mapping[str, str], depth: int = 3) -> str:
    """``value`` with ``var(--x)`` replaced by the property's value (or its fallback), a few levels deep."""
    for _ in range(depth):
        if "var(" not in value:
            break
        value = _VAR.sub(lambda m: props.get(m.group(1), m.group(2) or ""), value)
    return value


class _ColourTally:
    """Each colour once, with every place it was seen and how often."""

    def __init__(self) -> None:
        self.sources: Dict[RGB, List[str]] = {}
        self.uses: Counter[RGB] = Counter()

    def add(self, rgb: RGB, source: Optional[str]) -> None:
        self.uses[rgb] += 1
        seen = self.sources.setdefault(rgb, [])
        if source and source not in seen:
            seen.append(source)


def _tally_properties(tally: _ColourTally, props: Mapping[str, str]) -> None:
    for name, value in props.items():
        for rgb in colours_in(resolve_vars(value, props)):
            tally.add(rgb, f"CSS variable {name}" if BRAND_WORDS.search(name) else None)


def _rule_role(selector: str) -> Optional[str]:
    return next((role for role, pattern in _ROLES if pattern.search(selector)), None)


def _tally_rules(tally: _ColourTally, css: str, props: Mapping[str, str]) -> None:
    for selector, body in _RULE.findall(css):
        if _DARK_SELECTOR.search(selector):
            continue
        role = _rule_role(selector)
        for prop, value in _DECLARATION.findall(body):
            if prop.lower() not in _COLOUR_PROPERTIES or prop.startswith("--"):
                continue
            label = f"{role}: {prop.lower()}" if role else None
            for rgb in colours_in(resolve_vars(value, props)):
                tally.add(rgb, label)


def _colour_entry(rgb: RGB, tally: _ColourTally) -> Dict[str, Any]:
    return {
        "hex": to_hex(rgb),
        "found_in": tally.sources.get(rgb, [])[:4],
        "uses": tally.uses[rgb],
        "neutral": is_neutral(rgb),
        "contrast_on_white": round(contrast(rgb, WHITE), 2),
        "contrast_on_black": round(contrast(rgb, BLACK), 2),
    }


def site_colours(theme_colours: Sequence[Tuple[str, str]], css: str) -> List[Dict[str, Any]]:
    """The site's colours, the brand's first: named or role-bound colours ahead of neutrals, then by use."""
    tally = _ColourTally()
    for source, value in theme_colours:
        for rgb in colours_in(value):
            tally.add(rgb, source)
    props = custom_properties(css)
    _tally_properties(tally, props)
    _tally_rules(tally, css, props)
    ranked = sorted(tally.uses, key=lambda rgb: (is_neutral(rgb), not tally.sources.get(rgb), -tally.uses[rgb]))
    return [_colour_entry(rgb, tally) for rgb in ranked[:MAX_COLOURS]]


# ── fonts ──────────────────────────────────────────────────────────────────────────────────────

def _families(value: str) -> List[str]:
    """The named families in a ``font-family`` value, generics and anything that is not a name left out."""
    names = [part.strip().strip("'\"").strip() for part in value.split(",")]
    return [n for n in names if n and n.lower() not in _GENERIC_FAMILIES and not _NOT_A_FAMILY.search(n)]


def google_font_families(hrefs: Iterable[str]) -> List[str]:
    """The families a Google Fonts stylesheet link asks for (``css`` and ``css2`` forms)."""
    found: List[str] = []
    for href in hrefs:
        parsed = urlparse(href)
        if not parsed.netloc.endswith("fonts.googleapis.com"):
            continue
        for spec in parse_qs(parsed.query).get("family", []):
            for one in spec.split("|"):
                name = one.split(":")[0].replace("+", " ").strip()
                if name and name not in found:
                    found.append(name)
    return found


def _font_rules(css: str, props: Mapping[str, str]) -> List[Tuple[str, str]]:
    """(source, family) for every font-family a styled role uses, in the stylesheet's order."""
    found: List[Tuple[str, str]] = []
    for selector, body in _RULE.findall(css):
        role = _rule_role(selector) or ("CSS variable" if "--" in body else None)
        for prop, value in _DECLARATION.findall(body):
            # the `font` shorthand mixes sizes into the list, so only font-family and --*font* properties count
            is_family = prop.lower() == "font-family" or (prop.startswith("--") and "font" in prop.lower())
            if not is_family or role is None:
                continue
            for family in _families(resolve_vars(value, props))[:1]:
                found.append((f"{role}: {prop}", family))
    return found


def site_fonts(css: str, google_hrefs: Iterable[str]) -> List[Dict[str, Any]]:
    """The families the site sets, each with where: headings, body, buttons, @font-face, Google Fonts."""
    props = custom_properties(css)
    sources: Dict[str, List[str]] = {}

    def add(family: str, source: str) -> None:
        seen = sources.setdefault(family, [])
        if source not in seen:
            seen.append(source)

    for source, family in _font_rules(css, props):
        add(family, source)
    for block in _FONT_FACE.findall(css):
        for prop, value in _DECLARATION.findall(block):
            if prop.lower() == "font-family":
                for family in _families(value)[:1]:
                    add(family, "@font-face (the site serves this font)")
    for family in google_font_families(google_hrefs):
        add(family, "Google Fonts link")
    return [{"family": family, "found_in": where[:4]} for family, where in list(sources.items())[:MAX_FONTS]]


# ── logo, mark, name, social ───────────────────────────────────────────────────────────────────

def _attrs_text(tag: Any) -> str:
    classes = " ".join(tag.get("class") or [])
    return " ".join(str(tag.get(a) or "") for a in ("src", "alt", "id", "aria-label", "title")) + " " + classes


def _in_header(tag: Any) -> bool:
    return any(parent.name in ("header", "nav") for parent in tag.parents)


def _logo_from_img(img: Any, base_url: str) -> Optional[Dict[str, Any]]:
    src = img.get("src") or img.get("data-src") or ""
    owner = img.find_parent("a")
    says_logo = _LOGO_WORD.search(_attrs_text(img)) or (owner is not None and _LOGO_WORD.search(_attrs_text(owner)))
    if not src or src.startswith("data:") or not says_logo:
        return None
    where = "header" if _in_header(img) else "page"
    return {"url": urljoin(base_url, src), "alt": (img.get("alt") or "").strip(),
            "found_in": f"{where} <img> marked logo", "format": _format_of(src)}


def _format_of(url: str) -> str:
    path = urlparse(url).path.lower()
    return next((ext for ext in ("svg", "png", "webp", "jpg", "jpeg", "gif", "ico") if path.endswith("." + ext)), "")


def logo_candidates(soup: BeautifulSoup, base_url: str, notes: List[str]) -> List[Dict[str, Any]]:
    """Wordmark candidates, the header's first, then ``og:image`` as the fallback a page offers."""
    found = [logo for logo in (_logo_from_img(img, base_url) for img in soup.find_all("img")) if logo]
    found.sort(key=lambda logo: not logo["found_in"].startswith("header"))
    inline_svg = next((svg for svg in soup.find_all("svg") if _LOGO_WORD.search(_attrs_text(svg))
                       or (svg.find_parent("a") is not None and _LOGO_WORD.search(_attrs_text(svg.find_parent("a"))))),
                      None)
    if inline_svg is not None and not found:
        notes.append("The logo is drawn inline as SVG, so it has no file address: a person uploads it on the "
                     "Brand kit page.")
    og = _meta(soup, "og:image")
    if og:
        found.append({"url": urljoin(base_url, og), "alt": "", "found_in": "og:image (the share picture; often "
                      "a banner, not the logo)", "format": _format_of(og)})
    unique = {logo["url"]: logo for logo in reversed(found)}
    return [logo for logo in found if unique.get(logo["url"]) is logo][:MAX_LOGOS]


def _icon_size(link: Any) -> int:
    match = re.match(r"(\d+)x\d+", str(link.get("sizes") or ""))
    return int(match.group(1)) if match else 0


def mark_candidates(soup: BeautifulSoup, base_url: str) -> List[Dict[str, Any]]:
    """Square marks: apple-touch-icon and the site's icons, largest first, SVG ahead of bitmaps."""
    marks: List[Dict[str, Any]] = []
    for link in soup.find_all("link", href=True):
        rel = " ".join(link.get("rel") or []).lower()
        if "icon" not in rel:
            continue
        url = urljoin(base_url, link["href"])
        marks.append({"url": url, "found_in": f'<link rel="{rel}">', "size": _icon_size(link), "format": _format_of(url)})
    marks.sort(key=lambda m: (m["format"] != "svg", -m["size"], "apple" not in m["found_in"]))
    return marks[:MAX_MARKS]


def _meta(soup: BeautifulSoup, key: str) -> str:
    tag = soup.find("meta", attrs={"property": key}) or soup.find("meta", attrs={"name": key})
    return str(tag.get("content") or "").strip() if tag else ""


def site_identity(soup: BeautifulSoup) -> Dict[str, Any]:
    """The name and line the site gives itself, each with where it came from."""
    title = soup.title.get_text(" ", strip=True) if soup.title else ""
    h1 = soup.find("h1")
    names = [(v, s) for v, s in ((_meta(soup, "og:site_name"), "og:site_name"),
                                 (_meta(soup, "application-name"), "application-name"), (title, "<title>")) if v]
    lines = [(v, s) for v, s in ((_meta(soup, "description"), "meta description"),
                                 (_meta(soup, "og:description"), "og:description"),
                                 (h1.get_text(" ", strip=True) if h1 else "", "the first heading")) if v]
    return {"names": [{"value": v, "found_in": s} for v, s in names],
            "taglines": [{"value": v, "found_in": s} for v, s in lines]}


def _profile_handle(network: str, path: str) -> Optional[str]:
    parts = [p for p in path.split("/") if p]
    if network == "linkedin":
        return parts[1] if len(parts) >= 2 and parts[0] in ("company", "in", "school") else None
    if network == "youtube" and parts and parts[0] in ("c", "channel", "user") and len(parts) >= 2:
        return parts[1]
    if not parts or parts[0] in _NOT_A_PROFILE:
        return None
    return parts[0].lstrip("@") or None


def social_links(soup: BeautifulSoup, base_url: str) -> Dict[str, Dict[str, str]]:
    """The brand's own profile per network, from the links on the page (the footer's included)."""
    found: Dict[str, Dict[str, str]] = {}
    for anchor in soup.find_all("a", href=True):
        url = urljoin(base_url, anchor["href"])
        parsed = urlparse(url)
        host = parsed.netloc.lower().split(":")[0]
        network = next((name for name, pattern in _SOCIAL_HOSTS if pattern.search(host)), None)
        if network is None or network in found:
            continue
        handle = _profile_handle(network, parsed.path)
        if handle:
            found[network] = {"handle": handle, "url": url}
    return found


# ── the page ───────────────────────────────────────────────────────────────────────────────────

def stylesheet_links(soup: BeautifulSoup, base_url: str) -> Tuple[List[str], List[str]]:
    """(stylesheets to read, Google Fonts links) from the page's ``<link rel="stylesheet">``s, in order."""
    sheets: List[str] = []
    google: List[str] = []
    for link in soup.find_all("link", href=True):
        rel = " ".join(link.get("rel") or []).lower()
        if "stylesheet" not in rel and "preload" not in rel:
            continue
        url = urljoin(base_url, link["href"])
        if urlparse(url).netloc.endswith("fonts.googleapis.com"):
            google.append(url)
        elif "stylesheet" in rel and urlparse(url).scheme in ("http", "https") and url not in sheets:
            sheets.append(url)
    return sheets, google


def _theme_colours(soup: BeautifulSoup) -> List[Tuple[str, str]]:
    keys = ("theme-color", "msapplication-TileColor")
    return [(f'<meta name="{k}">', _meta(soup, k)) for k in keys if _meta(soup, k)]


def _inline_css(soup: BeautifulSoup) -> str:
    blocks = [tag.get_text() for tag in soup.find_all("style")]
    attrs = [f"[style]{{{tag['style']}}}" for tag in soup.find_all(style=True)]
    return "\n".join(blocks + attrs)


def _thin_page(soup: BeautifulSoup) -> bool:
    text = " ".join(soup.get_text(" ", strip=True).split())
    return len(text) < THIN_PAGE_CHARS and soup.find("script") is not None


def site_brand(html: str, base_url: str, stylesheets: Mapping[str, str], skipped: Sequence[str] = ()) -> Dict[str, Any]:
    """The brand evidence in a page and the stylesheets read for it (``{url: css}``).

    Every value says where it was found. ``skipped`` names the stylesheets that were not read.
    """
    soup = BeautifulSoup(html, "html.parser")
    notes: List[str] = []
    _, google = stylesheet_links(soup, base_url)
    css = without_dark_and_comments("\n".join([_inline_css(soup), *stylesheets.values()]))
    if _thin_page(soup):
        notes.append("The page has almost no text without JavaScript, so it may show only part of the brand.")
    if skipped:
        notes.append(f"{len(skipped)} stylesheet(s) were not read: " + ", ".join(skipped[:5]))
    return {
        "colours": site_colours(_theme_colours(soup), css),
        "fonts": site_fonts(css, google),
        "logos": logo_candidates(soup, base_url, notes),
        "marks": mark_candidates(soup, base_url),
        **site_identity(soup),
        "social": social_links(soup, base_url),
        "stylesheets_read": list(stylesheets),
        "notes": notes,
    }


__all__ = ["colours_in", "custom_properties", "google_font_families", "is_neutral", "resolve_vars",
           "site_brand", "site_colours", "site_fonts", "stylesheet_links"]
