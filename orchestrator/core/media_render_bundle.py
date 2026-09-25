"""The render bundle media-render takes, built from a social template and the brand kit.

PRD-251 S1.2 (D3, D4, D5). One builder for both callers: a Socials post's render
(``modules/socials/render.py``) and ``generate_document`` with a social format
(``modules/documents/generation_service.py``). It lives in core because those two
feature modules may not import each other (``orchestrator/.importlinter``). Each
caller hands in the workspace brand kit render-ready, an uploaded logo already
inlined as a data: URI (``modules/documents/brand_logo.brand_kit_for_render``).

The brand kit becomes:

* ``brand.tokens``: its colours and fonts. media-render declares each as a
  ``--brand-<name>`` custom property, so a template reads ``var(--brand-primary)``
  or ``var(--brand-heading-font)`` and never names a colour or a font (D4);
* ``files`` and ``brand.fonts``: an uploaded logo at ``assets/brand/logo.<ext>``,
  and the kit's font files (D5 ``font_files``) under ``assets/brand/fonts/``, each
  with its ``@font-face``;
* the variables ``brand.name``, ``brand.tagline`` and ``brand.logo`` (the staged
  logo's path, or a transparent pixel when there is no uploaded logo), and
  ``size.width`` / ``size.height`` for the size being rendered.

An external ``logo_url`` is never fetched: a render reads only the files its
bundle carries and our own storage (D9), so a logo reaches a render once it is
uploaded. Nothing here generates anything: the renderer assembles (D3).
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Mapping, Optional, Tuple

from core.social_templates import parse_size

logger = logging.getLogger(__name__)

# Token name → the brand kit field it reads.
COLOUR_TOKENS = (
    ("primary", "primary_color"),
    ("secondary", "secondary_color"),
    ("accent", "accent_color"),
    ("text", "text_color"),
)
BODY_FONT_TOKEN = "body-font"
HEADING_FONT_TOKEN = "heading-font"
# media-render's rule for a token value (services/media-render/media_render/bundle.py):
# one CSS value, nothing that ends a declaration, opens a block or fetches.
TOKEN_UNSAFE = re.compile(r"[;{}<>\\\n\r]|/\*|url\(|@import|expression\(", re.IGNORECASE)
MAX_TOKEN_CHARS = 200

BRAND_DIR = "assets/brand/"
FONTS_DIR = "assets/brand/fonts/"
LOGO_EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg"}
FONT_EXTENSIONS = {
    "font/woff2": "woff2",
    "application/font-woff2": "woff2",
    "font/woff": "woff",
    "application/font-woff": "woff",
    "font/ttf": "ttf",
    "font/otf": "otf",
}
FONT_FAMILY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,63}$")
FONT_WEIGHTS = frozenset({"normal", "bold"} | {str(weight) for weight in range(100, 1000, 100)})
FONT_STYLES = frozenset({"normal", "italic"})
_DATA_URI = re.compile(r"^data:([a-z0-9.+/-]+);base64,", re.IGNORECASE)
# What {{ brand.logo }} fills in when the kit has no uploaded logo: a 1x1 transparent GIF.
NO_LOGO = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"

VAR_BRAND_NAME = "brand.name"
VAR_BRAND_TAGLINE = "brand.tagline"
VAR_BRAND_LOGO = "brand.logo"
VAR_SIZE_WIDTH = "size.width"
VAR_SIZE_HEIGHT = "size.height"


def _token(name: str, value: Any) -> Optional[str]:
    text = value.strip() if isinstance(value, str) else ""
    if not text:
        return None
    if len(text) > MAX_TOKEN_CHARS or TOKEN_UNSAFE.search(text):
        logger.warning("[MediaRender] brand kit %s is not a single CSS value; the template's fallback applies", name)
        return None
    return text


def brand_tokens(kit: Mapping[str, Any]) -> Dict[str, str]:
    """The kit's colours and fonts as the ``--brand-*`` tokens a template reads."""
    body_font = kit.get("font_family")
    raw: Dict[str, Any] = {token: kit.get(field) for token, field in COLOUR_TOKENS}
    raw[BODY_FONT_TOKEN] = body_font
    # D5: the heading font is optional; without one, headings take the body font.
    raw[HEADING_FONT_TOKEN] = kit.get("heading_font") or body_font
    tokens = {name: _token(name, value) for name, value in raw.items()}
    return {name: value for name, value in tokens.items() if value is not None}


def _data_uri_type(value: Any) -> Optional[str]:
    match = _DATA_URI.match(value) if isinstance(value, str) else None
    return match.group(1).lower() if match else None


def _logo(kit: Mapping[str, Any]) -> Tuple[List[Dict[str, str]], str]:
    """The uploaded logo as a bundle file, and what ``{{ brand.logo }}`` fills in."""
    uri = kit.get("logo_url")
    ext = LOGO_EXTENSIONS.get(_data_uri_type(uri) or "")
    if ext is None:
        return [], NO_LOGO
    path = f"{BRAND_DIR}logo.{ext}"
    return [{"path": path, "data_uri": uri}], path


def _fonts(kit: Mapping[str, Any]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """The kit's font files (D5 ``font_files``, render-ready as data: URIs): bundle files and faces."""
    files: List[Dict[str, str]] = []
    faces: List[Dict[str, str]] = []
    for i, font in enumerate(kit.get("font_files") or []):
        font = font if isinstance(font, Mapping) else {}
        ext = FONT_EXTENSIONS.get(_data_uri_type(font.get("data_uri")) or "")
        family = font.get("family")
        weight, style = str(font.get("weight", 400)), str(font.get("style", "normal"))
        usable = isinstance(family, str) and FONT_FAMILY.match(family) and weight in FONT_WEIGHTS and style in FONT_STYLES
        if ext is None or not usable:
            logger.warning("[MediaRender] brand kit font file %d is not a usable font; skipped", i)
            continue
        path = f"{FONTS_DIR}font-{i}.{ext}"
        files.append({"path": path, "data_uri": font["data_uri"]})
        faces.append({"family": family, "weight": weight, "style": style, "path": path})
    return files, faces


def brand_name(kit: Mapping[str, Any], fallback: str = "") -> str:
    """The name a template shows: the kit's, its company's, else ``fallback`` (the workspace's)."""
    company = kit.get("company") if isinstance(kit.get("company"), Mapping) else {}
    return kit.get("name") or company.get("name") or fallback or ""


def render_size(blocks: Mapping[str, Any], size: Optional[str] = None) -> Tuple[int, int]:
    """``(width, height)`` of ``size``, one the template declares; its first size by default."""
    sizes = list(blocks.get("sizes") or [])
    chosen = size if size is not None else (sizes[0] if sizes else None)
    if chosen not in sizes:
        raise ValueError(f"size {chosen!r} is not one this template declares ({', '.join(map(str, sizes))})")
    return parse_size(chosen)


def build_bundle(
    *,
    workspace_id: Any,
    reference: str,
    blocks: Mapping[str, Any],
    values: Mapping[str, Any],
    brand_kit: Optional[Mapping[str, Any]],
    fallback_name: str = "",
    size: Optional[str] = None,
) -> Dict[str, Any]:
    """The bundle for one render of ``blocks`` (a checked social template) at ``size``.

    ``values`` are the template's own variables, already resolved
    (``core.social_templates.resolve_variables``); the brand and size variables
    are added here and always win over a same-named value.
    """
    kit = brand_kit or {}
    width, height = render_size(blocks, size)
    logo_files, logo = _logo(kit)
    font_files, faces = _fonts(kit)
    variables = {
        **dict(values),
        VAR_BRAND_NAME: brand_name(kit, fallback_name),
        VAR_BRAND_TAGLINE: kit.get("tagline") or "",
        VAR_BRAND_LOGO: logo,
        VAR_SIZE_WIDTH: width,
        VAR_SIZE_HEIGHT: height,
    }
    brand: Dict[str, Any] = {"tokens": brand_tokens(kit)}
    if faces:
        brand["fonts"] = faces
    bundle: Dict[str, Any] = {
        "workspace_id": str(workspace_id),
        "reference": reference,
        "composition": {"html": blocks["html"], "css": blocks.get("css") or ""},
        "variables": variables,
        "brand": brand,
    }
    if logo_files or font_files:
        bundle["files"] = logo_files + font_files
    audio = blocks.get("audio_plan")
    if isinstance(audio, dict) and audio:
        bundle["audio"] = audio
    return bundle


__all__ = [
    "NO_LOGO",
    "brand_name",
    "brand_tokens",
    "build_bundle",
    "render_size",
]
