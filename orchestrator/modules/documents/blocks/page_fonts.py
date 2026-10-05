"""The brand kit's uploaded fonts and heading font on a block document's page (F347, night 10b).

F347: every Branded PDF came out in DejaVu Serif, whatever font the kit named.
The kit's font stack was HTML-escaped into the stylesheet; F350
(``page_style.font_stack``) now writes it as CSS. Two of the kit's fonts still
never reached the PDF, and this sheet, printed after ``page_style``'s, adds them:

* the kit's uploaded woff2 files (``font_files``, inlined as data: URIs by
  ``brand_fonts.brand_kit_for_media_render``) as ``@font-face`` rules, so a kit
  font that is uploaded reaches the PDF though the server has no such font
  (the image carries DejaVu only);
* the headings' font (``heading_font``), checked as one CSS value the way the
  body font is; without one, headings keep the body font.

A face whose family, weight or style is not one the kit accepts, or whose file
is not a base64 woff2 data: URI, is left out: nothing here reaches the network
or can end the ``<style>``.

Pure: the kit in, CSS out.
"""
from __future__ import annotations

import re
from typing import Any, List, Mapping

from core.media_render_bundle import FONT_FAMILY, FONT_STYLES

from ..brand_kit import FONT_WEIGHT_VALUES
from .page_style import font_stack

WOFF2_DATA_URI = re.compile(r"^data:font/woff2;base64,[A-Za-z0-9+/]+={0,2}$")
HEADINGS = "h1, h2, h3, h4, h5, h6"


def _face(font: Any) -> str:
    """One uploaded face as an ``@font-face`` rule; empty for anything that is not a usable woff2 face."""
    if not isinstance(font, Mapping):
        return ""
    family, weight, style, uri = (font.get(key) for key in ("family", "weight", "style", "data_uri"))
    usable = (
        isinstance(family, str) and FONT_FAMILY.match(family) and weight in FONT_WEIGHT_VALUES
        and style in FONT_STYLES and isinstance(uri, str) and WOFF2_DATA_URI.match(uri)
    )
    if not usable:
        return ""
    return (
        f'\n  @font-face {{ font-family: "{family}"; src: url("{uri}") format("woff2"); '
        f"font-weight: {weight}; font-style: {style}; }}"
    )


def font_css(kit: Mapping[str, Any]) -> str:
    """The kit's uploaded faces and, when it names one, the headings' font, as CSS rules."""
    faces: List[str] = [face for face in map(_face, kit.get("font_files") or []) if face]
    heading = kit.get("heading_font")
    rule = f"\n  {HEADINGS} {{ font-family: {font_stack(heading)}; }}\n" if heading else "\n"
    return "".join(faces) + rule


__all__ = ["font_css"]
