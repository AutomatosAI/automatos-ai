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

F360 (night 10c): a kit that names Geist and Newsreader with no font files still
printed in DejaVu, and said nothing. The body's and the headings' stacks are now
resolved (``bundled_fonts.font_uses``): a family the code ships (Inter, Geist,
Newsreader) is added as ``@font-face`` rules from its bundled woff2 files unless
the kit uploaded that family itself, and when a role still prints in another
family than the kit names, the first page says so in its top-right margin
("Substitute font: DejaVu Sans for Brand Sans"), in the caption size and the muted
colour. That line is printed only when the PDF really falls back.

Pure: the kit in, CSS out (the bundled files are read once, and cached).
"""
from __future__ import annotations

import re
from typing import Any, List, Mapping

from core.media_render_bundle import FONT_FAMILY, FONT_STYLES

from ..brand_kit import FONT_WEIGHT_VALUES
from ..bundled_fonts import FontUse, bundled_faces, font_uses, substitutes
from . import design_tokens as t
from .page_style import css_string, font_stack

WOFF2_DATA_URI = re.compile(r"^data:font/woff2;base64,[A-Za-z0-9+/]+={0,2}$")
HEADINGS = "h1, h2, h3, h4, h5, h6"
SUBSTITUTE_LABEL = "Substitute font"
# The brand board's heading-step type samples: set in the headings' font too (F360).
BOARD_HEADING_SAMPLE = "board-type-heading"


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


def _uploaded(kit: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """The kit's uploaded faces that are usable (a valid family, weight, style and woff2 data: URI)."""
    return [font for font in kit.get("font_files") or [] if _face(font)]


def kit_font_uses(kit: Mapping[str, Any]) -> List[FontUse]:
    """The body's and (when set) the headings' font: the family named and the one the PDF prints in."""
    uploaded = {str(font["family"]).casefold() for font in _uploaded(kit)}
    heading = kit.get("heading_font")
    return font_uses(font_stack(kit.get("font_family")), font_stack(heading) if heading else None, uploaded)


def substitute_note(uses: List[FontUse]) -> str:
    """"Substitute font: DejaVu Sans for Brand Sans"; ``""`` when every role prints in the font it names."""
    found = substitutes(uses)
    if not found:
        return ""
    return f"{SUBSTITUTE_LABEL}: " + "; ".join(f"{use.prints_in} for {use.named}" for use in found)


def _substitute_mark(kit: Mapping[str, Any], note: str) -> str:
    """The first page's top-right margin line naming the substitute, in the caption size and muted colour."""
    if not note:
        return ""
    design = t.design(kit)
    return (
        f'\n  @page :first {{ @top-right {{ content: "{css_string(note)}"; '
        f"font-size: {design.type['caption'].size_pt}pt; color: {design.palette.muted}; vertical-align: middle; }} }}"
    )


def font_css(kit: Mapping[str, Any]) -> str:
    """The kit's uploaded faces, the bundled faces its stacks resolve to, the headings' font when it
    names one, and the substitute-font line when a role really falls back, as CSS rules."""
    uses = kit_font_uses(kit)
    faces: List[str] = [_face(font) for font in [*_uploaded(kit), *bundled_faces(uses)]]
    heading = kit.get("heading_font")
    rule = (f"\n  {HEADINGS} {{ font-family: {font_stack(heading)}; }}"
            f"\n  .{BOARD_HEADING_SAMPLE} {{ font-family: {font_stack(heading)}; }}\n") if heading else "\n"
    return "".join(faces) + _substitute_mark(kit, substitute_note(uses)) + rule


__all__ = ["BOARD_HEADING_SAMPLE", "SUBSTITUTE_LABEL", "font_css", "kit_font_uses", "substitute_note"]
