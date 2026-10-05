"""The stylesheet a block document prints with (PRD-167 S2; F350 and F356 design passes).

F350 (night 10b, 5 Oct): the Branded starters printed as a stack of same-size
lines under a logo that took the top fifth of the page, with no footer, no page
numbers, the invoice totals in the middle of the page, number columns wrapping
("£23.00/" over "kg"), and the brand kit's second colour nowhere. This sheet
gives every block document:

* a footer on every page: the company's name, the document's title (its first
  ``h1``: "Invoice INV-0042" on each page of a long invoice) and "Page X of Y",
  printed in WeasyPrint's page margin boxes;
* number columns (a data table's right- or centre-aligned columns) that keep to
  one line and take only their content's width, so text columns take the rest;
* rows that never split across pages, headings kept with what follows them, and
  short sections kept on one page (``KEEP_CLASS``, set by the renderer).

F356 (5 Oct): every size, gap and colour now comes from one design system
(``design_tokens``): a six-step type scale in two weights, a 4 pt spacing grid,
the kit's primary for the title and table headers, its accent for section
headings and rules, its text colour for body text. Tables have zebra rows,
hairlines and tabular figures. A paragraph left empty (an optional line whose
chips had nothing) takes no space. The starters' own rules, keyed on their block
ids, are ``page_starters``; the legacy Jinja starters print with this same sheet
(``legacy_jinja``'s ``document_style`` filter).

Brand strings land inside ``<style>``, where HTML entities are NOT decoded:

* colours are hex, normalised by ``design_tokens.palette``;
* the font stack is printed as it is when it is one safe CSS value
  (``core.media_render_bundle.TOKEN_UNSAFE``), else the default stack. It used to
  be HTML-escaped, so the default ``Inter, 'Segoe UI', …`` printed as
  ``Inter, &#x27;Segoe UI&#x27;…``: the ``;`` of ``&#x27;`` ended the declaration
  and every block PDF fell back to the viewer's serif font;
* the footer's company name is a CSS string, escaped by :func:`css_string`
  (every character but letters, digits and spaces as a CSS hex escape, so no
  quote, backslash or ``</style>`` survives).
"""

from __future__ import annotations

from dataclasses import asdict
from string import Template
from typing import Any, Dict

from core.media_render_bundle import MAX_TOKEN_CHARS, TOKEN_UNSAFE

from . import design_tokens as t
from .page_starters import STARTER_RULES

DEFAULT_FONT = "Inter, 'Segoe UI', system-ui, sans-serif"
# A section whose rendered HTML is at most this long is kept on one page.
KEEP_TOGETHER_MAX_HTML_CHARS = 1600
KEEP_CLASS = "keep"

_BASE = Template("""
  @page { size: A4; margin: 20mm 20mm 22mm 20mm; font-family: $font;
    @bottom-left { content: "$footer_name"; font-size: ${caption}pt; color: $muted; vertical-align: top; }
    @bottom-center { content: string(doctitle, first); font-size: ${caption}pt; color: $muted; vertical-align: top; }
    @bottom-right { content: "Page " counter(page) " of " counter(pages); font-size: ${caption}pt; color: $muted;
      vertical-align: top; }
  }
  body { font-family: $font; color: $text; line-height: $leading; font-size: ${body}pt; font-weight: $regular; }
  h1, h2, h3, h4, h5, h6 { color: $heading; font-weight: $bold; line-height: 1.2; margin: 0; break-after: avoid; }
  h1 { color: $title; font-size: ${title_pt}pt; margin: ${s2}pt 0 ${s1}pt 0; string-set: doctitle content(); }
  h2 { font-size: ${h2}pt; margin: ${s5}pt 0 ${s2}pt 0; }
  h3 { font-size: ${h3}pt; margin: ${s4}pt 0 ${s1}pt 0; }
  h4, h5, h6 { font-size: ${body}pt; margin: ${s3}pt 0 ${s1}pt 0; }
  p { margin: 0 0 ${s2}pt 0; orphans: 2; widows: 2; }
  p:empty { display: none; }
  strong { font-weight: $bold; }
  ul, ol { margin: 0 0 ${s2}pt 0; padding-left: ${s4}pt; }
  li { margin: 0 0 ${s1}pt 0; }
  .doc-section { margin-bottom: ${s3}pt; }
  .doc-section > h2:first-child { margin-top: ${s4}pt; }
  .doc-section p { white-space: pre-line; }
  .doc-section.$keep { break-inside: avoid; }
  .doc-image { display: block; margin: 0 0 ${s3}pt 0; }
  .doc-empty { color: $muted; font-style: italic; }
  .doc-empty:empty { display: none; }
  .page-break { page-break-after: always; }
  .unresolved-var { color: #b00020; background: #fde7ea; padding: 0 2px; border-radius: 2px; }
""")

_TABLES = Template("""
  .doc-table { border-collapse: collapse; width: 100%; margin: ${s2}pt 0 ${s4}pt 0; font-variant-numeric: tabular-nums; }
  .doc-table th { background: $header_fill; color: $header_text; font-weight: $bold; font-size: ${small}pt;
    text-align: left; padding: ${s1}pt ${s2}pt; }
  .doc-table td { border-bottom: ${hairline_pt}pt solid $hairline; padding: ${s1}pt ${s2}pt; vertical-align: top; }
  .doc-table tbody tr:nth-child(even) td { background: $zebra; }
  .doc-table tr { break-inside: avoid; }
  .doc-table th[style*="text-align:right"], .doc-table td[style*="text-align:right"],
  .doc-table th[style*="text-align:center"], .doc-table td[style*="text-align:center"] { white-space: nowrap; width: 1%; }
""")


def css_string(value: str) -> str:
    """``value`` safe inside a double-quoted CSS string: every character but letters,
    digits and spaces as a CSS hex escape (``\\26 `` for "&"). Pure."""
    return "".join(ch if ch.isalnum() and ch.isascii() or ch == " " else f"\\{ord(ch):x} " for ch in value)


def font_stack(value: Any) -> str:
    """``value`` when it is one safe CSS font stack, else the default stack. Pure."""
    text = str(value or "").strip()
    if not text or len(text) > MAX_TOKEN_CHARS or TOKEN_UNSAFE.search(text):
        return DEFAULT_FONT
    return text


def footer_name(brand_kit: Dict[str, Any]) -> str:
    """The name the footer prints: the company contact's, else the brand's; empty without a kit."""
    company = (brand_kit or {}).get("company") or {}
    return str(company.get("name") or (brand_kit or {}).get("name") or "").strip()


def style_tokens(brand_kit: Dict[str, Any]) -> Dict[str, Any]:
    """Every value the sheets substitute: the kit's colour roles and the type and spacing scales. Pure."""
    bk = brand_kit or {}
    return {
        **asdict(t.palette(bk)),
        "font": font_stack(bk.get("font_family")),
        "footer_name": css_string(footer_name(bk)),
        "keep": KEEP_CLASS,
        "title_pt": t.TITLE_PT, "h2": t.H2_PT, "h3": t.H3_PT, "body": t.BODY_PT, "small": t.SMALL_PT,
        "caption": t.CAPTION_PT, "regular": t.REGULAR, "bold": t.BOLD, "leading": t.LEADING,
        "s1": t.SPACE_1, "s2": t.SPACE_2, "s3": t.SPACE_3, "s4": t.SPACE_4, "s5": t.SPACE_5, "s6": t.SPACE_6,
        "hairline_pt": t.HAIRLINE_PT, "rule_pt": t.RULE_PT,
    }


def build_styles(brand_kit: Dict[str, Any]) -> str:
    """The stylesheet for a block document printed under ``brand_kit``. Pure."""
    tokens = style_tokens(brand_kit)
    return "".join(sheet.substitute(tokens) for sheet in (_BASE, _TABLES, *STARTER_RULES))


__all__ = [
    "DEFAULT_FONT", "KEEP_CLASS", "KEEP_TOGETHER_MAX_HTML_CHARS", "build_styles", "css_string", "font_stack",
    "footer_name", "style_tokens",
]
