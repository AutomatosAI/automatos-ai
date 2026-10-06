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

F356 (5 Oct): every size, gap and colour comes from one design system
(``design_tokens``). Tables have zebra rows, hairlines and tabular figures. A
paragraph left empty (an optional line whose chips had nothing) takes no space.
The starters' own rules, keyed on their block ids, are ``page_starters``; the
legacy Jinja starters print with this same sheet (``legacy_jinja``'s
``document_style`` filter).

PRD-255 (US-004): that design system is the kit's. Headings and the title print in
``heading`` (near-black), the title over a short rule in the accent; body text in
``ink`` on ``paper``; a table header on ``surface_2`` in ``heading`` (on the accent
only when ``accent_use`` is ``bold``), zebra rows on ``surface``, hairlines in
``rule``; links in the accent. Sizes, line heights and weights are the kit's
``type_scale``; gaps are multiples of ``spacing_unit_pt``, the page margins
``page_margin_mm``; the letterhead logo is ``logo_rules.letterhead_mm`` high with
its clear space beside it.

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
from .brand_board_style import BOARD, BOARD_TOKENS
from .page_starters import STARTER_RULES

DEFAULT_FONT = "Inter, 'Segoe UI', system-ui, sans-serif"
# A section whose rendered HTML is at most this long is kept on one page.
KEEP_TOGETHER_MAX_HTML_CHARS = 1600
KEEP_CLASS = "keep"
# The body's line height is printed as a ratio of its size, so smaller text inherits a proportional one.
LEADING_DECIMALS = 3

_BASE = Template("""
  @page { size: A4; margin: ${margin}mm; font-family: $font; background: $paper;
    @bottom-left { content: "$footer_name"; font-size: ${caption}pt; color: $muted; vertical-align: middle; }
    @bottom-center { content: string(doctitle, first); font-size: ${caption}pt; color: $muted; vertical-align: middle; }
    @bottom-right { content: "Page " counter(page) " of " counter(pages); font-size: ${caption}pt; color: $muted;
      vertical-align: middle; }
  }
  body { font-family: $font; color: $ink; font-size: ${body}pt; line-height: $body_leading; font-weight: $body_weight; }
  h1, h2, h3, h4, h5, h6 { color: $heading; margin: 0; break-after: avoid; }
  h1 { font-size: ${h1}pt; line-height: ${h1_line}pt; font-weight: $h1_weight; margin: ${s2}pt 0 ${s2}pt 0;
    string-set: doctitle content(); }
  h1::after { content: ""; display: block; width: ${title_rule_length}pt; border-top: ${title_rule_pt}pt solid $accent;
    margin: ${s1}pt 0 0 0; }
  h2 { font-size: ${h2}pt; line-height: ${h2_line}pt; font-weight: $h2_weight; margin: ${s5}pt 0 ${s1}pt 0; }
  h3 { font-size: ${h3}pt; line-height: ${h3_line}pt; font-weight: $h3_weight; margin: ${s4}pt 0 ${s1}pt 0; }
  h4, h5, h6 { font-size: ${body}pt; font-weight: $h3_weight; margin: ${s3}pt 0 ${s1}pt 0; }
  p { margin: 0 0 ${s2}pt 0; orphans: 2; widows: 2; }
  p:empty { display: none; }
  strong { font-weight: $bold; }
  a { color: $accent_text; }
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
  .doc-table th { background: $header_fill; color: $header_text; font-weight: $h3_weight; font-size: ${small}pt;
    line-height: ${small_line}pt; text-align: left; padding: ${s1}pt ${s2}pt; }
  .doc-table td { border-bottom: ${hairline_pt}pt solid $rule; padding: ${s1}pt ${s2}pt; vertical-align: top; }
  .doc-table tbody tr:nth-child(even) td { background: $surface; }
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


def _type_tokens(design: t.Design) -> Dict[str, Any]:
    """Each step's size (``h1``), line height (``h1_line``; ``h1_leading`` as a ratio) and weight (``h1_weight``)."""
    found: Dict[str, Any] = {}
    for name, step in design.type.items():
        found.update({name: step.size_pt, f"{name}_line": step.line_pt, f"{name}_weight": step.weight,
                      f"{name}_leading": round(step.line_pt / step.size_pt, LEADING_DECIMALS)})
    return found


def _layout_tokens(design: t.Design) -> Dict[str, Any]:
    """The gaps (``s1``..``s6``), rules, page margins and the letterhead logo's size."""
    return {
        **{f"s{step}": design.space(step) for step in range(1, len(t.SPACE_STEPS) + 1)},
        "hairline_pt": t.HAIRLINE_PT, "rule_pt": t.RULE_PT, "title_rule_pt": t.TITLE_RULE_PT,
        "title_rule_length": design.spacing_unit_pt * t.TITLE_RULE_UNITS,
        "margin": design.page_margin_mm,
        "logo_mm": design.logo_mm, "logo_clear_mm": design.logo_clear_mm,
    }


def style_tokens(brand_kit: Dict[str, Any]) -> Dict[str, Any]:
    """Every value the sheets substitute: the kit's colour roles, type scale, spacing and logo. Pure."""
    bk = brand_kit or {}
    design = t.design(bk)
    return {
        **asdict(design.palette),
        **_type_tokens(design),
        **_layout_tokens(design),
        **BOARD_TOKENS,
        "font": font_stack(bk.get("font_family")),
        "footer_name": css_string(footer_name(bk)),
        "keep": KEEP_CLASS,
        "bold": t.BOLD,
    }


def build_styles(brand_kit: Dict[str, Any]) -> str:
    """The stylesheet for a block document printed under ``brand_kit``. Pure."""
    tokens = style_tokens(brand_kit)
    return "".join(sheet.substitute(tokens) for sheet in (_BASE, _TABLES, *STARTER_RULES, BOARD))


__all__ = [
    "DEFAULT_FONT", "KEEP_CLASS", "KEEP_TOGETHER_MAX_HTML_CHARS", "build_styles", "css_string", "font_stack",
    "footer_name", "style_tokens",
]
