"""The stylesheet a block document prints with (PRD-167 S2; F350 design pass).

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
  short sections kept on one page (``KEEP_CLASS``, set by the renderer);
* the kit's accent colour (the owner's navy) on the rules: under the letterhead,
  under a report's or proposal's byline, above an invoice's total, beside a
  report's executive summary.

The starters' blocks carry stable ids (``modules/documents/presets.py``); the
renderer prints each block's id as ``data-block`` and the starter rules below
style them by it. A copy of a starter keeps its ids, so it keeps the look; a
block a person adds has an id of its own and prints with the plain rules.

Brand strings land inside ``<style>``, where HTML entities are NOT decoded:

* colours are hex, validated by the brand kit;
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

import html
from string import Template
from typing import Any, Dict

from core.media_render_bundle import MAX_TOKEN_CHARS, TOKEN_UNSAFE

DEFAULT_PRIMARY = "#1a1a2e"
DEFAULT_SECONDARY = "#16213e"
DEFAULT_ACCENT = "#0f3460"
DEFAULT_TEXT = "#1a1a2e"
DEFAULT_FONT = "Inter, 'Segoe UI', system-ui, sans-serif"
# Hex alpha suffixes: the soft row rules and muted print (footer, contact line).
RULE_ALPHA = "33"
MUTED_ALPHA = "b3"
TINT_ALPHA = "0d"
# A section whose rendered HTML is at most this long is kept on one page.
KEEP_TOGETHER_MAX_HTML_CHARS = 1600
KEEP_CLASS = "keep"

_BASE = Template("""
  @page { size: A4; margin: 2cm 2cm 2.2cm 2cm; font-family: $font;
    @bottom-left { content: "$footer_name"; font-size: 8pt; color: $text$muted; }
    @bottom-center { content: string(doctitle, first); font-size: 8pt; color: $text$muted; }
    @bottom-right { content: "Page " counter(page) " of " counter(pages); font-size: 8pt; color: $text$muted; }
  }
  body { font-family: $font; color: $text; line-height: 1.55; font-size: 10.5pt; }
  h1, h2, h3, h4, h5, h6 { color: $primary; margin: 1.3rem 0 0.4rem 0; line-height: 1.25; break-after: avoid; }
  h1 { font-size: 21pt; margin-top: 0.4rem; string-set: doctitle content(); }
  h2 { font-size: 13.5pt; }
  h3 { font-size: 12pt; }
  h4, h5, h6 { font-size: 10.5pt; }
  p { margin: 0.45rem 0; orphans: 2; widows: 2; }
  .doc-section { margin-bottom: 1.1rem; }
  .doc-section p { white-space: pre-line; }
  .doc-section.$keep { break-inside: avoid; }
  .doc-image { display: block; margin: 0 0 0.8rem 0; }
  .doc-empty { color: $secondary; font-style: italic; }
  .page-break { page-break-after: always; }
  .unresolved-var { color: #b00020; background: #fde7ea; padding: 0 2px; border-radius: 2px; }
""")

_TABLES = Template("""
  .doc-table { border-collapse: collapse; width: 100%; margin: 0.75rem 0 1rem 0; }
  .doc-table th { background: $primary; color: #fff; text-align: left; padding: 0.4rem 0.65rem; font-size: 9.5pt; }
  .doc-table td { border-bottom: 1px solid $secondary$rule; padding: 0.4rem 0.65rem; vertical-align: top; }
  .doc-table tr { break-inside: avoid; }
  .doc-table th[style*="text-align:right"], .doc-table td[style*="text-align:right"],
  .doc-table th[style*="text-align:center"], .doc-table td[style*="text-align:center"] { white-space: nowrap; width: 1%; }
""")

# The starters' own blocks, by id (presets.py): letterhead, letter, invoice, report.
_STARTERS = Template("""
  [data-block="lh-name"] { margin: 0.2rem 0 0.15rem 0; font-size: 13pt; }
  [data-block="lh-address"], [data-block="lh-contact"] { margin: 0; font-size: 9pt; color: $text$muted; }
  [data-block="lh-contact"], [data-block="byline"], [data-block="cover"] {
    padding-bottom: 0.6rem; border-bottom: 1.5pt solid $accent; margin-bottom: 1.3rem; }
  [data-block="byline"], [data-block="cover"] { color: $text$muted; }
  [data-block="date"] { text-align: right; margin: 0 0 1.1rem 0; }
  [data-block="to-name"], [data-block="to-company"], [data-block="to-address"] { margin: 0; line-height: 1.4; }
  [data-block="to-name"] { font-weight: 600; }
  [data-block="subject"] { font-weight: 600; margin: 1.4rem 0 1rem 0; }
  [data-block="greeting"] { margin: 0 0 0.6rem 0; }
  [data-block="body"] { white-space: pre-line; }
  [data-block="closing"] { margin-top: 1.4rem; }
  [data-block="sig-name"] { font-weight: 600; margin: 2rem 0 0 0; }
  [data-block="sig-email"] { margin: 0; color: $text$muted; }
  [data-block="meta"] { color: $text$muted; margin-top: 0; }
  [data-block="bill-to-label"] { margin: 1rem 0 0.1rem 0; font-size: 8.5pt; letter-spacing: 0.06em; color: $accent; }
  [data-block="bill-to"], [data-block="bill-to-address"], [data-block="bill-to-email"] { margin: 0; line-height: 1.4; }
  [data-block="bill-to"] { font-weight: 600; }
  [data-block="totals"] { width: auto; min-width: 50%; margin: 0 0 1.2rem auto; break-inside: avoid; }
  [data-block="totals"] td:last-child { text-align: right; white-space: nowrap; }
  [data-block="totals"] tr:last-child td { font-weight: 700; font-size: 11.5pt; border-top: 1.5pt solid $accent; border-bottom: none; }
  [data-block="s-summary"] { border-left: 3pt solid $accent; background: $accent$tint; padding: 0.3rem 0.9rem; }
  [data-block="s-summary"] h2 { margin-top: 0.4rem; }
  [data-block="signatures"] { break-inside: avoid; }
  [data-block="footer"] { margin-top: 1.4rem; font-size: 9pt; color: $text$muted; }
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


def build_styles(brand_kit: Dict[str, Any]) -> str:
    """The stylesheet for a block document printed under ``brand_kit``. Pure."""
    bk = brand_kit or {}
    tokens = {
        "primary": html.escape(bk.get("primary_color") or DEFAULT_PRIMARY, quote=True),
        "secondary": html.escape(bk.get("secondary_color") or DEFAULT_SECONDARY, quote=True),
        "accent": html.escape(bk.get("accent_color") or DEFAULT_ACCENT, quote=True),
        "text": html.escape(bk.get("text_color") or DEFAULT_TEXT, quote=True),
        "font": font_stack(bk.get("font_family")),
        "footer_name": css_string(footer_name(bk)),
        "rule": RULE_ALPHA, "muted": MUTED_ALPHA, "tint": TINT_ALPHA, "keep": KEEP_CLASS,
    }
    return "".join(sheet.substitute(tokens) for sheet in (_BASE, _TABLES, _STARTERS))


__all__ = ["KEEP_CLASS", "KEEP_TOGETHER_MAX_HTML_CHARS", "build_styles", "css_string", "font_stack", "footer_name"]
