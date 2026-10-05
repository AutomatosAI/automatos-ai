"""The starters' own rules, keyed on their block ids (F350; F356 design pass).

The starters' blocks carry stable ids (``modules/documents/presets.py``); the
renderer prints each block's id as ``data-block`` and these rules style them by
it. A copy of a starter keeps its ids, so it keeps the look; a block a person
adds has an id of its own and prints with the plain rules (``page_style``). The
legacy Jinja starters print the same ids, so they share the letterhead.

F356 (5 Oct): the letterhead logo sits beside the company block, not above it:
the renderer prints the letterhead as one row (``letterhead_run``), the logo on
the left, the company's name, address and contact line set right, and one rule
in the kit's accent under both. Every size, gap and colour is a
``design_tokens`` value, substituted by ``page_style.build_styles``.
"""
from __future__ import annotations

from string import Template

LETTERHEAD = Template("""
  .letterhead { display: table; width: 100%; border-bottom: ${rule_pt}pt solid $rule; padding-bottom: ${s3}pt;
    margin-bottom: ${s5}pt; }
  .lh-mark, .lh-company { display: table-cell; vertical-align: middle; }
  .lh-mark .doc-image { margin: 0; }
  .lh-company { text-align: right; }
  [data-block="lh-name"] { margin: 0; font-size: ${h3}pt; color: $heading; }
  [data-block="lh-address"], [data-block="lh-contact"] { margin: 0; font-size: ${small}pt; line-height: 1.4; color: $muted; }
""")

LETTER = Template("""
  [data-block="date"] { text-align: right; margin: 0 0 ${s4}pt 0; }
  [data-block="to-name"], [data-block="to-company"], [data-block="to-address"] { margin: 0; line-height: 1.4; }
  [data-block="to-name"] { font-weight: $bold; }
  [data-block="subject"] { font-weight: $bold; margin: ${s5}pt 0 ${s4}pt 0; }
  [data-block="greeting"] { margin: 0 0 ${s2}pt 0; }
  [data-block="body"] { white-space: pre-line; }
  [data-block="closing"] { margin-top: ${s5}pt; }
  [data-block="sig-name"] { font-weight: $bold; margin: ${s6}pt 0 0 0; }
  [data-block="sig-email"] { margin: 0; color: $muted; }
""")

INVOICE = Template("""
  [data-block="byline"], [data-block="meta"], [data-block="cover"] { color: $muted; font-size: ${small}pt; margin: 0 0 ${s4}pt 0; }
  [data-block="bill-to-label"] { margin: ${s4}pt 0 ${s1}pt 0; font-size: ${caption}pt; letter-spacing: 0.08em; color: $heading; }
  [data-block="bill-to"], [data-block="bill-to-address"], [data-block="bill-to-email"] { margin: 0; line-height: 1.4; }
  [data-block="bill-to"] { font-weight: $bold; }
  [data-block="bill-to-email"] { margin-bottom: ${s4}pt; }
  [data-block="totals"] { width: auto; min-width: 45%; margin: 0 0 ${s5}pt auto; break-inside: avoid; }
  [data-block="totals"] td { background: none !important; }
  [data-block="totals"] td:last-child { text-align: right; white-space: nowrap; }
  [data-block="totals"] tr:last-child td { font-weight: $bold; font-size: ${h3}pt; border-top: ${rule_pt}pt solid $rule; border-bottom: none; }
  [data-block="terms"] { margin-top: ${s4}pt; }
  [data-block="thanks"], [data-block="footer"] { margin-top: ${s4}pt; font-size: ${small}pt; color: $muted; }
""")

REPORT = Template("""
  [data-block="s-summary"] { border-left: 3pt solid $rule; background: $panel; padding: ${s1}pt ${s3}pt ${s1}pt ${s3}pt; }
  [data-block="s-summary"] h2 { margin-top: ${s2}pt; }
  [data-block="signatures"] { break-inside: avoid; }
""")

STARTER_RULES = (LETTERHEAD, LETTER, INVOICE, REPORT)

__all__ = ["STARTER_RULES"]
