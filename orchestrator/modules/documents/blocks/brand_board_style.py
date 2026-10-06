"""The brand board's own rules (PRD-255 US-009), keyed on its block ids and the parts' classes.

The board is one A4 page: the title, the logo row, the colour roles in one row of
swatches, the logo variants beside the voice, the type scale beside the spacing,
and the three applications. Its rows are sections (``board-row-*``) printed as
tables, which WeasyPrint lays out without surprises. Every colour, size and gap
is a ``design_tokens`` value (substituted by ``page_style.build_styles``); the
board's own sizes, in millimetres, are :data:`BOARD_TOKENS`.
"""
from __future__ import annotations

from string import Template

# The board's own sizes (mm): the logo row's logo, a variant's ground and the logo on it,
# a swatch's chip, the clear-space drawing's logo, and an application's frame.
BOARD_TOKENS = {
    "board_logo_mm": 18,
    "board_variant_mm": 16,
    "board_variant_logo_mm": 8,
    "board_chip_mm": 10,
    "board_chip_pad_mm": 1.5,
    "board_clear_logo_mm": 7,
    "board_app_mm": 42,
}

BOARD = Template("""
  .board-label { font-size: ${caption}pt; line-height: ${caption_line}pt; font-weight: $bold; letter-spacing: 0.1em;
    text-transform: uppercase; color: $heading; margin: 0 0 ${s1}pt 0; }
  .board-caption, .board-note { font-size: ${caption}pt; line-height: ${caption_line}pt; color: $muted; margin: 0; }
  .board-note { font-style: italic; }
  .board-part { margin: 0 0 ${s3}pt 0; }
  [data-block^="board-row"] { display: table; width: 100%; table-layout: fixed; margin: 0 0 ${s2}pt 0; }
  [data-block^="board-row"] > .board-part { display: table-cell; vertical-align: top; padding-right: ${s5}pt; }
  [data-block^="board-row"] > .board-part:last-child { padding-right: 0; }
  [data-block="board-variants"], [data-block="board-type"] { width: 58%; }
  .board-logo { display: table; width: 100%; }
  .board-logo-mark, .board-logo-name { display: table-cell; vertical-align: middle; }
  .board-logo-img { display: block; height: ${board_logo_mm}mm; width: auto; max-width: 100%; }
  .board-logo-name { text-align: right; }
  .board-name { font-size: ${h2}pt; line-height: ${h2_line}pt; font-weight: $h2_weight; color: $heading; margin: 0; }
  .board-tagline { color: $muted; margin: 0; }
  .board-wordmark { font-size: ${display}pt; line-height: ${display_line}pt; font-weight: $display_weight; color: $heading;
    margin: 0; }
  .board-swatches { display: table; width: 100%; table-layout: fixed; }
  .board-swatch { display: table-cell; padding-right: ${s1}pt; vertical-align: top; }
  .board-swatch:last-child { padding-right: 0; }
  .board-chip-colour { height: ${board_chip_mm}mm; border: ${hairline_pt}pt solid $rule; margin: 0 0 ${s1}pt 0; }
  .board-swatch-name { font-size: ${caption}pt; line-height: ${caption_line}pt; font-weight: $bold; color: $heading; margin: 0; }
  .board-hex { font-size: ${caption}pt; line-height: ${caption_line}pt; color: $ink; margin: 0; }
  .board-accent-rule { margin: ${s1}pt 0 0 0; }
  .board-variants { display: table; width: 100%; table-layout: fixed; }
  .board-variant { display: table-cell; padding-right: ${s2}pt; vertical-align: top; }
  .board-variant:last-child { padding-right: 0; }
  .board-ground { height: ${board_variant_mm}mm; border: ${hairline_pt}pt solid $rule; text-align: center;
    margin: 0 0 ${s1}pt 0; }
  .board-ground img { height: ${board_variant_logo_mm}mm; width: auto; max-width: 90%;
    margin-top: ${board_chip_pad_mm}mm; }
  .board-on-chip { display: inline-block; background: $paper; padding: ${board_chip_pad_mm}mm;
    margin-top: ${board_chip_pad_mm}mm; }
  .board-on-chip img { margin-top: 0; }
  .board-tone { margin: 0 0 ${s1}pt 0; font-size: ${small}pt; line-height: ${small_line}pt; }
  .board-tone strong { color: $heading; }
  .board-type-row { display: table; width: 100%; table-layout: fixed; }
  .board-type-label { display: table-cell; width: 32%; vertical-align: middle; }
  .board-type-cell { display: table-cell; vertical-align: middle; }
  .board-type-sample { color: $heading; white-space: nowrap; overflow: hidden; margin: 0; }
  .board-gap { margin: 0 0 ${s1}pt 0; }
  .board-gap-bar { display: inline-block; height: ${s2}pt; background: $surface_2; border-left: ${rule_pt}pt solid $heading;
    vertical-align: middle; margin-right: ${s2}pt; }
  .board-clear { display: inline-block; border: ${hairline_pt}pt dashed $muted; margin: ${s1}pt 0 0 0; }
  .board-clear img { display: block; height: ${board_clear_logo_mm}mm; width: auto; outline: ${hairline_pt}pt solid $rule; }
  .board-apps { display: table; width: 100%; table-layout: fixed; }
  .board-app { display: table-cell; padding-right: ${s3}pt; vertical-align: top; }
  .board-app:last-child { padding-right: 0; }
  .board-app-frame { height: ${board_app_mm}mm; overflow: hidden; border: ${hairline_pt}pt solid $rule;
    margin: 0 0 ${s1}pt 0; }
  .board-app-frame img { display: block; width: 100%; height: auto; }
  .board-social { position: relative; padding: ${s4}pt ${s3}pt; }
  .board-social-stripe { position: absolute; left: 0; top: 0; width: 100%; height: ${s1}pt; }
  .board-social-headline { font-size: ${h1}pt; line-height: ${h1_line}pt; font-weight: $display_weight; margin: ${s3}pt 0; }
  .board-social-brand { font-size: ${small}pt; line-height: ${small_line}pt; font-weight: $bold; margin: 0; }
""")

__all__ = ["BOARD", "BOARD_TOKENS"]
