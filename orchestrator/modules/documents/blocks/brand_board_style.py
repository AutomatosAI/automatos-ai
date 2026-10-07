"""The brand board's own rules (PRD-255 US-009), keyed on its block ids and the parts' classes.

The board is one A4 page: the title, the logo row, the colour roles in one row of
swatches, the logo variants beside the voice, the type scale beside the spacing,
and the three applications. Its rows are sections (``board-row-*``) printed as
tables, which WeasyPrint lays out without surprises. Every colour, size and gap
is a ``design_tokens`` value (substituted by ``page_style.build_styles``); the
board's own sizes, in millimetres, are :data:`BOARD_TOKENS`.

Nothing on the board overflows its box: WeasyPrint breaks the page on a child's own
height, clipped or not, so a miniature is cropped to its frame (``object-fit``) and
the social card's lines fit inside theirs. A type sample is one line, its height its
own line height, so it never wraps. Its rule outranks the base sheet's
``.doc-section p { white-space: pre-line }`` (the type scale sits in a section row):
without that, a sample wrapped at its own size and the clipped lines still pushed the
applications onto a second page.

F368 (night 10c): the sample was cut mid-word at the column's edge ("The quick |") on
every board. Each step's label now sits on its own line above its sample, so the
sample has the column's whole width, and the sample holds only the whole words that
fit it (``brand_board.fitted_sample``); the clip stays as a backstop only.
"""
from __future__ import annotations

from string import Template

# The board's own sizes (mm): the logo row's logo, a variant's ground and the logo on it,
# a swatch's chip, an application's frame, and the type scale's label column.
BOARD_TOKENS = {
    "board_logo_mm": 18,
    "board_variant_mm": 16,
    "board_variant_logo_mm": 8,
    "board_chip_mm": 10,
    "board_chip_pad_mm": 1.5,
    "board_app_mm": 34,
    "board_type_label_mm": 32,
    # The variants' and the type scale's share of their row (the type samples are fitted to it, F368).
    "board_type_share_pct": 58,
    # F376: the social card's short brand rule under sparing, as a share of its width
    # (a real card's is 160 of its 1080 design pixels).
    "board_social_rule_pct": 15,
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
  [data-block="board-variants"], [data-block="board-type"] { width: ${board_type_share_pct}%; }
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
  .board-part p.board-type-sample { color: $heading; white-space: nowrap; overflow: hidden; margin: 0; }
  .board-part p.board-font { margin: 0 0 ${s1}pt 0; }
  .board-part p.board-sublabel { margin: ${s3}pt 0 ${s1}pt 0; }
  .board-part p.board-type-label { font-family: $font; font-size: ${caption}pt; line-height: ${caption_line}pt;
    font-weight: $body_weight; color: $muted; margin: 0; }
  .board-gaps { margin: 0 0 ${s1}pt 0; }
  .board-gap { display: inline-block; vertical-align: bottom; margin-right: ${s3}pt; }
  .board-gap-bar { height: ${s2}pt; background: $surface_2; border-left: ${rule_pt}pt solid $heading; }
  .board-apps { display: table; width: 100%; table-layout: fixed; }
  .board-app { display: table-cell; padding-right: ${s3}pt; vertical-align: top; }
  .board-app:last-child { padding-right: 0; }
  .board-app-frame { box-sizing: border-box; height: ${board_app_mm}mm; overflow: hidden;
    border: ${hairline_pt}pt solid $rule; margin: 0 0 ${s1}pt 0; }
  .board-app-frame img { display: block; width: 100%; height: 100%; object-fit: cover; object-position: top; }
  .board-social { position: relative; padding: ${s3}pt; }
  .board-social-stripe { position: absolute; left: 0; top: 0; width: 100%; height: ${s1}pt; }
  .board-social-stripe.board-social-rule { left: ${s3}pt; width: ${board_social_rule_pct}%; }
  .board-social-headline { font-size: ${h3}pt; line-height: ${h3_line}pt; font-weight: $display_weight;
    margin: ${s2}pt 0; }
  .board-social-brand { font-size: ${small}pt; line-height: ${small_line}pt; font-weight: $bold; margin: 0; }
""")

__all__ = ["BOARD", "BOARD_TOKENS"]
