"""The brand board's parts as HTML, for the PDF (PRD-255 US-009).

Each ``brand`` block prints one part of the board from the kit (``brand_board``
reads it). Every string from the kit is HTML-escaped as it is written, like every
other block (``html_renderer``); colours are the palette's normalised hex. The
page's look is ``brand_board_style``.

The applications part embeds the Branded Invoice and Letter as page-1 PNGs
(``modules/documents/brand_board_miniatures``, imported when the part is drawn:
that module renders block documents itself) and draws the social card from the
kit's colours.
"""
from __future__ import annotations

import html
from typing import Any, Callable, Dict, Mapping

from core.brand_palette import ROLE_DERIVED

from . import brand_board as bb

# A role the owner did not set, derived from the kit's colours (FR-2), says so under its swatch.
DERIVED_NOTE = "derived"


def _esc(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _part(block: Any, label: str, inner: str) -> str:
    title = f'<p class="board-label">{_esc(label)}</p>' if label else ""
    return f'<div class="board-part" data-block="{_esc(block.id)}" data-part="{_esc(block.part)}">{title}{inner}</div>'


def _img(src: str, alt: str, css_class: str = "") -> str:
    classes = f' class="{css_class}"' if css_class else ""
    return f'<img{classes} src="{_esc(src)}" alt="{_esc(alt)}" />' if src else ""


def logo_html(block: Any, kit: Mapping[str, Any]) -> str:
    """The logo large, as uploaded (or the name and a line saying there is no logo), beside the name and tagline."""
    name, src = bb.brand_name(kit), bb.logo(kit)
    if src:
        mark = _img(src, "Logo", "board-logo-img")
    else:
        mark = f'<p class="board-wordmark">{_esc(name)}</p><p class="board-note">{_esc(bb.NO_LOGO_NOTE)}</p>'
    lines = f'<p class="board-label">{_esc(bb.BOARD_TITLE)}</p>' + "".join(
        f'<p class="{css}">{_esc(text)}</p>' for css, text in (("board-name", name), ("board-tagline", bb.tagline(kit))) if text
    )
    inner = f'<div class="board-logo"><div class="board-logo-mark">{mark}</div><div class="board-logo-name">{lines}</div></div>'
    return _part(block, "", inner)


def variants_html(block: Any, kit: Mapping[str, Any]) -> str:
    """The logo on light, on dark and in one colour; an unset variant as its fallback, saying so (FR-9)."""
    tiles = []
    for variant in bb.variants(kit):
        image = _img(variant.src, variant.label)
        if variant.chip:
            image = f'<span class="board-on-chip">{image}</span>'
        note = f'<p class="board-note">{_esc(variant.note)}</p>' if variant.note else ""
        tiles.append(
            f'<div class="board-variant"><div class="board-ground" style="background:{_esc(variant.ground)}">{image}</div>'
            f'<p class="board-caption">{_esc(variant.label)}</p>{note}</div>'
        )
    return _part(block, "Logo", f'<div class="board-variants">{"".join(tiles)}</div>')


def colours_html(block: Any, kit: Mapping[str, Any]) -> str:
    """Every colour role as a swatch with its name and hex; a derived role says so."""
    cells = []
    for swatch in bb.swatches(kit):
        derived = f'<p class="board-caption">{DERIVED_NOTE}</p>' if swatch.source == ROLE_DERIVED else ""
        cells.append(
            f'<div class="board-swatch" title="{_esc(swatch.job)}">'
            f'<div class="board-chip-colour" style="background:{_esc(swatch.hex)}"></div>'
            f'<p class="board-swatch-name">{_esc(swatch.label)}</p><p class="board-hex">{_esc(swatch.hex)}</p>{derived}</div>'
        )
    rule = f'<p class="board-caption board-accent-rule">{_esc(bb.accent_rule(kit))}</p>'
    return _part(block, "Colour", f'<div class="board-swatches">{"".join(cells)}</div>{rule}')


def type_html(block: Any, kit: Mapping[str, Any]) -> str:
    """Each step of the type scale: its size, line height and weight, and a sample set in it."""
    rows = "".join(
        f'<div class="board-type-row"><p class="board-caption board-type-label">{_esc(sample.label)}</p>'
        f'<div class="board-type-cell"><p class="board-type-sample" style="font-size:{sample.size_pt:g}pt;'
        f'line-height:{sample.line_pt:g}pt;font-weight:{sample.weight}">{_esc(bb.TYPE_SAMPLE)}</p></div></div>'
        for sample in bb.type_samples(kit)
    )
    return _part(block, "Type", rows)


def spacing_html(block: Any, kit: Mapping[str, Any]) -> str:
    """The spacing grid as bars, the page margin, and the logo's clear space."""
    grid = bb.spacing(kit)
    bars = "".join(
        f'<div class="board-gap"><div class="board-gap-bar" style="width:{gap:g}pt"></div>'
        f'<p class="board-caption">{gap:g}</p></div>'
        for gap in grid.gaps_pt
    )
    lines = (
        f'<p class="board-caption">A {grid.unit_pt:g} pt grid (gaps in pt); page margins {grid.margin_mm:g} mm.</p>'
        f'<p class="board-caption">Logo clear space: {grid.clear_space:.2g} of its height '
        f'({grid.clear_mm:.3g} mm round the {grid.logo_mm:g} mm letterhead logo).</p>'
    )
    return _part(block, "Spacing", f'<div class="board-gaps">{bars}</div>' + lines)


def voice_html(block: Any, kit: Mapping[str, Any]) -> str:
    """The tone words with their meanings, and the sign-off."""
    tones, sign_off = bb.voice(kit)
    words = "".join(
        f'<p class="board-tone"><strong>{_esc(tone["word"])}</strong>'
        + (f' {_esc(tone["meaning"])}' if tone["meaning"] else "") + "</p>"
        for tone in tones
    ) or f'<p class="board-note">{_esc(bb.NO_TONE_NOTE)}</p>'
    signed = f'<p class="board-caption">Signs off as {_esc(sign_off)}</p>' if sign_off else ""
    return _part(block, "Voice", words + signed)


def _social_card(kit: Mapping[str, Any]) -> str:
    paper, ink, accent = bb.social_colours(kit)
    name = bb.brand_name(kit)
    return (
        f'<div class="board-app-frame board-social" style="background:{_esc(paper)}">'
        f'<div class="board-social-stripe" style="background:{_esc(accent)}"></div>'
        f'<p class="board-social-headline" style="color:{_esc(ink)}">{_esc(bb.SOCIAL_SAMPLE_HEADLINE)}</p>'
        f'<p class="board-social-brand" style="color:{_esc(ink)}">{_esc(name)}</p></div>'
    )


def applications_html(block: Any, kit: Mapping[str, Any]) -> str:
    """The invoice and the letter printed with the kit (page 1, as PNGs) and the social card drawn from its colours."""
    from modules.documents.brand_board_miniatures import starter_miniatures

    drawn = starter_miniatures(kit)
    tiles = []
    for key, label in bb.APPLICATIONS:
        if key == bb.APPLICATION_SOCIAL:
            frame = _social_card(kit)
        else:
            frame = f'<div class="board-app-frame">{_img(bb.miniature(drawn, key), label)}</div>'
        tiles.append(f'<div class="board-app">{frame}<p class="board-caption">{_esc(label)}</p></div>')
    return _part(block, "Applications", f'<div class="board-apps">{"".join(tiles)}</div>')


PART_HTML: Dict[str, Callable[[Any, Mapping[str, Any]], str]] = {
    "logo": logo_html,
    "variants": variants_html,
    "colours": colours_html,
    "type": type_html,
    "spacing": spacing_html,
    "voice": voice_html,
    "applications": applications_html,
}


def brand_part_html(block: Any, kit: Mapping[str, Any]) -> str:
    """The HTML of one ``brand`` block, drawn from ``kit`` (the render-ready brand kit)."""
    return PART_HTML[block.part](block, kit or {})


__all__ = ["PART_HTML", "brand_part_html"]
