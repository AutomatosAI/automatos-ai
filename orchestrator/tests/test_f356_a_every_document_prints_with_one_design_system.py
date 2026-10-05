"""F356 (5 Oct): every document prints with one design system.

After F350's pass the owner said the documents "look more professional now… I
need them to be as professional as we can make them". Each renderer chose its
own sizes, gaps and colours, and the letterhead logo sat above the company
block at 22 mm. Now one module (``blocks/design_tokens``) holds a six-step type
scale, a 4 pt spacing grid and the colour roles, every colour from the kit; the
letterhead is a 14 mm logo beside the company block; tables have zebra rows,
hairlines and tabular figures. The Branded Invoice is rendered for real here
(WeasyPrint) and read back (pdfplumber).

PRD-255 (US-004): the design system is now the kit's own (type scale, spacing
unit, logo rules, colour roles), so these expectations read the kit: the title
and the table headers are no longer the primary ("a lot of orange in there"):
headings are the kit's ``heading``, the table header ``surface_2``, and the
letterhead logo is ``logo_rules.letterhead_mm`` high.
"""
from __future__ import annotations

import base64
import re
import struct
import zlib
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List

import pdfplumber
import pytest

from modules.documents.blocks import collect_variable_paths, render_document_html, validate_blocks
from core.brand_palette import derive_palette
from modules.documents.blocks import design_tokens as tokens
from modules.documents.blocks.page_style import build_styles
from modules.documents.brand_kit import get_brand_kit
from modules.documents.presets import preset_for
from modules.documents.variables.resolver import build_context, resolve_paths

ORANGE, NAVY = "#c44a1a", "#1d3658"
COMPANY = "Automatos AI"
NOW = datetime(2026, 10, 5, 9, 0, 0)
USER = SimpleNamespace(name="Gerard Kavanagh", email="gerard@automatos.app", username="gerard")
POINTS_PER_MM = 72 / 25.4


def _png(width: int = 11, height: int = 13) -> bytes:
    """A small opaque PNG, taller than wide like the owner's sailboat."""
    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    rows = b"".join(b"\x00" + b"\xc4\x4a\x1a" * width for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


KIT = {
    **get_brand_kit({"brand_kit": {
        "name": COMPANY, "primary_color": ORANGE, "secondary_color": NAVY, "accent_color": NAVY,
        "company": {"name": COMPANY, "address": "14 Wapping Quay, Bristol BS1 4RW", "email": "gerard@automatos.app"},
    }}),
    "logo_url": "data:image/png;base64," + base64.b64encode(_png()).decode("ascii"),
}


def _rendered(blocks: Dict[str, Any], data: Dict[str, Any]) -> Any:
    doc = validate_blocks(blocks)
    resolved = resolve_paths(build_context(USER, None, KIT, NOW, extra_data=data), collect_variable_paths(doc))
    return render_document_html(doc, resolved.values, KIT, title="F356", data=data)


def _page(tmp_path, blocks: Dict[str, Any], data: Dict[str, Any]) -> Any:
    from weasyprint import HTML

    path = tmp_path / "f356.pdf"
    HTML(string=_rendered(blocks, data).html).write_pdf(str(path))
    with pdfplumber.open(str(path)) as pdf:
        page = pdf.pages[0]
        return SimpleNamespace(words=page.extract_words(), images=[dict(i) for i in page.images],
                               lines=[dict(shape) for shape in (*page.lines, *page.rects, *page.curves)])


def _word(page: Any, text: str) -> Dict[str, Any]:
    return next(word for word in page.words if word["text"] == text)


def test_the_colour_roles_come_from_the_kit():
    owner = tokens.palette(KIT)
    roles = derive_palette(KIT)
    assert (owner.heading, owner.ink, owner.rule, owner.accent) == (
        roles["heading"], roles["ink"], roles["rule"], roles["accent"])
    assert ORANGE not in (owner.heading, owner.header_fill)  # the primary is the accent, not the headings
    assert (owner.header_fill, owner.header_text) == (roles["surface_2"], roles["heading"])  # sparing
    bold = tokens.palette({**KIT, "accent_use": "bold"})
    assert (bold.header_fill, bold.header_text) == (roles["accent"], "#ffffff")  # white reads on the accent
    yellow = tokens.palette({"primary_color": "#ffd400", "text_color": "#222222", "accent_use": "bold",
                             "palette": {"accent": "#ffd400"}})
    assert yellow.header_fill == "#ffd400" and yellow.header_text == yellow.heading  # white does not read on yellow


def test_the_sheet_uses_the_kits_type_scale_and_its_spacing_grid():
    style = build_styles(KIT)
    scale = tokens.type_scale(KIT).values()
    sizes = {float(size) for size in re.findall(r"font-size: ([\d.]+)pt", style)}
    assert sizes <= {step.size_pt for step in scale}, sizes
    weights = {int(weight) for weight in re.findall(r"font-weight: (\d+)", style)}
    assert weights <= {step.weight for step in scale} | {tokens.BOLD}, weights
    unit = tokens.design(KIT).spacing_unit_pt
    for value in re.findall(r"(?:margin|padding)[a-z-]*: ([^;]+);", style):
        for part in re.findall(r"(-?[\d.]+)pt", value):
            assert float(part) % unit == 0, (value, style)


def test_headings_take_the_heading_role_and_the_title_an_accent_rule():
    style = build_styles(KIT)
    roles, h1 = tokens.palette(KIT), tokens.type_scale(KIT)["h1"]
    assert f"h1, h2, h3, h4, h5, h6 {{ color: {roles.heading};" in style
    assert f"h1 {{ font-size: {h1.size_pt}pt; line-height: {h1.line_pt}pt;" in style
    assert f"solid {roles.accent};" in style.split("h1::after", 1)[1].split("}", 1)[0]
    assert f".doc-table th {{ background: {roles.surface_2}; color: {roles.heading};" in style
    assert f"border-bottom: {tokens.RULE_PT}pt solid {roles.rule}" in style  # the letterhead's rule
    assert "tbody tr:nth-child(even) td { background: #" in style and "font-variant-numeric: tabular-nums" in style


def test_the_letterhead_logo_sits_beside_the_company_block(tmp_path):
    invoice = preset_for("invoice")
    page = _page(tmp_path, invoice["blocks"], invoice["sample_data"]["data"])
    (logo,) = page.images
    name = _word(page, "Automatos")
    assert abs(logo["height"] - KIT["logo_rules"]["letterhead_mm"] * POINTS_PER_MM) < 1
    assert name["x0"] > logo["x1"], (name, logo)  # beside, not under
    assert logo["top"] <= name["top"] <= logo["bottom"], (name, logo)
    title = _word(page, "INV-0042")
    assert title["top"] > logo["bottom"]  # the document starts under the letterhead
    # A border is drawn as the box's outline (an even-odd fill), so the rule is where the box ends.
    rules = [line for line in page.lines if logo["bottom"] <= line["bottom"] <= title["top"] and line["width"] > 300]
    assert rules, [(round(line["top"]), round(line["bottom"]), round(line["width"])) for line in page.lines]


def test_a_logo_without_a_company_block_stays_where_it_was():
    blocks = {"blocks": [{"type": "image", "id": "logo", "source": "brand_logo", "alt": "Logo", "width_mm": 14},
                         {"type": "heading", "id": "title", "level": 1, "content": [{"type": "text", "text": "Hi"}]}]}
    html = _rendered(blocks, {}).html
    assert 'class="letterhead"' not in html and '<img class="doc-image" data-block="logo"' in html


def test_an_optional_line_left_empty_takes_no_space(tmp_path):
    def blocks(*middle: Dict[str, Any]) -> Dict[str, Any]:
        head = {"type": "heading", "id": "t", "level": 1, "content": [{"type": "text", "text": "Notes"}]}
        tail = {"type": "text", "id": "after", "content": [{"type": "text", "text": "Afterwards"}]}
        return {"blocks": [head, *middle, tail]}

    optional = {"type": "text", "id": "opt", "content": [{"type": "variable", "path": "data.extra", "fallback": ""}]}
    with_empty = _word(_page(tmp_path, blocks(optional), {}), "Afterwards")
    without = _word(_page(tmp_path, blocks(), {}), "Afterwards")
    assert abs(with_empty["top"] - without["top"]) < 0.5


def _rgb(colour: str) -> tuple:
    return tuple(round(int(colour[i:i + 2], 16) / 255, 2) for i in (1, 3, 5))


@pytest.mark.parametrize("category", ["invoice", "report"])
def test_table_header_cells_are_filled_with_surface_2_not_the_primary(tmp_path, category):
    preset = preset_for(category)
    page = _page(tmp_path, preset["blocks"], preset["sample_data"]["data"])
    fills: List[Any] = [rect.get("non_stroking_color") for rect in page.lines if rect.get("fill")]
    found = {tuple(round(c, 2) for c in fill[:3]) for fill in fills if isinstance(fill, (tuple, list))}
    assert _rgb(tokens.palette(KIT).surface_2) in found, found
    assert _rgb(ORANGE) not in found, found
