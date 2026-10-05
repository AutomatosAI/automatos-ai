"""PRD-255 Wave 1 US-004: every PDF and Word document prints with the kit's tokens.

Nights 10 and 10b: the kit was four colours with no roles, so the documents
painted the primary on the title and every table header ("a lot of orange in
there"). Now the kit's colour roles, type scale, spacing and logo rules drive the
block PDF and DOCX renderers and the legacy Jinja starters; amounts print in the
kit's currency, ``date.long`` in its date style, and the Branded Letter signs with
``brand.sign_off``.

The render tests are real (the F331/F350 pattern): each starter is printed by
WeasyPrint, page 1 is drawn with the F353 renderer (``pdf_first_page_png``,
pypdfium2) and its pixels are counted, and the text is read back with pdfplumber.
"""
from __future__ import annotations

import base64
import copy
import io
import struct
import zlib
from datetime import date, datetime
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Set, Tuple

import pdfplumber
import pytest
from jinja2.sandbox import SandboxedEnvironment

from modules.documents.amounts import amount_text, field_text
from modules.documents.blocks import (
    collect_variable_paths, legacy_render_data, render_document_docx, render_document_html, validate_blocks,
)
from modules.documents.blocks import design_tokens as tokens
from modules.documents.blocks.page_style import build_styles
from modules.documents.brand_kit import get_brand_kit
from modules.documents.legacy_jinja import legacy_brand, with_document_filters
from modules.documents.locale_text import currency_of, long_date
from modules.documents.presets import LETTER, MEETING_NOTES_BLOCKS, PRESETS, preset_for
from modules.documents.seed_templates import STARTER_TEMPLATES, seed_source
from modules.documents.thumbnails.render import pdf_first_page_png
from modules.documents.variables.catalog import KNOWN_PATHS
from modules.documents.variables.resolver import build_context, resolve_paths

COMPANY = "Automatos AI"
NOW = datetime(2026, 10, 5, 9, 0, 0)
USER = SimpleNamespace(name="Gerard Kavanagh", email="gerard@automatos.app", username="gerard")
POINTS_PER_MM = 72 / 25.4
# PRD-255 §2: at most about 15% of a starter's inked area is the accent.
ACCENT_SHARE_MAX = 0.15
# A pixel is ink when its channels differ from the paper's by more than this, summed
# (the 4% and 8% surface tints are not ink); it is the accent when every channel is
# within the tolerance of the accent's.
INK_THRESHOLD = 60
ACCENT_TOLERANCE = 24
# An accent element's anti-aliased edge: this many pixels round it.
EDGE_PX = 3
# Page 1 is drawn twice the thumbnail's width, so a 2 pt rule has pixels of its own colour.
RASTER_WIDTH_PX = 960
RASTER_MAX_HEIGHT_PX = 2000
# Sizes read back from a PDF, in points.
SIZE_TOLERANCE_PT = 0.2
GREY_LOGO = b"\x80\x80\x80"
LEGACY_STARTERS = ("Basic Report", "Invoice", "Executive Summary")
SEEDS = {seed["name"]: seed for seed in STARTER_TEMPLATES}


def _png(side: int = 8, rgb: bytes = GREY_LOGO) -> bytes:
    """A small opaque PNG logo. Grey: the logo is the owner's image, not the renderer's accent."""
    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    rows = b"".join(b"\x00" + rgb * side for _ in range(side))
    header = struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def _kit(**colours: Any) -> Dict[str, Any]:
    kit = get_brand_kit({"brand_kit": {
        "name": COMPANY, **colours,
        "company": {"name": COMPANY, "address": "14 Wapping Quay, Bristol BS1 4RW", "email": "gerard@automatos.app"},
    }})
    return {**kit, "logo_url": "data:image/png;base64," + base64.b64encode(_png()).decode("ascii")}


AUTOMATOS = _kit(primary_color="#c44a1a", secondary_color="#1d3658", text_color="#1a1a2e")
HARBOURLINE = _kit(primary_color="#1E3A5F", secondary_color="#C26A2E")
KITS = {"automatos": AUTOMATOS, "harbourline": HARBOURLINE}


# ---------------------------------------------------------------------------
# Rendering and reading back
# ---------------------------------------------------------------------------


def _block_html(blocks: Dict[str, Any], data: Dict[str, Any], kit: Dict[str, Any]) -> str:
    doc = validate_blocks(blocks)
    resolved = resolve_paths(build_context(USER, None, kit, NOW, extra_data=data), collect_variable_paths(doc))
    rendered = render_document_html(doc, resolved.values, kit, title="PRD-255", data=data)
    assert rendered.unresolved == []
    return rendered.html


def _legacy_html(name: str, kit: Dict[str, Any]) -> str:
    data = {"title": name, **copy.deepcopy(SEEDS[name]["sample_data"])}
    env = with_document_filters(SandboxedEnvironment(autoescape=True))
    return env.from_string(seed_source(SEEDS[name])).render(**legacy_render_data(data), brand=legacy_brand(kit))


def _pdf(html: str) -> bytes:
    from weasyprint import HTML

    return HTML(string=html).write_pdf()


@pytest.fixture(autouse=True)
def _sharp_raster(monkeypatch):
    """The F353 renderer, drawing the whole of page 1 at twice the thumbnail's width."""
    from modules.documents.thumbnails import render

    monkeypatch.setattr(render, "THUMBNAIL_WIDTH_PX", RASTER_WIDTH_PX)
    monkeypatch.setattr(render, "MAX_THUMBNAIL_HEIGHT_PX", RASTER_MAX_HEIGHT_PX)


def _pixels(pdf: bytes) -> Tuple[int, List[Tuple[int, int, int]]]:
    """Page 1, drawn by the F353 renderer: its width and its pixels, row by row."""
    from PIL import Image

    image = Image.open(io.BytesIO(pdf_first_page_png(pdf))).convert("RGB")
    return image.width, list(image.getdata())


def _rgb(colour: str) -> Tuple[int, int, int]:
    return tuple(int(colour[i:i + 2], 16) for i in (1, 3, 5))


def _is_accent(pixel: Tuple[int, int, int], accent: Tuple[int, int, int]) -> bool:
    return all(abs(a - b) <= ACCENT_TOLERANCE for a, b in zip(pixel, accent))


def _accent_share(pdf: bytes, kit: Dict[str, Any]) -> float:
    roles = tokens.palette(kit)
    paper, accent = _rgb(roles.paper), _rgb(roles.accent)
    _, pixels = _pixels(pdf)
    inked = [p for p in pixels if sum(abs(a - b) for a, b in zip(p, paper)) > INK_THRESHOLD]
    assert inked, "nothing printed"
    return sum(1 for p in inked if _is_accent(p, accent)) / len(inked)


def _chars(pdf: bytes, kit: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Page 1's characters above the footer (the footer repeats the title, in the caption's size)."""
    footer_top = tokens.design(kit).page_margin_mm * POINTS_PER_MM
    with pdfplumber.open(io.BytesIO(pdf)) as document:
        page = document.pages[0]
        return [c for c in page.chars if c["bottom"] < page.height - footer_top]


def _chars_of(chars: Iterable[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
    """The chars of the first run of ``text`` on the page."""
    chars = [c for c in chars if c["text"].strip()]
    letters = "".join(c["text"] for c in chars)
    wanted = text.replace(" ", "")
    start = letters.index(wanted)
    return chars[start:start + len(wanted)]


def _colour_of(char: Dict[str, Any]) -> Tuple[int, int, int]:
    return tuple(round(channel * 255) for channel in char["non_stroking_color"][:3])


# ---------------------------------------------------------------------------
# The accent is used sparingly (PRD-255 §2, FR-6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kit_name", sorted(KITS))
@pytest.mark.parametrize("preset", PRESETS, ids=[p["category"] for p in PRESETS])
def test_the_accent_covers_at_most_15_percent_of_each_block_starters_page_one(preset, kit_name):
    kit = KITS[kit_name]
    pdf = _pdf(_block_html(preset["blocks"], preset["sample_data"]["data"], kit))
    assert _accent_share(pdf, kit) <= ACCENT_SHARE_MAX


@pytest.mark.parametrize("name", LEGACY_STARTERS)
def test_the_accent_covers_at_most_15_percent_of_each_legacy_starters_page_one(name):
    assert _accent_share(_pdf(_legacy_html(name, AUTOMATOS)), AUTOMATOS) <= ACCENT_SHARE_MAX


def test_the_accent_covers_at_most_15_percent_of_meeting_notes():
    data = {"title": "Meeting Notes", **copy.deepcopy(SEEDS["Meeting Notes"]["sample_data"])}
    assert _accent_share(_pdf(_block_html(MEETING_NOTES_BLOCKS, data, AUTOMATOS)), AUTOMATOS) <= ACCENT_SHARE_MAX


def test_bold_puts_the_accent_on_the_table_header_and_sparing_does_not():
    accent = tokens.palette(AUTOMATOS).accent
    assert f".doc-table th {{ background: {accent};" not in build_styles(AUTOMATOS)
    assert f".doc-table th {{ background: {accent}; color: #ffffff;" in build_styles({**AUTOMATOS, "accent_use": "bold"})


# ---------------------------------------------------------------------------
# Headings are not the accent; the sizes are the kit's
# ---------------------------------------------------------------------------

HEADINGS = {"blocks": [
    {"type": "heading", "id": "t", "level": 1, "content": [{"type": "text", "text": "Quarterly Review"}]},
    {"type": "heading", "id": "s", "level": 2, "content": [{"type": "text", "text": "Findings Overview"}]},
    {"type": "heading", "id": "u", "level": 3, "content": [{"type": "text", "text": "Regional Detail"}]},
    {"type": "text", "id": "b", "content": [{"type": "text", "text": "Body copy line"}]},
]}
LEVELS = (("Quarterly Review", "h1"), ("Findings Overview", "h2"), ("Regional Detail", "h3"), ("Body copy line", "body"))


@pytest.mark.parametrize("kit_name", sorted(KITS))
def test_headings_print_in_the_heading_role_never_the_accent(kit_name):
    kit = KITS[kit_name]
    roles = tokens.palette(kit)
    chars = _chars(_pdf(_block_html(HEADINGS, {}, kit)), kit)
    for text, _ in LEVELS[:3]:
        colours = {_colour_of(c) for c in _chars_of(chars, text)}
        assert colours == {_rgb(roles.heading)}, (text, colours)
        assert _rgb(roles.accent) not in colours


def test_the_invoice_title_is_not_the_accent():
    preset = preset_for("invoice")
    chars = _chars(_pdf(_block_html(preset["blocks"], preset["sample_data"]["data"], AUTOMATOS)), AUTOMATOS)
    colours = {_colour_of(c) for c in _chars_of(chars, "Invoice INV-0042")}
    assert colours == {_rgb(tokens.palette(AUTOMATOS).heading)}


def test_the_type_sizes_match_the_kit():
    scale = {"h1": {"size_pt": 26, "line_pt": 32}, "h2": {"size_pt": 17, "line_pt": 22},
             "h3": {"size_pt": 13, "line_pt": 18}, "body": {"size_pt": 11, "line_pt": 16}}
    kit = {**AUTOMATOS, "type_scale": {**AUTOMATOS["type_scale"], **scale}}
    chars = _chars(_pdf(_block_html(HEADINGS, {}, kit)), kit)
    for text, step in LEVELS:
        sizes = {round(c["size"], 1) for c in _chars_of(chars, text)}
        assert all(abs(size - scale[step]["size_pt"]) <= SIZE_TOLERANCE_PT for size in sizes), (text, sizes)


def test_the_default_kit_prints_the_default_type_scale():
    chars = _chars(_pdf(_block_html(HEADINGS, {}, AUTOMATOS)), AUTOMATOS)
    for text, step in LEVELS:
        want = tokens.type_scale(AUTOMATOS)[step].size_pt
        assert all(abs(c["size"] - want) <= SIZE_TOLERANCE_PT for c in _chars_of(chars, text)), text


# ---------------------------------------------------------------------------
# Changing the accent changes only the accent's elements
# ---------------------------------------------------------------------------


def _accent_pixels(width: int, pixels: List[Tuple[int, int, int]], accent: str) -> Set[Tuple[int, int]]:
    rgb = _rgb(accent)
    return {(i % width, i // width) for i, p in enumerate(pixels) if _is_accent(p, rgb)}


def _near(point: Tuple[int, int], marked: Set[Tuple[int, int]]) -> bool:
    x, y = point
    return any((x + dx, y + dy) in marked for dx in range(-EDGE_PX, EDGE_PX + 1) for dy in range(-EDGE_PX, EDGE_PX + 1))


@pytest.mark.parametrize("category", ["report", "proposal", "invoice"])
def test_changing_the_accent_changes_only_the_accent_elements(category):
    preset = preset_for(category)
    data = preset["sample_data"]["data"]
    orange = {**AUTOMATOS, "palette": {"accent": "#b74518"}}
    green = {**AUTOMATOS, "palette": {"accent": "#1f6b3a"}}
    width, before = _pixels(_pdf(_block_html(preset["blocks"], data, orange)))
    _, after = _pixels(_pdf(_block_html(preset["blocks"], data, green)))
    marked = _accent_pixels(width, before, "#b74518") | _accent_pixels(width, after, "#1f6b3a")
    assert marked, "the accent prints somewhere"
    changed = [(i % width, i // width) for i, (a, b) in enumerate(zip(before, after)) if a != b]
    stray = [point for point in changed if not _near(point, marked)]
    assert not stray, stray[:20]


# ---------------------------------------------------------------------------
# Spacing, margins and the logo come from the kit
# ---------------------------------------------------------------------------


def test_the_margins_and_gaps_follow_the_kits_spacing():
    style = build_styles({**AUTOMATOS, "spacing_unit_pt": 5, "page_margin_mm": 25})
    assert "@page { size: A4; margin: 25.0mm;" in style
    assert "p { margin: 0 0 10.0pt 0;" in style  # two spacing units


def test_the_letterhead_logo_is_the_kits_letterhead_height(tmp_path):
    kit = {**AUTOMATOS, "logo_rules": {**AUTOMATOS["logo_rules"], "letterhead_mm": 22}}
    preset = preset_for("invoice")
    with pdfplumber.open(io.BytesIO(_pdf(_block_html(preset["blocks"], preset["sample_data"]["data"], kit)))) as pdf:
        (logo,) = pdf.pages[0].images
    assert abs(logo["height"] - 22 * POINTS_PER_MM) < 1


# ---------------------------------------------------------------------------
# The Word file maps the same tokens to its styles
# ---------------------------------------------------------------------------


def _docx(category: str, kit: Dict[str, Any]) -> Any:
    preset = preset_for(category)
    data = copy.deepcopy(preset["sample_data"]["data"])
    doc = validate_blocks(preset["blocks"])
    resolved = resolve_paths(build_context(USER, None, kit, NOW, extra_data=data), collect_variable_paths(doc))
    return render_document_docx(doc, resolved.values, kit, data=data).document


def test_the_word_styles_are_the_kits_type_and_roles():
    from docx.oxml.ns import qn

    design = tokens.design(AUTOMATOS)
    styles = _docx("report", AUTOMATOS).styles
    for name, step in (("Heading 1", "h1"), ("Heading 2", "h2"), ("Heading 3", "h3"), ("Caption", "caption")):
        assert styles[name].font.size.pt == design.type[step].size_pt, name
    for name in ("Heading 1", "Heading 2", "Heading 3"):
        assert str(styles[name].font.color.rgb) == design.palette.heading.lstrip("#").upper(), name
        assert styles[name].element.pPr.find(qn("w:pBdr")) is None, name  # no full-width rules under headings (PRD-243)
    assert str(styles["Caption"].font.color.rgb) == design.palette.muted.lstrip("#").upper()


def test_a_word_tables_header_is_the_accent_only_when_bold():
    from docx.oxml.ns import qn

    def header_fills(kit: Dict[str, Any]) -> Set[str]:
        row = _docx("invoice", kit).tables[0].rows[0]
        return {shd.get(qn("w:fill")) for cell in row.cells for shd in cell._tc.iter(qn("w:shd"))}

    roles = tokens.palette(AUTOMATOS)
    assert header_fills(AUTOMATOS) == {roles.surface_2.lstrip("#").upper()}
    assert header_fills({**AUTOMATOS, "accent_use": "bold"}) == {roles.accent.lstrip("#").upper()}


def test_the_word_letterhead_logo_is_the_kits_letterhead_height():
    from docx.oxml.ns import qn

    header = _docx("invoice", AUTOMATOS).sections[0].first_page_header
    (extent,) = list(header._element.iter(qn("wp:extent")))
    assert abs(int(extent.get("cy")) / 36000 - AUTOMATOS["logo_rules"]["letterhead_mm"]) < 0.1  # EMU per mm


# ---------------------------------------------------------------------------
# The legacy Jinja starters read brand.palette.* and brand.type.*
# ---------------------------------------------------------------------------


def test_a_legacy_template_gets_the_kits_roles_and_type_scale():
    brand = legacy_brand(AUTOMATOS)
    roles = tokens.palette(AUTOMATOS)
    assert brand["palette"]["accent"] == roles.accent and brand["palette"]["surface"] == roles.surface
    assert brand["type"]["h1"] == {"size_pt": 22.0, "line_pt": 28.0, "weight": 600}


def test_the_executive_summary_prints_its_figures_in_the_accent_on_the_surface():
    html = _legacy_html("Executive Summary", AUTOMATOS)
    roles = tokens.palette(AUTOMATOS)
    assert f"background: {roles.surface}; }}" in html and f"color: {roles.accent}; }}" in html
    assert AUTOMATOS["primary_color"] not in html  # no raw primary


# ---------------------------------------------------------------------------
# Currency and date style (FR-7, FR-8)
# ---------------------------------------------------------------------------


def test_an_amount_prints_in_the_kits_currency_with_two_decimals():
    assert amount_text(311, "GBP") == "£311.00"
    assert amount_text("-5", "EUR") == "-€5.00"
    assert amount_text(12.5, "SEK") == "SEK 12.50"
    assert amount_text("€19.50", "GBP") == "€19.50"  # given with its own currency: as given
    assert amount_text(311) == "311.00"  # no currency in the kit: none is printed
    assert field_text("quantity", 3, "GBP") == "3"


def test_a_kits_currency_is_read_leniently():
    assert currency_of({"currency": "gbp"}) == "GBP"
    assert currency_of({"currency": "pounds"}) == "" and currency_of({}) == "" and currency_of(None) == ""


def test_the_invoice_prints_its_amounts_in_the_kits_currency():
    kit = {**AUTOMATOS, "currency": "GBP"}
    data = {**copy.deepcopy(preset_for("invoice")["sample_data"]["data"]), "subtotal": 100, "tax": 20, "total": 120,
            "line_items": [{"description": "Roast", "quantity": 2, "unit_price": 50, "total": 100}]}
    html = _block_html(preset_for("invoice")["blocks"], data, kit)
    assert "£120.00" in html and "£50.00" in html and "£100.00" in html
    legacy = _legacy_html("Invoice", kit)
    assert "£1500.00" in legacy and "$" not in legacy.split("<body>", 1)[1]


def test_date_long_follows_the_date_style():
    assert long_date(date(2026, 10, 5)) == "5 October 2026"
    assert long_date(date(2026, 10, 5), "MMMM d, yyyy") == "October 5, 2026"
    context = build_context(USER, None, {**AUTOMATOS, "date_style": "MMMM d, yyyy"}, NOW)
    assert resolve_paths(context, ["date.long"]).values["date.long"] == "October 5, 2026"
    assert resolve_paths(build_context(USER, None, AUTOMATOS, NOW), ["date.long"]).values["date.long"] == "5 October 2026"


# ---------------------------------------------------------------------------
# brand.sign_off
# ---------------------------------------------------------------------------


def test_brand_sign_off_is_the_kits_sign_off_else_the_person_signing():
    assert "brand.sign_off" in KNOWN_PATHS
    signed = {**AUTOMATOS, "voice": {**AUTOMATOS["voice"], "sign_off": "The Automatos team"}}
    assert resolve_paths(build_context(USER, None, signed, NOW), ["brand.sign_off"]).values == {
        "brand.sign_off": "The Automatos team"}
    assert resolve_paths(build_context(USER, None, AUTOMATOS, NOW), ["brand.sign_off"]).values == {
        "brand.sign_off": USER.name}
    assert resolve_paths(build_context(None, None, AUTOMATOS, NOW), ["brand.sign_off"]).unresolved == ["brand.sign_off"]


def test_the_branded_letter_closes_with_brand_sign_off(tmp_path):
    (sig,) = [b for b in LETTER["blocks"]["blocks"] if b["id"] == "sig-name"]
    assert [run["path"] for run in sig["content"]] == ["brand.sign_off"]
    signed = {**AUTOMATOS, "voice": {**AUTOMATOS["voice"], "sign_off": "The Automatos team"}}
    with pdfplumber.open(io.BytesIO(_pdf(_block_html(LETTER["blocks"], LETTER["sample_data"]["data"], signed)))) as pdf:
        text = " ".join("".join(page.extract_text() or "" for page in pdf.pages).split())
    assert "Kind regards, The Automatos team" in text, text
