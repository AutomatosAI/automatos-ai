"""F350 (night 10b, 5 Oct): a Branded starter prints like a finished document.

The owner's review of the starters under his kit (Automatos AI, orange #c44a1a,
navy #1d3658, the sailboat logo): the logo took the top fifth of the page and was
a different size on every starter; the invoice's totals sat mid-page, not under
the Total column; "£23.00/kg" wrapped over two lines; the letter doubled the
agent's own "Dear ..."; the report took two pages for one page of content (a
forced page break before a two-line appendix) and the proposal left its second
page white; no page had a footer or a page number; the navy never showed; dates
printed US-style. Each starter is rendered for real here (WeasyPrint) with its
own sample data and read back (pdfplumber).
"""
from __future__ import annotations

import base64
import copy
import struct
import zlib
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pdfplumber
import pytest

from modules.documents.blocks import collect_variable_paths, render_document_html, validate_blocks
from modules.documents.blocks.page_style import KEEP_CLASS, build_styles, css_string
from modules.documents.brand_kit import get_brand_kit
from modules.documents.presets import LETTERHEAD_LOGO_MM, PRESETS, preset_for, preset_payload
from modules.documents.variables.resolver import build_context, resolve_paths

COMPANY = "Automatos AI"
NOW = datetime(2026, 10, 5, 9, 0, 0)
USER = SimpleNamespace(name="Gerard Kavanagh", email="gerard@automatos.app", username="gerard")
POINTS_PER_MM = 72 / 25.4
# How far apart two right edges may be and still read as one column (points).
SAME_EDGE_PT = 2.0
EXEC_SUMMARY = Path(__file__).resolve().parents[1] / "modules" / "documents" / "templates" / "executive_summary.html"


def _png(side: int = 8) -> bytes:
    """A small opaque PNG: the uploaded logo, inlined as the renderer receives it."""
    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    rows = b"".join(b"\x00" + b"\xc4\x4a\x1a" * side for _ in range(side))
    header = struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


KIT = {
    **get_brand_kit({"brand_kit": {
        "name": COMPANY, "primary_color": "#c44a1a", "secondary_color": "#1d3658", "accent_color": "#1d3658",
        "company": {"name": COMPANY, "address": "14 Wapping Quay, Bristol BS1 4RW", "email": "gerard@automatos.app"},
    }}),
    "logo_url": "data:image/png;base64," + base64.b64encode(_png()).decode("ascii"),
}


def _html(preset: Dict[str, Any], data: Dict[str, Any]) -> Any:
    doc = validate_blocks(preset["blocks"])
    resolved = resolve_paths(build_context(USER, None, KIT, NOW, extra_data=data), collect_variable_paths(doc))
    return render_document_html(doc, resolved.values, KIT, title=preset["name"], data=data)


@pytest.fixture
def printed(tmp_path):
    """Render ``preset`` with ``data`` (its sample by default) to a real PDF: its pages, read back."""
    def go(preset: Dict[str, Any], data: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        from weasyprint import HTML

        rendered = _html(preset, preset["sample_data"]["data"] if data is None else data)
        assert rendered.unresolved == []
        path = tmp_path / f"{preset['category']}.pdf"
        HTML(string=rendered.html).write_pdf(str(path))
        with pdfplumber.open(str(path)) as pdf:
            return [{
                "lines": [" ".join(line.split()) for line in (page.extract_text() or "").splitlines() if line.strip()],
                "words": page.extract_words(),
                "images": [dict(image) for image in page.images],
            } for page in pdf.pages]
    return go


def _words(page: Dict[str, Any], text: str) -> List[Dict[str, Any]]:
    return [word for word in page["words"] if word["text"] == text]


def test_every_starter_prints_its_logo_at_one_letterhead_size(printed):
    widths = set()
    for preset in PRESETS:
        (logo,) = printed(preset)[0]["images"]
        widths.add(round(logo["width"], 1))
    assert len(widths) == 1, widths
    assert abs(widths.pop() - LETTERHEAD_LOGO_MM * POINTS_PER_MM) < 1


@pytest.mark.parametrize("category", ["report", "proposal"])
def test_a_report_or_proposal_of_one_pages_content_prints_on_one_page(printed, category):
    pages = printed(preset_for(category))
    assert len(pages) == 1, [page["lines"] for page in pages]


def test_no_starter_forces_a_page_break():
    def kinds(blocks: List[Dict[str, Any]]) -> List[str]:
        return [kind for block in blocks for kind in [block["type"], *kinds(block.get("children", []))]]

    for preset in PRESETS:
        assert "page_break" not in kinds(preset["blocks"]["blocks"]), preset["name"]


@pytest.mark.parametrize("preset", PRESETS, ids=[p["category"] for p in PRESETS])
def test_every_page_has_a_footer_with_the_company_and_the_page_number(printed, preset):
    pages = printed(preset)
    for number, page in enumerate(pages, start=1):
        footer = [line for line in page["lines"] if f"Page {number} of {len(pages)}" in line]
        assert footer and COMPANY in footer[0], page["lines"]


def test_a_long_invoices_second_page_names_the_invoice(printed):
    invoice = preset_for("invoice")
    data = copy.deepcopy(invoice["sample_data"]["data"])
    data["line_items"] = [{"description": f"Harbour Espresso blend, 1 kg, week {n}", "quantity": 12,
                           "unit_price": "24.50", "total": "294.00"} for n in range(1, 41)]
    pages = printed(invoice, data)

    assert len(pages) >= 2
    footer = [line for line in pages[1]["lines"] if f"Page 2 of {len(pages)}" in line]
    assert footer and "Invoice INV-0042" in footer[0] and COMPANY in footer[0], pages[1]["lines"]
    assert any("Description" in line and "Total" in line for line in pages[1]["lines"])  # the header row repeats


def test_the_invoice_totals_sit_under_the_total_column(printed):
    (page,) = printed(preset_for("invoice"))
    header = min(_words(page, "Total"), key=lambda word: word["top"])
    for amount in ("3,600.00", "5,100.00", "1,173.00", "6,273.00"):  # a line total, subtotal, tax, total due
        (word,) = _words(page, amount)
        assert abs(word["x1"] - header["x1"]) <= SAME_EDGE_PT, (amount, word["x1"], header["x1"])


def test_a_price_per_kilo_column_keeps_each_price_on_one_line(printed):
    sheet = copy.deepcopy(preset_for("data"))
    for block in sheet["blocks"]["blocks"]:
        if block["id"] == "rows":
            block["columns"][1] = {"key": "value", "label": "Price / kg", "align": "right"}
    rows = [
        {"name": "Harbour Espresso (house blend)", "value": "£23.00/kg", "notes": "Our espresso for milk drinks; 12 kg minimum for the loan grinder"},
        {"name": "Finca La Esperanza, Colombia", "value": "£28.50/kg", "notes": "Filter or espresso, a long sweet finish with red apple"},
    ]
    (page,) = printed(sheet, {"title": "Wholesale Price List", "rows": rows})

    assert any("Price / kg" in line for line in page["lines"]), page["lines"]
    for price in ("£23.00/kg", "£28.50/kg"):
        assert _words(page, price), page["lines"]


def test_the_letter_takes_the_greeting_as_a_field_and_prints_it_once(printed):
    letter = preset_for("letter")
    payload = preset_payload(letter)
    assert "greeting" in payload["data_fields"]
    assert "the body has no greeting or sign-off: the template adds them" in letter["description"]
    runs = [run for block in letter["blocks"]["blocks"] for run in block.get("content", [])]
    assert not any("Dear" in run.get("text", "") for run in runs)

    (page,) = printed(letter)
    assert sum(line.count("Dear") for line in page["lines"]) == 1, page["lines"]
    assert "Dear Jordan," in page["lines"]
    assert "5 October 2026" in page["lines"]  # day month year, not "October 5, 2026"


def test_a_letter_sent_without_a_greeting_is_not_blocked():
    letter = preset_for("letter")
    data = {key: value for key, value in letter["sample_data"]["data"].items() if key != "greeting"}
    assert _html(letter, data).unresolved == []


def test_the_brand_font_reaches_the_page_unescaped():
    style = build_styles({**KIT, "font_family": "Inter, 'Segoe UI', sans-serif"})
    assert "font-family: Inter, 'Segoe UI', sans-serif;" in style
    assert "&#x27;" not in style
    assert "font-family: Inter, 'Segoe UI', system-ui, sans-serif;" in build_styles({**KIT, "font_family": "x; } body { color: red"})


def test_the_footers_company_name_cannot_break_out_of_the_stylesheet():
    name = 'Smith & Co "</style><script>'
    style = build_styles({**KIT, "company": {"name": name}})
    assert "</style>" not in style and "<script>" not in style
    assert css_string("Smith & Co") == "Smith \\26  Co"  # the escape's own space ends it; the real one follows


def test_a_short_section_is_kept_on_one_page_and_a_long_one_may_break():
    short = {"type": "section", "id": "s", "title": "Next steps", "children": [{"type": "text", "id": "t", "content": [{"type": "text", "text": "Ship it."}]}]}
    long = copy.deepcopy(short)
    long["children"][0]["content"][0]["text"] = "A finding. " * 400
    html = render_document_html(validate_blocks({"blocks": [short, {**long, "id": "l"}]}), {}, KIT).html
    assert f'<section class="doc-section {KEEP_CLASS}" data-block="s">' in html
    assert '<section class="doc-section" data-block="l">' in html


def test_the_executive_summarys_figures_are_not_navy_on_orange():
    card = next(line for line in EXEC_SUMMARY.read_text(encoding="utf-8").splitlines() if ".metric-card .value" in line)
    assert "accent_color" not in card and "color: white" in card
