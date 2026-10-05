"""F356 (5 Oct): each Branded starter has a finished layout.

The owner's asks, per starter: the Agreement's signatures never sit alone on a
page; the Data Sheet's figure column has a real header, not "Value"; the
Proposal opens with a cover header and can carry a pricing total; the Report can
carry a row of KPI tiles; and every starter opens with the same letterhead. An
optional part that is not sent prints nothing (no heading over nothing, no
"Total" with no amount). Rendered for real (WeasyPrint), read back (pdfplumber).
"""
from __future__ import annotations

import base64
import copy
import struct
import zlib
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List

import pdfplumber
import pytest

from modules.documents.blocks import collect_variable_paths, render_document_html, validate_blocks
from modules.documents.blocks.letterhead_run import split_letterhead
from modules.documents.brand_kit import get_brand_kit
from modules.documents.presets import PRESETS, preset_for
from modules.documents.variables.resolver import build_context, resolve_paths

COMPANY = "Automatos AI"
NOW = datetime(2026, 10, 5, 9, 0, 0)
USER = SimpleNamespace(name="Gerard Kavanagh", email="gerard@automatos.app", username="gerard")


def _png(side: int = 8) -> bytes:
    """A small opaque PNG: the logo, inlined as the renderer receives it."""
    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    rows = b"".join(b"\x00" + b"\xc4\x4a\x1a" * side for _ in range(side))
    header = struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


KIT = {
    **get_brand_kit({"brand_kit": {
        "name": COMPANY, "primary_color": "#c44a1a", "accent_color": "#1d3658",
        "company": {"name": COMPANY, "address": "14 Wapping Quay, Bristol BS1 4RW", "email": "gerard@automatos.app"},
    }}),
    "logo_url": "data:image/png;base64," + base64.b64encode(_png()).decode("ascii"),
}
# Same right edge, in points.
SAME_EDGE_PT = 2.0


def _rendered(preset: Dict[str, Any], data: Dict[str, Any]) -> Any:
    doc = validate_blocks(preset["blocks"])
    resolved = resolve_paths(build_context(USER, None, KIT, NOW, extra_data=data), collect_variable_paths(doc))
    return render_document_html(doc, resolved.values, KIT, title=preset["name"], data=data)


@pytest.fixture
def pages(tmp_path):
    """``preset`` rendered with ``data`` to a real PDF: each page's words and text lines."""
    def go(preset: Dict[str, Any], data: Dict[str, Any]) -> List[Any]:
        from weasyprint import HTML

        rendered = _rendered(preset, data)
        assert rendered.unresolved == []
        path = tmp_path / f"{preset['category']}.pdf"
        HTML(string=rendered.html).write_pdf(str(path))
        with pdfplumber.open(str(path)) as pdf:
            return [SimpleNamespace(words=page.extract_words(), text=page.extract_text() or "") for page in pdf.pages]
    return go


def _sample(category: str) -> Dict[str, Any]:
    return copy.deepcopy(preset_for(category)["sample_data"]["data"])


def _words(page: Any, text: str) -> List[Dict[str, Any]]:
    return [word for word in page.words if word["text"] == text]


@pytest.mark.parametrize("preset", PRESETS, ids=[p["category"] for p in PRESETS])
def test_every_starter_opens_with_the_letterhead(preset):
    head, _ = split_letterhead(validate_blocks(preset["blocks"]).blocks)
    assert [block.id for block in head] == ["logo", "lh-name", "lh-address", "lh-contact"]


def test_the_agreements_signatures_never_stand_alone_on_a_page(pages):
    data = _sample("contract")
    data["services"] = "\n\n".join(["The Provider will design, build and launch the Client's website. " * 6] * 4)
    printed = pages(preset_for("contract"), data)
    assert len(printed) >= 2
    (signed,) = [page for page in printed if "Signature:" in page.text]
    assert "6. Governing law" in signed.text and "Signed" in signed.text, signed.text


def test_the_data_sheets_figure_column_has_a_real_header():
    columns = next(b for b in preset_for("data")["blocks"]["blocks"] if b["id"] == "rows")["columns"]
    assert [column["label"] for column in columns] == ["Item", "Quantity", "Notes"]


def test_the_proposal_opens_with_a_cover_header_and_totals_its_pricing(pages):
    html = _rendered(preset_for("proposal"), _sample("proposal")).html
    assert '<section class="doc-section keep" data-block="cover-header">' in html
    (page,) = pages(preset_for("proposal"), _sample("proposal"))
    assert "PROPOSAL" in page.text and "A faster, clearer site" in page.text
    price = _words(page, "Price")[0]
    (total,) = _words(page, "€18,000")
    assert abs(total["x1"] - price["x1"]) <= SAME_EDGE_PT, (total, price)


def test_a_proposal_without_a_subtitle_or_total_prints_neither():
    data = {key: value for key, value in _sample("proposal").items() if key not in ("subtitle", "pricing_total")}
    rendered = _rendered(preset_for("proposal"), data)
    assert rendered.unresolved == []
    assert 'data-block="pricing-total"' not in rendered.html
    assert '<p data-block="subtitle"></p>' in rendered.html  # an empty line, which the sheet hides


def test_the_report_prints_its_kpis_as_one_row_of_tiles(pages):
    (page,) = pages(preset_for("report"), _sample("report"))
    tiles = [_words(page, value)[0] for value in ("€182k", "312", "61")]
    assert max(t["top"] for t in tiles) - min(t["top"] for t in tiles) < 1, tiles
    assert tiles[0]["x0"] < tiles[1]["x0"] < tiles[2]["x0"], tiles


def test_a_report_without_kpis_prints_no_tiles(pages):
    data = {key: value for key, value in _sample("report").items() if key != "kpis"}
    (page,) = pages(preset_for("report"), data)
    assert "€182k" not in page.text and "Executive summary" in page.text


def test_a_section_with_nothing_to_print_is_left_out():
    section = {"type": "section", "id": "s-dec", "title": "Decisions", "children": [
        {"type": "text", "id": "dec", "content": [{"type": "variable", "path": "data.decisions", "fallback": ""}]}]}
    preset = {"name": "x", "category": "general", "blocks": {"blocks": [section]}}
    assert "Decisions" not in _rendered(preset, {}).html
    assert "Decisions" in _rendered(preset, {"decisions": "Ship on Friday."}).html
