"""F369 (night 10c, document part): an invoice keeps its quantity, unit and unit price.

Auto's invoice (chat 76d4753a) for "12 kg Harbour Blend at £22.00 a kg" printed
"Qty 1 · £264.00": the Branded Invoice's line items had no place for a unit, so the
model folded "12 kg" into the description and priced the line as one item. A line
item may now carry ``unit``: it prints after the quantity ("12 kg") in the PDF and the
Word file, the unit price stays the price of one kilo, and the template tells the
agent so (its description, its sample, and the table's ``also_reads`` in the schema
answer).
"""
from __future__ import annotations

from datetime import datetime

import pytest

from modules.documents.blocks import collect_variable_paths, render_document_html, validate_blocks
from modules.documents.blocks.table_cells import cell_text
from modules.documents.brand_kit import get_brand_kit
from modules.documents.field_requirements import block_requirements
from modules.documents.presets import INVOICE
from modules.documents.variables.resolver import build_context, resolve_paths

KIT = get_brand_kit({"brand_kit": {"name": "Harbourline", "company": {"name": "Harbourline Coffee"}, "currency": "GBP"}})
LANTERN = {
    "client_name": "Lantern Kitchen", "invoice_number": "HL-W-1034", "due_date": "5 November 2026",
    "payment_terms": "30 days by bank transfer",
    "line_items": [
        {"description": "Harbour Blend", "quantity": 12, "unit": "kg", "unit_price": 22, "total": 264},
        {"description": "Carriage", "quantity": 1, "unit_price": 5, "total": 5},
    ],
    "subtotal": 269, "tax": 0, "total": 269,
}
ROW = LANTERN["line_items"][0]


def test_a_quantity_prints_with_its_unit_and_the_unit_price_stays_per_unit():
    assert cell_text(ROW, "quantity", 1, "GBP") == "12 kg"
    assert cell_text(ROW, "unit_price", 3, "GBP") == "£22.00"
    assert cell_text(ROW, "total", 4, "GBP") == "£264.00"
    assert cell_text(LANTERN["line_items"][1], "quantity", 1) == "1"  # no unit: the number alone
    assert cell_text({"quantity": "12 kg", "unit": "kg"}, "quantity", 0) == "12 kg"  # never "12 kg kg"
    assert cell_text({"description": "kg", "unit": "kg"}, "description", 0) == "kg"  # only the quantity takes it


def _invoice_html() -> str:
    doc = validate_blocks(INVOICE["blocks"])
    values = resolve_paths(build_context(None, None, KIT, datetime(2026, 10, 6), LANTERN), collect_variable_paths(doc)).values
    rendered = render_document_html(doc, values, KIT, data=LANTERN)
    assert rendered.unresolved == []
    return rendered.html


def test_the_pdf_invoice_prints_twelve_kilos_at_twenty_two_pounds():
    page = _invoice_html()

    assert '<td style="text-align:right">12 kg</td><td style="text-align:right">£22.00</td>' in page
    assert '<td style="text-align:right">£264.00</td>' in page


def test_the_word_invoice_prints_them_too():
    pytest.importorskip("docx")
    from modules.documents.blocks import render_document_docx

    document = render_document_docx(validate_blocks(INVOICE["blocks"]), {}, KIT, data=LANTERN).document
    cells = [cell.text for table in document.tables for row in table.rows for cell in row.cells]
    assert "12 kg" in cells and "£22.00" in cells


def test_the_template_tells_the_agent_to_send_the_unit():
    (items,) = block_requirements(INVOICE["blocks"])["tables"]
    assert "unit" in items["also_reads"]
    assert "Never fold a quantity into the description" in INVOICE["description"]
    assert any(row.get("unit") for row in INVOICE["sample_data"]["data"]["line_items"])
