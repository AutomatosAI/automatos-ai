"""F331 (night 10, 5 Oct): an invoice PDF made with no template shows its line items and totals.

Eleven invoice PDFs came out as a title on an empty page. With no template named,
the workspace's seeded "Basic Report" filled them: it prints a title, a byline,
metrics and sections, and Auto's invoice number, customer and item list were
dropped without a word. The calls are made here as Auto and the agents made them,
rendered for real (WeasyPrint) and read back line by line (pdfplumber). A quote
sent as sections alone still goes through Basic Report, as it did.
"""
from __future__ import annotations

import asyncio
import copy
import os
import uuid
from types import SimpleNamespace
from typing import Any, List, Tuple

import pdfplumber
import pytest

import modules.documents.generation_service as generation_service
from modules.documents.generation_service import DocumentGenerationService
from modules.documents.seed_templates import STARTER_TEMPLATES, TEMPLATES_DIR
from services.brand_rules import forget_cached_kits

WS = uuid.UUID("00000000-0000-0000-0000-0000000331a1")

# Auto's call on the night, its data as it arrived (no template_name).
AUTOS_INVOICE = {
    "sections": [{"title": "Invoice for Lantern Kitchen",
                  "content": "12 kg of Harbour Blend delivered on Friday 2 October 2026."}],
    "invoice_number": "HL-2026-0142",
    "customer_name": "Lantern Kitchen",
    "item_details": [{"item": "Harbour Blend", "quantity": "12 kg", "delivery_date": "2026-10-02"}],
}
# A session agent's call: line items and totals as keys beside the sections.
AGENTS_INVOICE = {
    "sections": [{"title": "October wholesale", "content": "Thank you for your order."}],
    "client_name": "Lantern Kitchen",
    "line_items": [
        {"description": "Harbour Blend 1 kg", "quantity": 12, "unit_price": "19.50", "total": "234.00"},
        {"description": "Decaf 1 kg", "quantity": 2, "unit_price": "24.00", "total": "48.00"},
    ],
    "subtotal": "282.00",
    "vat": "56.40",
    "total": "338.40",
}
QUOTE = {"sections": [{"title": "Quote for Lantern Kitchen",
                       "content": "Harbour Blend at £19.50 per kilo, delivered every Friday."}]}


class _NoRows:
    """The workspace has no brand kit: every lookup finds nothing (the defaults apply)."""

    def query(self, *args: Any) -> "_NoRows":
        return self

    def filter(self, *args: Any, **kwargs: Any) -> "_NoRows":
        return self

    def first(self) -> None:
        return None


def _basic_report() -> SimpleNamespace:
    """The "Basic Report" row every workspace is seeded with."""
    seed = next(t for t in STARTER_TEMPLATES if t["name"] == "Basic Report")
    with open(os.path.join(TEMPLATES_DIR, seed["template_file"]), encoding="utf-8") as handle:
        source = handle.read()
    return SimpleNamespace(id=uuid.uuid4(), name="Basic Report", blocks=None, template_content=source,
                           data_schema=seed["data_schema"], template_file_path=None)


@pytest.fixture
def make(monkeypatch, tmp_path):
    """generate() for real, no template named: the result and the PDF's printed lines."""
    monkeypatch.setattr(generation_service, "GENERATED_DIR", str(tmp_path))
    monkeypatch.setattr(generation_service, "is_storage_configured", lambda: False)
    forget_cached_kits()

    class _Resolver:  # the block fallback has no variable chips to resolve
        def __init__(self, db: Any):
            pass

        def resolve(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(values={}, unknown=[])

    monkeypatch.setattr(generation_service, "VariableResolver", _Resolver)
    basic_report = _basic_report()

    def go(title: str, data: dict) -> Tuple[Any, List[str]]:
        service = DocumentGenerationService(_NoRows(), WS)
        service.template_service = SimpleNamespace(
            get_template=lambda template_id, ws: None,
            get_template_by_name=lambda ws, name: basic_report if name == "Basic Report" else None,
        )
        result = asyncio.run(service.generate(title=title, format="pdf", data=copy.deepcopy(data), workspace_id=WS))
        with pdfplumber.open(result.path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        return result, [" ".join(line.split()) for line in text.splitlines() if line.strip()]

    yield go
    forget_cached_kits()


def _a_line_with(lines: List[str], *parts: str) -> bool:
    return any(all(part in line for part in parts) for line in lines)


def test_autos_invoice_shows_its_number_customer_and_items(make):
    result, lines = make("Invoice HL-2026-0142", AUTOS_INVOICE)

    assert _a_line_with(lines, "Invoice number", "HL-2026-0142"), lines
    assert _a_line_with(lines, "Customer name", "Lantern Kitchen"), lines
    assert _a_line_with(lines, "Item", "Quantity", "Delivery date"), lines
    assert _a_line_with(lines, "Harbour Blend", "12 kg", "2026-10-02"), lines
    assert _a_line_with(lines, "12 kg of Harbour Blend delivered on Friday 2 October 2026."), lines
    assert result.template_name is None  # Basic Report has no place for them, so it did not fill the page
    assert result.unused_keys == []


def test_an_agents_line_items_and_totals_are_on_the_page(make):
    result, lines = make("Invoice HL-2026-0150", AGENTS_INVOICE)

    assert _a_line_with(lines, "Description", "Quantity", "Unit price", "Total"), lines
    assert _a_line_with(lines, "Harbour Blend 1 kg", "12", "19.50", "234.00"), lines
    assert _a_line_with(lines, "Decaf 1 kg", "2", "24.00", "48.00"), lines
    assert _a_line_with(lines, "Subtotal", "282.00"), lines
    assert _a_line_with(lines, "Vat", "56.40"), lines
    assert any(line.startswith("Total") and "338.40" in line for line in lines), lines
    assert _a_line_with(lines, "Client name", "Lantern Kitchen"), lines
    assert result.unused_keys == []


def test_a_quote_sent_as_sections_alone_still_renders_through_basic_report(make):
    result, lines = make("Quote for Lantern Kitchen", QUOTE)

    assert result.template_name == "Basic Report" and result.template_lane == "legacy"
    assert _a_line_with(lines, "Harbour Blend at £19.50 per kilo, delivered every Friday."), lines
    assert result.unused_keys == []


def test_a_body_sent_beside_sections_is_not_hidden_by_them(make):
    data = {"content": "Payment within 14 days, please.", "sections": [{"title": "Order", "content": "Two bags."}],
            "highlights": ["Delivered on time"]}

    result, lines = make("Order note", data)

    assert _a_line_with(lines, "Payment within 14 days, please."), lines
    assert _a_line_with(lines, "Two bags.") and _a_line_with(lines, "Delivered on time"), lines
    assert result.template_name is None
