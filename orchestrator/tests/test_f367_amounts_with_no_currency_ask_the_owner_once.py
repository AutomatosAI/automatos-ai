"""F367 (night 10c): amounts that print with no currency sign are named, and the agent asks once.

c1's derived kit had currency "" and an invoice built from plain numbers printed
"Total due 269.00"; nothing said the kit had no currency. A document never invents
one, and the workspace has no locale or country setting to take one from, so the
generate_document answer now names the amounts that printed bare and tells the agent
to ask the owner once and save the answer to the brand kit. With a currency set,
PDF, Word and the spreadsheet print it alike.
"""
from __future__ import annotations

import asyncio
import copy
import uuid
from types import SimpleNamespace
from typing import Any

import services.brand_rules as brand_rules
from modules.documents.amounts import bare_amount_keys, field_text
from modules.documents.currency_notice import amounts_without_currency
from modules.documents.generation_service import DocumentGenerationService
from modules.documents.models import GeneratedDocument
from modules.documents.xlsx_letterhead import money_format
from modules.tools.formatting.result_formatter import ToolResultFormatter
from tests.f298_fixtures import FILENAME, RecordedService, call_tool, recorded

WS = uuid.UUID("00000000-0000-0000-0000-0000000367a1")
INVOICE = {
    "invoice_number": "HL-W-1034",
    "client_name": "Lantern Kitchen",
    "line_items": [{"description": "Harbour Blend", "quantity": 12, "unit": "kg", "unit_price": 22, "total": "264.00"}],
    "subtotal": 264, "tax": "5.00", "total": 269.0,
}
NO_CURRENCY = {"currency": ""}
GBP = {"currency": "GBP"}


def test_the_bare_amounts_in_the_data_and_its_rows_are_found_once_each():
    assert bare_amount_keys(INVOICE) == ["unit_price", "total", "subtotal", "tax"]
    assert bare_amount_keys({"total": "£269.00", "quantity": 12}) == []  # a sign of its own; not an amount


def test_only_a_kit_without_a_currency_names_them_and_only_for_priced_formats():
    assert amounts_without_currency(NO_CURRENCY, INVOICE, "pdf") == ["unit_price", "total", "subtotal", "tax"]
    assert amounts_without_currency(None, INVOICE, "xlsx") == ["unit_price", "total", "subtotal", "tax"]
    assert amounts_without_currency(GBP, INVOICE, "docx") == []
    assert amounts_without_currency(NO_CURRENCY, INVOICE, "social_image") == []


def _generate(monkeypatch: Any, kit: Any) -> GeneratedDocument:
    async def generate_pdf(template, data, workspace_id, title, user_id=None):
        return GeneratedDocument(path="/tmp/x.pdf", format="pdf", filename="x.pdf", size=1)

    async def kit_off_loop(db: Any, workspace_id: Any) -> Any:
        return kit

    monkeypatch.setattr(brand_rules, "kit_off_loop", kit_off_loop)
    service = DocumentGenerationService(SimpleNamespace(), WS)
    service.template_service = SimpleNamespace(get_template_by_name=lambda ws, name: None)
    monkeypatch.setattr(service, "generate_pdf", generate_pdf)
    return asyncio.run(service.generate(title="Invoice HL-W-1034", format="pdf", data=copy.deepcopy(INVOICE),
                                        workspace_id=WS))


def test_generate_names_the_unpriced_amounts_when_the_kit_has_no_currency(monkeypatch):
    assert _generate(monkeypatch, NO_CURRENCY).unpriced_keys == ["unit_price", "total", "subtotal", "tax"]
    assert _generate(monkeypatch, GBP).unpriced_keys == []


def test_the_agent_is_told_to_ask_the_owner_once(monkeypatch):
    recorded(monkeypatch)

    async def generate(self: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(
            filename=FILENAME, format="pdf", download_url=f"/api/documents/generated/{FILENAME}", size=8671,
            content="# Invoice", template_id=None, template_name="Branded Invoice", s3_key="k",
            unused_keys=[], unpriced_keys=["total", "subtotal"],
        )

    monkeypatch.setattr(RecordedService, "generate", generate)
    answer = call_tool({"title": "Invoice", "format": "pdf", "data": copy.deepcopy(INVOICE)})

    assert answer["results"][0]["amounts_without_currency"] == ["total", "subtotal"]
    (line,) = [line for line in ToolResultFormatter.format_for_llm(answer, "generate_document").splitlines()
               if line.startswith("NO CURRENCY SIGN")]
    assert "total, subtotal" in line and "Ask the owner once" in line and "platform_update_brand_kit" in line


def test_with_a_currency_every_format_prints_it_alike():
    assert field_text("total", 269, "GBP") == "£269.00"  # PDF and Word (amounts.field_text)
    assert money_format({"currency": "GBP"}) == '"£"#,##0.00'  # the spreadsheet
    assert money_format({"currency": ""}) is None
