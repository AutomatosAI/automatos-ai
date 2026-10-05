"""F345 (night 10b): a document with an unfilled field is never delivered.

The Template Studio promises it, and it held only for an empty chip on a block
template. Night 10b delivered three price lists with empty prices and invoices with
a blank Total cell (a table row's blank or wrong-key cell counted as filled), a
"Bill to: ·" (spaces counted as filled), an Invoice that said "BILL TO Client Name"
and a blank Executive Summary (legacy templates had no required check), and starters
that printed "Net 30", "0.00" and "the laws of Ireland" for a business that never
said so (silent business and legal fallbacks).

Each is driven here as the owner meets it: the document is blocked, naming the field
(and, in a table, the row and the column) to fill. Nothing renders a real PDF: the
block lane is read as HTML or made as a .docx, and the legacy checks run before
WeasyPrint is reached.
"""
from __future__ import annotations

import asyncio
import os
import uuid
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict

import pytest

import modules.documents.generation_service as generation_service
from modules.documents.blocks import collect_variable_paths, render_document_html, validate_blocks
from modules.documents.brand_kit import get_brand_kit
from modules.documents.models import EmptyDocumentError, UnresolvedDeliverableError
from modules.documents.presets import CONTRACT, INVOICE
from modules.documents.seed_templates import STARTER_TEMPLATES, TEMPLATES_DIR
from modules.documents.template_summary import summarize_template
from modules.documents.variables.resolver import build_context, resolve_paths

WS = uuid.UUID("00000000-0000-0000-0000-0000000345a1")
NOW = datetime(2026, 10, 5, 9, 0, 0)
KIT = get_brand_kit({"brand_kit": {"name": "Harbourline", "logo_url": "https://harbourline.example/logo.png",
                                   "company": {"name": "Harbourline Coffee Roasters", "address": "4 Quay St"}}})
USER = SimpleNamespace(name="Tom Byrne", email="tom@harbourline.example", username="tom")
PRICE_COLUMNS = [{"key": "description", "label": "Description"}, {"key": "total", "label": "Total", "align": "right"}]
INVOICE_ROWS = [
    {"description": "Harbour Blend 1 kg", "total": "234.00"},
    {"description": "Decaf 1 kg", "total": ""},
]


def _price_table(columns=None) -> Dict[str, Any]:
    return {"blocks": [{"type": "data_table", "id": "items", "path": "data.line_items",
                        "columns": columns or PRICE_COLUMNS}]}


def _rendered(blocks: Dict[str, Any], data: Dict[str, Any]):
    """A block template as the PDF lane renders it: chips resolved, then the page."""
    doc = validate_blocks(blocks)
    resolved = resolve_paths(build_context(USER, None, KIT, NOW, extra_data=data), collect_variable_paths(doc))
    return render_document_html(doc, resolved.values, KIT, data=data)


class _NoRows:
    """A session where nothing is stored: no brand kit, no profile, no user."""

    def query(self, *args: Any) -> "_NoRows":
        return self

    def filter(self, *args: Any, **kwargs: Any) -> "_NoRows":
        return self

    def order_by(self, *args: Any) -> "_NoRows":
        return self

    def first(self) -> None:
        return None


def _generate(monkeypatch, tmp_path, template: Any, fmt: str, data: Dict[str, Any]):
    """DocumentGenerationService.generate, the entry every caller uses, with ``template`` named."""
    monkeypatch.setattr(generation_service, "GENERATED_DIR", str(tmp_path))
    monkeypatch.setattr(generation_service, "is_storage_configured", lambda: False)
    service = generation_service.DocumentGenerationService(_NoRows(), WS)
    service.template_service = SimpleNamespace(get_template=lambda template_id, ws: template,
                                               get_template_by_name=lambda ws, name: template)
    return asyncio.run(service.generate(title="Invoice HL-0151", format=fmt, data=data, workspace_id=WS,
                                        template_name=getattr(template, "name", "T")))


def _legacy(name: str, **overrides: Any) -> SimpleNamespace:
    """A seeded legacy (Jinja) starter row, as every workspace has it."""
    seed = next(t for t in STARTER_TEMPLATES if t["name"] == name)
    with open(os.path.join(TEMPLATES_DIR, seed["template_file"]), encoding="utf-8") as handle:
        source = handle.read()
    row = dict(id=uuid.uuid4(), name=name, format="pdf", blocks=None, template_content=source,
               data_schema=seed["data_schema"], template_file_path=None)
    return SimpleNamespace(**{**row, **overrides})


# --------------------------------------------------------------------------- #
# A table's rows: every cell of every required column
# --------------------------------------------------------------------------- #


def test_a_blank_total_cell_blocks_the_invoice_naming_its_row_and_column(monkeypatch, tmp_path):
    pytest.importorskip("docx")
    template = SimpleNamespace(id=uuid.uuid4(), name="Price list", format="docx", blocks=_price_table(),
                               template_content=None, template_file_path=None, data_schema=None)

    with pytest.raises(UnresolvedDeliverableError) as blocked:
        _generate(monkeypatch, tmp_path, template, "docx", {"line_items": [dict(row) for row in INVOICE_ROWS]})

    assert blocked.value.unresolved == ["data.line_items[row 2].total"]
    assert "data.line_items[row 2].total" in str(blocked.value)


def test_a_row_without_the_columns_key_blocks_the_pdf_too():
    out = _rendered(_price_table(), {"line_items": [{"description": "Decaf 1 kg", "price": "48.00"}]})

    assert out.unresolved == ["data.line_items[row 1].total"]


def test_a_cell_of_spaces_is_empty():
    out = _rendered(_price_table(), {"line_items": [{"description": "Decaf 1 kg", "total": "   "}]})

    assert out.unresolved == ["data.line_items[row 1].total"]


def test_a_column_the_template_marks_optional_may_stay_empty():
    columns = [*PRICE_COLUMNS, {"key": "notes", "label": "Notes", "optional": True}]
    rows = [{"description": "Decaf 1 kg", "total": "48.00", "notes": ""}, {"description": "Harbour", "total": "9"}]

    assert _rendered(_price_table(columns), {"line_items": rows}).unresolved == []


# --------------------------------------------------------------------------- #
# A chip of spaces is empty
# --------------------------------------------------------------------------- #


def test_bill_to_sent_as_spaces_blocks_the_invoice():
    data = {**INVOICE["sample_data"]["data"], "client_name": "   "}

    out = _rendered(INVOICE["blocks"], data)

    assert out.unresolved == ["data.client_name"]
    assert "[[data.client_name]]" in out.html


# --------------------------------------------------------------------------- #
# Legacy (Jinja) templates: their required fields, and an empty page
# --------------------------------------------------------------------------- #

LEGACY_INVOICE = {
    "company": {"name": "Harbourline Coffee Roasters"},
    "invoice_number": "HL-0151",
    "line_items": [{"description": "Harbour Blend 1 kg", "quantity": 12, "unit_price": 19.5, "total": 234.0}],
    "subtotal": 234.0,
    "tax": 0,
    "total": 234.0,
    "payment_terms": "14 days",
}


def test_the_legacy_invoice_without_a_client_is_never_delivered(monkeypatch, tmp_path):
    with pytest.raises(UnresolvedDeliverableError) as blocked:
        _generate(monkeypatch, tmp_path, _legacy("Invoice"), "pdf", dict(LEGACY_INVOICE))

    assert blocked.value.unresolved == ["data.client"]


def test_the_legacy_invoices_client_needs_a_name_and_each_row_its_total(monkeypatch, tmp_path):
    data = {**LEGACY_INVOICE, "client": {"address": "12 Harbour Street"},
            "line_items": [{"description": "Decaf 1 kg", "quantity": 2, "unit_price": 24.0, "total": " "}]}

    with pytest.raises(UnresolvedDeliverableError) as blocked:
        _generate(monkeypatch, tmp_path, _legacy("Invoice"), "pdf", data)

    assert blocked.value.unresolved == ["data.client.name", "data.line_items[row 1].total"]


def test_an_executive_summary_without_highlights_is_not_a_blank_page(monkeypatch, tmp_path):
    with pytest.raises(UnresolvedDeliverableError) as blocked:
        _generate(monkeypatch, tmp_path, _legacy("Executive Summary"), "pdf", {"date": "2026-10-05"})

    assert blocked.value.unresolved == ["data.highlights"]


def test_a_legacy_page_that_shows_none_of_the_data_is_refused(monkeypatch, tmp_path):
    source = "<html><body><h1>{{ title }}</h1>{% if summary %}<p>{{ summary }}</p>{% endif %}</body></html>"
    template = SimpleNamespace(id=uuid.uuid4(), name="Owner summary", format="pdf", blocks=None,
                               template_content=source, data_schema={}, template_file_path=None)

    with pytest.raises(EmptyDocumentError) as blocked:
        _generate(monkeypatch, tmp_path, template, "pdf", {"overview": "Sales up 12% on September."})

    assert blocked.value.unresolved == ["data.summary"]
    assert "came out empty" in str(blocked.value) and "data.summary" in str(blocked.value)


def _check_basic_report(data: Dict[str, Any], title: str = "Monday dispatch checklist") -> None:
    from jinja2.sandbox import SandboxedEnvironment

    from modules.documents.legacy_guard import check_legacy_template
    from modules.documents.legacy_jinja import with_document_filters

    env = with_document_filters(SandboxedEnvironment(autoescape=True))
    check_legacy_template(env, _legacy("Basic Report"), data, title, True)


def test_a_body_sent_alone_prints_as_the_platforms_untitled_section_and_is_not_blocked():
    _check_basic_report({"content": "- [ ] Weigh the Harbour Blend bags"})  # F298: Auto's usual call


def test_a_section_the_caller_sent_without_its_title_is_still_blocked():
    with pytest.raises(UnresolvedDeliverableError) as blocked:
        _check_basic_report({"sections": [{"title": " ", "content": "Weigh the bags"}]})

    assert blocked.value.unresolved == ["data.sections[row 1].title"]


def test_a_body_sent_alone_without_a_document_title_is_blocked_on_the_title():
    with pytest.raises(UnresolvedDeliverableError) as blocked:
        _check_basic_report({"content": "Weigh the bags"}, title="")

    assert "data.title" in blocked.value.unresolved


# --------------------------------------------------------------------------- #
# Business and legal terms are asked for, never defaulted
# --------------------------------------------------------------------------- #


def test_a_contract_without_governing_law_asks_for_it():
    data = {k: v for k, v in CONTRACT["sample_data"]["data"].items() if k != "governing_law"}

    out = _rendered(CONTRACT["blocks"], data)

    assert out.unresolved == ["data.governing_law"]
    assert "laws of Ireland" not in out.html


def test_an_invoice_without_payment_terms_or_tax_asks_for_both():
    data = {k: v for k, v in INVOICE["sample_data"]["data"].items() if k not in ("payment_terms", "tax")}

    out = _rendered(INVOICE["blocks"], data)

    assert sorted(out.unresolved) == ["data.payment_terms", "data.tax"]
    assert "Net 30" not in out.html and ">0.00<" not in out.html
    assert "Thank you for your business." in out.html  # a cosmetic line keeps its words


# --------------------------------------------------------------------------- #
# The card and the agents can say what is required and what fills itself
# --------------------------------------------------------------------------- #


def test_the_studio_card_says_what_is_required_and_what_fills_itself():
    card = summarize_template(SimpleNamespace(id="t", name="Branded Invoice", format="pdf", blocks=INVOICE["blocks"],
                                              created_by="system"))

    assert {"data.payment_terms", "data.tax", "data.client_name", "data.line_items"} <= set(card["required_fields"])
    assert card["fallback_fields"]["data.client_address"] == ""
    assert "data.payment_terms" not in card["fallback_fields"]
    assert card["tables"] == [{"field": "line_items", "columns": ["description", "quantity", "unit_price", "total"],
                               "optional_columns": [], "required": True}]


def test_a_legacy_templates_card_reads_its_schema_and_its_defaults():
    card = summarize_template(_legacy("Invoice", created_by="system"))

    assert {"data.client", "data.client.name", "data.payment_terms", "data.tax"} <= set(card["required_fields"])
    assert card["fallback_fields"]["data.client.address"] == ""
    assert "data.client.name" not in card["fallback_fields"]
    (items,) = card["tables"]
    assert items["field"] == "line_items" and items["optional_columns"] == [] and items["required"] is True
