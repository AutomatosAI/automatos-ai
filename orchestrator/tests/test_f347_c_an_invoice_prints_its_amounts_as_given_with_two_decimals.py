"""F347 (night 10b): an invoice prints its amounts as given, with two decimals, and invents no currency.

The seeded legacy "Invoice" put a "$" before every amount, whatever the
business's currency, and the Branded Invoice printed an agent's ``"total": 311.0``
as "311.0". An amount now prints as it was given ("€24.00"), a bare number with
at least two decimals ("311.00"), and no currency is ever added. A quantity is
not an amount. The legacy row a workspace was seeded with takes the new source
when nobody edited it.

Rendered for real (WeasyPrint) and read back (pdfplumber).
"""
from __future__ import annotations

import copy
import hashlib
import os
import re
import uuid
from datetime import datetime
from types import SimpleNamespace

import pytest
from jinja2.sandbox import SandboxedEnvironment

from modules.documents.amounts import amount_text, field_text, is_amount_key
from modules.documents.blocks import collect_variable_paths, render_document_html, validate_blocks
from modules.documents.brand_kit import get_brand_kit
from modules.documents.data_coverage import legacy_template_keys
from modules.documents.legacy_jinja import with_document_filters
from modules.documents.presets import INVOICE
from modules.documents.seed_templates import (
    RETIRED_SEED_SOURCES,
    STARTER_TEMPLATES,
    TEMPLATES_DIR,
    holds_retired_source,
    refresh_retired_source,
)
from modules.documents.variables.resolver import build_context, resolve_paths
from tests.test_f347_a_a_branded_pdf_wears_the_brand_kits_font import block_template, pdf_lines, render_pdf

KIT_SETTINGS = {"brand_kit": {"name": "Harbourline Coffee Roasters", "company": {"name": "Harbourline Coffee Roasters"}}}
# An agent's invoice: amounts as JSON numbers, one as text with its currency.
AGENTS_INVOICE = {
    "client_name": "Lantern Kitchen",
    "invoice_number": "HL-2026-0142",
    "due_date": "19 October 2026",
    "line_items": [
        {"description": "Harbour Blend 1 kg", "quantity": 12, "unit_price": 19.5, "total": 234},
        {"description": "Decaf 1 kg", "quantity": 2, "unit_price": "€24.00", "total": 48.0},
    ],
    "subtotal": 282,
    "tax": 29.0,
    "total": 311.0,
}
INVOICE_SEED = next(t for t in STARTER_TEMPLATES if t["name"] == "Invoice")
# The invoice.html sources workspaces were seeded with, byte for byte, as each shipped:
# PRD-167's ("$" before every amount) and F347's (before F345 took out its placeholders).
# Kept as files so a later change to invoice.html cannot change what they are.
RETIRED_DIR = os.path.join(os.path.dirname(__file__), "fixtures", "retired_seeds")


def _invoice_source() -> str:
    with open(os.path.join(TEMPLATES_DIR, INVOICE_SEED["template_file"]), encoding="utf-8") as handle:
        return handle.read()


def _retired(name: str) -> str:
    with open(os.path.join(RETIRED_DIR, name), encoding="utf-8", newline="") as handle:
        return handle.read()


def _f347_source() -> str:
    return _retired("invoice_f347.html")


def _retired_source() -> str:
    return _retired("invoice_prd167.html")


def _branded_invoice_html(data: dict) -> str:
    doc = validate_blocks(INVOICE["blocks"])
    kit = get_brand_kit(KIT_SETTINGS)
    context = build_context(None, None, kit, datetime(2026, 10, 5), data)
    values = resolve_paths(context, collect_variable_paths(doc)).values
    return render_document_html(doc, values, kit, data=data).html


@pytest.mark.parametrize("value, printed", [
    (311, "311.00"), (311.0, "311.00"), ("311", "311.00"), (19.5, "19.50"), (-4, "-4.00"),
    (0.125, "0.125"), ("€19.50", "€19.50"), ("1,500.00", "1,500.00"), ("311 EUR", "311 EUR"),
    (None, ""), (True, "True"),
])
def test_an_amount_prints_as_given_a_bare_number_with_two_decimals(value, printed):
    assert amount_text(value) == printed


def test_only_an_amount_key_is_printed_as_an_amount():
    assert all(is_amount_key(key) for key in ("total", "unit_price", "line_total", "amount_due", "data.subtotal", "VAT"))
    assert not any(is_amount_key(key) for key in ("quantity", "tax_rate", "total_hours", "invoice_number"))
    assert field_text("quantity", 2) == "2" and field_text("data.total", 311.0) == "311.00"


def test_the_branded_invoice_prints_two_decimals_and_no_currency_it_was_not_given():
    html = _branded_invoice_html(AGENTS_INVOICE)

    assert "<td>311.00</td>" in html and "<td>282.00</td>" in html and "<td>29.00</td>" in html
    assert '<td style="text-align:right">19.50</td>' in html and '<td style="text-align:right">234.00</td>' in html
    assert '<td style="text-align:right">€24.00</td>' in html and '<td style="text-align:right">48.00</td>' in html
    assert '<td style="text-align:right">12</td>' in html  # a quantity is not an amount
    assert "311.0<" not in html and "$" not in html


def test_the_branded_invoice_pdf_prints_311_00(monkeypatch, tmp_path):
    pdf = render_pdf(monkeypatch, tmp_path, KIT_SETTINGS, block_template(INVOICE["blocks"]), AGENTS_INVOICE)
    text = "\n".join(pdf_lines(pdf))

    assert "311.00" in text and "282.00" in text and "19.50" in text and "€24.00" in text, text
    assert not re.search(r"311\.0(?!0)", text) and "$" not in text, text


def test_the_legacy_invoice_prints_no_dollar_sign(monkeypatch, tmp_path):
    template = SimpleNamespace(id=uuid.uuid4(), name="Invoice", blocks=None, template_content=_invoice_source(),
                               data_schema=INVOICE_SEED["data_schema"], template_file_path=None)
    data = {**copy.deepcopy(INVOICE_SEED["sample_data"]), "total": "€2,200.00"}

    text = "\n".join(pdf_lines(render_pdf(monkeypatch, tmp_path, {}, template, data)))

    assert "$" not in text, text
    assert "1500.00" in text and "150.00" in text and "2000.00" in text and "200.00" in text, text
    assert "€2,200.00" in text, text  # an amount sent as text prints as given (the old "%.2f" failed on it)


def test_the_amount_filter_renders_in_the_sandbox_and_the_coverage_parser_reads_it():
    env = with_document_filters(SandboxedEnvironment(autoescape=True))

    assert env.from_string("{{ total | amount }}").render(total=311.0) == "311.00"
    keys = legacy_template_keys(_invoice_source(), {})
    assert keys is not None and {"line_items", "subtotal", "total"} <= keys


def test_the_retired_source_is_the_one_a_workspace_was_seeded_with():
    retired = _retired_source()

    assert "$" in retired and "| amount" not in retired
    assert hashlib.sha256(retired.encode("utf-8")).hexdigest() in RETIRED_SEED_SOURCES["invoice.html"]


def test_a_workspace_seeded_while_f347_was_current_holds_a_retired_source_too():
    assert hashlib.sha256(_f347_source().encode("utf-8")).hexdigest() in RETIRED_SEED_SOURCES["invoice.html"]
    assert "| amount" in _f347_source() and "default('Company Name')" in _f347_source()


@pytest.mark.parametrize("retired", [_retired_source, _f347_source])
def test_a_seeded_invoice_nobody_edited_takes_the_new_source(retired):
    row = SimpleNamespace(template_content=retired(), created_by="system", is_active=True, updated_at=None)

    assert refresh_retired_source(row, INVOICE_SEED) == 1
    assert row.template_content == _invoice_source() and "$" not in row.template_content
    assert row.updated_at is not None
    assert refresh_retired_source(row, INVOICE_SEED) == 0  # already current


@pytest.mark.parametrize("row", [
    SimpleNamespace(template_content=_retired_source() + "<!-- mine -->", created_by="system", is_active=True),
    SimpleNamespace(template_content=_retired_source(), created_by="7", is_active=True),
    SimpleNamespace(template_content=_retired_source(), created_by="system", is_active=False),
])
def test_an_edited_a_users_or_a_deleted_invoice_is_left_alone(row):
    before = row.template_content

    assert not holds_retired_source(row, INVOICE_SEED)
    assert refresh_retired_source(row, INVOICE_SEED) == 0 and row.template_content == before
