"""PRD-243 — categories are layouts, and tables can fill from data.

* ``data_table`` block: schema rules (data.* path, unique keys), path collection,
  HTML/DOCX rendering from a list, the empty policy (unresolved unless
  ``empty_text``), list-field discovery for the Studio/tool schema;
* presets: one per category, unique names, every preset renders CLEAN against a
  filled brand kit + its own sample data (no unresolved, no unknown chips) — the
  gate that says a starter never ships a blocked document;
* starters refresh in place when platform-owned and drifted, never a user's row;
* headings no longer draw rules (the review complaint).

Pure — no DB, no WeasyPrint; the DOCX test skips if python-docx is absent.
"""

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace

import pytest

from modules.documents.blocks import (
    BlockValidationError,
    collect_list_fields,
    collect_variable_paths,
    render_document_html,
    validate_blocks,
)
from modules.documents.brand_kit import get_brand_kit
from modules.documents.presets import CATEGORIES, PRESETS, preset_for, preset_payload
from modules.documents.seed_templates import starter_columns, starter_outcome
from modules.documents.template_summary import summarize_template
from modules.documents.variables.resolver import build_context, resolve_paths

NOW = datetime(2026, 9, 15, 9, 0, 0)
KIT = get_brand_kit(
    {
        "brand_kit": {
            "name": "Acme",
            "logo_url": "https://acme.example/logo.png",
            "company": {"name": "Acme Ltd", "address": "1 Main St", "email": "hi@acme.example", "phone": "+353 1 555 0100", "website": "acme.example"},
        }
    }
)
USER = SimpleNamespace(name="Gerard Kavanagh", email="gerard@automatos.app", username="gerard")


def _table(path="data.line_items", empty_text=None, columns=None):
    block = {
        "type": "data_table",
        "id": "t",
        "path": path,
        "columns": columns or [{"key": "description", "label": "Description"}, {"key": "total", "label": "Total", "align": "right"}],
    }
    if empty_text is not None:
        block["empty_text"] = empty_text
    return {"blocks": [block]}


# --------------------------------------------------------------------------- #
# data_table block
# --------------------------------------------------------------------------- #


def test_data_table_requires_a_data_path_and_unique_keys():
    with pytest.raises(BlockValidationError) as exc:
        validate_blocks(_table(path="company.name"))
    assert any("data.*" in e["msg"] for e in exc.value.errors)
    with pytest.raises(BlockValidationError):
        validate_blocks(_table(columns=[{"key": "a"}, {"key": "a"}]))
    with pytest.raises(BlockValidationError):
        validate_blocks(_table(columns=[]))
    with pytest.raises(BlockValidationError):
        validate_blocks(_table(columns=[{"key": "not valid"}]))


def test_data_table_paths_and_list_fields_are_collected():
    doc = validate_blocks({"blocks": [{"type": "section", "id": "s", "title": "Items", "children": _table()["blocks"]}]})
    assert collect_variable_paths(doc) == {"data.line_items"}
    assert collect_list_fields(doc) == [{"field": "line_items", "columns": ["description", "total"]}]
    summary = summarize_template(SimpleNamespace(blocks={"blocks": _table()["blocks"]}, created_by="system"))
    assert summary["data_fields"] == ["line_items"]
    assert summary["list_fields"] == [{"field": "line_items", "columns": ["description", "total"]}]


def test_data_table_renders_rows_from_data_with_alignment():
    doc = validate_blocks(_table())
    data = {"line_items": [{"description": "Consulting", "total": "1,500.00"}, ["Positional row", "9.00"]]}
    out = render_document_html(doc, {}, KIT, data=data)
    assert out.unresolved == []
    assert "<th style=\"text-align:left\">Description</th>" in out.html
    assert "<th style=\"text-align:right\">Total</th>" in out.html
    assert "<td style=\"text-align:left\">Consulting</td>" in out.html
    assert "<td style=\"text-align:right\">9.00</td>" in out.html


def test_data_table_escapes_cell_values():
    doc = validate_blocks(_table())
    out = render_document_html(doc, {}, KIT, data={"line_items": [{"description": "<b>x</b>", "total": "1"}]})
    assert "<b>x</b>" not in out.html and "&lt;b&gt;x&lt;/b&gt;" in out.html


def test_data_table_empty_is_unresolved_unless_the_author_allows_it():
    doc = validate_blocks(_table())
    for data in ({}, {"line_items": []}, {"line_items": "not a list"}):
        out = render_document_html(doc, {}, KIT, data=data)
        assert out.unresolved == ["data.line_items"]
        assert "[[data.line_items]]" in out.html
    allowed = validate_blocks(_table(empty_text="No items this period."))
    out = render_document_html(allowed, {}, KIT, data={})
    assert out.unresolved == [] and "No items this period." in out.html


def test_data_table_renders_in_docx():
    pytest.importorskip("docx")
    from modules.documents.blocks import render_document_docx

    doc = validate_blocks(_table())
    out = render_document_docx(doc, {}, KIT, data={"line_items": [{"description": "Consulting", "total": "1,500.00"}]})
    assert out.unresolved == []
    table = out.document.tables[0]
    assert table.cell(0, 0).text == "Description" and table.cell(1, 1).text == "1,500.00"
    empty = render_document_docx(doc, {}, KIT, data={})
    assert empty.unresolved == ["data.line_items"]


# --------------------------------------------------------------------------- #
# presets
# --------------------------------------------------------------------------- #


def test_one_preset_per_category_with_unique_names():
    assert CATEGORIES == ["letter", "invoice", "report", "proposal", "contract", "data", "general"]
    names = [p["name"] for p in PRESETS]
    assert len(set(names)) == len(names)
    assert preset_for("INVOICE ")["category"] == "invoice"
    assert preset_for("nope")["category"] == "general"
    assert preset_for(None)["category"] == "general"


@pytest.mark.parametrize("preset", PRESETS, ids=[p["category"] for p in PRESETS])
def test_every_preset_renders_clean_with_its_sample_data(preset):
    """A starter must never ship a blocked document: with a filled brand kit and its
    own sample data every chip resolves and every data table has rows."""
    doc = validate_blocks(preset["blocks"])
    sample = preset["sample_data"]["data"]
    ctx = build_context(USER, None, KIT, NOW, extra_data=sample)
    resolved = resolve_paths(ctx, collect_variable_paths(doc))
    out = render_document_html(doc, resolved.values, KIT, title=preset["name"], data=sample)
    assert resolved.unknown == []
    assert out.unresolved == []
    assert 'src="https://acme.example/logo.png"' in out.html  # every preset carries the brand logo


@pytest.mark.parametrize("preset", PRESETS, ids=[p["category"] for p in PRESETS])
def test_preset_payload_is_derived_not_hand_kept(preset):
    payload = preset_payload(preset)
    assert payload["category"] == preset["category"] and payload["blocks"] == preset["blocks"]
    assert payload["data_fields"] and all(not f.startswith("data.") for f in payload["data_fields"])
    for lf in payload["list_fields"]:
        assert lf["field"] in payload["data_fields"] and lf["columns"]
    assert payload["includes"]


def test_invoice_and_report_presets_fill_tables_from_data():
    invoice = preset_payload(preset_for("invoice"))
    assert {"field": "line_items", "columns": ["description", "quantity", "unit_price", "total"]} in invoice["list_fields"]
    report = preset_payload(preset_for("report"))
    assert [lf["field"] for lf in report["list_fields"]] == ["metrics"]


# --------------------------------------------------------------------------- #
# starters refresh in place (platform-owned only)
# --------------------------------------------------------------------------- #


def test_starter_outcome_creates_refreshes_or_leaves_user_rows():
    preset = preset_for("letter")
    assert starter_outcome(None, preset) == "created"
    identical = SimpleNamespace(created_by="system", **starter_columns(preset))
    assert starter_outcome(identical, preset) == "unchanged"
    drifted = SimpleNamespace(created_by="system", **{**starter_columns(preset), "blocks": {"version": 1, "blocks": []}})
    assert starter_outcome(drifted, preset) == "refreshed"
    users_row = SimpleNamespace(created_by="user_abc", **{**starter_columns(preset), "blocks": {"version": 1, "blocks": []}})
    assert starter_outcome(users_row, preset) == "user_owned"


# --------------------------------------------------------------------------- #
# headings draw no rules
# --------------------------------------------------------------------------- #


def test_headings_no_longer_draw_rules():
    doc = validate_blocks({"blocks": [{"type": "heading", "id": "h", "level": 2, "content": [{"type": "text", "text": "Name"}]}]})
    html = render_document_html(doc, {}, KIT).html
    style = html[html.index("<style>") : html.index("</style>")]
    for line in style.splitlines():
        if line.strip().startswith(("h1", "h2", "h3", "h1,")):
            assert "border" not in line, line
