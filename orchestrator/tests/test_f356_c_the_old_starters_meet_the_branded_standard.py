"""F356 (5 Oct): the four old-style starters meet the Branded standard.

Basic Report, the plain Invoice, Data Export and Meeting Notes printed with their
own CSS or none: no letterhead, no logo on two of them, no footer on the Invoice,
and Meeting Notes had no template at all, so its own format (docx) could not
render. They now carry the letterhead, the footer and the shared design system,
under the names and data fields saved copies and agents already use. A row a
workspace was seeded with earlier is refreshed when it is still the platform's
own and unedited; a row anyone edited is left alone.
"""
from __future__ import annotations

import base64
import copy
import hashlib
import os
import struct
import zipfile
import zlib
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List

import pdfplumber
import pytest
from jinja2.sandbox import SandboxedEnvironment

from modules.documents.blocks import (
    collect_variable_paths, legacy_render_data, render_document_docx, render_document_html, validate_blocks,
)
from modules.documents.brand_kit import get_brand_kit
from modules.documents.data_coverage import block_template_keys, legacy_template_keys
from modules.documents.legacy_jinja import legacy_brand, with_document_filters
from modules.documents.presets import MEETING_NOTES_BLOCKS
from modules.documents.seed_templates import (
    RETIRED_SEED_SOURCES, STARTER_TEMPLATES, refresh_retired_source, seed_source, seed_starter_templates,
    takes_starter_blocks,
)
from modules.documents.variables.chip_text import chip_text
from modules.documents.variables.resolver import build_context, resolve_paths
from modules.documents.xlsx_render import write_xlsx

COMPANY = "Automatos AI"
NOW = datetime(2026, 10, 5, 9, 0, 0)
USER = SimpleNamespace(name="Gerard Kavanagh", email="gerard@automatos.app", username="gerard")
SEEDS = {seed["name"]: seed for seed in STARTER_TEMPLATES}
SAME_EDGE_PT = 2.0


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


def _sample(name: str) -> Dict[str, Any]:
    data = copy.deepcopy(SEEDS[name]["sample_data"])
    data.setdefault("title", name)
    return data


def _read(tmp_path, html: str) -> List[Any]:
    from weasyprint import HTML

    path = tmp_path / "starter.pdf"
    HTML(string=html).write_pdf(str(path))
    with pdfplumber.open(str(path)) as pdf:
        return [SimpleNamespace(words=p.extract_words(), text=p.extract_text() or "", images=list(p.images))
                for p in pdf.pages]


def _legacy_html(name: str, data: Dict[str, Any]) -> str:
    env = with_document_filters(SandboxedEnvironment(autoescape=True))
    # As generation_service hands it over: the kit with its colour roles and type scale (PRD-255).
    return env.from_string(seed_source(SEEDS[name])).render(**legacy_render_data(data), brand=legacy_brand(KIT))


def _block_render(data: Dict[str, Any]) -> Any:
    doc = validate_blocks(MEETING_NOTES_BLOCKS)
    resolved = resolve_paths(build_context(USER, None, KIT, NOW, extra_data=data), collect_variable_paths(doc))
    return doc, resolved.values, render_document_html(doc, resolved.values, KIT, title="Meeting Notes", data=data)


def test_the_names_and_formats_are_kept():
    assert [(s["name"], s["format"]) for s in STARTER_TEMPLATES] == [
        ("Basic Report", "pdf"), ("Invoice", "pdf"), ("Executive Summary", "pdf"), ("Meeting Notes", "docx"),
        ("Data Export", "xlsx"),
    ]


@pytest.mark.parametrize("name, keys", [
    ("Basic Report", {"title", "date", "author", "company_name", "sections", "metrics"}),
    ("Invoice", {"company", "client", "invoice_number", "date", "due_date", "line_items", "subtotal", "tax", "total", "payment_terms"}),
    ("Executive Summary", {"title", "date", "author", "highlights", "metrics", "recommendations"}),
])
def test_a_jinja_starter_still_reads_every_field_it_read(name, keys):
    read = legacy_template_keys(seed_source(SEEDS[name]), {})
    assert keys <= read, keys - read
    assert "document_style" not in read  # a filter, never counted as a data field


def test_meeting_notes_reads_its_schemas_fields():
    schema = SEEDS["Meeting Notes"]["data_schema"]["properties"]
    assert block_template_keys(MEETING_NOTES_BLOCKS) <= set(schema)
    assert {"title", "date", "attendees", "agenda", "notes", "action_items"} <= block_template_keys(MEETING_NOTES_BLOCKS)


@pytest.mark.parametrize("name", ["Basic Report", "Invoice", "Executive Summary"])
def test_a_jinja_starter_has_the_letterhead_and_footer(tmp_path, name):
    (page,) = _read(tmp_path, _legacy_html(name, _sample(name)))
    (logo,) = page.images
    lines = page.text.splitlines()
    assert any("Page 1 of 1" in line and COMPANY in line for line in lines), lines
    first = page.words[0]
    assert first["x0"] > logo["x1"] and logo["top"] <= first["top"] <= logo["bottom"], (first, logo)


def test_the_plain_invoices_totals_sit_under_its_total_column(tmp_path):
    (page,) = _read(tmp_path, _legacy_html("Invoice", _sample("Invoice")))
    header = min((w for w in page.words if w["text"] == "Total"), key=lambda w: w["top"])
    for amount in ("1500.00", "2000.00", "2200.00"):
        (word,) = [w for w in page.words if w["text"] == amount]
        assert abs(word["x1"] - header["x1"]) <= SAME_EDGE_PT, (amount, word, header)
    assert "$" not in page.text and "Invoice INV-001" in page.text


def test_meeting_notes_prints_attendees_decisions_and_an_actions_table(tmp_path):
    _, _, rendered = _block_render(_sample("Meeting Notes"))
    assert rendered.unresolved == []
    (page,) = _read(tmp_path, rendered.html)
    lines = [" ".join(line.split()) for line in page.text.splitlines()]
    assert "Alice, Bob, Carol" in lines, lines
    assert "Decisions" in lines and "Action Owner Due" in lines, lines
    assert any(line.startswith("Complete API integration Bob") for line in lines), lines


def test_meeting_notes_with_only_its_required_fields_prints_no_empty_sections():
    data = {"title": "Stand-up", "date": "2026-10-05", "attendees": ["Alice", "Bob"]}
    _, _, rendered = _block_render(data)
    assert rendered.unresolved == []
    for heading in ("Agenda", "Discussion", "Decisions", "Actions"):
        assert f"<h2>{heading}</h2>" not in rendered.html


def test_an_action_without_an_owner_or_due_date_is_not_blocked_but_one_without_a_task_is():
    base = {"title": "Stand-up", "date": "2026-10-05", "attendees": ["Alice"]}
    _, _, loose = _block_render({**base, "action_items": [{"task": "Book the venue"}]})
    assert loose.unresolved == [] and "Book the venue" in loose.html
    _, _, untasked = _block_render({**base, "action_items": [{"owner": "Bob"}]})
    assert untasked.unresolved and all("task" in path for path in untasked.unresolved)


def test_meeting_notes_renders_its_own_format():
    doc, values, _ = _block_render(_sample("Meeting Notes"))
    rendered = render_document_docx(doc, values, KIT, data=_sample("Meeting Notes"))
    assert rendered.unresolved == []
    cells = [cell.text for table in rendered.document.tables for row in table.rows for cell in row.cells]
    assert "Owner" in cells and "Complete API integration" in cells


def test_a_list_of_names_prints_as_a_list_not_as_python():
    assert chip_text("data.attendees", ["Alice", "Bob"]) == "- Alice\n- Bob"
    assert chip_text("data.total", 311.0) == "311.00"  # F347's amounts are unchanged
    assert chip_text("data.rows", [{"a": 1}]) == "[{'a': 1}]"  # not a list of names: as before


class _Seeder:
    """A session for the starter seeder: nothing exists yet, every row it adds is kept."""

    def __init__(self):
        self.rows: List[Any] = []

    def query(self, *args: Any) -> "_Seeder":
        return self

    def filter(self, *args: Any) -> "_Seeder":
        return self

    def first(self) -> None:
        return None

    def add(self, row: Any) -> None:
        self.rows.append(row)

    def commit(self) -> None:
        return None


def test_a_new_workspace_gets_meeting_notes_as_a_block_template():
    seeder = _Seeder()
    seed_starter_templates(seeder, "00000000-0000-0000-0000-000000000001")
    (row,) = [r for r in seeder.rows if r.name == "Meeting Notes"]
    assert row.blocks == MEETING_NOTES_BLOCKS and row.format == "docx"


def test_an_old_meeting_notes_row_takes_the_blocks_and_an_edited_one_is_left_alone():
    def row(**over: Any) -> SimpleNamespace:
        base = dict(created_by="system", is_active=True, blocks=None, template_content=None, template_file_path=None)
        return SimpleNamespace(**{**base, **over})

    seed = SEEDS["Meeting Notes"]
    old = row()
    assert refresh_retired_source(old, seed) == 1 and old.blocks == MEETING_NOTES_BLOCKS
    for kept in (row(created_by="user_1"), row(is_active=False), row(template_file_path="/x/notes.docx"),
                 row(template_content="<p>Our notes</p>"), row(blocks={"version": 1, "blocks": [{"type": "page_break", "id": "p"}]})):
        assert not takes_starter_blocks(kept, seed)


@pytest.mark.parametrize("name", ["Basic Report", "Invoice", "Executive Summary"])
def test_the_current_jinja_source_is_not_retired_and_earlier_ones_are(name):
    source = seed_source(SEEDS[name])
    retired = RETIRED_SEED_SOURCES[SEEDS[name]["template_file"]]
    assert hashlib.sha256(source.encode("utf-8")).hexdigest() not in retired
    assert len(retired) >= 2


def test_a_branded_data_export_has_a_letterhead_zebra_rows_and_a_printed_footer(tmp_path):
    brand = {"name": COMPANY, "logo": "", "colours": {"primary": "#c44a1a", "accent": "#1d3658", "text": "#1a1a2e"},
             "fonts": {"body": "", "heading": ""}}
    path = str(tmp_path / "export.xlsx")
    write_xlsx(path, "Sales Data", _sample("Data Export"), brand)
    with zipfile.ZipFile(path) as book:
        sheet = book.read("xl/worksheets/sheet1.xml").decode()
        strings = book.read("xl/sharedStrings.xml").decode()
        styles = book.read("xl/styles.xml").decode()
    assert "Sales Data" in strings and COMPANY in strings
    assert '<c r="A4"' in sheet  # the header row under the title and the company line
    assert 'ySplit="4"' in sheet and "<autoFilter" in sheet
    assert "Page &amp;P of &amp;N" in sheet and "<oddFooter>" in sheet
    assert '<fgColor rgb="FFC44A1A"/>' in styles and styles.count("<fill>") >= 4  # header and zebra fills
    assert os.path.getsize(path) > 0
