"""F331 (night 10, 5 Oct): the agent is told which data keys the page does not show.

The agent that calls generate_document cannot see the page. Eleven invoices went
out a title on an empty page and every card said "done", because the tool's answer
listed every key it was sent. A template the caller names may still have no place
for some keys; the document is made, and the answer now names those keys and says
not to call it done.
"""
from __future__ import annotations

import asyncio
import copy
import os
import uuid
from types import SimpleNamespace
from typing import Any

from modules.documents.data_coverage import unused_data_keys
from modules.documents.generation_service import DocumentGenerationService
from modules.documents.models import GeneratedDocument
from modules.documents.presets import INVOICE
from modules.documents.seed_templates import STARTER_TEMPLATES, TEMPLATES_DIR
from modules.tools.formatting.result_formatter import ToolResultFormatter
from tests.f298_fixtures import FILENAME, RecordedService, call_tool, recorded

WS = uuid.UUID("00000000-0000-0000-0000-0000000331b1")
AUTOS_INVOICE = {
    "sections": [{"title": "Invoice for Lantern Kitchen", "content": "12 kg of Harbour Blend."}],
    "invoice_number": "HL-2026-0142",
    "customer_name": "Lantern Kitchen",
    "item_details": [{"item": "Harbour Blend", "quantity": "12 kg", "delivery_date": "2026-10-02"}],
}


def _basic_report() -> SimpleNamespace:
    seed = next(t for t in STARTER_TEMPLATES if t["name"] == "Basic Report")
    with open(os.path.join(TEMPLATES_DIR, seed["template_file"]), encoding="utf-8") as handle:
        source = handle.read()
    return SimpleNamespace(id=uuid.uuid4(), name="Basic Report", blocks=None, template_content=source,
                           data_schema=seed["data_schema"], template_file_path=None)


def test_a_block_template_names_the_keys_it_has_no_place_for():
    branded_invoice = SimpleNamespace(blocks=INVOICE["blocks"], template_content=None)

    assert unused_data_keys(branded_invoice, {**AUTOS_INVOICE, "title": "Invoice"}, "pdf") == [
        "sections", "customer_name", "item_details",
    ]


def test_basic_report_names_the_keys_it_has_no_place_for():
    assert unused_data_keys(_basic_report(), AUTOS_INVOICE, "pdf") == [
        "invoice_number", "customer_name", "item_details",
    ]


def test_no_template_leaves_nothing_out():
    assert unused_data_keys(None, AUTOS_INVOICE, "pdf") == []


def test_generate_carries_the_keys_a_named_template_left_out(monkeypatch):
    async def generate_pdf(template, data, workspace_id, title, user_id=None):
        return GeneratedDocument(path="/tmp/x.pdf", format="pdf", filename="x.pdf", size=1)

    basic_report = _basic_report()
    service = DocumentGenerationService(SimpleNamespace(), WS)
    service.template_service = SimpleNamespace(get_template_by_name=lambda ws, name: basic_report)
    monkeypatch.setattr(service, "generate_pdf", generate_pdf)

    result = asyncio.run(service.generate(title="Invoice HL-2026-0142", format="pdf", data=copy.deepcopy(AUTOS_INVOICE),
                                          workspace_id=WS, template_name="Basic Report"))

    assert result.unused_keys == ["invoice_number", "customer_name", "item_details"]


def _made_with_keys_left_out(monkeypatch: Any) -> None:
    recorded(monkeypatch)

    async def generate(self: Any, **kwargs: Any) -> Any:
        RecordedService.asked.append(kwargs)
        return SimpleNamespace(
            filename=FILENAME, format="pdf", download_url=f"/api/documents/generated/{FILENAME}", size=8671,
            content="# Invoice", template_id=None, template_name="Branded Invoice", s3_key="k",
            unused_keys=["customer_name", "item_details"],
        )

    monkeypatch.setattr(RecordedService, "generate", generate)


def _call() -> dict:
    return call_tool({"title": "Invoice HL-2026-0142", "format": "pdf", "template_name": "Branded Invoice",
                      "data": copy.deepcopy(AUTOS_INVOICE)})


def test_the_tools_answer_lists_the_keys_the_page_does_not_show(monkeypatch):
    _made_with_keys_left_out(monkeypatch)

    (made,) = _call()["results"]

    assert made["unused_data_keys"] == ["customer_name", "item_details"]


def test_the_agent_reads_that_the_page_lacks_them_and_is_not_done(monkeypatch):
    _made_with_keys_left_out(monkeypatch)

    summary = ToolResultFormatter.format_for_llm(_call(), "generate_document")

    (line,) = [line for line in summary.splitlines() if line.startswith("NOT IN THE DOCUMENT")]
    assert "customer_name, item_details" in line and "Do not call it done" in line


def test_a_document_that_shows_every_key_says_nothing_of_the_kind(monkeypatch):
    recorded(monkeypatch)

    answer = call_tool({"title": "Price list", "format": "pdf", "data": {"content": "* Decaf: £24.00"}})

    assert answer["results"][0]["unused_data_keys"] == []
    assert "NOT IN THE" not in ToolResultFormatter.format_for_llm(answer, "generate_document")
