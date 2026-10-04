"""F298 (night 8, #0249 and #0394.2): a generated PDF prints markdown as formatting.

The owner asked for a printable checklist and got three unusable PDFs: "## … - [ ]"
in one paragraph, then the "[ ]" jobs run together under headings. #0394.2's
price list printed "* Harbour Blend: £19.50 per kilo + VAT * Christmas Blend…" on
one line with literal "*". Both lanes a PDF with no chosen template renders
through are rendered for real here (WeasyPrint) and read back line by line
(pdfplumber): the seeded "Basic Report" legacy template every workspace gets,
and the block fallback a workspace without it uses.
"""
from __future__ import annotations

import asyncio
import os
import re
import uuid
from types import SimpleNamespace
from typing import Any, List

import pdfplumber
import pytest

import modules.documents.generation_service as generation_service
from modules.documents.blocks.markdown_body import CHECKED_BOX, UNCHECKED_BOX
from modules.documents.generation_service import DocumentGenerationService
from modules.documents.seed_templates import STARTER_TEMPLATES, TEMPLATES_DIR

WS = uuid.UUID("00000000-0000-0000-0000-0000000298b1")

CHECKLIST = (
    "## Before the collection\n"
    "- [ ] Check Thursday's roast is bagged and labelled\n"
    "- [ ] Count the boxes against the printed labels\n"
    "- [x] Print the Royal Mail labels from Shopify\n"
)
# As the agent wrote it on #0394.2: the list straight under a bold line, no blank line between.
PRICE_LIST = (
    "**Harbourline Coffee Roasters - Wholesale Price List**\n"
    "*   Harbour Blend: £19.50 per kilo + VAT\n"
    "*   Christmas Blend: £21.00 per kilo + VAT\n"
    "*   Decaf: £24.00 per kilo + VAT\n"
    "\n"
    "Minimum order: *3 kg*\n"
    "\n"
    "| Blend | Bag |\n"
    "|---|---|\n"
    "| Harbour Blend | 1 kg |\n"
)
JOBS = [
    f"{UNCHECKED_BOX} Check Thursday's roast is bagged and labelled",
    f"{UNCHECKED_BOX} Count the boxes against the printed labels",
    f"{CHECKED_BOX} Print the Royal Mail labels from Shopify",
]
PRICES = ["Harbour Blend: £19.50 per kilo + VAT", "Christmas Blend: £21.00 per kilo + VAT", "Decaf: £24.00 per kilo + VAT"]


class _NoRows:
    """The workspace has no brand kit: every lookup finds nothing (the defaults apply)."""

    def query(self, *args: Any) -> "_NoRows":
        return self

    def filter(self, *args: Any, **kwargs: Any) -> "_NoRows":
        return self

    def first(self) -> None:
        return None


def _basic_report() -> SimpleNamespace:
    seed = next(t for t in STARTER_TEMPLATES if t["name"] == "Basic Report")
    with open(os.path.join(TEMPLATES_DIR, seed["template_file"]), encoding="utf-8") as handle:
        return SimpleNamespace(blocks=None, template_content=handle.read(), data_schema=seed["data_schema"])


@pytest.fixture
def render(monkeypatch, tmp_path):
    """generate_pdf for real into tmp_path; the file's text, one entry per printed line."""
    monkeypatch.setattr(generation_service, "GENERATED_DIR", str(tmp_path))
    monkeypatch.setattr(generation_service, "is_storage_configured", lambda: False)

    class _Resolver:  # the block fallback has no variable chips to resolve
        def __init__(self, db: Any):
            pass

        def resolve(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(values={}, unknown=[])

    monkeypatch.setattr(generation_service, "VariableResolver", _Resolver)

    def go(template: Any, data: dict) -> List[str]:
        service = DocumentGenerationService(_NoRows(), WS)
        result = asyncio.run(service.generate_pdf(template, data, WS, "Tom's Monday Dispatch Checklist"))
        with pdfplumber.open(result.path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        return [" ".join(line.split()) for line in text.splitlines() if line.strip()]

    return go


LANES = [pytest.param(_basic_report, id="basic-report-template"), pytest.param(lambda: None, id="block-fallback")]


def _squashed(text: str) -> str:
    return re.sub(r"\s+", "", text)


@pytest.mark.parametrize("template", LANES)
def test_each_job_prints_on_its_own_line_after_its_box(render, template):
    lines = render(template(), {"sections": [{"title": "Monday dispatch", "content": CHECKLIST}]})

    for job in JOBS:
        assert _squashed(job) in [_squashed(line) for line in lines], (job, lines)
    assert "Before the collection" in lines
    printed = "\n".join(lines)
    assert "[ ]" not in printed and "[x]" not in printed and "##" not in printed


@pytest.mark.parametrize("template", LANES)
def test_a_price_list_prints_one_price_per_line_without_asterisks(render, template):
    lines = render(template(), {"sections": [{"title": "December prices", "content": PRICE_LIST}]})

    for price in PRICES:
        (line,) = [line for line in lines if price in line]
        assert sum(other in line for other in PRICES) == 1, line
    assert "Harbourline Coffee Roasters - Wholesale Price List" in lines
    assert "Minimum order: 3 kg" in lines
    assert any("Harbour Blend" in line and "1 kg" in line for line in lines)
    assert "*" not in "\n".join(lines)
    assert not any(line.startswith("|") or line.endswith("|") or "---" in line for line in lines)


@pytest.mark.parametrize("template", LANES)
def test_a_body_sent_as_content_alone_is_printed(render, template):
    lines = render(template(), {"content": CHECKLIST})

    for job in JOBS:
        assert _squashed(job) in [_squashed(line) for line in lines], (job, lines)
