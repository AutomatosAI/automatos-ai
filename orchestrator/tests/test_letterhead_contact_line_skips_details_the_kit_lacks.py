"""The letterhead's contact line prints only the details the brand kit has.

``presets.letterhead()`` joins email, phone and website with "  ·  ", each chip
falling back to "". A kit with an email and nothing else printed
"gerard@automatos.app  ·    ·  " in every Branded starter's letterhead, in the
PDF and the Word file alike. Now, in a block with a chip, a separator run prints
only between two pieces that printed something. A block with no chip is printed
as its author typed it.
"""
from __future__ import annotations

import re

import pytest

from modules.documents.blocks import render_document_html, validate_blocks
from modules.documents.blocks.text_body import segments
from modules.documents.brand_kit import get_brand_kit
from modules.documents.presets import letterhead


def _text(content, values):
    missing = []
    return "".join(seg.text for seg in segments(content, values, missing)), missing


def _contact_runs():
    doc = validate_blocks({"version": 1, "blocks": letterhead()})
    return next(block for block in doc.blocks if block.id == "lh-contact").content


def _block(*runs):
    doc = validate_blocks({"version": 1, "blocks": [{"type": "text", "id": "b", "content": list(runs)}]})
    return doc.blocks[0].content


@pytest.mark.parametrize("given, printed", [
    ({"company.email": "a@b.ie"}, "a@b.ie"),
    ({"company.phone": "01 234"}, "01 234"),
    ({"company.email": "a@b.ie", "company.website": "b.ie"}, "a@b.ie  ·  b.ie"),
    ({"company.email": "a@b.ie", "company.phone": "01 234", "company.website": "b.ie"}, "a@b.ie  ·  01 234  ·  b.ie"),
    ({}, ""),
])
def test_the_contact_line_joins_only_the_details_given(given, printed):
    assert _text(_contact_runs(), given) == (printed, [])


def test_a_block_with_no_chip_prints_its_separators_as_typed():
    runs = _block({"type": "text", "text": "Mon"}, {"type": "text", "text": "  ·  "}, {"type": "text", "text": ""})
    assert _text(runs, {}) == ("Mon  ·  ", [])


def test_a_space_between_two_chips_prints_only_when_both_are_filled():
    runs = _block(
        {"type": "variable", "path": "data.first"}, {"type": "text", "text": " "},
        {"type": "variable", "path": "data.last", "fallback": ""},
    )
    assert _text(runs, {"data.first": "Jo", "data.last": "Smith"}) == ("Jo Smith", [])
    assert _text(runs, {"data.first": "Jo"}) == ("Jo", [])


def test_a_separator_before_words_is_kept():
    runs = _block(
        {"type": "text", "text": "Date: "}, {"type": "variable", "path": "date.long"}, {"type": "text", "text": "  ·  "},
        {"type": "text", "text": "Due: "}, {"type": "variable", "path": "data.due_date"},
    )
    assert _text(runs, {"date.long": "6 October 2026", "data.due_date": "5 November 2026"}) == (
        "Date: 6 October 2026  ·  Due: 5 November 2026", [],
    )


def test_a_missing_chip_still_shows_and_keeps_its_separator():
    runs = _block(
        {"type": "variable", "path": "company.email", "fallback": ""}, {"type": "text", "text": "  ·  "},
        {"type": "variable", "path": "data.reference"},
    )
    assert _text(runs, {"company.email": "a@b.ie"}) == ("a@b.ie  ·  data.reference", ["data.reference"])


def test_the_pdf_letterhead_has_no_dangling_separator():
    kit = get_brand_kit({"brand_kit": {"name": "Harbourline", "company": {"name": "Harbourline", "email": "a@b.ie"}}})
    doc = validate_blocks({"version": 1, "blocks": letterhead()})
    values = {"company.name": "Harbourline", "company.email": "a@b.ie"}
    html = render_document_html(doc, values, kit).html
    contact = re.search(r'data-block="lh-contact"[^>]*>(.*?)</', html, re.S).group(1)
    assert "a@b.ie" in contact and "·" not in contact


def test_the_word_letterhead_has_no_dangling_separator():
    pytest.importorskip("docx")
    from tests.test_f356_d_the_word_file_matches_the_pdf import _docx  # its kit has an email, no phone or website

    header = _docx("invoice").sections[0].first_page_header
    company_cell = header.tables[0].rows[0].cells[1]
    lines = [p.text for p in company_cell.paragraphs]
    assert "gerard@automatos.app" in lines
    assert not any("·" in line for line in lines)
