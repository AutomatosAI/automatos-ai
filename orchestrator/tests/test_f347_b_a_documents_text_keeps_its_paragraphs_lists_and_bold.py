"""F347 (night 10b): a document's text keeps its paragraphs, line breaks, lists and bold.

A Branded template's body is a chip filled with what an agent wrote. In the PDF
and the Word file alike, its two paragraphs ran into one, "**twice a week**"
printed with its asterisks, and its "- " and "1. " lines ran inline. Now a blank
line ends a paragraph, a single line break stays one, bold and italic are
marks, and list lines are list items; nothing else is read, and every piece of
text is escaped, so a ``<script>`` an agent wrote prints as text.

The PDF is rendered for real (WeasyPrint) and read back (pdfplumber); the Word
file is read back with python-docx.
"""
from __future__ import annotations

import re

import pdfplumber
import pytest

from modules.documents.blocks import render_document_docx, render_document_html, validate_blocks
from modules.documents.brand_kit import get_brand_kit
from tests.test_f347_a_a_branded_pdf_wears_the_brand_kits_font import block_template, pdf_lines, render_pdf

KIT = get_brand_kit(None)
DOC = {"version": 1, "blocks": [
    {"type": "heading", "id": "title", "level": 1, "content": [{"type": "text", "text": "Roasting notes"}]},
    {"type": "text", "id": "body", "content": [{"type": "variable", "path": "data.body"}]},
]}
BODY = (
    "We roast on Mondays.\n"
    "\n"
    "We deliver **twice a week**, *before noon*.\n"
    "Call ahead for Saturdays.\n"
    "\n"
    "- Harbour Blend\n"
    "- Christmas Blend\n"
    "\n"
    "1. Weigh the beans\n"
    "2. Seal the bags"
)


def _html(body: str, blocks=None) -> str:
    rendered = render_document_html(validate_blocks(blocks or DOC), {"data.body": body}, KIT)
    return rendered.html.split("<body>", 1)[1]


def test_paragraphs_line_breaks_lists_and_emphasis_are_html():
    html = _html(BODY)

    assert '<div class="doc-text" data-block="body"><p>We roast on Mondays.</p>' in html  # the starters' styles still reach it
    assert ("<p>We deliver <strong>twice a week</strong>, <em>before noon</em>.<br />"
            "Call ahead for Saturdays.</p>") in html
    assert "<ul><li>Harbour Blend</li><li>Christmas Blend</li></ul>" in html
    assert "<ol><li>Weigh the beans</li><li>Seal the bags</li></ol>" in html
    assert "*" not in html


def test_a_numbered_list_keeps_its_first_number():
    assert '<ol start="3"><li>Roast</li><li>Pack</li></ol>' in _html("3. Roast\n4. Pack")


def test_script_in_a_value_is_escaped_and_html_never_passes_through():
    html = _html("<script>alert(1)</script>\n**<b>loud</b>**\n- <img src=x onerror=alert(1)>")

    assert "<script>" not in html and "<b>" not in html and "<img" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<strong>&lt;b&gt;loud&lt;/b&gt;</strong>" in html
    assert "<li>&lt;img src=x onerror=alert(1)&gt;</li>" in html


def test_a_list_marker_mid_line_and_the_authors_own_text_print_as_written():
    blocks = {"version": 1, "blocks": [
        {"type": "text", "id": "note", "content": [{"type": "text", "text": "Note: "}, {"type": "variable", "path": "data.body"}]},
        {"type": "text", "id": "typed", "content": [{"type": "text", "text": "2 **x** 3", "marks": ["italic"]}]},
    ]}
    html = _html("- not a list", blocks)

    assert '<p data-block="note">Note: - not a list</p>' in html
    assert '<p data-block="typed"><em>2 **x** 3</em></p>' in html  # what the author typed keeps its own marks only


def test_a_chip_with_no_value_in_a_body_is_still_marked_and_blocks():
    rendered = render_document_html(validate_blocks(DOC), {}, KIT)

    assert "[[data.body]]" in rendered.html and rendered.unresolved == ["data.body"]


def test_the_pdf_keeps_two_paragraphs_two_and_prints_no_asterisks(monkeypatch, tmp_path):
    pdf = render_pdf(monkeypatch, tmp_path, {}, block_template(DOC), {"body": BODY})
    lines = pdf_lines(pdf)

    assert "We roast on Mondays." in lines, lines
    assert "We deliver twice a week, before noon." in lines, lines
    assert "Call ahead for Saturdays." in lines, lines
    assert not any("*" in line for line in lines), lines
    harbour = [line for line in lines if "Harbour Blend" in line]
    assert len(harbour) == 1 and "Christmas" not in harbour[0] and "- " not in harbour[0], lines
    assert any(re.fullmatch(r"1\.\s*Weigh the beans", line) for line in lines), lines
    assert any(re.fullmatch(r"2\.\s*Seal the bags", line) for line in lines), lines


def test_the_pdf_sets_the_bold_words_in_a_bold_face(monkeypatch, tmp_path):
    pdf = render_pdf(monkeypatch, tmp_path, {}, block_template(DOC), {"body": BODY})
    with pdfplumber.open(pdf) as document:
        words = [word for page in document.pages for word in page.extract_words(extra_attrs=["fontname"])]
    font_of = {word["text"]: word["fontname"] for word in words}

    assert font_of["twice"] != font_of["deliver"], font_of
    assert font_of["twice"] == font_of.get("week", font_of.get("week,")), font_of  # the bold run, one face
    assert font_of["before"] not in (font_of["twice"], font_of["deliver"]), font_of  # the italic run


def test_the_word_file_keeps_paragraphs_lists_and_bold():
    pytest.importorskip("docx")
    rendered = render_document_docx(validate_blocks(DOC), {"data.body": BODY}, KIT)
    paragraphs = rendered.document.paragraphs[1:]  # after the heading

    texts = [p.text for p in paragraphs]
    assert texts[0] == "We roast on Mondays."
    assert texts[1] == "We deliver twice a week, before noon.\nCall ahead for Saturdays."
    assert not any("*" in text for text in texts), texts
    assert any(run.bold and run.text == "twice a week" for run in paragraphs[1].runs)
    assert any(run.italic and run.text == "before noon" for run in paragraphs[1].runs)
    bullets = [p.text for p in paragraphs if p.style.name == "List Bullet"]
    assert bullets == ["Harbour Blend", "Christmas Blend"], texts
    assert texts[-2:] == ["1. Weigh the beans", "2. Seal the bags"], texts
