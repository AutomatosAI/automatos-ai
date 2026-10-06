"""F374 (night 10c): a .md report's card picture reads the markdown as the report view does.

The RESEARCHER sweep and TRACKER pass cards drew their lines that start with a backticked
ticket number as full-size code, and their bullets out of line. The thumbnail page was
made by the blog's renderer (Python-Markdown with nl2br): a list written straight under a
line of text folded into that paragraph as "- `F331` …" lines, items nested two spaces a
level came out flat, and the page had no style for code. The report view
(react-markdown with GFM) reads the same text as CommonMark. The page now goes through
the document renderer's reading of an agent's markdown (``view_html``) with the view's
rules and sizes: a list under a line is a list, two-space nesting nests at any depth, a
single line break is a space, a fence is code, and inline code is a small chip in its line.
"""
from __future__ import annotations

from bs4 import BeautifulSoup

from modules.documents.thumbnails.html_sources import MARKDOWN_CLASS, MARKDOWN_CSS, document_html

TRACKER_PASS = (
    "**TRACKER pass**\n"
    "- `F331` blank invoice PDFs\n"
    "- `F333` only c1 mounted locally\n"
)
RESEARCHER_SWEEP = (
    "## RESEARCHER sweep\n"
    "- Branded documents\n"
    "  - `F331` blank invoice PDFs\n"
    "    - routed to FIXER\n"
    "- Socials\n"
)


def _body(markdown: str) -> BeautifulSoup:
    page = BeautifulSoup(document_html(markdown.encode(), ".md"), "html.parser")
    return page.find("div", class_=MARKDOWN_CLASS)


def test_a_list_under_a_line_of_text_is_a_list_with_its_code_inline():
    body = _body(TRACKER_PASS)

    items = body.find("ul").find_all("li", recursive=False)
    assert [li.get_text() for li in items] == ["F331 blank invoice PDFs", "F333 only c1 mounted locally"]
    assert [li.code.get_text() for li in items] == ["F331", "F333"]
    assert body.find("pre") is None
    assert "- " not in body.p.get_text()


def test_items_nested_two_spaces_a_level_nest_at_every_depth():
    top = _body(RESEARCHER_SWEEP).find("ul")

    items = top.find_all("li", recursive=False)
    assert [li.find(string=True).strip() for li in items] == ["Branded documents", "Socials"]
    child = items[0].ul.li
    assert child.code.get_text() == "F331"
    assert child.ul.li.get_text().strip() == "routed to FIXER"


def test_a_single_line_break_is_a_space_as_in_the_view():
    body = _body("`F331` blank invoice PDFs\n`F333` only c1 mounted locally\n")

    assert body.find("br") is None
    assert len(body.find_all("p")) == 1
    assert len(body.find_all("code")) == 2


def test_a_fenced_block_is_code_and_its_lines_stay_as_written():
    body = _body("Run this:\n```\n- not a list\n\n## not a heading\n```\nDone.\n")

    assert body.pre.code.get_text() == "- not a list\n\n## not a heading\n"
    assert body.find("li") is None and body.find("h2") is None
    assert body.find_all("p")[-1].get_text() == "Done."


def test_raw_html_in_a_report_is_text_not_markup():
    body = _body("<script>alert(1)</script> and <b>bold</b>\n")

    assert body.find("script") is None and body.find("b") is None
    assert "alert(1)" in body.get_text()


def test_the_page_styles_code_as_a_chip_and_headings_at_the_views_sizes():
    # The view: 14px body; h1 1.5rem (24px), h2 1.25rem (20px), h3 1.0625rem (17px); code 0.875em.
    css = " ".join(MARKDOWN_CSS.split())
    assert ".md code { font-family: monospace; font-size: 0.875em; font-weight: normal;" in css
    assert ".md h1 { font-size: 1.714em;" in css
    assert ".md h2 { font-size: 1.429em;" in css
    assert ".md h3 { font-size: 1.214em; }" in css
    assert MARKDOWN_CSS in document_html(b"# Weekly report\n", ".md")
