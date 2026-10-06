"""F360 (night 10c): a PDF prints in the kit's fonts, or says it prints a substitute.

The night's kit named Geist (body) and Newsreader (headings) with no font files
uploaded; every PDF came out in DejaVu Sans and DejaVu Serif (pdffonts) and
nothing said so, while the Word file was set in Geist. Inter, Geist and Newsreader
now ship with the code (``modules/documents/fonts``, OFL), so a kit naming them
prints in them; a family the kit names but neither uploaded nor the code ships is
named on the first page ("Substitute font: DejaVu Sans for Brand Sans") and on the
brand board, and only then. The Word file's headings take the heading font, as
the PDF's do.

The PDF tests are real (WeasyPrint prints, pdfplumber reads back), the F347 way.
"""
from __future__ import annotations

import io
import re
from typing import Any, Dict, Set

import pdfplumber
import pytest

import tests.test_f347_a_a_branded_pdf_wears_the_brand_kits_font as f347
from modules.documents import bundled_fonts
from modules.documents.blocks import brand_board as bb
from modules.documents.blocks import render_document_docx, render_document_html, validate_blocks
from modules.documents.blocks.page_fonts import font_css, kit_font_uses, substitute_note
from modules.documents.blocks.page_style import css_string
from modules.documents.brand_kit import get_brand_kit
from modules.documents.presets import BRAND_BOARD

# The night's kit (c1): Geist and Newsreader named, no font files.
NIGHT_KIT = {
    "name": "Automatos AI",
    "font_family": "Geist, Inter, 'Segoe UI', system-ui, sans-serif",
    "heading_font": "Newsreader, Georgia, serif",
    "font_files": [],
}
UNKNOWN_KIT = {"font_family": "'Brand Sans', sans-serif", "heading_font": "'Brand Serif', serif"}
FACE_RULE = re.compile(r'@font-face \{ font-family: "([^"]+)"; src: url\("data:font/woff2;base64,')


def _faces(css: str) -> Dict[str, int]:
    found: Dict[str, int] = {}
    for family in FACE_RULE.findall(css):
        found[family] = found.get(family, 0) + 1
    return found


def _kit(raw: Dict[str, Any]) -> Dict[str, Any]:
    return get_brand_kit({"brand_kit": raw})


def test_every_bundled_face_is_a_complete_woff2_and_each_family_carries_its_licence():
    for family in bundled_fonts.BUNDLED_FAMILIES:
        assert (bundled_fonts.FONTS_DIR / f"OFL-{family}.txt").read_text().find("SIL Open Font License") > 0
        for weight, style in bundled_fonts.BUNDLED_FACE_SET:
            data = bundled_fonts.face_file(family, weight, style).read_bytes()
            assert data[:4] == b"wOF2" and int.from_bytes(data[8:12], "big") == len(data), (family, weight, style)


def test_the_nights_kit_resolves_to_the_bundled_geist_and_newsreader_with_no_substitute():
    uses = kit_font_uses(_kit(NIGHT_KIT))

    assert [(use.role, use.named, use.prints_in, use.substitute) for use in uses] == [
        ("Body", "Geist", "Geist", False), ("Headings", "Newsreader", "Newsreader", False),
    ]
    css = font_css(_kit(NIGHT_KIT))
    assert _faces(css) == {"Geist": 4, "Newsreader": 4}
    assert "Substitute font" not in css and "@top-right" not in css


def test_the_default_kit_prints_in_the_bundled_inter():
    css = font_css(_kit({}))
    assert _faces(css) == {"Inter": 4} and "@top-right" not in css


def test_a_family_neither_uploaded_nor_bundled_is_named_as_a_substitute_on_the_first_page():
    uses = kit_font_uses(_kit(UNKNOWN_KIT))
    note = substitute_note(uses)

    assert note == "Substitute font: DejaVu Sans for Brand Sans; DejaVu Serif for Brand Serif"
    css = font_css(_kit(UNKNOWN_KIT))
    assert f'@page :first {{ @top-right {{ content: "{css_string(note)}";' in css
    assert _faces(css) == {}  # nothing bundled is added for a family the code does not ship


def test_a_system_or_generic_family_is_never_called_a_substitute():
    for stack in ("'DejaVu Sans Mono', monospace", "serif", "system-ui, sans-serif"):
        assert substitute_note(kit_font_uses(_kit({"font_family": stack}))) == "", stack


def test_an_uploaded_face_of_a_bundled_family_wins_over_the_bundled_one():
    upload = {"family": "Geist", "weight": 400, "style": "normal", "data_uri": "data:font/woff2;base64,AAAA"}
    kit = {**_kit(NIGHT_KIT), "font_files": [upload]}

    css = font_css(kit)
    assert _faces(css) == {"Geist": 1, "Newsreader": 4}
    assert 'src: url("data:font/woff2;base64,AAAA")' in css


def test_the_brand_board_names_the_fonts_and_a_substitute():
    assert [line.text for line in bb.font_lines(_kit(NIGHT_KIT))] == ["Body: Geist", "Headings: Newsreader"]
    lines = bb.font_lines(_kit(UNKNOWN_KIT))
    assert all(line.substitute for line in lines)
    assert lines[0].text == ("Body: Brand Sans is not uploaded, so PDFs print it in DejaVu Sans. "
                             "Upload its woff2 on the Brand kit page.")

    page = render_document_html(validate_blocks(BRAND_BOARD["blocks"]), {}, _kit(NIGHT_KIT), data={}).html
    assert '<p class="board-caption board-font">Headings: Newsreader</p>' in page
    assert 'class="board-type-sample board-type-heading"' in page  # the heading steps' samples, in Newsreader


def test_the_word_files_headings_take_the_heading_font():
    page = {"version": 1, "blocks": [
        {"type": "heading", "id": "title", "level": 1, "content": [{"type": "text", "text": "Price list"}]},
        {"type": "text", "id": "body", "content": [{"type": "text", "text": "Harbour Blend"}]},
    ]}
    document = render_document_docx(validate_blocks(page), {}, _kit(NIGHT_KIT)).document

    assert document.styles["Heading 1"].font.name == "Newsreader"
    assert document.styles["Normal"].font.name == "Geist"
    heading = next(p for p in document.paragraphs if p.text == "Price list")
    assert all(run.font.name is None for run in heading.runs)  # the style's font, not the body's


def _pdf_fonts(pdf: bytes) -> Set[str]:
    with pdfplumber.open(io.BytesIO(pdf)) as document:
        return {f347.flat(char["fontname"]) for page in document.pages for char in page.chars}


def _print(kit: Dict[str, Any]) -> bytes:
    from weasyprint import HTML

    return HTML(string=render_document_html(validate_blocks(f347.PAGE), {"data.body": f347.DATA["body"]}, kit).html).write_pdf()


def test_the_nights_kit_prints_in_geist_and_newsreader_and_no_dejavu():
    fonts = _pdf_fonts(_print(_kit(NIGHT_KIT)))

    assert any("Geist" in name for name in fonts), fonts
    assert any("Newsreader" in name for name in fonts), fonts
    assert not any("DejaVu" in name for name in fonts), fonts


@pytest.mark.parametrize("kit, marked", [(UNKNOWN_KIT, True), (NIGHT_KIT, False)])
def test_the_first_page_says_substitute_font_only_when_the_pdf_falls_back(kit, marked):
    with pdfplumber.open(io.BytesIO(_print(_kit(kit)))) as document:
        text = " ".join((document.pages[0].extract_text() or "").split())

    assert ("Substitute font: DejaVu Sans for Brand Sans" in text) is marked, text
