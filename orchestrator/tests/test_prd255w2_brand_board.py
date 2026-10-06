"""PRD-255 Wave 2 US-009: the brand board, the kit on one page (and as a social card).

The board is a block starter ("Brand Board", category ``brand``) whose parts are
``brand`` blocks, each drawn from the kit itself: the logo large and its variants
on light and dark (an unset variant shown as its FR-9 fallback, never invented),
the colour roles as swatches with their hex codes, the type scale, the spacing and
the logo's clear space, the tone words with their meanings, and three miniature
applications (the Branded Invoice and Letter printed with the kit, page 1 as PNGs,
and a social card drawn from the kit's colours). It has no data field to fill.

The render tests are real (the W1 pattern): WeasyPrint prints the board, pdfplumber
reads it back, python-docx writes the Word file.
"""
from __future__ import annotations

import base64
import io
import re
import struct
import zlib
from typing import Any, Dict
from uuid import UUID

import pdfplumber
import pytest

from core.brand_palette import PALETTE_ROLES, effective_palette
from core.media_render_bundle import build_bundle
from core.social_brand_rule import brand_literals
from core.social_templates import SOCIAL_IMAGE, resolve_variables
from modules.documents import brand_logo as bl
from modules.documents import seed_templates
from modules.documents.blocks import (
    BlockValidationError, collect_list_fields, collect_variable_paths, render_document_docx, render_document_html,
    validate_blocks,
)
from modules.documents.blocks import brand_board as bb
from modules.documents.blocks.schema import BRAND_PARTS
from modules.documents.brand_board_miniatures import starter_page
from modules.documents.brand_fonts import brand_kit_for_media_render
from modules.documents.brand_kit import get_brand_kit
from modules.documents.presets import BRAND_BOARD, INVOICE, LETTER, PRESETS
from modules.documents.social_starters import SOCIAL_BRAND_STARTER_SLUGS, social_starters
from modules.documents.template_summary import STARTER_CREATOR

WS = UUID("00000000-0000-0000-0000-0000000255b9")
COMPANY = "Automatos AI"
A4_PT = (595.28, 841.89)
PAGE_TOLERANCE_PT = 1.0
TONES = [
    {"word": "Clear", "meaning": "Says what it means in plain words."},
    {"word": "Calm", "meaning": "Never shouts; the work speaks."},
    {"word": "Exact", "meaning": "Numbers and names are right, every time."},
]
SIGN_OFF = "The Automatos team"
PNG_URI = "data:image/png;base64,"


def _png(side: int = 8, rgb: bytes = b"\x80\x80\x80") -> bytes:
    """A small opaque PNG: a stand-in for an uploaded logo."""
    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    rows = b"".join(b"\x00" + rgb * side for _ in range(side))
    header = struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


def _uri(data: bytes) -> str:
    return PNG_URI + base64.b64encode(data).decode("ascii")


LOGO, DARK_LOGO, MONO_LOGO = _uri(_png()), _uri(_png(rgb=b"\xf0\xf0\xf0")), _uri(_png(rgb=b"\x10\x10\x10"))


def _kit(**over: Any) -> Dict[str, Any]:
    """The Automatos kit, render-ready (its logo inlined, as ``brand_kit_for_media_render`` hands it on)."""
    kit = get_brand_kit({"brand_kit": {
        "name": COMPANY, "tagline": "Agents that do the work", "primary_color": "#c44a1a",
        "secondary_color": "#1d3658", "text_color": "#1a1a2e",
        "company": {"name": COMPANY, "address": "14 Wapping Quay, Bristol BS1 4RW", "email": "hello@automatos.app"},
        "voice": {"tone": TONES, "sign_off": SIGN_OFF},
    }})
    return {**kit, "logo_url": LOGO, **over}


AUTOMATOS = _kit()


def _board_html(kit: Dict[str, Any]) -> str:
    rendered = render_document_html(validate_blocks(BRAND_BOARD["blocks"]), {}, kit, title="Brand Board", data={})
    assert rendered.unresolved == []
    return rendered.html


def _pdf(page: str) -> bytes:
    from weasyprint import HTML

    return HTML(string=page).write_pdf()


def _part(page: str, block_id: str) -> str:
    """The HTML of the board part ``block_id``, up to the next part."""
    start = page.index(f'class="board-part" data-block="{block_id}"')
    following = page.find('class="board-part"', start + 1)
    return page[start:following if following != -1 else len(page)]


def _parts(blocks):
    """Every brand part in a block list, sections walked."""
    found = []
    for block in blocks:
        found += [block["part"]] if block["type"] == "brand" else _parts(block.get("children") or [])
    return found


# ---------------------------------------------------------------------------
# The starter: blocks drawn from the kit, nothing to fill
# ---------------------------------------------------------------------------


def test_the_brand_board_is_a_brand_starter_with_no_data_field():
    assert (BRAND_BOARD["name"], BRAND_BOARD["category"], BRAND_BOARD["format"]) == ("Brand Board", "brand", "pdf")
    doc = validate_blocks(BRAND_BOARD["blocks"])
    assert collect_variable_paths(doc) == set() and collect_list_fields(doc) == []
    assert sorted(_parts(BRAND_BOARD["blocks"]["blocks"])) == sorted(BRAND_PARTS)
    assert BRAND_BOARD not in PRESETS  # a starter of its own, not a category the picker starts from


def test_a_brand_block_names_a_part_and_anything_else_is_refused_by_field():
    with pytest.raises(BlockValidationError) as caught:
        validate_blocks({"blocks": [{"type": "brand", "id": "b", "part": "logos"}]})
    assert [error["loc"] for error in caught.value.errors] == ["blocks.0.part"]
    with pytest.raises(BlockValidationError):
        validate_blocks({"blocks": [{"type": "brand", "id": "b", "part": "logo", "src": "https://x"}]})


def test_each_brand_part_appears_once_so_a_template_cannot_multiply_the_miniature_renders():
    repeated = {"blocks": [
        {"type": "brand", "id": "a", "part": "applications"},
        {"type": "section", "id": "s", "children": [{"type": "brand", "id": "b", "part": "applications"}]},
    ]}
    with pytest.raises(BlockValidationError) as caught:
        validate_blocks(repeated)
    assert [error["loc"] for error in caught.value.errors] == ["blocks.1.children.0.part"]


# ---------------------------------------------------------------------------
# The render: one A4 page, every role's hex, the voice, the applications
# ---------------------------------------------------------------------------


def test_the_board_prints_the_automatos_kit_on_one_a4_page_with_every_roles_hex():
    pdf = _pdf(_board_html(AUTOMATOS))
    with pdfplumber.open(io.BytesIO(pdf)) as document:
        assert len(document.pages) == 1
        page = document.pages[0]
        assert abs(page.width - A4_PT[0]) <= PAGE_TOLERANCE_PT and abs(page.height - A4_PT[1]) <= PAGE_TOLERANCE_PT
        text = "".join((page.extract_text() or "").split())
    roles, _ = effective_palette(AUTOMATOS)
    for role in PALETTE_ROLES:
        assert roles[role].upper() in text, role
    for tone in TONES:
        assert tone["word"] in text
    assert "BRANDBOARD" in text.upper() and COMPANY.replace(" ", "") in text


def test_each_swatch_is_filled_with_its_role_and_a_derived_role_says_so():
    page = _board_html(_kit(palette={"accent": "#9a3d1a"}))
    colours = _part(page, "board-colours")
    for swatch in bb.swatches(_kit(palette={"accent": "#9a3d1a"})):
        assert f'style="background:{swatch.hex}"' in colours and f'<p class="board-hex">{swatch.hex}</p>' in colours
    swatches = colours.split('class="board-swatch"')[1:]
    (accent,) = [swatch for swatch in swatches if "#9A3D1A" in swatch]
    assert ">derived<" not in accent  # the one role the owner set
    assert sum(">derived<" in swatch for swatch in swatches) == len(PALETTE_ROLES) - 1
    assert "Accent use: sparing" in colours


def test_the_logo_prints_large_as_uploaded_beside_the_name_and_tagline():
    logo = _part(_board_html(AUTOMATOS), "board-logo")
    assert f'src="{LOGO}"' in logo and "Agents that do the work" in logo and COMPANY in logo
    assert bb.BOARD_TITLE in logo


def test_an_unset_variant_is_its_fr9_fallback_and_never_invented():
    variants = _part(_board_html(AUTOMATOS), "board-variants")
    assert variants.count(f'src="{LOGO}"') == 3  # on light, on dark (on its chip), one colour: the logo each time
    assert '<span class="board-on-chip">' in variants
    assert bb.NO_DARK_LOGO_NOTE in variants and bb.NO_MONO_LOGO_NOTE in variants


def test_uploaded_variants_are_shown_as_uploaded():
    variants = _part(_board_html(_kit(logo_dark_url=DARK_LOGO, logo_mono_url=MONO_LOGO)), "board-variants")
    assert f'src="{DARK_LOGO}"' in variants and f'src="{MONO_LOGO}"' in variants
    assert "board-on-chip" not in variants and bb.NO_DARK_LOGO_NOTE not in variants and bb.NO_MONO_LOGO_NOTE not in variants


def test_a_kit_without_a_logo_prints_its_name_and_says_so():
    kit = {**AUTOMATOS, "logo_url": ""}
    page = _board_html(kit)
    assert "<img" not in _part(page, "board-logo") and bb.NO_LOGO_NOTE in _part(page, "board-logo")
    assert "<img" not in _part(page, "board-variants")


def test_the_type_scale_spacing_and_clear_space_are_the_kits():
    kit = _kit(type_scale={"h1": {"size_pt": 26, "line_pt": 32}}, spacing_unit_pt=5, logo_rules={"clear_space": 1})
    page = _board_html(kit)
    assert "H1 26/32 pt, 600" in _part(page, "board-type") and "font-size:26pt;line-height:32pt" in page
    spacing = _part(page, "board-spacing")
    assert "A 5 pt grid" in spacing and "width:40pt" in spacing  # the sixth gap: 8 units
    assert "Logo clear space: 1 of its height (16 mm round the 16 mm letterhead logo)" in spacing


def test_the_voice_prints_each_tone_word_with_its_meaning_and_the_sign_off():
    voice = _part(_board_html(AUTOMATOS), "board-voice")
    for tone in TONES:
        assert f"<strong>{tone['word']}</strong> {tone['meaning']}" in voice
    assert f"Signs off as {SIGN_OFF}" in voice
    assert bb.NO_TONE_NOTE in _part(_board_html(_kit(voice={})), "board-voice")


def test_the_applications_are_the_invoice_and_letter_printed_with_the_kit_and_a_social_card():
    apps = _part(_board_html(AUTOMATOS), "board-applications")
    assert apps.count(f'<img src="{PNG_URI}') == 2  # page 1 of the invoice and the letter, drawn
    paper, ink, accent = bb.social_colours(AUTOMATOS)
    assert f'board-social" style="background:{paper}"' in apps and f"background:{accent}" in apps
    assert f'style="color:{ink}"' in apps
    invoice, letter = (starter_page(preset, AUTOMATOS, _now()) for preset in (INVOICE, LETTER))
    assert "INV-0042" in invoice and COMPANY in invoice and 'class="unresolved-var"' not in invoice
    assert SIGN_OFF in letter and 'class="unresolved-var"' not in letter


def _now():
    from datetime import datetime

    return datetime(2026, 10, 6, 9, 0, 0)


def test_a_miniature_that_cannot_be_drawn_leaves_its_frame(monkeypatch):
    import modules.documents.brand_board_miniatures as minis

    def broken(page: str) -> str:
        raise RuntimeError("pdfium is unwell")

    monkeypatch.setattr(minis, "page_png_uri", broken)
    apps = _part(_board_html(AUTOMATOS), "board-applications")
    assert "<img" not in apps and apps.count('class="board-app-frame"') == 2


# ---------------------------------------------------------------------------
# The Word file
# ---------------------------------------------------------------------------


def test_the_word_board_carries_every_swatch_its_hex_the_logo_and_the_miniatures():
    pytest.importorskip("docx")
    rendered = render_document_docx(validate_blocks(BRAND_BOARD["blocks"]), {}, AUTOMATOS, data={})
    assert rendered.unresolved == []
    xml = rendered.document.element.body.xml
    for swatch in bb.swatches(AUTOMATOS):
        assert f'w:fill="{swatch.hex.lstrip("#")}"' in xml and swatch.hex in xml
    assert len(re.findall(r"<pic:pic[\s>]", xml)) == 6  # the logo, its three tiles, the invoice and the letter
    for tone in TONES:
        assert tone["word"] in xml and tone["meaning"] in xml
    assert bb.NO_DARK_LOGO_NOTE in xml and bb.SOCIAL_SAMPLE_HEADLINE in xml


# ---------------------------------------------------------------------------
# Seeded for every workspace, under the starter-refresh rule
# ---------------------------------------------------------------------------


class _Session:
    """Keeps the template rows the seeder adds; answers its lookup by workspace and name."""

    def __init__(self):
        self.rows, self.commits, self._criteria = [], 0, {}

    def query(self, _model):
        self._criteria = {}
        return self

    def filter(self, *criteria):
        for criterion in criteria:
            self._criteria[criterion.left.key] = criterion.right.value
        return self

    def first(self):
        return next((r for r in self.rows if all(getattr(r, k) == v for k, v in self._criteria.items())), None)

    def add(self, row):
        self.rows.append(row)

    def commit(self):
        self.commits += 1


def _board_row(db: _Session):
    return next(row for row in db.rows if row.name == BRAND_BOARD["name"])


def test_every_workspace_is_seeded_the_brand_board_and_a_drifted_one_is_refreshed():
    db = _Session()
    seed_templates.seed_starter_templates(db, WS)
    row = _board_row(db)
    assert (row.category, row.format, row.created_by, row.blocks) == ("brand", "pdf", STARTER_CREATOR, BRAND_BOARD["blocks"])
    count = len(db.rows)
    seed_templates.seed_starter_templates(db, WS)
    assert len(db.rows) == count  # once
    row.blocks = {"version": 1, "blocks": []}
    seed_templates.seed_starter_templates(db, WS)
    assert _board_row(db).blocks == BRAND_BOARD["blocks"]


def test_a_persons_own_brand_board_is_never_touched():
    db = _Session()
    seed_templates.seed_starter_templates(db, WS)
    row = _board_row(db)
    row.created_by, row.blocks = "user_7", {"version": 1, "blocks": []}
    seed_templates.seed_starter_templates(db, WS)
    assert _board_row(db).blocks == {"version": 1, "blocks": []}


# ---------------------------------------------------------------------------
# The social card: 4:5 and 9:16, from the kit's tokens
# ---------------------------------------------------------------------------


def _social_board():
    (starter,) = [s for s in social_starters(SOCIAL_IMAGE) if s["slug"] in SOCIAL_BRAND_STARTER_SLUGS]
    return starter


def test_the_social_brand_board_is_a_4_5_and_9_16_image_from_the_kits_tokens():
    starter = _social_board()
    blocks = starter["blocks"]
    assert (starter["name"], starter["format"], blocks["sizes"]) == ("Brand board", SOCIAL_IMAGE, ["1080x1350", "1080x1920"])
    assert brand_literals(blocks["html"], blocks.get("css") or "") == []
    assert "{{ brand.logo }}" in blocks["html"] and "{{ brand.logo_on_dark }}" in blocks["html"]
    values = resolve_variables(blocks["variables_schema"], starter["sample_data"])
    assert values.missing == [] and values.invalid == []
    for size in blocks["sizes"]:
        bundle = build_bundle(workspace_id=WS, reference="brand-board", blocks=blocks, values=values.values,
                              brand_kit=AUTOMATOS, size=size, fmt=SOCIAL_IMAGE)
        assert bundle["still"] == {"at": [0.0]}
        swatch_tokens = re.findall(r'data-token="([a-z-]+)"', blocks["html"])
        assert len(swatch_tokens) == 6 and set(swatch_tokens) <= set(bundle["brand"]["tokens"])


# ---------------------------------------------------------------------------
# The one-colour logo reaches the render
# ---------------------------------------------------------------------------


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(bl.config, "DOCUMENT_STORAGE_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(bl, "is_storage_configured", lambda: False)
    return tmp_path


def test_the_render_inlines_an_uploaded_one_colour_logo(storage):
    mono = _png(side=12, rgb=b"\x00\x00\x00")
    kit = get_brand_kit({"brand_kit": {"logo_mono_path": bl.save_brand_logo(WS, mono, stem=bl.LOGO_MONO_STEM)}})
    assert brand_kit_for_media_render(kit)["logo_mono_url"] == _uri(mono)
    assert "logo_mono_url" not in brand_kit_for_media_render(get_brand_kit({"brand_kit": {}}))
