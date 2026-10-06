"""Brand kit at generation (D, night 10 prep): generate_document's spreadsheet renders in the brand.

The PDF, Word and social renders read the workspace's brand kit; the spreadsheet did
not. Its header row was hard-coded "#1a1a2e" with white text and it had no logo, so a
branded workspace's Excel deliverable came out in someone else's colours. It now takes
the kit's colours for the header, the kit's body font, and the uploaded logo above the
table. A workspace without a kit gets the sheet it always got.

PRD-255 (US-005): the header is no longer the primary. It is the kit's table header
role (``surface_2`` with ``heading`` text; the accent only under ``accent_use: bold``),
the same as the kit's documents (``blocks.design_tokens``).
"""
from __future__ import annotations

import asyncio
import struct
import zipfile
import zlib
from types import SimpleNamespace as NS
from uuid import UUID

import pytest

WS = UUID("6d0b5c1e-8f1a-4c2b-9d3e-0a1b2c3d4e61")
DATA = {"columns": ["Coffee", "Kg left"], "rows": [["Kirinyaga", 41], ["Guji Shakiso", 55.5]]}


def png(width: int = 8, height: int = 4) -> bytes:
    """A real (tiny, opaque white) PNG: signature, IHDR, IDAT, IEND."""
    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + b"\xff\xff\xff" * width for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


@pytest.fixture(autouse=True)
def fresh_kits():
    from services.brand_rules import forget_cached_kits

    forget_cached_kits()
    yield
    forget_cached_kits()


@pytest.fixture
def make_sheet(tmp_path, monkeypatch):
    import modules.documents.brand_logo as bl
    import modules.documents.generation_service as gs

    monkeypatch.setattr(gs, "GENERATED_DIR", str(tmp_path / "generated"))
    monkeypatch.setattr(gs, "is_storage_configured", lambda: False)
    monkeypatch.setattr(bl.config, "DOCUMENT_STORAGE_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(bl, "is_storage_configured", lambda: False)

    def make(settings):
        db = NS(get=lambda model, key: NS(settings=settings), query=lambda *a, **k: None)
        service = gs.DocumentGenerationService(db, WS)
        result = asyncio.run(service.generate(title="Green coffee stock", format="xlsx", data=dict(DATA),
                                              workspace_id=WS))
        with zipfile.ZipFile(result.path) as book:
            names = set(book.namelist())
            return NS(styles=book.read("xl/styles.xml").decode(), media=[n for n in names if n.startswith("xl/media/")],
                      sheet=book.read("xl/worksheets/sheet1.xml").decode())
    return make


def _argb(hex_colour: str) -> str:
    return "FF" + hex_colour.lstrip("#").upper()


def test_the_sheet_takes_the_kits_colour_font_and_logo(make_sheet):
    from modules.documents.blocks.design_tokens import palette
    from modules.documents.brand_kit import get_brand_kit
    from modules.documents.brand_logo import save_brand_logo

    kit = {"name": "Harbourline Coffee Roasters", "primary_color": "#e96235", "font_family": "Georgia, serif",
           "logo_path": save_brand_logo(WS, png())}
    sheet = make_sheet({"brand_kit": kit})
    header = palette(get_brand_kit({"brand_kit": kit})).header_fill
    assert f'<fgColor rgb="{_argb(header)}"/>' in sheet.styles and '<fgColor rgb="FF1A1A2E"/>' not in sheet.styles
    assert '<fgColor rgb="FFE96235"/>' not in sheet.styles    # sparing: the primary paints no header
    assert 'val="Georgia"' in sheet.styles
    assert sheet.media == ["xl/media/image1.png"]
    assert '<c r="A1"' not in sheet.sheet    # the table sits under the logo


def test_a_light_accent_under_bold_gets_dark_header_text(make_sheet):
    from modules.documents.blocks.design_tokens import palette
    from modules.documents.brand_kit import get_brand_kit

    kit = {"palette": {"accent": "#f1e9dd"}, "text_color": "#2b2118", "accent_use": "bold"}
    sheet = make_sheet({"brand_kit": kit})
    roles = palette(get_brand_kit({"brand_kit": kit}))
    assert roles.header_fill == "#f1e9dd" and roles.header_text == roles.heading
    assert '<fgColor rgb="FFF1E9DD"/>' in sheet.styles and f'<color rgb="{_argb(roles.heading)}"/>' in sheet.styles


def test_without_a_kit_the_sheet_is_the_one_it_always_was(make_sheet):
    sheet = make_sheet({})
    assert '<fgColor rgb="FF1A1A2E"/>' in sheet.styles and sheet.media == [] and '<c r="A1"' in sheet.sheet
    assert "Georgia" not in sheet.styles and "Inter" not in sheet.styles
