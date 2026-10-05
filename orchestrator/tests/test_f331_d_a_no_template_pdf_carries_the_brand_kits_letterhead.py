"""F331 retest (night 10, 5 Oct): a PDF made with no template carries the brand kit's letterhead.

Deliverable 837dd5fa, a session agent's generate_document with no template, came
out with the kit's navy headings but no logo, no company name and no contact line,
though the workspace's kit has an uploaded logo. The no-template render now starts
with the Branded Letter's letterhead whenever the workspace has a kit. Rendered for
real (WeasyPrint), the uploaded logo read from the store, and read back (pdfplumber).
"""
from __future__ import annotations

import asyncio
import struct
import uuid
import zlib
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pdfplumber
import pytest

import modules.documents.brand_logo as brand_logo
import modules.documents.generation_service as generation_service
from modules.documents.brand_kit import get_brand_kit
from modules.documents.generation_service import DocumentGenerationService
from modules.documents.letterhead import letterhead_blocks
from modules.documents.variables.resolver import build_context, resolve_paths
from services.brand_rules import forget_cached_kits

WS = uuid.UUID("00000000-0000-0000-0000-0000000331d1")
COMPANY = "Harbourline Coffee Roasters"
EMAIL = "orders@harbourline.example"
LOGO_PATH = f"{WS}/brand/logo.png"
KIT = {"name": COMPANY, "logo_path": LOGO_PATH,
       "company": {"name": COMPANY, "email": EMAIL, "address": "4 Quay Street, Galway"}}
# As generate() hands it on: the call's title is in the data.
DATA = {"title": "Invoice HL-2026-0142",
        "sections": [{"title": "October wholesale", "content": "12 kg of Harbour Blend."}],
        "invoice_number": "HL-2026-0142"}


def _png(side: int = 8) -> bytes:
    """A small opaque PNG, the logo the store holds."""
    def chunk(kind: bytes, body: bytes) -> bytes:
        return struct.pack(">I", len(body)) + kind + body + struct.pack(">I", zlib.crc32(kind + body) & 0xFFFFFFFF)

    rows = b"".join(b"\x00" + b"\xff\x6b\x35" * side for _ in range(side))
    header = struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


class _Rows:
    def __init__(self, row: Any):
        self._row = row

    def filter(self, *args: Any, **kwargs: Any) -> "_Rows":
        return self

    def first(self) -> Any:
        return self._row


class _Workspace:
    """A session whose one workspace has ``settings``."""

    def __init__(self, settings: Dict[str, Any]):
        self._workspace = SimpleNamespace(settings=settings)

    def get(self, model: Any, key: Any) -> Any:
        return self._workspace

    def query(self, *args: Any) -> _Rows:
        return _Rows(self._workspace)


@pytest.fixture
def render(monkeypatch, tmp_path):
    """generate_pdf for real with no template, for a workspace with ``settings``: (result, lines, images)."""
    monkeypatch.setattr(generation_service, "GENERATED_DIR", str(tmp_path))
    monkeypatch.setattr(generation_service, "is_storage_configured", lambda: False)
    monkeypatch.setattr(brand_logo, "load_brand_logo", lambda path: _png() if path == LOGO_PATH else None)

    def go(settings: Dict[str, Any]) -> Tuple[Any, List[str], int]:
        class _Resolver:  # the chips resolve against this workspace's kit, as the platform resolves them
            def __init__(self, db: Any):
                pass

            def resolve(self, workspace_id: Any, user_id: Any, paths: Any, extra_data: Any = None) -> Any:
                context = build_context(None, None, get_brand_kit(settings), datetime(2026, 10, 5), extra_data)
                return resolve_paths(context, paths)

        monkeypatch.setattr(generation_service, "VariableResolver", _Resolver)
        forget_cached_kits()
        service = DocumentGenerationService(_Workspace(settings), WS)
        result = asyncio.run(service.generate_pdf(None, dict(DATA), WS, "Invoice HL-2026-0142"))
        with pdfplumber.open(result.path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
            images = sum(len(page.images) for page in pdf.pages)
        return result, [" ".join(line.split()) for line in text.splitlines() if line.strip()], images

    yield go
    forget_cached_kits()


def test_a_kit_with_a_logo_puts_the_logo_and_the_company_on_top(render):
    result, lines, images = render({"brand_kit": KIT})

    assert images == 1
    assert lines[0] == COMPANY, lines
    assert any(EMAIL in line for line in lines) and "4 Quay Street, Galway" in lines, lines
    assert any("12 kg of Harbour Blend." in line for line in lines), lines
    assert result.unresolved == []


def test_a_kit_without_a_logo_still_names_the_company_and_is_not_blocked(render):
    result, lines, images = render({"brand_kit": {"name": COMPANY, "company": {"email": EMAIL}}})

    assert images == 0
    assert lines[0] == COMPANY, lines
    assert result.unresolved == []


def test_a_workspace_with_no_kit_gets_the_page_it_always_got(render):
    result, lines, images = render({})

    assert images == 0
    assert COMPANY not in "\n".join(lines) and EMAIL not in "\n".join(lines)
    assert lines[0] == "Invoice HL-2026-0142", lines
    assert result.unresolved == []


def test_no_kit_no_letterhead():
    assert letterhead_blocks(None, {"logo_url": "data:image/png;base64,AA=="}) == []
