"""PRD-255 Wave 2 US-010: the brand board on the Brand kit page, as a PDF and a PNG.

``GET /api/documents/brand-kit/board?format=pdf|png`` prints the Brand Board
starter from the CALLER's kit (the workspace of the request context; another
workspace's kit is never read), render-ready (an uploaded logo inlined), in a
child process (``brand_board_render.render_board_isolated``), and streams it
with private, never-stored cache headers so a save shows at once. No workspace
is 404; an unknown format is 422 and draws nothing; a failed draw is a clear 500
that leaks nothing.

The route tests fake the child process; the render tests are real (WeasyPrint,
pypdfium2, and one real child process), as F353's are.
"""
from __future__ import annotations

import inspect
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

import pdfplumber  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from PIL import Image  # noqa: E402

import api.document_brand_kit as brand_kit_routes  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from modules.documents import brand_logo  # noqa: E402
from modules.documents.brand_board_render import (  # noqa: E402
    BOARD_PNG_WIDTH_PX, render_board, render_board_isolated,
)
from modules.documents.brand_kit import get_brand_kit  # noqa: E402
from modules.documents.thumbnails.render import ThumbnailError  # noqa: E402

OWNER = "00000000-0000-0000-0000-0000000255a1"
STRANGER = "00000000-0000-0000-0000-0000000255a2"
BOARD_ROUTE = "/api/documents/brand-kit/board"
PDF_BYTES, PNG_BYTES = b"%PDF-1.7 the board", b"\x89PNG\r\n\x1a\nthe board"
INLINED_LOGO = "data:image/png;base64,aW5saW5lZA=="
A4_RATIO = 841.89 / 595.28
MANIFEST = Path(__file__).resolve().parents[1] / "reports" / "route-manifest.json"


class _Workspaces:
    """``db.query(Workspace).filter(Workspace.id == x).first()`` over a dict of workspaces, recording each id asked."""

    def __init__(self, workspaces: Dict[str, Any]):
        self.workspaces, self.asked = workspaces, []
        self._id = None

    def query(self, _model):
        return self

    def filter(self, clause):
        value = getattr(clause.right, "value", None)
        self._id = str(value) if value is not None else None
        self.asked.append(self._id)
        return self

    def first(self):
        return self.workspaces.get(self._id)


def _ctx(workspace_id):
    return RequestContext(
        workspace_id=workspace_id,
        user=UserContext(id="owner-1", clerk_user_id="clerk-owner-1", system_role="user"),
        auth_type="clerk",
    )


@pytest.fixture
def board(monkeypatch):
    """The documents router over OWNER's workspace (and only OWNER's), the child process faked."""
    import api.document_generation as documents_module

    drawn: List[Dict[str, Any]] = []

    def fake_render(kit, fmt):
        drawn.append({"kit": kit, "fmt": fmt})
        return PDF_BYTES if fmt == "pdf" else PNG_BYTES

    monkeypatch.setattr(brand_kit_routes, "render_board_isolated", fake_render)
    owner = SimpleNamespace(id=OWNER, name="Acme", settings={"brand_kit": {"name": "Acme Studio", "primary_color": "#1d3658"}})
    db = _Workspaces({OWNER: owner})

    def client(workspace_id=OWNER) -> TestClient:
        app = FastAPI()
        app.include_router(documents_module.router)
        app.dependency_overrides[get_request_context_hybrid] = lambda: _ctx(workspace_id)
        app.dependency_overrides[get_db] = lambda: db
        return TestClient(app, raise_server_exceptions=False)

    return SimpleNamespace(client=client, drawn=drawn, db=db, owner=owner)


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt, body, media_type", [("pdf", PDF_BYTES, "application/pdf"), ("png", PNG_BYTES, "image/png")])
def test_the_board_is_streamed_as_a_pdf_or_a_png_never_kept_by_a_cache(board, fmt, body, media_type):
    resp = board.client().get(BOARD_ROUTE, params={"format": fmt})
    assert resp.status_code == 200, resp.text
    assert resp.content == body
    assert resp.headers["content-type"] == media_type
    assert resp.headers["cache-control"] == "private, no-store"
    assert resp.headers["content-disposition"] == f'inline; filename="brand-board.{fmt}"'
    assert [d["fmt"] for d in board.drawn] == [fmt]


def test_the_board_is_a_pdf_when_no_format_is_asked(board):
    resp = board.client().get(BOARD_ROUTE)
    assert resp.status_code == 200 and resp.content == PDF_BYTES


def test_the_board_is_printed_from_the_callers_own_kit(board):
    board.client().get(BOARD_ROUTE, params={"format": "png"})
    (kit,) = [d["kit"] for d in board.drawn]
    assert kit["name"] == "Acme Studio" and kit["primary_color"] == "#1d3658"
    assert board.db.asked == [OWNER]


def test_the_uploaded_logo_reaches_the_board_inlined(board, monkeypatch):
    board.owner.settings["brand_kit"]["logo_path"] = f"{OWNER}/brand/logo.png"
    monkeypatch.setattr(brand_logo, "logo_data_uri", lambda path: INLINED_LOGO if path.endswith("logo.png") else None)
    board.client().get(BOARD_ROUTE)
    assert board.drawn[0]["kit"]["logo_url"] == INLINED_LOGO


def test_another_workspace_never_reads_the_owners_kit(board):
    resp = board.client(STRANGER).get(BOARD_ROUTE, params={"format": "png"})
    assert resp.status_code == 404
    assert board.drawn == [] and board.db.asked == [STRANGER]
    assert PNG_BYTES not in resp.content


def test_no_workspace_is_404_and_nothing_is_read(board):
    resp = board.client(None).get(BOARD_ROUTE)
    assert resp.status_code == 404
    assert board.drawn == [] and board.db.asked == []


def test_an_unknown_format_is_422_and_nothing_is_drawn(board):
    resp = board.client().get(BOARD_ROUTE, params={"format": "svg"})
    assert resp.status_code == 422
    assert board.drawn == []


def test_a_failed_draw_is_a_clear_500_that_leaks_nothing(board, monkeypatch):
    def broken(kit, fmt):
        raise ThumbnailError("the render failed (exit 1): Traceback in weasyprint/layout.py")

    monkeypatch.setattr(brand_kit_routes, "render_board_isolated", broken)
    resp = board.client().get(BOARD_ROUTE)
    assert resp.status_code == 500
    assert resp.json()["detail"] == brand_kit_routes.BOARD_RENDER_FAILED
    assert "weasyprint" not in resp.text.lower()


def test_the_route_is_a_plain_def_behind_the_request_context():
    route = next(r for r in brand_kit_routes.router.routes if getattr(r, "path", "") == "/brand-kit/board")
    assert route.methods == {"GET"}
    assert not inspect.iscoroutinefunction(brand_kit_routes.get_brand_board)
    assert get_request_context_hybrid in [d.call for d in route.dependant.dependencies]


def test_the_route_is_in_the_committed_manifest():
    manifest = json.loads(MANIFEST.read_text())
    assert {"method": "GET", "path": BOARD_ROUTE} in manifest["routes"]
    assert manifest["route_count"] == len(manifest["routes"])


# ---------------------------------------------------------------------------
# The render (real)
# ---------------------------------------------------------------------------


def _kit() -> Dict[str, Any]:
    return {**get_brand_kit({"brand_kit": {"name": "Acme Studio", "tagline": "Built to last"}}), "font_files": []}


def _pdf_text(pdf: bytes) -> str:
    with pdfplumber.open(io.BytesIO(pdf)) as document:
        return "".join((document.pages[0].extract_text() or "").split())


def test_the_pdf_is_the_brand_board_of_the_kit():
    pdf = render_board(_kit(), "pdf")
    assert pdf.startswith(b"%PDF")
    text = _pdf_text(pdf)
    assert "AcmeStudio" in text and "BRANDBOARD" in text.upper()


def test_the_png_is_the_whole_first_page_sharp_enough_to_download():
    image = Image.open(io.BytesIO(render_board(_kit(), "png")))
    assert image.format == "PNG" and image.width == BOARD_PNG_WIDTH_PX
    assert abs(image.height - BOARD_PNG_WIDTH_PX * A4_RATIO) <= 2  # uncut: the card thumbnail's cap is not applied


def test_an_unknown_format_is_refused_before_anything_is_printed():
    with pytest.raises(ThumbnailError):
        render_board(_kit(), "svg")
    with pytest.raises(ThumbnailError):
        render_board_isolated(_kit(), "svg")


def test_the_child_process_prints_the_same_board():
    pdf = render_board_isolated(_kit(), "pdf")
    assert pdf.startswith(b"%PDF") and "AcmeStudio" in _pdf_text(pdf)
