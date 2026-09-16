"""PRD-242 S3 — an uploaded brand logo is stored under the document root and
INLINED at render time, so neither renderer needs a public URL.

Pure at the boundary: a tmp storage root, object storage unconfigured (the
mirror is skipped), no WeasyPrint, no python-docx document build.
"""

from __future__ import annotations

import base64
from uuid import uuid4

import pytest

import modules.documents.brand_logo as bl
from modules.documents.blocks import render_document_html, validate_blocks
from modules.documents.blocks.docx_renderer import _inline_image_bytes, _safe_image_bytes
from modules.documents.brand_kit import get_brand_kit, validate_brand_kit
from modules.documents.variables.resolver import build_context, resolve_paths

from datetime import datetime

WS = uuid4()


def png_bytes(width: int = 120, height: int = 40) -> bytes:
    """A PNG signature + IHDR chunk (enough header for sniffing + dimensions)."""
    ihdr = width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x06\x00\x00\x00"
    return b"\x89PNG\r\n\x1a\n" + (13).to_bytes(4, "big") + b"IHDR" + ihdr + b"\x00" * 16


def jpeg_bytes(width: int = 120, height: int = 40) -> bytes:
    """SOI + APP0 + SOF0 (the segment that carries the dimensions)."""
    app0 = b"\xff\xe0" + (16).to_bytes(2, "big") + b"JFIF\x00" + b"\x00" * 9
    sof0 = b"\xff\xc0" + (17).to_bytes(2, "big") + b"\x08" + height.to_bytes(2, "big") + width.to_bytes(2, "big") + b"\x03" + b"\x00" * 9
    return b"\xff\xd8" + app0 + sof0 + b"\x00" * 16


PNG = png_bytes()
JPEG = jpeg_bytes()


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(bl.config, "DOCUMENT_STORAGE_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(bl, "is_storage_configured", lambda: False)
    return tmp_path


def test_sniff_accepts_png_jpeg_only():
    assert bl.sniff_image_type(PNG) == ("image/png", ".png")
    assert bl.sniff_image_type(JPEG) == ("image/jpeg", ".jpg")
    assert bl.sniff_image_type(b"<svg xmlns='http://www.w3.org/2000/svg'/>") is None
    assert bl.sniff_image_type(b"") is None


def test_save_validates_type_and_size(storage):
    with pytest.raises(bl.BrandLogoError, match="PNG or JPEG"):
        bl.save_brand_logo(WS, b"GIF89a" + b"\x00" * 10)
    with pytest.raises(bl.BrandLogoError, match="empty"):
        bl.save_brand_logo(WS, b"")
    with pytest.raises(bl.BrandLogoError, match="MB or smaller"):
        bl.save_brand_logo(WS, PNG + b"\x00" * bl.MAX_LOGO_BYTES)


def test_image_dimensions_read_png_and_jpeg_headers():
    assert bl.image_dimensions(png_bytes(640, 480)) == (640, 480)
    assert bl.image_dimensions(jpeg_bytes(300, 200)) == (300, 200)
    # a truncated / header-less file is unreadable, never a crash
    assert bl.image_dimensions(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8) is None
    assert bl.image_dimensions(b"\xff\xd8\xff\xd9") is None
    assert bl.image_dimensions(b"GIF89a") is None


def test_save_rejects_pixel_floods_and_unreadable_headers(storage):
    # A tiny file declaring a huge canvas would make the PDF rasteriser allocate it.
    with pytest.raises(bl.BrandLogoError, match="px"):
        bl.save_brand_logo(WS, png_bytes(bl.MAX_LOGO_DIMENSION + 1, 10))
    with pytest.raises(bl.BrandLogoError, match="px"):
        bl.save_brand_logo(WS, jpeg_bytes(10, 20000))
    with pytest.raises(bl.BrandLogoError, match="header"):
        bl.save_brand_logo(WS, b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)
    # the cap itself is allowed
    assert bl.save_brand_logo(WS, png_bytes(bl.MAX_LOGO_DIMENSION, bl.MAX_LOGO_DIMENSION)).endswith(".png")


def test_save_load_replace_delete_roundtrip(storage):
    path = bl.save_brand_logo(WS, PNG)
    assert path == f"{WS}/brand/logo.png"
    assert (storage / path).read_bytes() == PNG
    assert bl.load_brand_logo(path) == PNG
    assert bl.logo_mime(path) == "image/png"

    # Re-upload as JPEG: the stale PNG twin must not linger.
    path2 = bl.save_brand_logo(WS, JPEG)
    assert path2.endswith(".jpg")
    assert not (storage / path).exists()
    assert bl.load_brand_logo(path2) == JPEG

    bl.delete_brand_logo(path2)
    assert bl.load_brand_logo(path2) is None


def test_load_refuses_paths_outside_the_storage_root(storage):
    (storage.parent / "secret.png").write_bytes(PNG)
    assert bl.load_brand_logo("../secret.png") is None


def test_data_uri_and_render_ready_kit(storage):
    path = bl.save_brand_logo(WS, PNG)
    uri = bl.logo_data_uri(path)
    assert uri == "data:image/png;base64," + base64.b64encode(PNG).decode()

    kit = {"logo_path": path, "logo_url": "", "name": "Acme"}
    rendered = bl.brand_kit_for_render(kit)
    assert rendered["logo_url"] == uri
    assert kit["logo_url"] == ""  # input untouched (no in-place mutation)

    # No stored bytes anywhere → the kit is returned as-is (external URL, if any, survives).
    missing = bl.brand_kit_for_render({"logo_path": f"{WS}/brand/logo.jpg", "logo_url": "https://x/y.png"})
    assert missing["logo_url"] == "https://x/y.png"
    assert bl.brand_kit_for_render({"logo_path": "", "logo_url": "https://x/y.png"})["logo_url"] == "https://x/y.png"


def test_html_renderer_embeds_the_inlined_logo(storage):
    path = bl.save_brand_logo(WS, PNG)
    kit = bl.brand_kit_for_render({**get_brand_kit(None), "logo_path": path})
    doc = validate_blocks({"blocks": [{"type": "image", "id": "logo", "source": "brand_logo", "alt": "Logo", "width_mm": 40}]})
    out = render_document_html(doc, {}, kit, title="t")
    assert 'src="data:image/png;base64,' in out.html
    assert out.unresolved == []


def test_docx_renderer_decodes_inline_data_uris():
    uri = "data:image/png;base64," + base64.b64encode(PNG).decode()
    assert _inline_image_bytes(uri).read() == PNG
    assert _safe_image_bytes(uri).read() == PNG
    assert _inline_image_bytes("data:text/plain;base64,aGk=") is None
    assert _inline_image_bytes("data:image/png;base64,***not-base64***") is None
    assert _inline_image_bytes("data:image/png,rawbytes") is None


def test_external_logo_url_must_be_http_or_empty():
    from modules.documents.brand_kit import is_acceptable_logo_url

    assert is_acceptable_logo_url("") and is_acceptable_logo_url(None)
    assert is_acceptable_logo_url("https://cdn.acme.com/logo.png")
    assert is_acceptable_logo_url("http://cdn.acme.com/logo.png")
    assert not is_acceptable_logo_url("javascript:alert(1)")
    assert not is_acceptable_logo_url("file:///etc/passwd")
    assert not is_acceptable_logo_url("/relative/logo.png")


def test_brand_kit_model_carries_logo_path_but_clients_cannot_set_it():
    kit = get_brand_kit({"brand_kit": {"logo_path": "ws/brand/logo.png"}})
    assert kit["logo_path"] == "ws/brand/logo.png"
    # A PUT patch cannot point the kit at an arbitrary stored file — the field is dropped.
    merged = validate_brand_kit({"logo_path": "../../etc/passwd", "name": "Acme"}, {"logo_path": "ws/brand/logo.png"})
    assert merged["logo_path"] == "ws/brand/logo.png"
    assert merged["name"] == "Acme"


def test_brand_logo_url_chip_resolves_to_the_stream_route_for_uploads():
    kit = get_brand_kit({"brand_kit": {"logo_path": "ws/brand/logo.png"}})
    ctx = build_context(None, None, kit, datetime(2026, 9, 11))
    res = resolve_paths(ctx, ["brand.logo_url"])
    assert res.values["brand.logo_url"] == bl.BRAND_LOGO_ROUTE
    # An external URL still wins when both are present.
    kit2 = get_brand_kit({"brand_kit": {"logo_path": "ws/brand/logo.png", "logo_url": "https://cdn/x.png"}})
    assert resolve_paths(build_context(None, None, kit2, datetime(2026, 9, 11)), ["brand.logo_url"]).values["brand.logo_url"] == "https://cdn/x.png"
