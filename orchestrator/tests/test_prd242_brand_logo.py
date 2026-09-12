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

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 32
WS = uuid4()


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
