"""F343: a PDF with an image renders through the platform's guarded URL fetcher.

A fresh image build installed WeasyPrint 70.0 (released 8 Sep 2026; requirements said
only ``>=62.0``). 70 removed ``weasyprint.default_url_fetcher``, which
``generation_service._safe_url_fetcher`` (PRD-156 S4, the SSRF guard) calls for every
``data:`` URI. Every PDF with a logo or chart then failed: "PDF generation failed:
'function' object has no attribute '_fail_on_errors'", a 500 on /api/documents/generate.
requirements.txt now keeps WeasyPrint below 70; this test fails on any WeasyPrint the
guarded fetcher can't drive.
"""
from __future__ import annotations

import base64
import io


def _png() -> str:
    """A real 8x8 PNG (Pillow ships with WeasyPrint), base64 for a data: URI."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "#1E3A5F").save(buf, "PNG")
    return base64.b64encode(buf.getvalue()).decode()


def test_a_pdf_with_an_inline_image_renders_through_the_guarded_fetcher():
    from weasyprint import HTML

    from modules.documents.generation_service import _safe_url_fetcher

    html = f'<h1>Harbourline</h1><img src="data:image/png;base64,{_png()}">'
    pdf = HTML(string=html, url_fetcher=_safe_url_fetcher).write_pdf()
    assert pdf.startswith(b"%PDF")


def test_the_guard_still_refuses_a_file_url():
    import pytest

    from modules.documents.generation_service import _safe_url_fetcher

    with pytest.raises(ValueError):
        _safe_url_fetcher("file:///etc/passwd")
