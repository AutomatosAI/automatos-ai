"""F347 (night 10b): a branded PDF is set in the brand kit's font.

Every Branded PDF came out in DejaVu Serif (pdffonts listed DejaVu only), even
with the kit's font changed to Courier New, while the .docx wore the kit's
Geist. The block renderer HTML-escaped the kit's font stack into its ``<style>``
(``font-family: Inter, &#x27;Segoe UI&#x27;, …``, seen in the Studio's
preview-blocks HTML): CSS does not read HTML entities, so WeasyPrint dropped the
declaration and used its default serif (F350, #969, now writes the stack as CSS;
this proves it in the PDF). An uploaded font file never reached the PDF either:
the PDF's kit carried the logo only. Nor did the kit's heading font.

Rendered for real (WeasyPrint) and read back (pdfplumber): the fonts the PDF's
text is set in. DejaVu Sans Mono is a font the CI runner and the image both
have (fonts-dejavu-core); "CI Block" is a woff2 drawn by
``scripts/ci/ci_block_font.py``, uploaded to the kit.
"""
from __future__ import annotations

import asyncio
import importlib.util
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Set

import pdfplumber
import pytest

import modules.documents.brand_fonts as brand_fonts
import modules.documents.generation_service as generation_service
from modules.documents.blocks import render_document_html, validate_blocks
from modules.documents.blocks.page_fonts import font_css
from modules.documents.brand_fonts import brand_kit_for_media_render
from modules.documents.blocks.page_style import DEFAULT_FONT
from modules.documents.brand_kit import get_brand_kit
from modules.documents.generation_service import DocumentGenerationService
from modules.documents.variables.resolver import build_context, resolve_paths
from services.brand_rules import forget_cached_kits

_ROOT = Path(__file__).resolve().parents[2]
WS = uuid.UUID("00000000-0000-0000-0000-0000000347a1")
MONO = "'DejaVu Sans Mono', monospace"


def _load_block_font():
    spec = importlib.util.spec_from_file_location("ci_block_font", _ROOT / "scripts" / "ci" / "ci_block_font.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BLOCK_FONT = _load_block_font()
WOFF2 = BLOCK_FONT.block_font_woff2()
FONT_ID = "f347" + "0" * 28
FONT_PATH = f"{WS}/brand/fonts/{FONT_ID}.woff2"
UPLOADED_FONT = {"id": FONT_ID, "family": BLOCK_FONT.FAMILY, "weight": 400, "style": "normal",
                 "path": FONT_PATH, "file_name": "ci-block.woff2", "bytes": len(WOFF2)}

PAGE = {"version": 1, "blocks": [
    {"type": "heading", "id": "title", "level": 1, "content": [{"type": "text", "text": "Harbourline price list"}]},
    {"type": "text", "id": "body", "content": [{"type": "variable", "path": "data.body"}]},
]}
DATA = {"body": "Harbour Blend at 19.50 per kilo, delivered every Friday."}


def block_template(blocks: Dict[str, Any]) -> SimpleNamespace:
    """A Branded template row: its blocks, no legacy source."""
    return SimpleNamespace(id=uuid.uuid4(), name="Branded Page", blocks=blocks, template_content=None,
                           data_schema=None, template_file_path=None)


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


def render_pdf(monkeypatch, tmp_path, settings: Dict[str, Any], template: Any, data: Dict[str, Any],
               fonts: Optional[Dict[str, bytes]] = None) -> Path:
    """generate_pdf for real for a workspace with ``settings``; the stored font files are ``fonts`` (path → bytes)."""
    monkeypatch.setattr(generation_service, "GENERATED_DIR", str(tmp_path))
    monkeypatch.setattr(generation_service, "is_storage_configured", lambda: False)
    monkeypatch.setattr(brand_fonts, "load_brand_font", lambda font: (fonts or {}).get(font.get("path")))

    class _Resolver:  # the chips resolve against this workspace's kit, as the platform resolves them
        def __init__(self, db: Any):
            pass

        def resolve(self, workspace_id: Any, user_id: Any, paths: Any, extra_data: Any = None) -> Any:
            context = build_context(None, None, get_brand_kit(settings), datetime(2026, 10, 5), extra_data)
            return resolve_paths(context, paths)

    monkeypatch.setattr(generation_service, "VariableResolver", _Resolver)
    forget_cached_kits()
    service = DocumentGenerationService(_Workspace(settings), WS)
    result = asyncio.run(service.generate_pdf(template, dict(data), WS, "F347"))
    forget_cached_kits()
    return Path(result.path)


def pdf_lines(path: Path) -> List[str]:
    """The PDF's text, one line per printed line, its spacing normalised."""
    with pdfplumber.open(path) as pdf:
        text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    return [" ".join(line.split()) for line in text.splitlines() if line.strip()]


def pdf_fonts(path: Path) -> Set[str]:
    """The fonts the PDF's text is set in, by name, without spaces or the subset prefix."""
    with pdfplumber.open(path) as pdf:
        names = {char["fontname"] for page in pdf.pages for char in page.chars}
    return {name.split("+")[-1].replace(" ", "") for name in names}


def _style(html: str) -> str:
    return html.split("<style>", 1)[1].split("</style>", 1)[0]


def test_the_kits_font_stack_is_in_the_page_as_css_not_html_escaped():
    kit = get_brand_kit({"brand_kit": {"font_family": MONO}})
    style = _style(render_document_html(validate_blocks(PAGE), {"data.body": "x"}, kit).html)

    assert f"font-family: {MONO};" in style
    assert "&#x27;" not in style


def test_the_pdf_is_set_in_the_kits_font(monkeypatch, tmp_path):
    pdf = render_pdf(monkeypatch, tmp_path, {"brand_kit": {"font_family": MONO}}, block_template(PAGE), DATA)

    fonts = pdf_fonts(pdf)
    assert fonts and all("DejaVuSansMono" in name for name in fonts), fonts
    assert any("Harbour Blend at 19.50 per kilo" in line for line in pdf_lines(pdf))


def test_the_pdfs_headings_are_set_in_the_kits_heading_font(monkeypatch, tmp_path):
    settings = {"brand_kit": {"font_family": MONO, "heading_font": "'DejaVu Serif', serif"}}

    pdf = render_pdf(monkeypatch, tmp_path, settings, block_template(PAGE), DATA)

    with pdfplumber.open(pdf) as document:
        words = [word for page in document.pages for word in page.extract_words(extra_attrs=["fontname"])]
    fonts_of: Dict[str, Set[str]] = {}
    for word in words:
        fonts_of.setdefault(word["text"], set()).add(word["fontname"].replace(" ", ""))
    # The heading is in the heading font (the footer repeats the title in the body font).
    assert any("DejaVuSerif" in name for name in fonts_of["Harbourline"]), fonts_of
    assert all("DejaVuSansMono" in name for name in fonts_of["Friday."]), fonts_of


def test_an_uploaded_font_file_reaches_the_pdf(monkeypatch, tmp_path):
    settings = {"brand_kit": {"font_family": f'"{BLOCK_FONT.FAMILY}", sans-serif', "font_files": [UPLOADED_FONT]}}

    pdf = render_pdf(monkeypatch, tmp_path, settings, block_template(PAGE), DATA, fonts={FONT_PATH: WOFF2})

    fonts = pdf_fonts(pdf)
    assert any(BLOCK_FONT.FAMILY.replace(" ", "") in name for name in fonts), fonts


def test_an_uploaded_font_file_is_a_font_face_in_the_page(monkeypatch):
    monkeypatch.setattr(brand_fonts, "load_brand_font", lambda font: WOFF2)
    kit = brand_kit_for_media_render(get_brand_kit({"brand_kit": {"font_files": [UPLOADED_FONT]}}))

    style = _style(render_document_html(validate_blocks(PAGE), {"data.body": "x"}, kit).html)

    assert f'@font-face {{ font-family: "{BLOCK_FONT.FAMILY}"; src: url("data:font/woff2;base64,' in style
    assert 'format("woff2"); font-weight: 400; font-style: normal; }' in style


def test_the_headings_font_rule_and_an_unsafe_heading_font():
    assert "h1, h2, h3, h4, h5, h6 { font-family: 'DejaVu Serif', serif; }" in font_css({"heading_font": "'DejaVu Serif', serif"})
    unsafe = font_css({"heading_font": "x; } body { color: red"})
    assert "color: red" not in unsafe and f"font-family: {DEFAULT_FONT};" in unsafe
    assert "h1" not in font_css({"heading_font": ""})  # no heading font: headings keep the body font


@pytest.mark.parametrize("font", [
    {"family": "Brand Sans", "weight": 400, "style": "normal", "data_uri": "https://evil.example/f.woff2"},
    {"family": "Brand Sans", "weight": 400, "style": "normal", "data_uri": 'data:font/woff2;base64,AA==") } body {'},
    {"family": 'Brand"; } body { x', "weight": 400, "style": "normal", "data_uri": "data:font/woff2;base64,AA=="},
    {"family": "Brand Sans", "weight": 450, "style": "normal", "data_uri": "data:font/woff2;base64,AA=="},
    {"family": "Brand Sans", "weight": 400, "style": "oblique", "data_uri": "data:font/woff2;base64,AA=="},
])
def test_a_font_file_that_is_not_a_usable_woff2_face_is_left_out(font):
    assert "@font-face" not in font_css({"font_files": [font]})
