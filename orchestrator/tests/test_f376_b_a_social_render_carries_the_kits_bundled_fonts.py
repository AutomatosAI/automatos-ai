"""F376 (night 11, 7 Oct), fonts: a social render carries the faces the code ships, as the PDFs do.

The night's kit named Geist (body) and Newsreader (headings) with no font files
uploaded. F360 bundled Inter, Geist and Newsreader for the PDFs only: the
media-render bundle carried uploaded faces alone, and media-render has only
Liberation and DejaVu, so every post printed in Times-like, DejaVu-like and
Courier-like stand-ins. Pins:

* the render-ready kit (``brand_fonts.brand_kit_for_media_render``) carries, in
  ``bundled_font_files``, the bundled faces of each family its stacks resolve to and
  it did not upload; ``font_files`` stays what the owner uploaded;
* the bundle stages them as font files with their ``@font-face``, after the uploads,
  and media-render's own parser takes the bundle (inside its file and face limits);
* an uploaded face of a bundled family wins: that family is not bundled again;
* the PDF path (the board, ``page_fonts``) still counts only the uploads as uploads,
  so it adds each bundled family once.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict

_ORCH = Path(__file__).resolve().parents[1]
_MEDIA_RENDER = _ORCH.parent / "services" / "media-render"
if str(_MEDIA_RENDER) not in sys.path:
    sys.path.append(str(_MEDIA_RENDER))

from core.media_render_bundle import BUNDLED_FONT_FILES, FONTS_DIR, build_bundle  # noqa: E402
from core.social_templates import SOCIAL_IMAGE, resolve_variables  # noqa: E402
from media_render.bundle import parse_bundle  # noqa: E402  (media-render's own parser: stdlib only)
from media_render.config import load_settings  # noqa: E402
from modules.documents import brand_fonts, bundled_fonts  # noqa: E402
from modules.documents.blocks.page_fonts import font_css  # noqa: E402
from modules.documents.brand_kit import get_brand_kit  # noqa: E402
from modules.documents.social_starters import social_starters  # noqa: E402

NIGHT_KIT = {
    "name": "Tide Cafe",
    "primary_color": "#c44a1a",
    "secondary_color": "#dcd2bd",
    "text_color": "#1a1814",
    "font_family": "Geist, Inter, 'Segoe UI', system-ui, sans-serif",
    "heading_font": "Newsreader, Georgia, serif",
}
WOFF2 = b"wOF2" + b"\x00" * 60
FACE_RULE = re.compile(r'@font-face \{ font-family: "([^"]+)"; src: url\("data:font/woff2;base64,')


def _count(families) -> Dict[str, int]:
    found: Dict[str, int] = {}
    for family in families:
        found[family] = found.get(family, 0) + 1
    return found


def _rendered(raw):
    return brand_fonts.brand_kit_for_media_render(get_brand_kit({"brand_kit": raw}))


def _title_card_bundle(kit):
    starter = next(s for s in social_starters(SOCIAL_IMAGE) if s["slug"] == "title-card")
    values = resolve_variables(starter["blocks"]["variables_schema"], starter["sample_data"]).values
    return build_bundle(workspace_id="ws", reference="r", blocks=starter["blocks"], values=values,
                        brand_kit=kit, size="1080x1350", fmt=SOCIAL_IMAGE)


def test_the_render_ready_kit_carries_the_bundled_faces_of_the_families_it_names():
    kit = _rendered(NIGHT_KIT)
    assert kit["font_files"] == []  # nothing uploaded: font_files stays the owner's uploads
    faces = kit[BUNDLED_FONT_FILES]
    assert _count(face["family"] for face in faces) == {"Geist": 4, "Newsreader": 4}
    assert {(face["weight"], face["style"]) for face in faces} == set(bundled_fonts.BUNDLED_FACE_SET)
    assert all(face["data_uri"].startswith("data:font/woff2;base64,") for face in faces)


def test_the_bundle_stages_them_after_the_uploads_and_media_render_takes_it():
    bundle = _title_card_bundle(_rendered(NIGHT_KIT))
    fonts = bundle["brand"]["fonts"]
    assert _count(face["family"] for face in fonts) == {"Geist": 4, "Newsreader": 4}
    staged = {entry["path"] for entry in bundle["files"]}
    assert all(face["path"] in staged and face["path"].startswith(FONTS_DIR) for face in fonts)
    settings = load_settings({})
    parsed = parse_bundle(bundle, settings, {})
    assert len(parsed.files) <= settings.max_files and len(fonts) <= settings.max_brand_tokens


def test_an_uploaded_face_of_a_bundled_family_is_not_bundled_again(monkeypatch):
    monkeypatch.setattr(brand_fonts, "load_brand_font", lambda font: WOFF2)
    upload = {"id": "0" * 32, "family": "Geist", "weight": 700, "style": "normal",
              "path": f"ws-1/brand/fonts/{'0' * 32}.woff2", "file_name": "geist.woff2", "bytes": len(WOFF2)}
    kit = _rendered({**NIGHT_KIT, "font_files": [upload]})
    assert [font["family"] for font in kit["font_files"]] == ["Geist"]
    assert _count(face["family"] for face in kit[BUNDLED_FONT_FILES]) == {"Newsreader": 4}
    fonts = _title_card_bundle(kit)["brand"]["fonts"]
    assert fonts[0]["family"] == "Geist" and fonts[0]["path"] == f"{FONTS_DIR}font-0.woff2"  # the upload first


def test_the_pdf_path_still_counts_only_the_uploads_and_adds_each_bundled_family_once():
    css = font_css(_rendered(NIGHT_KIT))
    assert _count(FACE_RULE.findall(css)) == {"Geist": 4, "Newsreader": 4}
    assert "Substitute font" not in css
