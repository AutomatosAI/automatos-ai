"""PRD-251 Wave 1, US-108 (S1.3, D5) — the brand kit extended in place.

Pins:

* **Every new field is validated.** ``heading_font`` is one font stack (it becomes
  ``--brand-heading-font``); ``font_family`` stays the body font and a heading
  without its own font takes it. ``social_handles`` are checked against each
  network's handle rule (any other toolkit takes the generic rule), stored
  without the "@". ``voice`` takes three to five tone words, or none, and at most
  fifty banned phrases. ``logo_mark_url`` is http(s) or empty. The stored files
  (``font_files``, ``logo_mark_path``, ``logo_path``) are written by their upload
  routes only: a PUT cannot point the kit at a stored file. A stored field that
  fails validation takes its default and the rest of the kit is kept.
* **Font files: type and size caps.** Only a complete woff2 holding one face is
  accepted (the header is read, the declared type ignored), 2 MB at most, six to
  a kit; the face (family, weight, style) is checked, and the same face uploaded
  again replaces its file. The CI's "CI Block" font (scripts/ci/ci_block_font.py)
  is a real woff2: fontTools reads it with the real Brotli decoder.
* **An uploaded woff2 is stored, served and inlined into a render bundle.** The
  upload route stores it under the document root, the stream route serves the
  bytes, and the Socials render's kit (``brand_kit_for_media_render``) carries it
  as a data: URI, which the bundle stages with its ``@font-face``: media-render's
  own parser accepts the bundle, and the heading font names the uploaded face.
* **The logo mark** is a square PNG/JPEG stored like the logo; the bundle stages
  it and fills ``{{ brand.logo_mark }}``, falling back to the logo; every seeded
  template shows the mark.
* **A full kit fits one bundle**: six fonts, a logo and a mark at their caps stay
  inside media-render's default limits.
"""
from __future__ import annotations

import base64
import importlib.util
import io
import math
import os
import struct
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
_ROOT = _ORCH.parent
_MEDIA_RENDER = _ROOT / "services" / "media-render"
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))
if str(_MEDIA_RENDER) not in sys.path:
    sys.path.append(str(_MEDIA_RENDER))

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from pydantic import ValidationError  # noqa: E402

import core.auth.workspace_permission as permission_mod  # noqa: E402
import modules.documents.brand_logo as bl  # noqa: E402
from core.auth.dependencies import RequestContext, UserContext  # noqa: E402
from core.auth.hybrid import get_request_context_hybrid  # noqa: E402
from core.database.database import get_db  # noqa: E402
from core.media_render_bundle import NO_LOGO, brand_tokens, build_bundle  # noqa: E402
from core.social_templates import resolve_variables  # noqa: E402
from media_render.bundle import parse_bundle  # noqa: E402  (media-render's own parser: stdlib only)
from media_render.config import load_settings  # noqa: E402
from modules.documents.brand_fonts import (  # noqa: E402
    MAX_FONT_BYTES,
    WOFF2_MIME,
    BrandFontError,
    add_brand_font,
    brand_kit_for_media_render,
    check_woff2,
    remove_brand_font,
)
from modules.documents.brand_kit import (  # noqa: E402
    DEFAULT_FONT,
    MAX_BANNED_PHRASES,
    MAX_FONT_FILES,
    MAX_SOCIAL_HANDLES,
    BrandFontFile,
    BrandKit,
    get_brand_kit,
    validate_brand_kit,
)
from modules.documents.social_starters import social_starters  # noqa: E402

WS = uuid.UUID("00000000-0000-0000-0000-0000000000d1")
FONTS_ROUTE = "/api/documents/brand-kit/fonts"
MARK_ROUTE = "/api/documents/brand-kit/logo-mark"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BLOCK_FONT = _load("ci_block_font", _ROOT / "scripts" / "ci" / "ci_block_font.py")
WOFF2 = BLOCK_FONT.block_font_woff2()


def png_bytes(width: int, height: int) -> bytes:
    """A PNG signature + IHDR chunk (enough header for sniffing + dimensions)."""
    ihdr = width.to_bytes(4, "big") + height.to_bytes(4, "big") + b"\x08\x06\x00\x00\x00"
    return b"\x89PNG\r\n\x1a\n" + (13).to_bytes(4, "big") + b"IHDR" + ihdr + b"\x00" * 16


def woff2_header(*, flavor: int = 0x00010000, num_tables: int = 10, length: int = 64, size: int = 64) -> bytes:
    """A woff2 header claiming ``length`` bytes, padded to ``size``."""
    header = struct.pack(">4sIIHH", b"wOF2", flavor, length, num_tables, 0)
    return header + b"\x00" * (size - len(header))


def _data_uri(mime: str, data: bytes) -> str:
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


@pytest.fixture
def storage(tmp_path, monkeypatch):
    monkeypatch.setattr(bl.config, "DOCUMENT_STORAGE_DIR", str(tmp_path), raising=False)
    monkeypatch.setattr(bl, "is_storage_configured", lambda: False)
    return tmp_path


def _ctx():
    return RequestContext(
        workspace_id=WS,
        user=UserContext(id="member-1", clerk_user_id="clerk-member-1", system_role="user"),
        auth_type="clerk",
    )


@pytest.fixture
def api(storage, monkeypatch):
    """The documents router (the brand kit routes ride it) over one workspace held in memory."""
    import api.document_generation as documents_module

    role = {"value": "owner"}
    monkeypatch.setattr(permission_mod, "resolve_workspace_role", lambda db, ctx: role["value"])
    workspace = SimpleNamespace(id=WS, name="Acme", settings={})
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = workspace
    app = FastAPI()
    app.include_router(documents_module.router)
    app.dependency_overrides[get_request_context_hybrid] = _ctx
    app.dependency_overrides[get_db] = lambda: db
    return SimpleNamespace(client=TestClient(app), db=db, workspace=workspace, role=role, storage=storage)


def _upload_font(api, data: bytes = WOFF2, *, family: str = "Brand Display", weight: str = "700", style: str = "normal"):
    return api.client.post(
        FONTS_ROUTE,
        files={"file": ("brand-display-700.woff2", data, "font/woff2")},
        data={"family": family, "weight": weight, "style": style},
    )


# ---------------------------------------------------------------------------
# The new fields, validated
# ---------------------------------------------------------------------------


def test_an_old_kit_reads_with_the_new_fields_empty():
    stored = {"name": "Acme", "primary_color": "#0055aa", "logo_path": f"{WS}/brand/logo.png"}
    for kit in (get_brand_kit({"brand_kit": stored}), get_brand_kit(None)):
        assert kit["heading_font"] == "" and kit["font_files"] == []
        assert kit["logo_mark_url"] == "" and kit["logo_mark_path"] == ""
        assert kit["social_handles"] == {} and kit["voice"] == {"tone": [], "banned_phrases": []}
    assert get_brand_kit({"brand_kit": stored})["primary_color"] == "#0055aa"


def test_font_family_stays_the_body_font_and_the_heading_font_falls_back_to_it():
    assert "body_font" not in BrandKit.model_fields
    kit = get_brand_kit({"brand_kit": {"font_family": "Inter, sans-serif"}})
    assert brand_tokens(kit)["heading-font"] == brand_tokens(kit)["body-font"] == "Inter, sans-serif"
    kit = validate_brand_kit({"heading_font": '  "Brand Display", Georgia, serif  '}, kit)
    assert kit["heading_font"] == '"Brand Display", Georgia, serif'
    assert brand_tokens(kit)["heading-font"] == '"Brand Display", Georgia, serif'
    assert brand_tokens(kit)["body-font"] == "Inter, sans-serif"
    assert get_brand_kit(None)["font_family"] == DEFAULT_FONT


@pytest.mark.parametrize(
    "stack",
    ["Inter; } body { color: red", "url(https://evil.example/f.woff2)", "Inter<script>", "a\\62 c", "x" * 201, "Inter\nsans"],
)
def test_the_heading_font_is_one_font_stack(stack):
    with pytest.raises(ValidationError, match="heading_font"):
        validate_brand_kit({"heading_font": stack})


@pytest.mark.parametrize(
    "toolkit, handle, stored",
    [
        ("twitter", "@acme_hq", "acme_hq"),
        ("twitter", "a" * 15, "a" * 15),
        ("instagram", "acme.studio", "acme.studio"),
        ("instagram", "a" * 30, "a" * 30),
        ("tiktok", "@acme.hq", "acme.hq"),
        ("youtube", "Acme-HQ", "Acme-HQ"),
        ("linkedin", "acme-inc", "acme-inc"),
        ("mastodon", "acme_hq.social", "acme_hq.social"),  # any connected toolkit: the generic rule
    ],
)
def test_a_handle_that_fits_its_network_is_stored_without_the_at(toolkit, handle, stored):
    assert validate_brand_kit({"social_handles": {toolkit: handle}})["social_handles"] == {toolkit: stored}


@pytest.mark.parametrize(
    "toolkit, handle",
    [
        ("twitter", "acme-hq"),  # X takes letters, digits and underscores
        ("twitter", "a" * 16),
        ("instagram", ".acme"),
        ("instagram", "acme."),
        ("instagram", "ac..me"),
        ("instagram", "a" * 31),
        ("tiktok", "a"),
        ("tiktok", "acme."),
        ("tiktok", "a" * 25),
        ("youtube", "ab"),
        ("youtube", "a" * 31),
        ("linkedin", "acme_inc"),
        ("linkedin", "ab"),
        ("mastodon", "acme hq"),
        ("mastodon", "a" * 101),
    ],
)
def test_a_handle_that_breaks_its_network_rule_is_refused(toolkit, handle):
    with pytest.raises(ValidationError, match=f"the {toolkit} handle"):
        validate_brand_kit({"social_handles": {toolkit: handle}})


def test_handles_are_keyed_by_toolkit_an_empty_one_is_removed_and_the_map_is_replaced():
    kit = validate_brand_kit({"social_handles": {"LinkedIn": "acme-inc", "twitter": "@acme"}})
    assert kit["social_handles"] == {"linkedin": "acme-inc", "twitter": "acme"}
    # A PUT carries the whole map: a network left out, or given no handle, is removed.
    kit = validate_brand_kit({"social_handles": {"twitter": "", "instagram": "acme"}}, kit)
    assert kit["social_handles"] == {"instagram": "acme"}
    with pytest.raises(ValidationError, match="toolkit name"):
        validate_brand_kit({"social_handles": {"not a toolkit!": "acme"}})
    many = {f"network_{i}": "acme" for i in range(MAX_SOCIAL_HANDLES + 1)}
    with pytest.raises(ValidationError, match=f"at most {MAX_SOCIAL_HANDLES} social handles"):
        validate_brand_kit({"social_handles": many})


@pytest.mark.parametrize("count", [0, 3, 4, 5])
def test_three_to_five_tone_words_or_none(count):
    words = ["warm", "plain", "bold", "precise", "curious"][:count]
    assert validate_brand_kit({"voice": {"tone": words}})["voice"]["tone"] == words


@pytest.mark.parametrize("count", [1, 2, 6])
def test_any_other_number_of_tone_words_is_refused(count):
    words = ["warm", "plain", "bold", "precise", "curious", "dry"][:count]
    with pytest.raises(ValidationError, match=f"give 3 to 5 tone words, or none \\(got {count}\\)"):
        validate_brand_kit({"voice": {"tone": words}})


def test_tone_words_are_trimmed_counted_once_and_each_one_short_line_with_a_letter():
    kit = validate_brand_kit({"voice": {"tone": [" Warm ", "warm", "", "plain-spoken", "bold"]}})
    assert kit["voice"]["tone"] == ["Warm", "plain-spoken", "bold"]
    for bad, reason in ((["warm", "plain", "x" * 33], "at most 32"), (["warm", "plain", "123"], "a letter"), (["warm", "plain\nbold", "dry"], "one line")):
        with pytest.raises(ValidationError, match=reason):
            validate_brand_kit({"voice": {"tone": bad}})


def test_banned_phrases_are_capped_and_the_voice_merges_key_by_key():
    phrases = [f"phrase {i}" for i in range(MAX_BANNED_PHRASES)]
    kit = validate_brand_kit({"voice": {"tone": ["warm", "plain", "bold"], "banned_phrases": phrases}})
    assert kit["voice"]["banned_phrases"] == phrases
    with pytest.raises(ValidationError, match=f"at most {MAX_BANNED_PHRASES} banned phrases"):
        validate_brand_kit({"voice": {"banned_phrases": phrases + ["one more"]}})
    for bad in ("x" * 121, "game\nchanger"):
        with pytest.raises(ValidationError):
            validate_brand_kit({"voice": {"banned_phrases": [bad]}})
    # A patch naming only the banned phrases keeps the tone words.
    merged = validate_brand_kit({"voice": {"banned_phrases": ["game-changer", "Game-Changer"]}}, kit)
    assert merged["voice"] == {"tone": ["warm", "plain", "bold"], "banned_phrases": ["game-changer"]}


def test_a_client_patch_cannot_point_the_kit_at_a_stored_file():
    stored = {"logo_path": f"{WS}/brand/logo.png", "logo_mark_path": f"{WS}/brand/logo-mark.png", "font_files": []}
    merged = validate_brand_kit(
        {
            "logo_mark_path": "../../etc/passwd",
            "font_files": [{"id": "a" * 32, "family": "X", "path": "../../etc/passwd"}],
            "name": "Acme",
        },
        stored,
    )
    assert (merged["logo_mark_path"], merged["font_files"], merged["name"]) == (f"{WS}/brand/logo-mark.png", [], "Acme")


def _font(**overrides):
    entry = {"id": uuid.uuid4().hex, "family": "Brand Display", "weight": 700, "style": "normal",
             "path": f"{WS}/brand/fonts/{'b' * 32}.woff2"}
    return {**entry, **overrides}


@pytest.mark.parametrize(
    "overrides, field",
    [
        ({"id": "not-an-id"}, "id"),
        ({"family": "Brand;Sans"}, "family"),
        ({"family": "-Brand"}, "family"),
        ({"weight": 450}, "weight"),
        ({"weight": 1000}, "weight"),
        ({"style": "oblique"}, "style"),
        ({"path": "../../etc/passwd"}, "path"),
        ({"path": f"{WS}/brand/fonts/{'b' * 32}.ttf"}, "path"),
        ({"bytes": -1}, "bytes"),
    ],
)
def test_a_font_file_entry_is_checked(overrides, field):
    with pytest.raises(ValidationError, match=field):
        BrandFontFile.model_validate(_font(**overrides))


def test_a_kit_holds_six_distinct_font_files_at_most():
    assert len(BrandKit.model_validate({"font_files": [_font() for _ in range(MAX_FONT_FILES)]}).font_files) == MAX_FONT_FILES
    with pytest.raises(ValidationError, match=f"at most {MAX_FONT_FILES} font files"):
        BrandKit.model_validate({"font_files": [_font() for _ in range(MAX_FONT_FILES + 1)]})
    twin = _font()
    with pytest.raises(ValidationError, match="appears once"):
        BrandKit.model_validate({"font_files": [twin, twin]})


def test_a_bad_stored_field_takes_its_default_and_the_rest_of_the_kit_is_kept():
    stored = {"name": "Acme", "primary_color": "#0055aa", "social_handles": {"twitter": "not a handle!"}}
    kit = get_brand_kit({"brand_kit": stored})
    assert (kit["name"], kit["primary_color"], kit["social_handles"]) == ("Acme", "#0055aa", {})


# ---------------------------------------------------------------------------
# Font files: the type and size caps
# ---------------------------------------------------------------------------


def test_the_ci_block_font_is_a_real_woff2():
    from fontTools.ttLib import TTFont  # the orchestrator's WeasyPrint brings fontTools with Brotli

    font = TTFont(io.BytesIO(WOFF2))
    assert font.flavor == "woff2"
    assert font["maxp"].numGlyphs == 96 and font["head"].unitsPerEm == 1000
    assert font["name"].getDebugName(1) == BLOCK_FONT.FAMILY
    cmap = font.getBestCmap()
    assert len(cmap) == 95  # space to tilde
    glyph = font["glyf"][cmap[ord("A")]]
    assert glyph.numberOfContours == 1 and (glyph.xMin, glyph.yMin, glyph.xMax, glyph.yMax) == BLOCK_FONT.BLOCK
    assert font["glyf"][cmap[ord(" ")]].numberOfContours == 0
    assert len(WOFF2) % 4 == 0 and len(WOFF2) < 8 * 1024


def test_check_woff2_takes_one_complete_woff2_face():
    check_woff2(WOFF2)
    cases = [
        (b"", "empty"),
        (b"\x00\x01\x00\x00" + b"\x00" * 60, "woff2"),  # a TTF
        (b"wOFF" + b"\x00" * 60, "woff2"),  # a woff (1)
        (b"wOF2" + b"\x00" * 10, "header is incomplete"),
        (WOFF2 + b"\x00\x00\x00\x00", "incomplete"),  # the header says fewer bytes than arrived
        (woff2_header(flavor=0x74746366), "collection"),
        (woff2_header(flavor=0x12345678), "TrueType or OpenType"),
        (woff2_header(num_tables=0), "no font tables"),
        (woff2_header(length=MAX_FONT_BYTES + 1, size=MAX_FONT_BYTES + 1), "MB or smaller"),
    ]
    for data, reason in cases:
        with pytest.raises(BrandFontError, match=reason):
            check_woff2(data)


def test_a_face_is_checked_and_the_same_face_replaces_its_file(storage):
    fonts = add_brand_font(WS, [], WOFF2, family=" Brand Display ", weight=700, style="Normal", file_name="bd.woff2")
    (first,) = fonts
    assert (first["family"], first["weight"], first["style"], first["bytes"]) == ("Brand Display", 700, "normal", len(WOFF2))
    assert first["path"] == f"{WS}/brand/fonts/{first['id']}.woff2"
    assert (storage / first["path"]).read_bytes() == WOFF2
    # The same face again: one entry, a new file, the old file gone.
    (second,) = add_brand_font(WS, fonts, WOFF2, family="brand display", weight=700, style="normal")
    assert second["id"] != first["id"] and not (storage / first["path"]).exists()
    for face, reason in (({"family": "Brand;Display"}, "family name"), ({"weight": 450}, "weight"), ({"style": "oblique"}, "style")):
        kwargs = {"family": "Brand Display", "weight": 700, "style": "normal", **face}
        with pytest.raises(BrandFontError, match=reason):
            add_brand_font(WS, [second], WOFF2, **kwargs)
    full = [_font(weight=weight) for weight in (100, 200, 300, 400, 500, 600)]
    with pytest.raises(BrandFontError, match=f"{MAX_FONT_FILES} font files at most"):
        add_brand_font(WS, full, WOFF2, family="Another", weight=400, style="normal")
    assert sorted(p.name for p in (storage / str(WS) / "brand" / "fonts").iterdir()) == [f"{second['id']}.woff2"]
    assert remove_brand_font([second], "0" * 32) is None
    assert remove_brand_font([second], second["id"]) == [] and not (storage / second["path"]).exists()


# ---------------------------------------------------------------------------
# An uploaded woff2: stored, served, inlined into a render bundle
# ---------------------------------------------------------------------------


def _render_blocks():
    """A small social image whose headline reads the heading font."""
    html = (
        "<!doctype html><html><head><meta charset=\"UTF-8\" /><style>"
        "h1 { font-family: var(--brand-heading-font, sans-serif); font-weight: 700; }"
        "</style></head><body><div id=\"root\" data-composition-id=\"main\" data-start=\"0\" data-duration=\"1\" "
        "data-width=\"{{ size.width }}\" data-height=\"{{ size.height }}\">"
        "<img src=\"{{ brand.logo_mark }}\" alt=\"\" /><h1>{{ headline }}</h1>"
        "<audio id=\"mix\" src=\"assets/audio/mix.wav\" data-start=\"0\" data-duration=\"1\" data-track-index=\"1\"></audio>"
        "</div></body></html>"
    )
    return {"html": html, "variables_schema": {"headline": {"type": "text", "max_chars": 60}}, "sizes": ["1080x1350"]}


def test_an_uploaded_woff2_is_stored_served_and_inlined_into_a_render_bundle(api):
    import api.socials as socials_api

    response = _upload_font(api)
    assert response.status_code == 200, response.text
    (font,) = response.json()["font_files"]
    assert (font["family"], font["weight"], font["style"]) == ("Brand Display", 700, "normal")
    assert (font["file_name"], font["bytes"]) == ("brand-display-700.woff2", len(WOFF2))
    # Stored: under the document root, and on the kit the workspace keeps.
    assert (api.storage / font["path"]).read_bytes() == WOFF2
    assert api.workspace.settings["brand_kit"]["font_files"] == [font]
    api.db.commit.assert_called()

    # Served back to the UI.
    served = api.client.get(f"{FONTS_ROUTE}/{font['id']}")
    assert served.status_code == 200 and served.content == WOFF2
    assert served.headers["content-type"] == WOFF2_MIME

    # Inlined: the Socials render's kit carries it as a data: URI ...
    kit = validate_brand_kit({"heading_font": '"Brand Display", sans-serif'}, api.workspace.settings["brand_kit"])
    api.workspace.settings = {"brand_kit": kit}
    rendered_kit = socials_api._render_brand_kit(api.workspace.settings)
    assert rendered_kit["font_files"] == [
        {"family": "Brand Display", "weight": 700, "style": "normal", "data_uri": _data_uri(WOFF2_MIME, WOFF2)}
    ]
    # ... and the bundle stages it with its @font-face, the heading font naming it.
    blocks = _render_blocks()
    values = resolve_variables(blocks["variables_schema"], {"headline": "On brand"}).values
    bundle = build_bundle(workspace_id=WS, reference="r", blocks=blocks, values=values, brand_kit=rendered_kit, fmt="social_image")
    font_path = "assets/brand/fonts/font-0.woff2"
    assert {"path": font_path, "data_uri": _data_uri(WOFF2_MIME, WOFF2)} in bundle["files"]
    assert bundle["brand"]["fonts"] == [{"family": "Brand Display", "weight": "700", "style": "normal", "path": font_path}]
    assert bundle["brand"]["tokens"]["heading-font"] == '"Brand Display", sans-serif'
    parsed = parse_bundle(bundle, load_settings({}), {})
    assert [(f.path, f.data) for f in parsed.files] == [(font_path, WOFF2)]
    assert '@font-face{font-family:"Brand Display";src:url("assets/brand/fonts/font-0.woff2") format("woff2");font-weight:700' in parsed.composition.html
    assert '--brand-heading-font:"Brand Display", sans-serif;' in parsed.composition.html

    # Removed: gone from the kit and from storage.
    removed = api.client.delete(f"{FONTS_ROUTE}/{font['id']}")
    assert removed.status_code == 200 and removed.json()["font_files"] == []
    assert not (api.storage / font["path"]).exists()
    assert api.client.get(f"{FONTS_ROUTE}/{font['id']}").status_code == 404
    assert api.client.delete(f"{FONTS_ROUTE}/{font['id']}").status_code == 404


def test_a_refused_font_upload_says_why_and_stores_nothing(api):
    cases = [
        (b"\x00\x01\x00\x00" + b"\x00" * 60, {}, "woff2"),
        (WOFF2, {"family": "Brand;Display"}, "family name"),
        (WOFF2, {"weight": "450"}, "weight"),
        (WOFF2, {"style": "oblique"}, "style"),
    ]
    for data, face, reason in cases:
        response = _upload_font(api, data, **face)
        assert response.status_code == 422 and reason in response.json()["detail"], response.text
    assert _upload_font(api, weight="bold").status_code == 422  # not a number
    assert not (api.storage / str(WS)).exists()
    api.db.commit.assert_not_called()
    # Only a workspace manager uploads.
    api.role["value"] = "editor"
    assert _upload_font(api).status_code == 403
    assert not (api.storage / str(WS)).exists()


def test_brand_kit_for_media_render_inlines_the_logo_the_mark_and_the_fonts_and_documents_do_not(storage):
    logo = png_bytes(400, 100)
    mark = png_bytes(128, 128)
    fonts = add_brand_font(WS, [], WOFF2, family="Brand Display", weight=700, style="normal")
    kit = get_brand_kit({"brand_kit": {
        "logo_path": bl.save_brand_logo(WS, logo),
        "logo_mark_path": bl.save_brand_logo_mark(WS, mark),
        "font_files": fonts,
    }})
    rendered = brand_kit_for_media_render(kit)
    assert rendered["logo_url"] == _data_uri("image/png", logo)
    assert rendered["logo_mark_url"] == _data_uri("image/png", mark)
    assert rendered["font_files"] == [{"family": "Brand Display", "weight": 700, "style": "normal", "data_uri": _data_uri(WOFF2_MIME, WOFF2)}]
    assert kit["logo_mark_url"] == "" and "data_uri" not in kit["font_files"][0]  # the input is untouched
    # The PDF / DOCX path inlines the logo only.
    documents = bl.brand_kit_for_render(kit)
    assert documents["logo_mark_url"] == "" and documents["font_files"] == fonts
    # A font whose bytes are gone is left out of the render.
    (storage / fonts[0]["path"]).unlink()
    assert brand_kit_for_media_render(kit)["font_files"] == []


# ---------------------------------------------------------------------------
# The logo mark
# ---------------------------------------------------------------------------


def test_a_logo_mark_is_a_square_logo_stored_beside_the_logo(storage):
    logo_path = bl.save_brand_logo(WS, png_bytes(400, 100))
    mark_path = bl.save_brand_logo_mark(WS, png_bytes(512, 512))
    assert (logo_path, mark_path) == (f"{WS}/brand/logo.png", f"{WS}/brand/logo-mark.png")
    assert bl.load_brand_logo(logo_path) == png_bytes(400, 100)
    assert bl.save_brand_logo_mark(WS, png_bytes(120, 100)) == mark_path  # a little off square is fine
    with pytest.raises(bl.BrandLogoError, match="must be square"):
        bl.save_brand_logo_mark(WS, png_bytes(400, 100))
    with pytest.raises(bl.BrandLogoError, match="PNG or JPEG"):
        bl.save_brand_logo_mark(WS, b"GIF89a" + b"\x00" * 10)
    assert (storage / logo_path).exists() and (storage / mark_path).exists()


def test_the_logo_mark_routes_upload_serve_and_remove_it(api):
    uploaded = api.client.post(MARK_ROUTE, files={"file": ("mark.png", png_bytes(256, 256), "image/png")})
    assert uploaded.status_code == 200, uploaded.text
    body = uploaded.json()
    assert body["logo_mark_path"] == f"{WS}/brand/logo-mark.png" and body["logo_mark_route"] == MARK_ROUTE
    assert api.workspace.settings["brand_kit"]["logo_mark_path"] == body["logo_mark_path"]
    served = api.client.get(MARK_ROUTE)
    assert served.status_code == 200 and served.content == png_bytes(256, 256)
    assert served.headers["content-type"] == "image/png"
    wide = api.client.post(MARK_ROUTE, files={"file": ("wide.png", png_bytes(800, 200), "image/png")})
    assert wide.status_code == 422 and "square" in wide.json()["detail"]
    removed = api.client.delete(MARK_ROUTE)
    assert removed.status_code == 200 and removed.json()["logo_mark_path"] == ""
    assert api.client.get(MARK_ROUTE).status_code == 404


def test_the_bundle_stages_the_mark_and_falls_back_to_the_logo():
    blocks = _render_blocks()
    values = {"headline": "Hi"}
    logo_uri = _data_uri("image/png", png_bytes(400, 100))
    mark_uri = _data_uri("image/png", png_bytes(128, 128))

    def bundle(kit):
        return build_bundle(workspace_id=WS, reference="r", blocks=blocks, values=values, brand_kit=kit, fmt="social_image")

    both = bundle({"logo_url": logo_uri, "logo_mark_url": mark_uri})
    assert both["files"] == [
        {"path": "assets/brand/logo.png", "data_uri": logo_uri},
        {"path": "assets/brand/logo-mark.png", "data_uri": mark_uri},
    ]
    assert (both["variables"]["brand.logo"], both["variables"]["brand.logo_mark"]) == ("assets/brand/logo.png", "assets/brand/logo-mark.png")
    logo_only = bundle({"logo_url": logo_uri, "logo_mark_url": "https://cdn.example/mark.png"})  # never fetched
    assert logo_only["variables"]["brand.logo_mark"] == logo_only["variables"]["brand.logo"] == "assets/brand/logo.png"
    assert bundle({})["variables"]["brand.logo_mark"] == NO_LOGO
    parsed = parse_bundle(both, load_settings({}), {})
    assert 'src="assets/brand/logo-mark.png"' in parsed.composition.html


def test_every_seeded_template_shows_the_mark_where_it_showed_the_logo():
    for starter in social_starters():
        html = starter["blocks"]["html"]
        assert "{{ brand.logo_mark }}" in html, starter["slug"]
        assert "{{ brand.logo }}" not in html, starter["slug"]


def test_the_put_saves_the_new_fields_and_keeps_the_stored_files(api):
    response = api.client.put("/api/documents/brand-kit", json={
        "heading_font": '"Brand Display", serif',
        "logo_mark_url": "https://cdn.example/mark.png",
        "social_handles": {"twitter": "@acme", "linkedin": "acme-inc"},
        "voice": {"tone": ["warm", "plain", "bold"], "banned_phrases": ["game-changer"]},
        "font_files": [_font()],
        "logo_mark_path": "../../etc/passwd",
    })
    assert response.status_code == 200, response.text
    kit = response.json()
    assert kit["heading_font"] == '"Brand Display", serif' and kit["logo_mark_url"] == "https://cdn.example/mark.png"
    assert kit["social_handles"] == {"twitter": "acme", "linkedin": "acme-inc"}
    assert kit["voice"] == {"tone": ["warm", "plain", "bold"], "banned_phrases": ["game-changer"]}
    assert kit["font_files"] == [] and kit["logo_mark_path"] == ""
    # A refusal names the field and the rule (the detail serialises: no exception objects in it).
    for body, loc, rule in (
        ({"voice": {"tone": ["warm", "plain"]}}, ["voice", "tone"], "give 3 to 5 tone words"),
        ({"social_handles": {"twitter": "acme-hq"}}, ["social_handles"], "the twitter handle"),
        ({"heading_font": "Inter; } body {"}, ["heading_font"], "one font stack"),
    ):
        refused = api.client.put("/api/documents/brand-kit", json=body)
        assert refused.status_code == 422, refused.text
        (error,) = refused.json()["detail"]["errors"]
        assert error["loc"] == loc and rule in error["msg"]
    assert api.client.put("/api/documents/brand-kit", json={"logo_mark_url": "javascript:alert(1)"}).status_code == 422


# ---------------------------------------------------------------------------
# A full kit fits one bundle
# ---------------------------------------------------------------------------


def test_a_full_brand_kit_fits_one_media_render_bundle():
    settings = load_settings({})

    def inline(mime: str, size: int) -> int:
        return len(f"data:{mime};base64,") + 4 * math.ceil(size / 3)

    files = [inline(WOFF2_MIME, MAX_FONT_BYTES)] * MAX_FONT_FILES + [inline("image/png", bl.MAX_LOGO_BYTES)] * 2
    assert max(MAX_FONT_BYTES, bl.MAX_LOGO_BYTES) <= settings.max_asset_bytes
    assert len(files) <= settings.max_files
    # The brand files, with a megabyte to spare for the composition and the rest of the bundle.
    assert sum(files) + 1024 * 1024 <= settings.max_bundle_bytes


# ---------------------------------------------------------------------------
# The CI render: the Title card with the uploaded heading font
# ---------------------------------------------------------------------------


def test_the_ci_driver_renders_the_title_card_with_an_uploaded_heading_font_and_mark():
    driver = _load("social_template_previews", _ROOT / "scripts" / "ci" / "social_template_previews.py")
    kit = driver.heading_font_kit({**driver.KIT, "logo_url": driver.logo_png(driver.KIT["primary_color"])})
    assert "logo_url" not in kit and kit["logo_mark_url"].startswith("data:image/png;base64,")
    assert kit["heading_font"] == f'"{BLOCK_FONT.FAMILY}", {driver.KIT["heading_font"]}'
    (face,) = kit["font_files"]
    assert base64.b64decode(face["data_uri"].split(",", 1)[1]) == WOFF2
    starter = next(s for s in social_starters("social_image") if s["slug"] == driver.HEADING_FONT_TEMPLATE)
    bundle = driver.image_bundle_for(starter, kit, starter["blocks"]["sizes"][0])
    parsed = parse_bundle(bundle, load_settings({}), {})
    assert [f.path for f in parsed.files] == ["assets/brand/logo-mark.png", "assets/brand/fonts/font-0.woff2"]
    assert '@font-face{font-family:"CI Block";src:url("assets/brand/fonts/font-0.woff2") format("woff2");font-weight:900' in parsed.composition.html
    assert 'src="assets/brand/logo-mark.png"' in parsed.composition.html
    # The measured colour is the accent words' alone on this kit.
    tokens = bundle["brand"]["tokens"]
    accent = driver.hex_rgb(tokens[driver.HEADING_FONT_TOKEN])
    for name, value in tokens.items():
        if name != driver.HEADING_FONT_TOKEN and value.startswith("#") and len(value) == 7:
            assert max(abs(a - b) for a, b in zip(driver.hex_rgb(value), accent)) > driver.PROBE_TOKEN_TOLERANCE, name


def test_the_ink_measure_tells_solid_blocks_from_strokes_and_ignores_stray_pixels():
    driver = _load("social_template_previews", _ROOT / "scripts" / "ci" / "social_template_previews.py")
    ink, paper = (198, 83, 45), (240, 232, 219)

    def image(inked):
        return driver.encode_png(300, 200, lambda x, y: (*(ink if inked(x, y) else paper), 255))

    def text(width):
        return lambda x, y: (50 <= y < 110 and 20 <= x < 280 and (x - 20) % 43 < width) or (x, y) in ((5, 5), (6, 6))

    blocks, strokes = driver.ink_fill(image(text(40)), ink), driver.ink_fill(image(text(12)), ink)
    assert blocks[1] >= driver.BLOCK_FILL > strokes[1]
    assert blocks[0] == 60 * sum(1 for x in range(20, 280) if (x - 20) % 43 < 40)  # the two stray pixels are not counted
    assert driver.ink_fill(image(lambda x, y: False), ink) == (0, 0.0)
