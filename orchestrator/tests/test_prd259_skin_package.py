"""PRD-259 S-003 — the .autoskin package reader: services/skins/package.py read_package.

A package is a zip holding skin.json and only the asset paths §3 names. Each refusal
carries a code and a message; a valid package gives the validated skin, the manifest
``{path: {type, size, sha256}}`` and the files to store. Zips are built in memory
(``tests/fixtures/skin_packages.py``); nothing touches storage.
"""
from __future__ import annotations

import hashlib
import io
import json
import stat
import struct
import sys
import warnings
import zipfile
from pathlib import Path

import pytest

ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(ORCH_ROOT))

from config import config  # noqa: E402
from services.skins.package import (  # noqa: E402
    MAX_MEMBERS, PACKAGE_ERROR_CODES, SkinPackageError, read_package,
)
from services.skins.schema import MASCOT_STATES, SKIN_ERROR_CODES  # noqa: E402
from tests.fixtures.skin_packages import (  # noqa: E402
    MP4, PNG, WEBM, base_skin, build_zip, full_package, mascot_files, package, woff2,
)

FONT_SKIN = base_skin(fonts={"sans": "assets/fonts/brand.woff2", "serif": "newsreader", "mono": "geist-mono"})
CUSTOM_SKIN = base_skin(mascot={"mode": "custom"})


def _refusal(data: bytes) -> SkinPackageError:
    with pytest.raises(SkinPackageError) as caught:
        read_package(data)
    return caught.value


def _codes(data: bytes) -> list:
    return [error["code"] for error in _refusal(data).errors]


# A central directory entry: 46 fixed bytes, then the name, the extra field and the comment.
CENTRAL_FIXED = 46
FLAG_BITS_AT = 8  # general purpose bit flag (bit 0: encrypted)
SIZE_AT = 24  # uncompressed size


def _patched(data: bytes, name: str, offset: int, value: bytes) -> bytes:
    """``data`` with ``value`` written at ``offset`` into ``name``'s central directory entry."""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        position = zf.start_dir
    out = bytearray(data)
    while out[position:position + 4] == b"PK\x01\x02":
        name_len, extra_len, comment_len = struct.unpack("<HHH", out[position + 28:position + 34])
        entry_name = out[position + CENTRAL_FIXED:position + CENTRAL_FIXED + name_len].decode("utf-8")
        if entry_name == name:
            out[position + offset:position + offset + len(value)] = value
            return bytes(out)
        position += CENTRAL_FIXED + name_len + extra_len + comment_len
    raise AssertionError(f"{name} is not in the zip")


# ---------------------------------------------------------------------------
# The valid packages
# ---------------------------------------------------------------------------


def test_a_full_package_gives_the_skin_the_manifest_and_the_files():
    result = read_package(full_package())

    assert result.skin["slug"] == "studio-own"
    assert result.skin["fonts"]["sans"] == "assets/fonts/brand.woff2"
    assert result.skin["mascot"]["mode"] == "custom"
    expected_paths = {"assets/preview.png", "assets/fonts/brand.woff2", *mascot_files(MASCOT_STATES)}
    assert set(result.manifest) == expected_paths == set(result.files)
    assert "skin.json" not in result.manifest
    font = result.manifest["assets/fonts/brand.woff2"]
    assert font == {"type": "font/woff2", "size": 1024, "sha256": hashlib.sha256(woff2()).hexdigest()}
    assert result.manifest["assets/preview.png"]["type"] == "image/png"
    assert result.manifest["assets/mascot/idle.webm"]["type"] == "video/webm"
    assert result.manifest["assets/mascot/idle.mp4"]["type"] == "video/mp4"
    assert result.manifest["assets/mascot/idle.png"]["type"] == "image/png"
    assert result.files["assets/fonts/brand.woff2"].data == woff2()
    assert result.files["assets/fonts/brand.woff2"].content_type == "font/woff2"


def test_a_minimal_package_holds_only_its_preview():
    result = read_package(package())
    assert set(result.manifest) == {"assets/preview.png"}
    assert result.skin["fonts"] == {"sans": "geist", "serif": "newsreader", "mono": "geist-mono"}


def test_a_skin_without_assets_needs_no_assets_folder():
    skin = {key: value for key, value in base_skin().items() if key != "preview"}
    assert read_package(package(skin, files={})).manifest == {}


def test_the_folder_entries_a_zip_tool_writes_are_skipped():
    extra = [("assets/", b""), ("assets/fonts/", b"")]
    assert set(read_package(package(extra=extra)).manifest) == {"assets/preview.png"}


# ---------------------------------------------------------------------------
# The package itself
# ---------------------------------------------------------------------------


def test_a_package_over_the_cap_is_refused_as_too_large(monkeypatch):
    monkeypatch.setattr(config, "SKIN_MAX_PACKAGE_MB", 1)
    error = _refusal(b"PK\x03\x04" + b"\x00" * (1024 * 1024))
    assert [e["code"] for e in error.errors] == ["PACKAGE_TOO_LARGE"]
    assert error.too_large


@pytest.mark.parametrize("data", [b"", b"not a zip", b"PK\x03\x04 but not really a zip", b"%PDF-1.7"])
def test_anything_but_a_zip_is_refused(data):
    assert _codes(data) == ["NOT_A_ZIP"]


def test_more_entries_than_a_skin_can_hold_are_refused():
    extra = [(f"assets/fonts/f{i}.woff2", woff2()) for i in range(MAX_MEMBERS)]
    assert _codes(package(extra=extra)) == ["TOO_MANY_FILES"]


def test_a_flood_of_tiny_entries_is_refused_before_zipfile_parses_them(monkeypatch):
    import services.skins.package as package_module

    flood = build_zip([(f"{i:x}", b"") for i in range(5000)], compression=zipfile.ZIP_STORED)
    assert len(flood) < 1024 * 1024
    opened = []
    monkeypatch.setattr(package_module.zipfile, "ZipFile", lambda *args, **kwargs: opened.append(args))
    assert _codes(flood) == ["TOO_MANY_FILES"]
    assert opened == []  # refused from the end record alone


# ---------------------------------------------------------------------------
# Member paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["../x", "assets/../../skin.json", "/etc/passwd", "/skin.json", "C:/x", "assets\\fonts\\a.woff2"])
def test_an_unsafe_path_is_refused(name):
    assert _codes(package(extra=[(name, b"x")])) == ["UNSAFE_PATH"]


def test_a_symlink_member_is_refused():
    link = zipfile.ZipInfo("assets/fonts/brand.woff2")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    assert _codes(package(FONT_SKIN, extra=[(link, b"/etc/passwd")])) == ["SYMLINK"]


def test_a_duplicate_name_is_refused():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # zipfile warns on the duplicate it is asked to write
        data = package(extra=[("assets/preview.png", PNG)])
    assert _codes(data) == ["DUPLICATE_FILE"]


def test_an_encrypted_member_is_refused():
    data = _patched(package(), "assets/preview.png", FLAG_BITS_AT, (0x1).to_bytes(2, "little"))
    assert _codes(data) == ["ENCRYPTED"]


@pytest.mark.parametrize("name", [
    "assets/preview.svg", "assets/theme.css", "assets/app.js", "index.html", "assets/fonts/brand.ttf",
    "assets/fonts/brand.woff", "assets/mascot/dance.webm", "assets/mascot/idle.gif", "assets/other/preview.png",
    "skin.json.bak", "assets/fonts/sub/brand.woff2",
])
def test_a_member_outside_the_allowed_paths_is_refused(name):
    assert _codes(package(extra=[(name, b"x")])) == ["NOT_ALLOWED"]


def test_a_svg_a_css_and_a_js_are_each_named_in_the_refusal():
    extra = [("assets/logo.svg", b"<svg/>"), ("assets/skin.css", b"*{}"), ("assets/skin.js", b"alert(1)")]
    error = _refusal(package(extra=extra))
    assert [(e["code"], e["path"]) for e in error.errors] == [
        ("NOT_ALLOWED", "assets/logo.svg"), ("NOT_ALLOWED", "assets/skin.css"), ("NOT_ALLOWED", "assets/skin.js"),
    ]


# ---------------------------------------------------------------------------
# Sizes: the declared total before any read, then counted while reading
# ---------------------------------------------------------------------------


def test_a_high_ratio_bomb_is_refused_from_the_central_directory(monkeypatch):
    monkeypatch.setattr(config, "SKIN_MAX_PACKAGE_MB", 1)
    bomb = package(files={"assets/preview.png": PNG + b"\x00" * (4 * 1024 * 1024)})
    assert len(bomb) < 64 * 1024  # 4 MB of zeros deflates to a few KB
    assert _codes(bomb) == ["UNCOMPRESSED_TOO_LARGE"]


def test_a_member_that_unpacks_past_its_declared_size_is_refused_while_reading():
    honest = package(files={"assets/preview.png": PNG + b"\x01" * 4000})
    lying = _patched(honest, "assets/preview.png", SIZE_AT, (100).to_bytes(4, "little"))  # claims 100 bytes
    assert _codes(lying) == ["CORRUPT_FILE"]


# ---------------------------------------------------------------------------
# skin.json
# ---------------------------------------------------------------------------


def test_a_package_without_skin_json_is_refused():
    assert _codes(build_zip([("assets/preview.png", PNG)])) == ["MISSING_SKIN_JSON"]


@pytest.mark.parametrize("raw", [b"{not json", b"\xff\xfe\x00", b"[1, 2"])
def test_skin_json_that_is_not_json_is_refused(raw):
    assert _codes(build_zip([("skin.json", raw), ("assets/preview.png", PNG)])) == ["BAD_JSON"]


def test_skin_json_goes_through_the_validator():
    skin = {**base_skin(), "css": "body{background:url(//x)}"}
    skin["tokens"]["light"]["foreground"] = skin["tokens"]["light"]["background"]
    codes = _codes(package(skin))
    assert "UNKNOWN_FIELD" in codes
    assert "CONTRAST" in codes


# ---------------------------------------------------------------------------
# What skin.json names against what the package holds
# ---------------------------------------------------------------------------


def test_an_undeclared_extra_file_is_refused():
    error = _refusal(package(extra=[("assets/fonts/extra.woff2", woff2())]))
    assert [(e["code"], e["path"]) for e in error.errors] == [("UNDECLARED_FILE", "assets/fonts/extra.woff2")]


def test_mascot_files_without_a_custom_mascot_are_undeclared():
    assert set(_codes(package(extra=list(mascot_files(["idle"]).items())))) == {"UNDECLARED_FILE"}


def test_a_named_font_that_is_missing_is_refused():
    error = _refusal(package(FONT_SKIN))
    assert [(e["code"], e["path"]) for e in error.errors] == [("MISSING_FILE", "assets/fonts/brand.woff2")]


def test_a_named_preview_that_is_missing_is_refused():
    assert _codes(package(files={})) == ["MISSING_FILE"]


def test_a_custom_mascot_needs_every_state_in_every_format():
    files = {"assets/preview.png": PNG, **mascot_files(MASCOT_STATES)}
    del files["assets/mascot/sleeping.mp4"]
    assert _codes(package(CUSTOM_SKIN, files)) == ["MASCOT_INCOMPLETE"]


def test_a_custom_mascot_with_no_files_is_refused():
    assert _codes(package(CUSTOM_SKIN)) == ["MASCOT_INCOMPLETE"]


# ---------------------------------------------------------------------------
# Fonts, mascot and preview by their bytes, and their caps
# ---------------------------------------------------------------------------


def test_a_png_renamed_woff2_is_refused():
    assert _codes(package(FONT_SKIN, {"assets/preview.png": PNG, "assets/fonts/brand.woff2": PNG})) == ["NOT_WOFF2"]


def test_a_401_kb_woff2_is_refused():
    big = woff2(401 * 1024)
    assert _codes(package(FONT_SKIN, {"assets/preview.png": PNG, "assets/fonts/brand.woff2": big})) == ["FONT_TOO_LARGE"]


def test_a_400_kb_woff2_is_accepted():
    files = {"assets/preview.png": PNG, "assets/fonts/brand.woff2": woff2(400 * 1024)}
    assert read_package(package(FONT_SKIN, files)).manifest["assets/fonts/brand.woff2"]["size"] == 400 * 1024


def test_a_woff2_with_a_broken_header_is_refused():
    broken = woff2()[:-10]  # the header still claims the full length
    files = {"assets/preview.png": PNG, "assets/fonts/brand.woff2": broken}
    assert _codes(package(FONT_SKIN, files)) == ["BAD_FONT_FILE"]


@pytest.mark.parametrize("path,data", [
    ("assets/mascot/idle.webm", PNG), ("assets/mascot/idle.mp4", WEBM), ("assets/mascot/idle.png", MP4),
    ("assets/mascot/idle.webm", b"<svg onload=alert(1)>"),
])
def test_mascot_media_is_sniffed_by_its_bytes(path, data):
    files = {"assets/preview.png": PNG, **mascot_files(MASCOT_STATES), path: data}
    assert _codes(package(CUSTOM_SKIN, files)) == ["BAD_MASCOT_FILE"]


def test_the_mascot_files_together_are_capped(monkeypatch):
    monkeypatch.setattr(config, "SKIN_MAX_MASCOT_MB", 1)
    files = {"assets/preview.png": PNG, **mascot_files(MASCOT_STATES, fill=30 * 1024)}  # 36 x 30 KB > 1 MB
    assert _codes(package(CUSTOM_SKIN, files)) == ["MASCOT_TOO_LARGE"]


def test_a_preview_that_is_not_png_is_refused():
    assert _codes(package(files={"assets/preview.png": b"GIF89a" + b"\x00" * 32})) == ["NOT_PNG"]


def test_a_preview_over_its_cap_is_refused(monkeypatch):
    monkeypatch.setattr(config, "SKIN_MAX_PREVIEW_KB", 1)
    assert _codes(package(files={"assets/preview.png": PNG + b"\x00" * 2048})) == ["PREVIEW_TOO_LARGE"]


# ---------------------------------------------------------------------------
# Every refusal has a known code and a message
# ---------------------------------------------------------------------------


def test_every_refusal_carries_a_known_code_a_path_and_a_message():
    error = _refusal(package(extra=[("../x", b"x"), ("assets/a.js", b"x")]))
    for item in error.errors:
        assert item["code"] in PACKAGE_ERROR_CODES + SKIN_ERROR_CODES
        assert set(item) == {"code", "path", "message"}
        assert item["message"]
    assert json.dumps(error.errors)  # the route sends them as JSON
