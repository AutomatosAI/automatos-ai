"""PRD-261 — theme packages ship phone screenshots, and the Marketplace listing answers them per look.

The storefront contract (30 Oct 2026): theme.json's optional ``phonePreviews`` lists
``assets/previews/phone/<NAME>.png`` or ``.jpg`` files, PNG or JPEG by their magic bytes, at most
512 KB each and 24 in all; web ``previews`` hold 24 too. Pinned here:

* the manifest check accepts a PNG and a JPEG in the phone folder, and refuses another folder or
  extension, a list over 24, one naming a file twice and one that isn't a list of paths; web previews
  take 24 and refuse 25, fonts still stop at 16;
* the package reader types them ``image/png`` and ``image/jpeg`` and refuses bytes that aren't the
  format the extension says, a file over 512 KB, a named file the package lacks and a phone file
  outside its folder or formats; 24 web and 24 phone screenshots read;
* the files route serves a ``.jpg`` as ``image/jpeg``, with the headers a PNG preview gets;
* the card's ``phoneCover`` is the first phone screenshot theme.json lists that the package holds; the
  detail puts each under its preset by the web previews' rule (``<preset>`` or ``<preset>-<screen>``,
  the longest preset id wins) and the rest in ``otherPhonePreviews``; a package without them answers
  as before, with ``null`` and ``[]`` in the new fields.

The route tests reuse the files route's world (``test_prd261_theme_files.py``, its ``package`` fixture
overridden below) and the listings' app (``test_prd261_theme_listings.py``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from services.themes.manifest import (  # noqa: E402
    MAX_FILES_LISTED, MAX_PREVIEWS, ManifestResult, referenced_files, validate_manifest,
)
from services.themes.package import (  # noqa: E402
    IMAGE_MAX_BYTES, JPEG_SIGNATURE, PNG_SIGNATURE, ThemePackageError, read_theme_package,
)
from services.themes.serve import module_url  # noqa: E402
from tests.fixtures.themes import build  # noqa: E402
from tests.test_prd261_theme_files import (  # noqa: E402,F401 — `world` is a fixture
    _assert_serving_headers, _file, _person, _row, world,
)
from tests.test_prd261_theme_listings import (  # noqa: E402,F401 — `api` is a fixture
    PLAY_LISTING, PREVIEWS, _listings, _package as _listing_package, api,
)

KB = 1024
PNG = PNG_SIGNATURE + bytes(8)
JPEG = JPEG_SIGNATURE + b"\xe0" + bytes(8)
WEB_PNG = "assets/previews/default.png"
PHONE_PNG = "assets/previews/phone/default.png"
PHONE_JPG = "assets/previews/phone/night-chat.jpg"
# The listings' Play has presets ember and ember-dark (test_prd261_theme_listings.PRESETS).
PHONE = ("assets/previews/phone/ember.png", "assets/previews/phone/ember-chat.jpg",
         "assets/previews/phone/ember-dark-chat.jpg", "assets/previews/phone/team.png")
# What the card and the detail answered before phone screenshots, and the new fields.
CARD_KEYS = frozenset({
    "listingId", "themeId", "name", "description", "publisher", "signature", "version", "packageId",
    "compatible", "uses", "pricing", "cover", "revoked", "installed",
})
DETAIL_KEYS = frozenset({
    "presets", "otherPreviews", "readme", "changelog", "base", "tone", "minWidth", "pages", "routes",
    "latest", "versions",
})
PRESET_KEYS = frozenset({"id", "default", "previews"})
# Headers that differ file by file; every other one a PNG preview gets, a JPEG gets too.
PER_FILE_HEADERS = frozenset({"content-type", "content-length", "etag"})


def _manifest(**fields) -> ManifestResult:
    return validate_manifest({**build.hello_manifest(), **fields})


def _codes(result: ManifestResult) -> List[str]:
    return [error["code"] for error in result.errors]


def _phone_package(
    phone: Mapping[str, bytes], listed: Optional[Sequence[str]] = None, web: Optional[Mapping[str, bytes]] = None,
) -> bytes:
    """The hello package with ``phone`` (and ``web``) screenshots, listed in theme.json (``listed`` overrides)."""
    fields = {"phonePreviews": list(phone) if listed is None else list(listed)}
    fields = {**fields, "previews": list(web)} if web else fields
    return build.build_package(replace={build.MANIFEST: build.changed_manifest(fields), **phone, **(web or {})})


def _refusals(data: bytes) -> List[Tuple[str, str]]:
    try:
        read_theme_package(data)
    except ThemePackageError as exc:
        return [(error["code"], error["path"]) for error in exc.errors]
    return []


# ---------------------------------------------------------------------------
# theme.json
# ---------------------------------------------------------------------------


def test_a_png_and_a_jpg_phone_screenshot_are_accepted():
    result = _manifest(phonePreviews=[PHONE_PNG, PHONE_JPG])
    assert result.errors == []
    assert result.manifest["phonePreviews"] == [PHONE_PNG, PHONE_JPG]


@pytest.mark.parametrize("path", [
    "assets/previews/default.png",  # the web previews' folder
    "assets/phone/default.png",  # outside assets/previews
    "assets/previews/phone/home/default.png",  # a folder below phone/
    "assets/previews/phone/default.webp",
    "assets/previews/phone/default.jpeg",
    "assets/previews/phone/default.JPG",
    "assets/previews/phone/default.svg",
    "assets/previews/phone/-default.png",  # a name starts with a letter or a digit
])
def test_a_phone_screenshot_outside_its_folder_or_formats_is_refused(path):
    result = _manifest(phonePreviews=[path])
    assert result.manifest is None and _codes(result) == ["BAD_ASSET_PATH"]
    assert result.errors[0]["path"] == "phonePreviews"


@pytest.mark.parametrize("value", [
    [f"assets/previews/phone/shot-{n}.jpg" for n in range(MAX_PREVIEWS + 1)],
    [PHONE_PNG, PHONE_PNG],
    PHONE_PNG,
    [7],
], ids=["25 screenshots", "one named twice", "not a list", "not a path"])
def test_a_phone_list_over_the_cap_naming_a_file_twice_or_not_of_paths_is_refused(value):
    assert _codes(_manifest(phonePreviews=value)) == ["BAD_ASSET_PATH"]


def test_web_and_phone_lists_each_take_24_and_fonts_still_stop_at_16():
    assert (MAX_PREVIEWS, MAX_FILES_LISTED) == (24, 16)
    web = [f"assets/previews/shot-{n}.png" for n in range(MAX_PREVIEWS)]
    phone = [f"assets/previews/phone/shot-{n}.jpg" for n in range(MAX_PREVIEWS)]
    assert _manifest(previews=web, phonePreviews=phone).errors == []
    assert _codes(_manifest(previews=[*web, "assets/previews/one-more.png"])) == ["BAD_ASSET_PATH"]
    fonts = [f"assets/fonts/face-{n}.woff2" for n in range(MAX_FILES_LISTED + 1)]
    assert _codes(_manifest(fonts=fonts)) == ["BAD_ASSET_PATH"]


def test_a_phone_screenshot_is_not_a_web_preview():
    assert _codes(_manifest(previews=[PHONE_PNG])) == ["BAD_ASSET_PATH"]


def test_the_files_a_manifest_names_include_its_phone_screenshots():
    result = _manifest(previews=[WEB_PNG], phonePreviews=[PHONE_JPG, PHONE_PNG])
    assert referenced_files(result.manifest) == [WEB_PNG, PHONE_JPG, PHONE_PNG]


# ---------------------------------------------------------------------------
# The package reader
# ---------------------------------------------------------------------------


def test_a_png_and_a_jpg_phone_screenshot_read_as_png_and_jpeg():
    package = read_theme_package(_phone_package({PHONE_PNG: PNG, PHONE_JPG: JPEG}))
    assert package.files[PHONE_PNG].content_type == "image/png"
    assert package.files[PHONE_JPG].content_type == "image/jpeg"
    assert package.file_manifest[PHONE_JPG]["type"] == "image/jpeg"
    assert package.manifest["phonePreviews"] == [PHONE_PNG, PHONE_JPG]


@pytest.mark.parametrize("path, data", [
    (PHONE_JPG, PNG), (PHONE_PNG, JPEG), (PHONE_JPG, b"not an image"), (PHONE_PNG, b""),
], ids=["a PNG named .jpg", "a JPEG named .png", "text named .jpg", "an empty .png"])
def test_bytes_that_arent_the_format_the_extension_says_are_refused(path, data):
    assert _refusals(_phone_package({path: data})) == [("BAD_FILE_TYPE", path)]


@pytest.mark.parametrize("path, data", [(PHONE_PNG, PNG), (PHONE_JPG, JPEG)], ids=["png", "jpg"])
def test_a_phone_screenshot_over_512_kb_is_refused_and_one_of_512_kb_reads(path, data):
    assert IMAGE_MAX_BYTES == 512 * KB
    at_cap = data + bytes(IMAGE_MAX_BYTES - len(data))
    assert _refusals(_phone_package({path: at_cap})) == []
    assert _refusals(_phone_package({path: at_cap + b"\x00"})) == [("FILE_TOO_LARGE", path)]


def test_a_named_phone_screenshot_the_package_lacks_is_refused():
    assert _refusals(_phone_package({}, listed=[PHONE_JPG])) == [("MISSING_FILE", PHONE_JPG)]


@pytest.mark.parametrize("path, code", [
    ("assets/previews/phone/default.webp", "NOT_ALLOWED"),
    ("assets/previews/phone/default.jpeg", "NOT_ALLOWED"),
    ("assets/previews/phone/home/default.png", "NOT_ALLOWED"),
    ("assets/phone/default.png", "NOT_ALLOWED"),
    ("assets/previews/phone/default.svg", "SVG_NOT_ALLOWED"),
])
def test_a_phone_file_outside_its_folder_or_formats_is_refused(path, code):
    assert _refusals(build.build_package(replace={path: PNG})) == [(code, path)]


def test_24_web_and_24_phone_screenshots_read():
    web = {f"assets/previews/shot-{n}.png": PNG for n in range(MAX_PREVIEWS)}
    phone = {f"assets/previews/phone/shot-{n}.jpg": JPEG for n in range(MAX_PREVIEWS)}
    package = read_theme_package(_phone_package(phone, web=web))
    assert (len(package.manifest["previews"]), len(package.manifest["phonePreviews"])) == (MAX_PREVIEWS, MAX_PREVIEWS)
    assert {package.files[path].content_type for path in phone} == {"image/jpeg"}


# ---------------------------------------------------------------------------
# The files route
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def package():
    """``world``'s package: hello with a web preview and a PNG and a JPEG phone screenshot."""
    return read_theme_package(_phone_package({PHONE_PNG: PNG, PHONE_JPG: JPEG}, web={WEB_PNG: PNG}))


def _comparable(headers: Mapping[str, str]) -> Dict[str, str]:
    return {name.lower(): value for name, value in headers.items() if name.lower() not in PER_FILE_HEADERS}


def test_a_jpg_phone_screenshot_is_served_as_image_jpeg_with_a_png_previews_headers(world):
    _row(world, "marketplace", status="approved", reviewed=True)
    world.db.commit()
    web, phone_png, phone_jpg = (_file(world, _person("bob"), path) for path in (WEB_PNG, PHONE_PNG, PHONE_JPG))
    assert [resp.status_code for resp in (web, phone_png, phone_jpg)] == [200, 200, 200]
    assert phone_jpg.content == JPEG
    assert phone_jpg.headers["content-type"] == "image/jpeg"
    assert web.headers["content-type"] == phone_png.headers["content-type"] == "image/png"
    assert _comparable(phone_jpg.headers) == _comparable(web.headers) == _comparable(phone_png.headers)
    _assert_serving_headers(phone_jpg)


# ---------------------------------------------------------------------------
# The listing
# ---------------------------------------------------------------------------


def _image_row(path: str) -> Dict[str, object]:
    return {"type": "image/jpeg" if path.endswith(".jpg") else "image/png", "size": 1, "sha256": "34" * 32}


def _with_phone(api, listed: Sequence[str] = PHONE, held: Sequence[str] = PHONE):
    """The listings' Play listing ``listed`` phone screenshots in theme.json, holding the ``held`` ones."""
    row = _listing_package(api.db, phonePreviews=list(listed))
    row.file_manifest = {**row.file_manifest, **{path: _image_row(path) for path in held}}
    api.db.commit()
    return row


def _detail(api) -> Dict[str, object]:
    resp = api.client.get(f"/api/marketplace/themes/{PLAY_LISTING}")
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_the_cards_phone_cover_is_the_first_phone_screenshot_theme_json_lists(api):
    play = _with_phone(api, listed=[PHONE[3], PHONE[1], PHONE[0]])
    card = _listings(api)[PLAY_LISTING]
    assert card["phoneCover"] == module_url(play.digest, PHONE[3])
    assert card["cover"] == module_url(play.digest, PREVIEWS[0])
    assert _detail(api)["phoneCover"] == card["phoneCover"]


def test_the_detail_puts_each_phone_screenshot_under_its_look_by_the_web_previews_rule(api):
    play = _with_phone(api)
    url = lambda path: module_url(play.digest, path)  # noqa: E731
    detail = _detail(api)
    phone = {preset["id"]: preset["phonePreviews"] for preset in detail["presets"]}
    assert phone == {
        "ember": [{"screen": None, "url": url(PHONE[0])}, {"screen": "chat", "url": url(PHONE[1])}],
        "ember-dark": [{"screen": "chat", "url": url(PHONE[2])}],  # the longest preset id wins
    }
    assert detail["otherPhonePreviews"] == [url(PHONE[3])]
    web = {preset["id"]: [shot["url"] for shot in preset["previews"]] for preset in detail["presets"]}
    assert web == {"ember": [url(PREVIEWS[0]), url(PREVIEWS[1])], "ember-dark": [url(PREVIEWS[2])]}
    assert detail["otherPreviews"] == [url(PREVIEWS[3])]


def test_a_phone_screenshot_the_package_doesnt_hold_is_left_out(api):
    play = _with_phone(api, held=PHONE[1:])
    detail = _detail(api)
    assert detail["phoneCover"] == module_url(play.digest, PHONE[1])
    assert detail["presets"][0]["phonePreviews"] == [{"screen": "chat", "url": module_url(play.digest, PHONE[1])}]


def test_a_listing_without_phone_screenshots_answers_as_before_with_empty_phone_fields(api):
    play = _listing_package(api.db)
    card, detail = _listings(api)[PLAY_LISTING], _detail(api)
    assert set(card) == CARD_KEYS | {"phoneCover"} and card["phoneCover"] is None
    assert set(detail) == CARD_KEYS | DETAIL_KEYS | {"phoneCover", "otherPhonePreviews"}
    assert detail["otherPhonePreviews"] == [] and detail["phoneCover"] is None
    assert [set(preset) for preset in detail["presets"]] == [PRESET_KEYS | {"phonePreviews"}] * 2
    assert [preset["phonePreviews"] for preset in detail["presets"]] == [[], []]
    assert card["cover"] == detail["cover"] == module_url(play.digest, PREVIEWS[0])
    assert detail["otherPreviews"] == [module_url(play.digest, PREVIEWS[3])]
