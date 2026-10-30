"""PRD-270 US-270-03c (§5.3, §5.5, K15) — Cloud listings in the local storefront.

A version a Sync with Automatos Cloud wrote (``source`` 'cloud') is listed like any other, and says so:
the card and the detail carry ``source``. This install may not hold its bytes, so its screenshots load
from Automatos Cloud: every cover, phone cover, preset preview and other preview is
``<AUTOMATOS_CLOUD_URL origin>/api/marketplace/catalog/files/<digest>/<path>``, for the screenshots
theme.json lists and the row holds only, and its readme and changelog are null (the catalogue serves
screenshots and the package only). A local row answers as before, with ``source`` null, whatever the
setting says. Official, the versions map, compatibility, the installed state and revocations read a
Cloud row exactly as a local one. A changed ``AUTOMATOS_CLOUD_URL`` moves the URLs; one that is empty
or not an origin leaves a Cloud row without screenshots, never with a files-route URL.

The real listings router runs on the Themes tab's test app (``test_prd261_theme_listings.py``'s ``api``
fixture: in-memory SQLite, the real gates); its rows are built as that file builds them.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from config import config  # noqa: E402
from core.models.themes import THEME_SOURCE_CLOUD, ThemeInstallation, ThemeRevocation  # noqa: E402
import services.themes.listings as listings_mod  # noqa: E402
from services.themes.serve import module_url  # noqa: E402
from tests.test_prd261_theme_listings import (  # noqa: E402,F401 — `api` is a fixture
    FUTURE_ENGINES, OTHER_RFC_KEY, PLAY, PLAY_LISTING, PREVIEWS, WS_A, _file, _keys, _listings, _package,
    _sign, _signable, api,
)

CLOUD = "https://cloud.example.test"
MIRROR = "https://mirror.example.test:8443"
LOOPBACK = "http://127.0.0.1:8000"
CATALOG_FILES = "/api/marketplace/catalog/files"
README, CHANGELOG = "README.md", "CHANGELOG.md"
# Play's presets are ember and ember-dark (test_prd261_theme_listings.PRESETS); team.jpg is no preset's.
PHONE = ("assets/previews/phone/ember.png", "assets/previews/phone/ember-dark-chat.jpg", "assets/previews/phone/team.jpg")
PHONE_NOT_HELD = "assets/previews/phone/ember-settings.png"  # listed in theme.json, not in the package
UNLISTED = "assets/previews/unlisted.png"  # in the package, listed nowhere in theme.json
CARD_KEYS = frozenset({
    "listingId", "themeId", "name", "description", "publisher", "signature", "version", "packageId",
    "compatible", "uses", "pricing", "cover", "phoneCover", "revoked", "installed", "source",
})
DETAIL_KEYS = CARD_KEYS | {
    "presets", "otherPreviews", "otherPhonePreviews", "readme", "changelog", "base", "tone", "minWidth",
    "pages", "routes", "latest", "versions",
}
# What a Cloud row answers exactly as a local one (everything but its URLs and its source).
SAME_FIELDS = (
    "listingId", "themeId", "name", "description", "publisher", "signature", "version", "packageId", "compatible",
    "uses", "pricing", "revoked", "installed", "base", "tone", "minWidth", "pages", "routes", "latest", "versions",
)


def _listed(api, version="1.0.0", source=THEME_SOURCE_CLOUD, **manifest_extra):  # noqa: F811 — the imported fixture
    """A listed Play version that also holds phone screenshots, a changelog and a screenshot theme.json doesn't list."""
    row = _package(api.db, version, phonePreviews=[*PHONE, PHONE_NOT_HELD], **manifest_extra)
    held = {path: _file("12") for path in (*PHONE, UNLISTED)}
    row.file_manifest = {**row.file_manifest, CHANGELOG: _file("cd"), **held}
    row.source = source
    api.db.commit()
    return row


def _detail(api) -> Dict[str, Any]:  # noqa: F811 — the imported fixture
    resp = api.client.get(f"/api/marketplace/themes/{PLAY_LISTING}")
    assert resp.status_code == 200, resp.text
    return resp.json()


def _answers(api):  # noqa: F811 — the imported fixture
    return _listings(api)[PLAY_LISTING], _detail(api)


def _cloud_url(origin: str, row, path: str) -> str:
    return f"{origin}{CATALOG_FILES}/{row.digest}/{path}"


def _strings(value: Any) -> List[str]:
    """Every string in a JSON answer, depth first."""
    if isinstance(value, dict):
        return [text for item in value.values() for text in _strings(item)]
    if isinstance(value, list):
        return [text for item in value for text in _strings(item)]
    return [value] if isinstance(value, str) else []


def _file_urls(answer: Dict[str, Any]) -> List[str]:
    return [text for text in _strings(answer) if "/files/" in text]


def _screenshots(answer: Dict[str, Any]) -> Dict[str, Any]:
    """Every screenshot field of a detail: the covers, each preset's lists and the unclaimed ones."""
    return {key: answer[key] for key in ("cover", "phoneCover", "presets", "otherPreviews", "otherPhonePreviews")}


def _set_cloud_url(monkeypatch, value: str) -> None:
    monkeypatch.setattr(config, "AUTOMATOS_CLOUD_URL", value)


# ---------------------------------------------------------------------------
# A Cloud row: its source, its screenshots from Automatos Cloud, no readme or changelog
# ---------------------------------------------------------------------------


def test_a_cloud_rows_card_and_detail_carry_source_cloud_and_cloud_urls_for_listed_screenshots_only(
    api, monkeypatch,  # noqa: F811 — the imported fixture
):
    _set_cloud_url(monkeypatch, CLOUD)
    row = _listed(api)
    card, detail = _answers(api)
    url = lambda path: _cloud_url(CLOUD, row, path)  # noqa: E731
    assert set(card) == CARD_KEYS and set(detail) == DETAIL_KEYS
    assert card["source"] == detail["source"] == "cloud"
    assert card["cover"] == detail["cover"] == url(PREVIEWS[0])
    assert card["phoneCover"] == detail["phoneCover"] == url(PHONE[0])
    assert detail["presets"] == [
        {"id": "ember", "default": True,
         "previews": [{"screen": None, "url": url(PREVIEWS[0])}, {"screen": "chat", "url": url(PREVIEWS[1])}],
         "phonePreviews": [{"screen": None, "url": url(PHONE[0])}]},
        {"id": "ember-dark", "default": False,
         "previews": [{"screen": "chat", "url": url(PREVIEWS[2])}],
         "phonePreviews": [{"screen": "chat", "url": url(PHONE[1])}]},
    ]
    assert detail["otherPreviews"] == [url(PREVIEWS[3])] and detail["otherPhonePreviews"] == [url(PHONE[2])]
    assert {README, CHANGELOG} <= set(row.file_manifest)  # held, yet the catalogue serves neither
    assert detail["readme"] is None and detail["changelog"] is None
    for path in (README, CHANGELOG, UNLISTED, "dist/entry.js"):  # held, but no screenshot theme.json lists
        assert listings_mod._file_url(row, path) is None, path
    listed = {*PREVIEWS, *PHONE}
    assert {text[len(url("")):] for text in _file_urls(detail)} == listed
    assert all(text.startswith(url("")) for text in _file_urls(detail) + _file_urls(card))
    assert module_url(row.digest, "") not in "".join(_strings(detail))  # never this install's files route


def test_a_local_rows_answer_is_unchanged_but_for_source_none_whatever_the_setting(api, monkeypatch):  # noqa: F811
    row = _listed(api, source=None)
    _set_cloud_url(monkeypatch, "")
    card_off, detail_off = _answers(api)
    _set_cloud_url(monkeypatch, CLOUD)
    card, detail = _answers(api)
    assert (card, detail) == (card_off, detail_off)  # the Cloud setting never moves a local row
    assert set(card) == CARD_KEYS and set(detail) == DETAIL_KEYS
    assert card["source"] is None and detail["source"] is None
    url = lambda path: module_url(row.digest, path)  # noqa: E731
    assert card["cover"] == url(PREVIEWS[0]) and card["phoneCover"] == url(PHONE[0])
    assert detail["readme"] == url(README) and detail["changelog"] == url(CHANGELOG)
    assert detail["otherPreviews"] == [url(PREVIEWS[3])] and detail["otherPhonePreviews"] == [url(PHONE[2])]
    assert detail["presets"][1]["phonePreviews"] == [{"screen": "chat", "url": url(PHONE[1])}]
    assert all(text.startswith(url("")) for text in _file_urls(detail))
    assert CLOUD not in "".join(_strings(detail))


# ---------------------------------------------------------------------------
# Official, versions, compatibility, installs and revocations read a Cloud row as a local one
# ---------------------------------------------------------------------------


def test_an_automatos_cloud_row_signed_by_an_active_key_is_official(api, monkeypatch, tmp_path):  # noqa: F811
    _set_cloud_url(monkeypatch, CLOUD)
    key_id = _keys(monkeypatch, tmp_path)
    row = _signable(api.db)
    row.source = THEME_SOURCE_CLOUD
    api.db.commit()
    _sign(api.db, row)
    verified = {"state": "verified", "keyId": key_id}
    card, detail = _answers(api)
    assert card["source"] == "cloud"
    assert card["publisher"] == {"name": "automatos", "official": True} and card["signature"] == verified
    assert detail["publisher"]["official"] is True and detail["versions"]["1.0.0"]["signature"] == verified
    _keys(monkeypatch, tmp_path, status="retired")  # listings' rule: still its signer's and listed, never Official
    card = _listings(api)[PLAY_LISTING]
    assert card["publisher"] == {"name": "automatos", "official": False}
    assert card["signature"] == {"state": "invalid", "keyId": key_id}


def test_a_cloud_row_whose_signature_fails_is_not_listed_as_a_local_one_isnt(api, monkeypatch, tmp_path):  # noqa: F811
    _keys(monkeypatch, tmp_path)
    row = _signable(api.db)
    row.source = THEME_SOURCE_CLOUD
    api.db.commit()
    _sign(api.db, row, OTHER_RFC_KEY)  # a key the keys file doesn't hold (C-05c: the version is disabled)
    assert PLAY_LISTING not in _listings(api)
    assert api.client.get(f"/api/marketplace/themes/{PLAY_LISTING}").status_code == 404


def test_an_unsigned_automatos_cloud_row_is_not_official(api, monkeypatch, tmp_path):  # noqa: F811
    _keys(monkeypatch, tmp_path)
    _listed(api)
    card = _listings(api)[PLAY_LISTING]
    assert card["source"] == "cloud"
    assert card["publisher"] == {"name": "automatos", "official": False}
    assert card["signature"] == {"state": "unsigned", "keyId": None}


def test_versions_compatibility_installs_and_revocations_read_a_cloud_row_as_a_local_one(api, monkeypatch):  # noqa: F811
    _set_cloud_url(monkeypatch, CLOUD)
    rows = [_listed(api, "1.0.0", source=None), _listed(api, "1.1.0", source=None),
            _listed(api, "2.0.0", source=None, engines=FUTURE_ENGINES)]
    api.db.add(ThemeInstallation(workspace_id=WS_A, theme_id=PLAY, package_id=rows[0].id))
    api.db.add(ThemeRevocation(theme_id=PLAY, level="warn", reason="Contrast issue in one preset."))
    api.db.commit()
    local_card, local_detail = _answers(api)
    for row in rows:
        row.source = THEME_SOURCE_CLOUD
    api.db.commit()
    card, detail = _answers(api)
    assert card["source"] == detail["source"] == "cloud" and local_card["source"] is None
    for field in SAME_FIELDS:
        assert detail[field] == local_detail[field], field
        assert card.get(field) == local_card.get(field), field
    assert [(preset["id"], preset["default"]) for preset in detail["presets"]] == [("ember", True), ("ember-dark", False)]
    assert detail["version"] == detail["latest"] == "1.1.0" and detail["compatible"] is True
    assert detail["versions"]["2.0.0"]["compatible"] is False and detail["versions"]["1.1.0"]["compatible"] is True
    assert detail["installed"]["version"] == "1.0.0" and detail["installed"]["fromListing"] is True
    assert detail["installed"]["updatePackageId"] == str(rows[1].id)
    assert detail["revoked"] == {"level": "warn", "reason": "Contrast issue in one preset."}
    assert detail["cover"] == _cloud_url(CLOUD, rows[1], PREVIEWS[0])


def test_a_disable_revocation_unlists_a_cloud_version_as_it_does_a_local_one(api, monkeypatch):  # noqa: F811
    _set_cloud_url(monkeypatch, CLOUD)
    older, newer = _listed(api, "1.0.0"), _listed(api, "1.1.0")
    api.db.add(ThemeRevocation(digest=newer.digest, level="disable", reason="Withdrawn."))
    api.db.commit()
    card, detail = _answers(api)
    assert card["version"] == "1.0.0" and card["packageId"] == str(older.id)
    assert set(detail["versions"]) == {"1.0.0"}


# ---------------------------------------------------------------------------
# The setting moves the URLs
# ---------------------------------------------------------------------------


def test_a_changed_automatos_cloud_url_moves_the_urls_with_it(api, monkeypatch):  # noqa: F811
    row = _listed(api)
    for setting, origin in ((CLOUD, CLOUD), (f"{MIRROR}/", MIRROR), (LOOPBACK, LOOPBACK)):
        _set_cloud_url(monkeypatch, setting)
        card, detail = _answers(api)
        assert card["cover"] == _cloud_url(origin, row, PREVIEWS[0]), setting
        assert card["phoneCover"] == _cloud_url(origin, row, PHONE[0]), setting
        assert detail["otherPhonePreviews"] == [_cloud_url(origin, row, PHONE[2])], setting
        assert all(text.startswith(_cloud_url(origin, row, "")) for text in _file_urls(detail)), setting


def test_an_empty_or_unusable_setting_leaves_a_cloud_row_without_screenshots_never_a_files_route_url(
    api, monkeypatch,  # noqa: F811 — the imported fixture
):
    _listed(api)
    for setting in ("", f"{CLOUD}/api", "http://cloud.example.test", "https://user:pw@cloud.example.test"):
        _set_cloud_url(monkeypatch, setting)
        card, detail = _answers(api)
        assert card["source"] == "cloud" and card["cover"] is None and card["phoneCover"] is None, setting
        assert _screenshots(detail) == {
            "cover": None, "phoneCover": None, "otherPreviews": [], "otherPhonePreviews": [],
            "presets": [{"id": "ember", "default": True, "previews": [], "phonePreviews": []},
                        {"id": "ember-dark", "default": False, "previews": [], "phonePreviews": []}],
        }, setting
        assert _file_urls(detail) == [] and detail["readme"] is None and detail["changelog"] is None, setting
