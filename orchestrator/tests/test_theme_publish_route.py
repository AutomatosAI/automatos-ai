"""PRD-261 — ``POST /api/marketplace/themes/publish``: the Design Kit's publish job lists, approves and signs.

The flow through the real router (``tests/helpers_theme_publish.py`` sets up the app, SQLite, the
fake S3, the RFC signing key and a stubbed review stage). Pins (the kit ↔ core publish contract):

* a signed, reviewed version under an allowlisted publisher is 200 ``published``: one approved,
  reviewed Marketplace row submitted by the service user, the kit's summary as its passing report
  note, an approval, a signature request and a ``signed`` event by the RFC key, its files stored,
  and the publisher claimed for the service user; a reserved publisher (``automatos``) too;
* the same bytes again are 200 ``unchanged``: no row, event, publisher or file written, no stage run;
* D3: core's ``review_required`` approves, ``blocked`` is 422 and stores nothing;
* a version not above the highest held, or held with other files, is 409;
* a publisher outside ``THEME_PUBLISH_PUBLISHERS`` is 403, ``automatos`` included when it isn't
  listed (the service user's admin's say exists only under the allowlist); someone else's is 403;
* 422 for a refused package, no ``source/``, an envelope from another key or for other bytes, a
  summary for other bytes or a failed run, a revoked digest and a missing part; each stores nothing
  and runs no stage;
* the envelope and summary may come as text fields or as file parts.
"""
from __future__ import annotations

import json

import pytest

from api.marketplace_themes_publish import SUMMARY_MAX_BYTES
from core.models.themes import ThemePackage, ThemePublisher, ThemeReviewEvent, ThemeRevocation
from services.themes import signature_verify
from tests import helpers_theme_publish as hp

PUBLISHED_EVENTS = ["approved", "report_note", "scanned", "signature_requested", "signed", "submitted"]


@pytest.fixture
def env(monkeypatch, tmp_path):
    yield from hp.publish_env(monkeypatch, tmp_path)


def _published(response, data, theme_id=hp.THEME_ID, version="1.0.0"):
    assert response.status_code == 200, response.text
    assert response.json() == {"status": "published", "id": theme_id, "version": version, "digest": hp.digest_of(data)}


def _events(env, package_id):
    return sorted(row.event for row in hp.rows(env, ThemeReviewEvent) if row.package_id == package_id)


# --- published -------------------------------------------------------------------------------


def test_a_signed_reviewed_version_is_published_approved_and_signed(env):
    data = hp.package()
    _published(hp.post(env.client, data), data)

    (row,) = hp.rows(env, ThemePackage)
    assert (row.owner_type, row.status, row.reviewed, row.submitted_by) == ("marketplace", "approved", True, hp.ACTOR_PK)
    assert row.signature == json.loads(hp.envelope(data))["signature"]
    assert signature_verify.package_verdict(row).verified
    assert _events(env, row.id) == PUBLISHED_EVENTS
    events = {event.event: event for event in hp.rows(env, ThemeReviewEvent)}
    assert events["report_note"].detail == {"summary": hp.summary(row.digest).strip(), "outcome": "pass"}
    assert events["signed"].detail["keyId"] == hp.SIGNER["keyId"]
    assert {event.actor_user_id for event in events.values()} == {hp.ACTOR_PK}
    assert set(env.s3.objects) == {row.storage_prefix + path for path in row.file_manifest}
    assert [(p.publisher, p.owner_user_id, p.reserved) for p in hp.rows(env, ThemePublisher)] == [
        ("acme-e2e", hp.ACTOR_PK, False)]
    assert env.stage.calls == [row.digest]


def test_the_reserved_publisher_is_published_by_the_service_user(env):
    data = hp.package(hp.OFFICIAL_ID)
    _published(hp.post(env.client, data, review_summary=hp.summary(hp.digest_of(data), theme_id=hp.OFFICIAL_ID)),
               data, theme_id=hp.OFFICIAL_ID)
    (row,) = hp.rows(env, ThemePackage)
    assert row.status == "approved" and signature_verify.package_verdict(row).verified
    assert [(p.publisher, p.reserved) for p in hp.rows(env, ThemePublisher)] == [("automatos", True)]


def test_the_envelope_and_summary_may_be_file_parts(env):
    data = hp.package()
    _published(hp.post(env.client, data, as_files=True), data)


def test_the_same_bytes_again_are_unchanged_and_write_nothing(env):
    data = hp.package()
    _published(hp.post(env.client, data), data)
    before = hp.snapshot(env)

    again = hp.post(env.client, data)

    assert again.status_code == 200, again.text
    assert again.json() == {"status": "unchanged", "id": hp.THEME_ID, "version": "1.0.0", "digest": hp.digest_of(data)}
    assert hp.snapshot(env) == before
    assert len(env.stage.calls) == 1


def test_a_newer_version_is_published_next_to_the_first(env):
    first, second = hp.package(), hp.package(version="1.1.0")
    _published(hp.post(env.client, first), first)
    _published(hp.post(env.client, second), second, version="1.1.0")
    assert sorted(row.version for row in hp.rows(env, ThemePackage) if row.status == "approved") == ["1.0.0", "1.1.0"]


# --- D3: core's own verdict ------------------------------------------------------------------


def test_review_required_is_approved_on_this_endpoint(env):
    env.stage = hp.stage("review_required")
    data = hp.package()
    _published(hp.post(env.client, data), data)
    (row,) = hp.rows(env, ThemePackage)
    assert (row.scan_verdict, row.status) == ("review_required", "approved")


@pytest.mark.parametrize("verdict, code", [("blocked", "REVIEW_BLOCKED"), ("not_run", "REVIEW_NOT_RUN")])
def test_a_blocked_or_not_run_verdict_is_422_and_stores_nothing(env, verdict, code):
    env.stage = hp.stage(verdict)
    hp.refused(hp.post(env.client, hp.package()), 422, code)
    assert hp.nothing_stored(env) and len(env.stage.calls) == 1


# --- 409: versions ---------------------------------------------------------------------------


def test_a_version_not_above_the_highest_is_409(env):
    newer = hp.package(version="1.1.0")
    _published(hp.post(env.client, newer), newer, version="1.1.0")
    before = hp.snapshot(env)

    hp.refused(hp.post(env.client, hp.package(version="1.0.0")), 409, "VERSION_NOT_HIGHER")
    hp.refused(hp.post(env.client, hp.package(version="1.1.0", source=b"export default 1\n")), 409,
               "VERSION_NOT_HIGHER")
    assert hp.snapshot(env) == before and len(env.stage.calls) == 1


# --- 403: publishers -------------------------------------------------------------------------


def test_a_publisher_outside_the_allowlist_is_403(env):
    data = hp.package(hp.UNLISTED_ID)
    detail = hp.refused(hp.post(env.client, data), 403, "PUBLISHER_NOT_LISTED")
    assert "other-e2e" in detail["message"] and hp.nothing_stored(env)


def test_the_reserved_publisher_needs_the_allowlist_too(env, monkeypatch):
    hp.configure(monkeypatch, publishers="acme-e2e")
    data = hp.package(hp.OFFICIAL_ID)
    hp.refused(hp.post(env.client, data), 403, "PUBLISHER_NOT_LISTED")
    assert hp.nothing_stored(env)


def test_someone_elses_publisher_is_403(env):
    hp.add(env, ThemePublisher(publisher="acme-e2e", owner_user_id=hp.PERSON_PK, reserved=False))
    hp.refused(hp.post(env.client, hp.package()), 403, "PUBLISHER_TAKEN")
    assert hp.rows(env, ThemePackage) == [] and env.s3.objects == {} and env.stage.calls == []


# --- 422: the package, the envelope, the summary ---------------------------------------------


def _no_source() -> bytes:
    manifest = hp.build.changed_manifest({"id": hp.THEME_ID})
    return hp.build.build_package(replace={hp.build.MANIFEST: manifest})


def _cases():
    data = hp.package()
    digest = hp.digest_of(data)
    other = hp.package(source=b"export default 2\n")
    return {
        "a refused package": ({"data": b"not a zip"}, "PACKAGE_REFUSED"),
        "no source/": ({"data": _no_source()}, "SOURCE_REQUIRED"),
        "another key's envelope": ({"data": data, "signature": hp.envelope(data, hp.OTHER_SIGNER)}, "SIGNATURE_BAD"),
        "an envelope for other bytes": ({"data": data, "signature": hp.envelope(other)}, "SIGNATURE_MISMATCH"),
        "an envelope naming another version": (
            {"data": data, "signature": hp.envelope(data, version="9.9.9")}, "SIGNATURE_MISMATCH"),
        "not an envelope": ({"data": data, "signature": "base64-only"}, "SIGNATURE_BAD"),
        "a summary for other bytes": (
            {"data": data, "review_summary": hp.summary(hp.digest_of(other))}, "SUMMARY_NOT_THIS_VERSION"),
        "a failed review run": ({"data": data, "review_summary": hp.summary(digest, "fail")}, "SUMMARY_NOT_THIS_VERSION"),
        "a blank summary": ({"data": data, "review_summary": "   "}, "SUMMARY_NOT_THIS_VERSION"),
    }


@pytest.mark.parametrize("case", sorted(_cases()))
def test_each_refusal_is_422_stores_nothing_and_runs_no_stage(env, case):
    request, code = _cases()[case]
    data = request.pop("data")
    if code in ("PACKAGE_REFUSED", "SOURCE_REQUIRED"):
        request.setdefault("signature", "{}")
        request.setdefault("review_summary", "-")
    hp.refused(hp.post(env.client, data, **request), 422, code)
    assert hp.nothing_stored(env) and env.stage.calls == []


def test_a_revoked_digest_is_422(env):
    data = hp.package()
    hp.add(env, ThemeRevocation(digest=hp.digest_of(data), level="disable", reason="withdrawn"))
    hp.refused(hp.post(env.client, data), 422, "DIGEST_DISABLED")
    assert hp.rows(env, ThemePackage) == [] and env.s3.objects == {} and env.stage.calls == []


@pytest.mark.parametrize("missing", ["signature", "review_summary"])
def test_a_missing_part_is_422(env, missing):
    data = hp.package()
    form = {"signature": hp.envelope(data), "review_summary": hp.summary(hp.digest_of(data))}
    form.pop(missing)
    response = env.client.post(hp.THEME_PUBLISH_ROUTE, headers={"Authorization": f"Bearer {hp.KEY}"},
                               files={"package": ("p.autotheme", data, "application/zip")}, data=form)
    detail = hp.refused(response, 422, "PART_MISSING")
    assert detail["errors"][0]["path"] == missing and hp.nothing_stored(env)


def _unreadable(env, form):
    response = env.client.post(hp.THEME_PUBLISH_ROUTE, headers={"Authorization": f"Bearer {hp.KEY}"},
                               files={"package": ("p.autotheme", hp.package(), "application/zip")}, data=form)
    hp.refused(response, 400, "PARTS_UNREADABLE")
    assert hp.nothing_stored(env)


def test_a_text_part_over_its_cap_is_400_in_the_endpoint_shape(env):
    _unreadable(env, {"signature": "{}", "review_summary": "x" * (SUMMARY_MAX_BYTES + 2)})


def test_too_many_parts_is_400_in_the_endpoint_shape(env):
    _unreadable(env, {"signature": "{}", "review_summary": "-", "extra": "-", "more": "-"})


def test_a_package_sent_as_a_text_field_is_a_missing_part(env):
    response = env.client.post(hp.THEME_PUBLISH_ROUTE, headers={"Authorization": f"Bearer {hp.KEY}"},
                               data={"package": "PK", "signature": "{}", "review_summary": "-"})
    assert hp.refused(response, 422, "PART_MISSING")["errors"][0]["path"] == "package"
