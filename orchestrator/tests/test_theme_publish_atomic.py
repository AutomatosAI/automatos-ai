"""PRD-261 — a theme publish is all or nothing, and its own limits.

Pins (``services/themes/publish.py``):

* a failure after the submission's rows are flushed (approve refuses, sign refuses, an unexpected
  error, the files can't be stored) rolls everything back: no pending, approved or half-signed row,
  no event, no publisher claimed, no file;
* bytes already held go on from their row: a pending submission (a person submitted it) is noted,
  approved and signed; an approved one without a signature is only signed; a rejected one is 409;
* Q9 never applies to the service user, which has its own quota (``THEME_PUBLISH_PER_DAY``, 429);
* another publish or submission of the same bytes in flight is 409 ``PUBLISH_BUSY``; every review
  slot busy is 503 with ``Retry-After``.
"""
from __future__ import annotations

from uuid import uuid4

import pytest

from config import config
from core.models.themes import ThemePackage, ThemeReviewEvent
from services.themes import review, review_limits
from services.themes.choices import CONFLICT
from services.themes.package import read_theme_package
from services.themes.publish import CLAIM_SCOPE
from services.themes.review import ThemeReviewError
from services.themes.signing import SigningError
from services.themes.upload_claim import upload_claim
from tests import helpers_theme_publish as hp


@pytest.fixture
def env(monkeypatch, tmp_path):
    yield from hp.publish_env(monkeypatch, tmp_path)


def _held(data: bytes, status: str, submitted_by=hp.PERSON_PK, verdict="pass") -> ThemePackage:
    """A Marketplace row for these bytes, as the submit route and the review queue leave one (its id set
    here: the row is detached once ``hp.add`` commits it)."""
    checked = read_theme_package(data)
    manifest = checked.manifest
    return ThemePackage(
        id=uuid4(), theme_id=manifest["id"], version=manifest["version"], owner_type="marketplace", workspace_id=None,
        digest=checked.digest, manifest=manifest, file_manifest=checked.file_manifest,
        storage_prefix=f"themes/{manifest['id']}/{manifest['version']}/", status=status,
        reviewed=status == "approved", scan_report={"verdict": verdict}, scan_verdict=verdict,
        submitted_by=submitted_by,
    )


def _events(env, package_id):
    return sorted(row.event for row in hp.rows(env, ThemeReviewEvent) if row.package_id == package_id)


# --- all or nothing --------------------------------------------------------------------------


def _refuse_approve(*args, **kwargs):
    raise ThemeReviewError(CONFLICT, "approve refused")


def _refuse_sign(*args, **kwargs):
    raise SigningError(CONFLICT, "sign refused")


@pytest.mark.parametrize("step, failure", [("approve", _refuse_approve), ("sign", _refuse_sign)])
def test_a_refusal_after_the_save_rolls_everything_back(env, monkeypatch, step, failure):
    monkeypatch.setattr(review, step, failure)
    detail = hp.refused(hp.post(env.client, hp.package()), 422, "REVIEW_REFUSED")
    assert detail["message"] == f"{step} refused"
    assert hp.nothing_stored(env) and len(env.stage.calls) == 1


def test_an_unexpected_error_while_signing_rolls_everything_back(env, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("the database went away")

    monkeypatch.setattr(review, "sign", broken)
    response = hp.post(hp.client(env, raise_server_exceptions=False), hp.package())
    assert response.status_code == 500
    assert hp.nothing_stored(env)


def test_a_store_failure_is_503_and_leaves_no_row(env):
    env.s3.fail = True
    hp.refused(hp.post(env.client, hp.package()), 503, "STORE_FAILED")
    assert hp.nothing_stored(env)


def test_a_publish_after_a_refused_one_succeeds_from_scratch(env, monkeypatch):
    data = hp.package()
    real_sign = review.sign
    monkeypatch.setattr(review, "sign", _refuse_sign)
    hp.refused(hp.post(env.client, data), 422, "REVIEW_REFUSED")
    monkeypatch.setattr(review, "sign", real_sign)
    assert hp.post(env.client, data).json()["status"] == "published"
    (row,) = hp.rows(env, ThemePackage)
    assert row.status == "approved" and row.signature


# --- bytes already held ----------------------------------------------------------------------


def test_a_pending_submission_of_the_same_bytes_is_approved_and_signed(env):
    data = hp.package(hp.OFFICIAL_ID)
    held = _held(data, "pending")
    held_id = held.id
    hp.add(env, held)
    summary = hp.summary(hp.digest_of(data), theme_id=hp.OFFICIAL_ID)

    assert hp.post(env.client, data, review_summary=summary).json()["status"] == "published"

    (row,) = hp.rows(env, ThemePackage)
    assert (row.id, row.status, row.submitted_by) == (held_id, "approved", hp.PERSON_PK) and row.signature
    assert _events(env, row.id) == ["approved", "report_note", "signature_requested", "signed"]
    assert env.stage.calls == []  # its stored verdict stands; the stage never runs twice on the same bytes


def test_a_pending_submission_core_blocked_stays_refused(env):
    data = hp.package(hp.OFFICIAL_ID)
    hp.add(env, _held(data, "pending", verdict="blocked"))
    summary = hp.summary(hp.digest_of(data), theme_id=hp.OFFICIAL_ID)
    hp.refused(hp.post(env.client, data, review_summary=summary), 422, "REVIEW_BLOCKED")
    (row,) = hp.rows(env, ThemePackage)
    assert row.status == "pending" and _events(env, row.id) == []


def test_an_approved_version_without_a_signature_is_only_signed(env):
    data = hp.package(hp.OFFICIAL_ID)
    hp.add(env, _held(data, "approved"))
    summary = hp.summary(hp.digest_of(data), theme_id=hp.OFFICIAL_ID)

    assert hp.post(env.client, data, review_summary=summary).json()["status"] == "published"

    (row,) = hp.rows(env, ThemePackage)
    assert row.status == "approved" and row.signature
    assert _events(env, row.id) == ["signature_requested", "signed"]
    assert hp.post(env.client, data, review_summary=summary).json()["status"] == "unchanged"


def test_a_rejected_version_with_the_same_bytes_is_409(env):
    data = hp.package(hp.OFFICIAL_ID)
    hp.add(env, _held(data, "rejected"))
    summary = hp.summary(hp.digest_of(data), theme_id=hp.OFFICIAL_ID)
    hp.refused(hp.post(env.client, data, review_summary=summary), 409, "VERSION_CLOSED")


# --- Q9 and the quota ------------------------------------------------------------------------


def test_q9_never_applies_to_the_service_user(env):
    assert config.THEMES_ONE_OPEN_SUBMISSION is True
    hp.add(env, _held(hp.package("acme-e2e/other"), "pending", submitted_by=hp.ACTOR_PK))
    assert hp.post(env.client, hp.package()).json()["status"] == "published"


def test_the_service_user_has_its_own_daily_quota(env, monkeypatch):
    monkeypatch.setattr(config, "THEME_PUBLISH_PER_DAY", 1)
    monkeypatch.setattr(config, "THEME_SUBMISSIONS_PER_DAY", 1)
    assert hp.post(env.client, hp.package()).json()["status"] == "published"
    hp.refused(hp.post(env.client, hp.package(version="1.1.0")), 429, "PUBLISH_QUOTA")
    monkeypatch.setattr(config, "THEME_PUBLISH_PER_DAY", 2)
    assert hp.post(env.client, hp.package(version="1.1.0")).json()["status"] == "published"


# --- busy ------------------------------------------------------------------------------------


def test_the_same_bytes_in_flight_are_409(env):
    data = hp.package()
    session = env.sessions()
    try:
        with upload_claim(session, CLAIM_SCOPE, hp.digest_of(data)) as claimed:
            assert claimed
            hp.refused(hp.post(env.client, data), 409, "PUBLISH_BUSY")
    finally:
        session.close()
    assert hp.nothing_stored(env)


def test_every_review_slot_busy_is_503_with_retry_after(env, monkeypatch):
    monkeypatch.setattr(config, "THEME_REVIEW_MAX_CONCURRENT", 1)
    with review_limits.review_slot() as slot:
        assert slot
        response = hp.post(env.client, hp.package())
    hp.refused(response, 503, "REVIEWS_BUSY")
    assert response.headers["retry-after"] == str(review_limits.BUSY_RETRY_SECONDS)
    assert hp.nothing_stored(env)
