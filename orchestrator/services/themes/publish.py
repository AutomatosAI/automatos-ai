"""PRD-261 — publish a theme version with the publish key: list it, approve it and store its signature.

``POST /api/marketplace/themes/publish`` (``api/marketplace_themes_publish.py``) hands this module a
read package that ships ``source/``, the kit's signature envelope and the kit review's summary,
with the key's settings (``publish_key.py``). The kit side is ``theme-publish.yml`` in the Design
Kit: the merge of a version bump is the only approval (D2), and every allowlisted publisher is
signed (D4). Order, and nothing is written until every check has passed (``publish_checks.py``):

1. the publisher is allowlisted, the service user exists, and the publisher is the service user's
   or unclaimed: 403 (503 without the user). A reserved publisher (``automatos``) takes it as an
   admin's submission would, through ``publish_submitter`` only;
2. the envelope verifies under an active key and names this id, version and digest: 422;
3. the summary names this digest with ``Outcome: pass`` (the hosted approval note's rule), and no
   live ``disable`` covers the version: 422;
4. these exact bytes approved with a signature an active key verifies: 200 ``unchanged``, nothing
   written. Otherwise, in a review slot (503 when every slot is busy) and under the Marketplace's
   claim on the digest, the submit route's (409 while another submission of these bytes runs):
5. bytes already held go on from their row: a pending version with its stored verdict, an approved
   one whose signature no longer verifies is only signed, a rejected or deactivated one is 409. New
   bytes: the version must be above the highest held (409), within ``THEME_PUBLISH_PER_DAY`` (429),
   then core's review stage runs inline on its own token hold, and its verdict must be ``pass`` or
   ``review_required`` (D3): 422 otherwise;
6. one transaction: the submission (``save_submission``, Q9 off for this user), the summary as the
   version's report note (``review.add_report_note``), ``review.approve``, ``review.sign`` with the
   envelope's signature for the approval's request, the files, then the one commit. A failure at any
   step rolls all of it back, so a refused publish leaves no pending, approved or half-signed row.

The service user is a platform admin nowhere: its admin's say over a reserved publisher exists only
in the ``Submitter`` this module builds, after the allowlist passed, and no route but this one
accepts the key. Synchronous (a plain session): the route is a plain ``def``.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Optional
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from core.models.themes import ThemePackage
from services.themes import publish_checks as checks
from services.themes import review, review_limits, signature_verify, signing
from services.themes.choices import ThemeInstallError
from services.themes.package import ThemePackage as CheckedThemePackage
from services.themes.publish_key import PublishSettings
from services.themes.records import publisher_of
from services.themes.signature_state import feed_checked
from services.themes.signed_upload import VerifiedSignature
from services.themes.storage import theme_prefix, write_theme_files
from services.themes.submissions import (
    PublisherReserved, ReviewStage, SubmissionConflict, Submitter, save_submission, submission_by_digest,
)
from services.themes.upload_claim import upload_claim

logger = logging.getLogger(__name__)

THEME_PUBLISH_ROUTE = "/api/marketplace/themes/publish"
CLAIM_SCOPE = "marketplace"  # the submit route's scope: a publish and a submission of the same bytes never overlap
STATUS_PUBLISHED, STATUS_UNCHANGED = "published", "unchanged"


@dataclass(frozen=True)
class PublishUpload:
    """What the route read: the checked package (it ships ``source/``), its size, the envelope and the summary."""

    checked: CheckedThemePackage
    package_bytes: int
    envelope: bytes
    summary: str


@dataclass(frozen=True)
class _Checked:
    """An upload every read-only check passed: its verified signature and its trimmed summary."""

    upload: PublishUpload
    signature: VerifiedSignature
    summary: str


def publish_submitter(settings: PublishSettings) -> Submitter:
    """The service user as a submitter: an admin's say under the allowlisted publishers (checked before
    it is used), and no Q9 (it has its own daily quota)."""
    return Submitter(user_pk=settings.actor_pk, is_admin=True, one_open_rule=False)


def _answer(status: str, package: ThemePackage) -> Dict[str, Any]:
    return {"status": status, "id": package.theme_id, "version": package.version, "digest": package.digest}


def _checked(db: Session, upload: PublishUpload, settings: PublishSettings) -> _Checked:
    """Steps 1 to 3: every check that needs no claim, in the contract's order."""
    checked = upload.checked
    publisher = publisher_of(checked.manifest["id"])
    checks.require_listed(publisher, settings)
    checks.require_actor(db, settings.actor_pk)
    checks.require_owner(db, publisher, publish_submitter(settings))
    signature = checks.verified_signature(db, checked, upload.envelope)
    summary = checks.checked_summary(upload.summary, checked.digest)
    checks.require_not_disabled(db, checked)
    return _Checked(upload=upload, signature=signature, summary=summary)


def _is_published(db: Session, package: ThemePackage) -> bool:
    """True when these bytes are approved with a signature an active key the hosted feed didn't revoke verifies."""
    if package.status != checks.STATUS_APPROVED or not package.signature:
        return False
    try:
        verdict = signature_verify.package_verdict(package)
    except signature_verify.SigningKeysError as broken:
        raise checks.signing_keys_refusal(broken) from broken
    return feed_checked(db, verdict).verified


@contextmanager
def _all_or_nothing(db: Session) -> Iterator[None]:
    """One transaction: committed when the block ends, rolled back (and the failure re-raised) when any step,
    or the commit itself, fails."""
    try:
        yield
        db.commit()
    except BaseException:
        db.rollback()
        raise


def _save(db: Session, request: _Checked, settings: PublishSettings,
          report: Optional[Dict[str, Any]]) -> ThemePackage:
    """The pending submission's rows, flushed and not committed (``store`` None: the files come last)."""
    checked = request.upload.checked
    prefix = theme_prefix(checked.manifest["id"], checked.manifest["version"])
    try:
        return save_submission(db, checked, publish_submitter(settings), prefix, report, store=None).package
    except (SubmissionConflict, PublisherReserved) as clash:
        raise checks.submission_refusal(clash) from clash
    except IntegrityError as clash:
        logger.warning("[Themes] Concurrent submission of %s while publishing it", checked.manifest["id"])
        raise checks.busy() from clash


def _request_id(db: Session, package: ThemePackage, actor_pk: int) -> UUID:
    """The version's open signature request (approve records one); a new one for an approved row without any."""
    request = signing.open_request(db, package.id)
    if request is not None:
        return request.id
    return UUID(signing.request_signature(db, package, actor_pk)["requestId"])


def _approve_and_sign(db: Session, package: ThemePackage, request: _Checked, actor_pk: int) -> None:
    """A pending version: the summary as its report note, then approve. Then store the envelope's signature.

    Each step is flushed before the next reads it (the app's sessions don't autoflush).
    """
    try:
        if package.status == checks.STATUS_PENDING:
            review.add_report_note(db, package.id, request.summary, review.NOTE_PASS, actor_pk)
            db.flush()
            review.approve(db, package.id, actor_pk)
            db.flush()
        request_id = _request_id(db, package, actor_pk)
        db.flush()
        review.sign(db, package.id, request_id, request.signature.signature, actor_pk)
        db.flush()
    except ThemeInstallError as refusal:
        raise checks.review_refusal(refusal) from refusal
    except signature_verify.SigningKeysError as broken:
        raise checks.signing_keys_refusal(broken) from broken


def _store(prefix: str, checked: CheckedThemePackage) -> None:
    try:
        write_theme_files(prefix, checked.files)
    except Exception as exc:
        logger.exception("[Themes] Storing %d published files under %s failed", len(checked.files), prefix)
        raise checks.store_failed() from exc


def _write(db: Session, request: _Checked, settings: PublishSettings, existing: Optional[ThemePackage],
           report: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Step 6, all or nothing: the rows, the note, approve, sign, then the files, then the one commit."""
    with _all_or_nothing(db):
        package = existing if existing is not None else _save(db, request, settings, report)
        _approve_and_sign(db, package, request, settings.actor_pk)
        if existing is None:
            _store(package.storage_prefix, request.upload.checked)
        answer = _answer(STATUS_PUBLISHED, package)
    logger.info("[Themes] Published %s@%s (%s) with the publish key", answer["id"], answer["version"],
                answer["digest"])
    return answer


def _publish_new(db: Session, request: _Checked, settings: PublishSettings, stage: ReviewStage) -> Dict[str, Any]:
    """New bytes: version and quota, core's review stage on its own token hold (kept until the commit), then write."""
    upload = request.upload
    checks.require_new_version(db, upload.checked, publish_submitter(settings))
    checks.require_quota(db, settings.actor_pk)
    with review_limits.token_hold(lambda: review_limits.tokens_left_today(db)) as tokens:
        db.rollback()  # the read transaction ends: no connection waits idle in it while the stage runs
        report = stage(upload.checked, upload.package_bytes, token_budget=tokens)
        checks.require_verdict(report.get("verdict"))
        return _write(db, request, settings, None, report)


def _publish_claimed(db: Session, request: _Checked, settings: PublishSettings, stage: ReviewStage) -> Dict[str, Any]:
    """Under the claim: bytes held already go on from their row, new bytes are reviewed first.

    The read starts afresh (``rollback`` ends the first read and expires its rows), so a row another
    publish of these bytes committed before this one took the claim is seen as it is now.
    """
    db.rollback()
    existing = submission_by_digest(db, request.upload.checked.digest)
    if existing is None:
        return _publish_new(db, request, settings, stage)
    if _is_published(db, existing):
        return _answer(STATUS_UNCHANGED, existing)
    checks.require_open(existing)
    return _write(db, request, settings, existing, None)


def publish_version(db: Session, upload: PublishUpload, settings: PublishSettings,
                    stage: ReviewStage) -> Dict[str, Any]:
    """Publish one version: ``{status: published|unchanged, id, version, digest}``, or :class:`PublishRefused`."""
    request = _checked(db, upload, settings)
    digest = upload.checked.digest
    existing = submission_by_digest(db, digest)
    if existing is not None and _is_published(db, existing):
        return _answer(STATUS_UNCHANGED, existing)
    with review_limits.review_slot() as slot:
        if not slot:
            raise checks.reviews_busy()
        with upload_claim(db, CLAIM_SCOPE, digest) as claimed:
            if not claimed:
                raise checks.busy()
            return _publish_claimed(db, request, settings, stage)


__all__ = ["STATUS_PUBLISHED", "STATUS_UNCHANGED", "THEME_PUBLISH_ROUTE", "PublishUpload", "publish_submitter",
           "publish_version"]
