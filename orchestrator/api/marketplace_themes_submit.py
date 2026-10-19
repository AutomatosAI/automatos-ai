"""PRD-261 C-01a — ``POST /api/marketplace/themes/submit``: a theme submitted to the Marketplace.

A workspace member submits a ``.autotheme`` for review:

1. the Marketplace switch ``THEMES_MARKETPLACE`` must be on, else 403 with a plain sentence;
2. the package is read and checked as a private upload is (``services/themes/package.py``:
   zip, manifest, engines, htmlClass, sizes, CSS scope, looks): 413 over the cap, 400 with codes;
3. in SaaS the caller must resolve to a person (a publisher has an owner), else 403;
4. publisher ownership, Q9 and the version rules (``services/themes/submissions.py``), read
   first so the stage never runs for a refused package, and again under a row lock at save: a
   reserved publisher (``automatos/…``) without ``caller_is_admin`` → 403; another author's
   publisher, a version clash or (Q9 on) a second open submission → 409;
5. the automated review stage runs inline, before the response (``submission_review_stage``),
   unless these exact bytes were submitted already; one submission of the same bytes runs at a
   time (``services/themes/upload_claim.py``, scope ``marketplace``: an overlapping one is 409,
   try again, so a resent request never pays for the stage twice); the request's read
   transaction ends before the stage, so no connection waits idle in transaction while it runs.
   The LLM check reads ``source/`` only on the hosted edition, where the isolated job rebuilds
   ``dist/`` from it; the local edition never has that job, so it reads ``dist/``;
6. the files are stored under ``themes/<publisher>/<slug>/<version>/`` before the commit, so a
   failed write leaves no row (503); the row is ``owner_type`` 'marketplace', ``status``
   'pending', ``reviewed`` false, with the stage's report and verdict, and its review events.

201 with the submission; 200 with the existing one when these exact bytes were submitted
already. Nothing is published: a person reviews every version (C-03a, D3).

Gates, in order: ``members:read``, the Marketplace switch, then a fresh step-up (A-07c,
``STEP_UP_MAX_AGE_MINUTES``), so a caller refused anyway is never asked to reverify. C1-wire
registers the router in ``main.py`` with a body-size limit for ``THEME_SUBMIT_ROUTE``; tests
mount it on a test app. A plain ``def`` (a synchronous session and S3 client, in the threadpool).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from config import config
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.principal import resolve_user_pk
from core.auth.roles import caller_is_admin
from core.auth.step_up import require_step_up
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from services.themes.package import ThemePackage as CheckedThemePackage
from services.themes.package import ThemePackageError, cap_bytes, read_theme_package
from services.themes.review_stage import run_review_stage
from services.themes.storage import theme_prefix, write_theme_files
from services.themes.upload_claim import upload_claim
from services.themes.submissions import (
    MARKETPLACE_OFF, NO_PERSON, SWITCH_OFF, THEME_SUBMIT_ROUTE, PublisherReserved, ReviewStage, SavedSubmission,
    SubmissionConflict, Submitter, check_submission, marketplace_switch, save_submission, submission_by_digest,
    submission_conflict,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/marketplace/themes", tags=["Theme Marketplace"])

SUBMIT_PERMISSION = "members:read"
MEMBER_GATE = require_workspace_permission(SUBMIT_PERMISSION)
STEP_UP = require_step_up(config.STEP_UP_MAX_AGE_MINUTES)
SUBMIT_PATH = THEME_SUBMIT_ROUTE[len(router.prefix):]

SUBMIT_REFUSED = "This theme can't be submitted."
STORE_FAILED = "The theme's files couldn't be stored. Try again."
BUSY = "This theme is being submitted already. Try again in a moment."
LOCAL_EDITION = "local"
CLAIM_SCOPE = "marketplace"


def require_marketplace() -> None:
    """403 with a plain sentence while ``THEMES_MARKETPLACE`` is off."""
    if marketplace_switch() == SWITCH_OFF:
        raise HTTPException(status_code=403, detail=MARKETPLACE_OFF)


def marketplace_review(checked: CheckedThemePackage, package_bytes: int) -> Dict[str, Any]:
    """C-02a's stage on a submission. On the hosted edition the LLM check reads ``source/`` when the
    package ships it, because the isolated job (C-02b) rebuilds ``dist/`` from it and approve needs
    its passing note. The local edition never has that job (approve skips the note there), so it
    reads ``dist/``: the code the browser runs."""
    return run_review_stage(checked, package_bytes, prefer_source=not config.IS_LOCAL_EDITION)


def submission_review_stage() -> ReviewStage:
    """The automated review stage run inline at submit (C-02a, wired at C1-wire).

    Its verdict is never ``not_run``: with no LLM key it is ``review_required`` ("LLM stage did
    not run"), which a person may approve (C-03a). Tests override this dependency to give a verdict.
    """
    return marketplace_review


def _read_upload(file: UploadFile) -> Tuple[CheckedThemePackage, int]:
    """The upload read up to one byte past the cap, then checked: 413 over the cap, 400 with the codes."""
    data = file.file.read(cap_bytes("package") + 1)
    try:
        return read_theme_package(data), len(data)
    except ThemePackageError as exc:
        status = 413 if exc.too_large else 400
        logger.info("[Themes] Submission refused (%d): %s", status, [error["code"] for error in exc.errors])
        raise HTTPException(status_code=status, detail={"message": SUBMIT_REFUSED, "errors": exc.errors}) from exc


def _submitter(db: Session, ctx: RequestContext) -> Submitter:
    """The caller as a submitter; in SaaS one with no ``users`` row is refused (403)."""
    user_pk = resolve_user_pk(db, ctx)
    local = str(getattr(config, "AUTH_EDITION", "") or "").strip().lower() == LOCAL_EDITION
    if user_pk is None and not local:
        raise HTTPException(status_code=403, detail=NO_PERSON)
    return Submitter(user_pk=user_pk, is_admin=caller_is_admin(getattr(ctx, "user", None)))


def _store(prefix: str, checked: CheckedThemePackage) -> None:
    try:
        write_theme_files(prefix, checked.files)
    except Exception as exc:
        logger.exception("[Themes] Storing %d submitted files under %s failed", len(checked.files), prefix)
        raise HTTPException(status_code=503, detail=STORE_FAILED) from exc


def _after_integrity_error(db: Session, checked: CheckedThemePackage, submitter: Submitter,
                           exc: IntegrityError) -> SavedSubmission:
    """A concurrent submission won a unique index: its row when it holds these bytes and the
    caller may submit under its publisher, else 409 naming the clash it left."""
    db.rollback()
    message = submission_conflict(db, checked.manifest, checked.digest, submitter)
    existing = submission_by_digest(db, checked.digest)
    if message is None and existing is not None:
        return SavedSubmission(existing, created=False)
    logger.warning("[Themes] Concurrent submission of %s", checked.manifest["id"])
    db.rollback()
    raise HTTPException(status_code=409, detail=message or BUSY) from exc


def _refusal(db: Session, exc: Exception) -> HTTPException:
    """A refused submission: its transaction rolled back, 403 for a reserved publisher, else 409."""
    db.rollback()
    status = 403 if isinstance(exc, PublisherReserved) else 409
    return HTTPException(status_code=status, detail=str(exc))


def _report(db: Session, checked: CheckedThemePackage, stage: ReviewStage, package_bytes: int) -> Dict[str, Any]:
    """The stage's report; the existing one when these bytes were submitted already (reused as it is).

    The read transaction ends before the stage, so the session holds no connection while it runs;
    ``save_submission`` checks everything again under its row lock.
    """
    existing = submission_by_digest(db, checked.digest)
    if existing is not None:
        return dict(existing.scan_report or {"verdict": existing.scan_verdict})
    db.rollback()
    return stage(checked, package_bytes)


def _save_claimed(db: Session, checked: CheckedThemePackage, submitter: Submitter, stage: ReviewStage,
                  package_bytes: int) -> SavedSubmission:
    """Refuse early (before the review stage spends anything), run the stage inline, then store."""
    manifest = checked.manifest
    prefix = theme_prefix(manifest["id"], manifest["version"])
    try:
        check_submission(db, manifest, checked.digest, submitter)
        report = _report(db, checked, stage, package_bytes)
        return save_submission(db, checked, submitter, prefix, report, store=lambda: _store(prefix, checked))
    except (PublisherReserved, SubmissionConflict) as exc:
        raise _refusal(db, exc) from exc
    except IntegrityError as exc:
        return _after_integrity_error(db, checked, submitter, exc)


def _save(db: Session, checked: CheckedThemePackage, submitter: Submitter, stage: ReviewStage,
          package_bytes: int) -> SavedSubmission:
    """Review and save under the (marketplace, digest) claim: 409 while another submission of these bytes runs."""
    with upload_claim(db, CLAIM_SCOPE, checked.digest) as claimed:
        if not claimed:
            logger.info("[Themes] Submission of %s refused: the same bytes are being submitted", checked.manifest["id"])
            db.rollback()
            raise HTTPException(status_code=409, detail=BUSY)
        return _save_claimed(db, checked, submitter, stage, package_bytes)


def submission_item(saved: SavedSubmission) -> Dict[str, Any]:
    """The submission as the Publish theme form reads it (C-01b)."""
    package = saved.package
    return {
        "package": {
            "id": str(package.id), "themeId": package.theme_id, "version": package.version,
            "name": (package.manifest or {}).get("name"), "digest": package.digest,
            "ownerType": package.owner_type, "status": package.status, "reviewed": bool(package.reviewed),
            "scanVerdict": package.scan_verdict, "scanReport": package.scan_report,
        },
        "duplicate": not saved.created,
    }


@router.post(
    SUBMIT_PATH, status_code=201,
    dependencies=[Depends(MEMBER_GATE), Depends(require_marketplace), Depends(STEP_UP)],
)
def submit_theme(
    response: Response,
    file: UploadFile = File(...),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
    stage: ReviewStage = Depends(submission_review_stage),
) -> Dict[str, Any]:
    """Submit a .autotheme for Marketplace review; returns the pending submission and its stage verdict."""
    checked, package_bytes = _read_upload(file)
    submitter = _submitter(db, ctx)
    saved = _save(db, checked, submitter, stage, package_bytes)
    if not saved.created:
        response.status_code = 200
    package = saved.package
    logger.info("[Themes] Submission of %s@%s: %s, verdict %s", package.theme_id, package.version,
                "new" if saved.created else "duplicate", package.scan_verdict)
    return submission_item(saved)
