"""PRD-261 A-07a — ``POST /api/themes/import``: a workspace's private theme upload.

A workspace admin (``workspace:manage``) uploads a ``.autotheme`` into THIS workspace only:

1. the platform switch and the workspace setting must allow private uploads
   (``services/themes/records.py``), else 403 with a plain sentence;
2. the package is read and checked (``services/themes/package.py``: zip, manifest, engines,
   htmlClass, sizes, CSS scope, looks): 413 over the package cap, 400 with the codes;
3. an optional ``signature`` part (C-05d, a detached ``.sig.json``, ``services/themes/signed_upload.py``)
   is checked against the upload's own files before anything is stored: 413 over
   ``ENVELOPE_MAX_BYTES``, 400 ``SIGNATURE_BAD``/``SIGNATURE_MISMATCH`` (``SIGNATURE_BAD`` too for a
   key the hosted feed revoked, in the local edition), 503 when the keys file doesn't load; a
   verified one is stored on the package row. Without the part nothing changes;
4. an id under a reserved publisher (``automatos/…``) needs ``caller_is_admin`` or a verified
   signature, else 403;
5. the automated review stage runs inline (C-02a, ``services/themes/review_stage.py``; the LLM
   stage reads ``dist/``, the code that runs, since no isolated rebuild ever checks a private
   upload's ``source/``) unless these bytes are already here, and never for a version that would be
   refused (409, checked read-only first); the request's read transaction ends before the stage,
   so no connection waits idle in transaction while it runs; one upload of the same bytes runs at
   a time (``services/themes/upload_claim.py``: an overlapping one is 409, try again, so a resent
   request never pays for the stage twice); the files are stored under
   ``themes/_workspace/<ws>/<publisher>/<slug>/<version>/`` before the commit, so a failed
   write leaves no row (503). What it may spend (``services/themes/review_limits.py``): a review
   slot in this process is taken first, before the claim's connection (503 with ``Retry-After``
   when ``THEME_REVIEW_MAX_CONCURRENT`` stages already run); a workspace past
   ``THEME_UPLOADS_PER_DAY`` in 24 hours is 429; the LLM check gets what is left of
   ``THEME_REVIEW_DAILY_TOKENS``;
6. the package row (``owner_type`` 'workspace', ``reviewed`` false) and an install row are
   written; a newer version of an installed theme becomes the install's pending update.

201 with the package (its verdict and plain checks; the LLM check's score and findings are the
platform reviewer's alone, ``review_stage.member_view``) and the install; 200 with the existing ones when these exact bytes are
already in this workspace (never a misleading 409); 409 for a version clash, or while the same
bytes are still being uploaded.

Gates, in order: ``workspace:manage``, the workspace's "who may install" setting (A-07e's
``require_install_role``), the private-upload switch and setting (``require_private_uploads``; a
refused upload is never asked to reverify) and a fresh step-up (A-07c, ``STEP_UP_MAX_AGE_MINUTES``). B2-wire
registers the router in ``main.py`` with a body-size limit for ``THEME_IMPORT_ROUTE``
(``services/themes/package.py``); tests mount it on a test app. A plain ``def`` (a synchronous session
and S3 client, run in the threadpool).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from config import config
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.principal import resolve_user_pk
from core.auth.step_up import require_step_up
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from core.models.themes import ThemeInstallation, ThemePackage
from services.themes.package import ThemePackage as CheckedThemePackage
from services.themes.package import ThemePackageError, cap_bytes, read_theme_package
from services.themes.records import (
    SavedUpload, UploadConflict, active_installation, package_by_digest, private_upload_refusal,
    reserved_publisher_refusal, save_private_upload, upload_conflict,
)
from services.themes import review_limits
from services.themes.review_stage import member_view, run_review_stage
from services.themes.settings import require_install_role
from services.themes.signature_verify import SigningKeysError
from services.themes.signed_upload import (
    ENVELOPE_MAX_BYTES, SignatureRefused, VerifiedSignature, check_upload_signature,
)
from services.themes.storage import theme_prefix, write_theme_files
from services.themes.upload_claim import upload_claim

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/themes", tags=["Themes"])

THEME_IMPORT_PERMISSION = "workspace:manage"
MANAGE_GATE = require_workspace_permission(THEME_IMPORT_PERMISSION)
INSTALL_ROLE_GATE = require_install_role()
STEP_UP = require_step_up(config.STEP_UP_MAX_AGE_MINUTES)

IMPORT_REFUSED = "This theme can't be installed."
STORE_FAILED = "The theme's files couldn't be stored. Try again."
BUSY = "This theme is being uploaded already. Try again in a moment."
SIGNATURES_UNAVAILABLE = "Theme signatures can't be checked right now, so nothing was installed. Try again later."
STATUS_INSTALLED = "installed"
STATUS_UPDATE_AVAILABLE = "update_available"
STATUS_STORED = "stored"


def _read_upload(file: UploadFile) -> Tuple[CheckedThemePackage, int]:
    """The upload read up to one byte past the cap, then checked: 413 over the cap, 400 with the codes."""
    data = file.file.read(cap_bytes("package") + 1)
    try:
        return read_theme_package(data), len(data)
    except ThemePackageError as exc:
        status = 413 if exc.too_large else 400
        logger.info("[Themes] Upload refused (%d): %s", status, [error["code"] for error in exc.errors])
        raise HTTPException(status_code=status, detail={"message": IMPORT_REFUSED, "errors": exc.errors}) from exc


def _read_signature(
    db: Session, part: Optional[UploadFile], checked: CheckedThemePackage,
) -> Optional[VerifiedSignature]:
    """The upload's verified signature (C-05d), or None without the part; 413/400/503 otherwise."""
    if part is None:
        return None
    data = part.file.read(ENVELOPE_MAX_BYTES + 1)
    try:
        verified = check_upload_signature(db, checked, data)
    except SignatureRefused as exc:
        logger.info("[Themes] Upload signature refused (%d): %s", exc.status_code, exc.error["code"])
        raise HTTPException(status_code=exc.status_code, detail={"message": IMPORT_REFUSED, "errors": [exc.error]}) from exc
    except SigningKeysError as exc:
        logger.error("[Themes] The signing keys did not load, so an upload signature can't be checked: %s", exc)
        raise HTTPException(status_code=503, detail=SIGNATURES_UNAVAILABLE) from exc
    logger.info("[Themes] Upload of %s@%s carries a signature verified by %s",
                checked.manifest["id"], checked.manifest["version"], verified.key_id)
    return verified


def _store(prefix: str, checked: CheckedThemePackage) -> None:
    try:
        write_theme_files(prefix, checked.files)
    except Exception as exc:
        logger.exception("[Themes] Storing %d files under %s failed", len(checked.files), prefix)
        raise HTTPException(status_code=503, detail=STORE_FAILED) from exc


def _install_status(package: ThemePackage, install: Optional[ThemeInstallation]) -> str:
    if install is not None and install.package_id == package.id:
        return STATUS_INSTALLED
    if install is not None and install.pending_package_id == package.id:
        return STATUS_UPDATE_AVAILABLE
    return STATUS_STORED


def _install_item(install: Optional[ThemeInstallation]) -> Optional[Dict[str, Any]]:
    if install is None:
        return None
    pending = install.pending_package_id
    return {
        "id": str(install.id), "themeId": install.theme_id, "packageId": str(install.package_id),
        "pendingPackageId": str(pending) if pending else None, "isDefault": bool(install.is_default),
    }


def upload_item(saved: SavedUpload) -> Dict[str, Any]:
    """The upload as the Install theme dialog reads it (A-07d): the package, its scan and the install."""
    package = saved.package
    return {
        "package": {
            "id": str(package.id), "themeId": package.theme_id, "version": package.version,
            "name": (package.manifest or {}).get("name"), "digest": package.digest,
            "ownerType": package.owner_type, "reviewed": bool(package.reviewed),
            "scanVerdict": package.scan_verdict, "scanReport": member_view(package.scan_report),
        },
        "install": _install_item(saved.installation),
        "status": _install_status(package, saved.installation),
        "duplicate": not saved.created,
    }


def _refuse(status: int, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail=message)


def require_private_uploads(
    ctx: RequestContext = Depends(get_request_context_hybrid), db: Session = Depends(get_db),
) -> None:
    """403 with the plain reason when uploads are off for this caller.

    Listed before the step-up, so a caller who can't upload anyway is never asked to reverify.
    """
    refusal = private_upload_refusal(db, ctx)
    if refusal is not None:
        raise _refuse(403, refusal)


def _after_integrity_error(db: Session, ctx: RequestContext, checked: CheckedThemePackage, exc: IntegrityError) -> SavedUpload:
    """A concurrent upload won the unique index: its package when it holds these bytes, else 409.

    The 409 names the clash the winner left (same version, other files; or not newer) when there is
    one, and asks to try again only when a retry could succeed.
    """
    db.rollback()
    existing = package_by_digest(db, ctx.workspace_id, checked.digest)
    if existing is None:
        logger.warning("[Themes] Concurrent upload of %s in workspace %s", checked.manifest["id"], ctx.workspace_id)
        raise _refuse(409, upload_conflict(db, ctx.workspace_id, checked.manifest) or BUSY) from exc
    return SavedUpload(existing, active_installation(db, ctx.workspace_id, existing.theme_id), created=False)


def _review(db: Session, ctx: RequestContext, checked: CheckedThemePackage, package_bytes: int,
            tokens: review_limits.TokenHold) -> Dict[str, Any]:
    """The automated stage's report (C-02a), skipped when these bytes are already in this workspace.

    Their existing row is reused as it is (``save_private_upload``), so a second run would only
    spend an LLM call; a version ``save_private_upload`` would refuse is 409 before the stage, and
    a workspace past ``THEME_UPLOADS_PER_DAY`` is 429. The stage spends from ``tokens``, the
    request's hold on what is left of the day's review tokens.
    The read transaction ends before the stage, so the session holds no connection while it runs
    (``save_private_upload`` checks again under its row lock). Called under the upload claim, so
    an overlapping upload of the same bytes never runs it too. The route is a plain ``def``: this
    runs in a threadpool thread, with no loop.
    """
    existing = package_by_digest(db, ctx.workspace_id, checked.digest)
    if existing is not None:
        return dict(existing.scan_report or {"verdict": existing.scan_verdict})
    conflict = upload_conflict(db, ctx.workspace_id, checked.manifest)
    if conflict is not None:
        db.rollback()
        logger.info("[Themes] Upload of %s in workspace %s refused before the review stage: version clash",
                    checked.manifest["id"], ctx.workspace_id)
        raise _refuse(409, conflict)
    quota = review_limits.upload_quota_refusal(db, ctx.workspace_id)
    if quota is not None:
        db.rollback()
        raise _refuse(429, quota)
    db.rollback()
    return run_review_stage(checked, package_bytes, include_source=False, token_budget=tokens)


def _save(db: Session, ctx: RequestContext, checked: CheckedThemePackage, package_bytes: int,
          signature: Optional[str]) -> SavedUpload:
    """Review and save in a review slot, taken before the claim holds a connection: 503 with
    ``Retry-After`` while ``THEME_REVIEW_MAX_CONCURRENT`` stages run here. ``signature`` is the
    upload's verified one (C-05d), or None."""
    with review_limits.review_slot() as slot:
        if not slot:
            logger.info("[Themes] Upload of %s in workspace %s refused: every review slot is busy",
                        checked.manifest["id"], ctx.workspace_id)
            db.rollback()
            raise HTTPException(status_code=503, detail=review_limits.REVIEWS_BUSY, headers=review_limits.busy_headers())
        return _save_in_slot(db, ctx, checked, package_bytes, signature)


def _save_in_slot(db: Session, ctx: RequestContext, checked: CheckedThemePackage, package_bytes: int,
                  signature: Optional[str]) -> SavedUpload:
    """Review and save under the (workspace, digest) claim: 409 while another upload of these bytes runs."""
    with upload_claim(db, ctx.workspace_id, checked.digest) as claimed:
        if not claimed:
            logger.info("[Themes] Upload of %s in workspace %s refused: the same bytes are being uploaded",
                        checked.manifest["id"], ctx.workspace_id)
            raise _refuse(409, BUSY)
        return _save_claimed(db, ctx, checked, package_bytes, signature)


def _save_claimed(db: Session, ctx: RequestContext, checked: CheckedThemePackage, package_bytes: int,
                  signature: Optional[str]) -> SavedUpload:
    """Review, then store, under one token hold (``review_limits.token_hold``): what the stage
    spent stays held in this process until its row is committed, when ``tokens_used_today`` sees it."""
    with review_limits.token_hold(lambda: review_limits.tokens_left_today(db)) as tokens:
        return _review_and_store(db, ctx, checked, package_bytes, tokens, signature)


def _review_and_store(db: Session, ctx: RequestContext, checked: CheckedThemePackage, package_bytes: int,
                      tokens: review_limits.TokenHold, signature: Optional[str]) -> SavedUpload:
    manifest = checked.manifest
    prefix = theme_prefix(manifest["id"], manifest["version"], ctx.workspace_id)
    report = _review(db, ctx, checked, package_bytes, tokens)
    try:
        return save_private_upload(
            db, ctx.workspace_id, resolve_user_pk(db, ctx), checked, prefix, report,
            store=lambda: _store(prefix, checked), signature=signature,
        )
    except UploadConflict as exc:
        db.rollback()
        raise _refuse(409, str(exc)) from exc
    except IntegrityError as exc:
        return _after_integrity_error(db, ctx, checked, exc)


@router.post(
    "/import", status_code=201,
    dependencies=[
        Depends(MANAGE_GATE), Depends(INSTALL_ROLE_GATE), Depends(require_private_uploads), Depends(STEP_UP),
    ],
)
def import_theme(
    response: Response,
    file: UploadFile = File(...),
    signature: Optional[UploadFile] = File(None),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Upload a .autotheme (and, optionally, its .sig.json) into this workspace only; returns the
    package, its scan and the install."""
    checked, package_bytes = _read_upload(file)
    verified = _read_signature(db, signature, checked)
    refusal = reserved_publisher_refusal(checked.manifest["id"], ctx.user, signed=verified is not None)
    if refusal is not None:
        raise _refuse(403, refusal)
    saved = _save(db, ctx, checked, package_bytes, None if verified is None else verified.signature)
    if not saved.created:
        response.status_code = 200
    package = saved.package
    logger.info("[Themes] Upload of %s@%s in workspace %s: %s", package.theme_id, package.version, ctx.workspace_id,
                _install_status(package, saved.installation))
    return upload_item(saved)
