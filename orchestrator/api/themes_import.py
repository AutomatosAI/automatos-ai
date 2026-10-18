"""PRD-261 A-07a — ``POST /api/themes/import``: a workspace's private theme upload.

A workspace admin (``workspace:manage``) uploads a ``.autotheme`` into THIS workspace only:

1. the platform switch and the workspace setting must allow private uploads
   (``services/themes/records.py``), else 403 with a plain sentence;
2. the package is read and checked (``services/themes/package.py``: zip, manifest, engines,
   htmlClass, sizes, CSS scope, looks): 413 over the package cap, 400 with the codes;
3. an id under a reserved publisher (``automatos/…``) needs ``caller_is_admin``, else 403;
4. the scan report is built (``services/themes/scan.py``); the files are stored under
   ``themes/_workspace/<ws>/<publisher>/<slug>/<version>/`` before the commit, so a failed
   write leaves no row (503);
5. the package row (``owner_type`` 'workspace', ``reviewed`` false) and an install row are
   written; a newer version of an installed theme becomes the install's pending update.

201 with the package and the install; 200 with the existing ones when these exact bytes are
already in this workspace (never a misleading 409); 409 for a version clash.

The router is registered by B2-wire (``main.py``, the route manifest, the body-size limit and
the step-up dependency); tests mount it on a test app. A plain ``def`` (a synchronous session
and S3 client, run in the threadpool).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from fastapi import APIRouter, Depends, File, HTTPException, Response, UploadFile
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.principal import resolve_user_pk
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from core.models.themes import ThemeInstallation, ThemePackage
from services.themes.package import ThemePackage as CheckedThemePackage
from services.themes.package import ThemePackageError, cap_bytes, read_theme_package
from services.themes.records import (
    SavedUpload, UploadConflict, active_installation, conflict_after_race, package_by_digest, private_upload_refusal,
    reserved_publisher_refusal, save_private_upload,
)
from services.themes.scan import build_scan_report
from services.themes.storage import theme_prefix, write_theme_files

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/themes", tags=["Themes"])

THEME_IMPORT_PERMISSION = "workspace:manage"
MANAGE_GATE = require_workspace_permission(THEME_IMPORT_PERMISSION)

IMPORT_REFUSED = "This theme can't be installed."
STORE_FAILED = "The theme's files couldn't be stored. Try again."
BUSY = "This theme is being uploaded already. Try again in a moment."
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
            "scanVerdict": package.scan_verdict, "scanReport": package.scan_report,
        },
        "install": _install_item(saved.installation),
        "status": _install_status(package, saved.installation),
        "duplicate": not saved.created,
    }


def _refuse(status: int, message: str) -> HTTPException:
    return HTTPException(status_code=status, detail=message)


def _after_integrity_error(db: Session, ctx: RequestContext, checked: CheckedThemePackage, exc: IntegrityError) -> SavedUpload:
    """A concurrent upload won the unique index: its package when it holds these bytes, else 409.

    The 409 names the clash the winner left (same version, other files; or not newer) when there is
    one, and asks to try again only when a retry could succeed.
    """
    db.rollback()
    existing = package_by_digest(db, ctx.workspace_id, checked.digest)
    if existing is None:
        logger.warning("[Themes] Concurrent upload of %s in workspace %s", checked.manifest["id"], ctx.workspace_id)
        raise _refuse(409, conflict_after_race(db, ctx.workspace_id, checked.manifest) or BUSY) from exc
    return SavedUpload(existing, active_installation(db, ctx.workspace_id, existing.theme_id), created=False)


def _save(db: Session, ctx: RequestContext, checked: CheckedThemePackage, package_bytes: int) -> SavedUpload:
    manifest = checked.manifest
    prefix = theme_prefix(manifest["id"], manifest["version"], ctx.workspace_id)
    report = build_scan_report(checked.files, package_bytes)
    try:
        return save_private_upload(
            db, ctx.workspace_id, resolve_user_pk(db, ctx), checked, prefix, report,
            store=lambda: _store(prefix, checked),
        )
    except UploadConflict as exc:
        db.rollback()
        raise _refuse(409, str(exc)) from exc
    except IntegrityError as exc:
        return _after_integrity_error(db, ctx, checked, exc)


@router.post("/import", status_code=201, dependencies=[Depends(MANAGE_GATE)])
def import_theme(
    response: Response,
    file: UploadFile = File(...),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Upload a .autotheme into this workspace only; returns the package, its scan and the install."""
    refusal = private_upload_refusal(db, ctx)
    if refusal is not None:
        raise _refuse(403, refusal)
    checked, package_bytes = _read_upload(file)
    refusal = reserved_publisher_refusal(checked.manifest["id"], ctx.user)
    if refusal is not None:
        raise _refuse(403, refusal)
    saved = _save(db, ctx, checked, package_bytes)
    if not saved.created:
        response.status_code = 200
    package = saved.package
    logger.info("[Themes] Upload of %s@%s in workspace %s: %s", package.theme_id, package.version, ctx.workspace_id,
                _install_status(package, saved.installation))
    return upload_item(saved)
