"""PRD-259 — skins in Studio: the skin routes.

* ``GET /api/skins/schema`` serves the automatos-skin/v1 rule set
  (``services/skins/schema.py``) so the Skin Kit can sync its copy from Studio.
* ``POST /api/skins/import`` loads a .autoskin into THIS workspace only: a private
  skin (``owner_type`` 'workspace', approved, no review), installed at once. The
  same slug again replaces its content and version (the developer loop).
* ``GET /api/skins/installed``: this workspace's installed skins, for the style dropdown.
* ``GET /api/skins/{skin_id}/assets/{asset_path}``: one file of a skin this workspace
  may see, only when the stored manifest names it, served as its checked type.
  Anything else is 404, never 403: no other workspace learns a skin exists.

Every route is a plain ``def`` (a synchronous session and S3 client, run in the
threadpool); every query is filtered by the request context's workspace
(``services/skins/records.py``).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from config import config
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.principal import resolve_user_pk
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from services.skins.assets import ASSET_TYPES
from services.skins.package import SkinPackage, SkinPackageError, package_max_bytes, read_package
from services.skins.records import (
    StaleFiles, installed_item, list_installed, manifest_of, save_private_import, visible_skin,
)
from services.skins.schema import SKIN_ID_PATTERN, schema_document
from services.skins.storage import delete_skin_files, read_skin_file, skin_prefix, write_skin_files

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/skins", tags=["Skins"])

# The gate the workspace upload routes use (the brand-kit logo upload is one).
SKIN_IMPORT_PERMISSION = "workspace:manage"
_MANAGE = Depends(require_workspace_permission(SKIN_IMPORT_PERMISSION))

IMPORT_REFUSED = "This skin can't be imported."
ASSET_NOT_FOUND = "Skin asset not found."


@router.get("/schema")
def get_skin_schema(ctx: RequestContext = Depends(get_request_context_hybrid)) -> Dict[str, Any]:
    """The skin format's rule set: allowed tokens, contrast pairs, fonts, mascot, shape and text limits."""
    return schema_document()


def _read_upload_package(file: UploadFile) -> SkinPackage:
    """The upload read up to one byte past the cap, then checked: 413 over the cap, 400 with the codes."""
    data = file.file.read(package_max_bytes() + 1)
    try:
        return read_package(data)
    except SkinPackageError as exc:
        status = 413 if exc.too_large else 400
        logger.info("[Skins] Import refused (%d): %s", status, [error["code"] for error in exc.errors])
        raise HTTPException(status_code=status, detail={"message": IMPORT_REFUSED, "errors": exc.errors}) from exc


def _store_files(prefix: str, package: SkinPackage) -> None:
    try:
        write_skin_files(prefix, package.files)
    except Exception as exc:
        logger.exception("[Skins] Storing %d files under %s failed", len(package.files), prefix)
        raise HTTPException(status_code=503, detail="The skin's files couldn't be stored. Try again.") from exc


def _remove_stale(stale: StaleFiles) -> None:
    """Remove the files a re-import replaced. The import has committed by now, so a failure
    here leaves only unreferenced files behind: it is logged in full, not turned into an error."""
    if not stale.paths:
        return
    try:
        delete_skin_files(stale.prefix, stale.paths)
    except Exception:
        logger.exception("[Skins] Removing %d replaced files under %s failed", len(stale.paths), stale.prefix)


@router.post("/import", dependencies=[_MANAGE])
def import_skin(
    file: UploadFile = File(...),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Load a .autoskin into this workspace only, installed at once; returns the installed skin."""
    package = _read_upload_package(file)
    skin = package.skin
    prefix = skin_prefix(skin["slug"], skin["version"], ctx.workspace_id)
    user_pk = resolve_user_pk(db, ctx)
    try:
        row, stale = save_private_import(
            db, ctx.workspace_id, user_pk, package, prefix, store=lambda: _store_files(prefix, package),
        )
    except IntegrityError as exc:
        db.rollback()
        logger.warning("[Skins] Concurrent import of %s in workspace %s", skin["slug"], ctx.workspace_id)
        raise HTTPException(status_code=409, detail="This skin is being imported already. Try again.") from exc
    _remove_stale(stale)
    logger.info("[Skins] Imported %s@%s into workspace %s", skin["slug"], skin["version"], ctx.workspace_id)
    return installed_item(row)


@router.get("/installed")
def get_installed_skins(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> List[Dict[str, Any]]:
    """This workspace's active installations, joined to their skins."""
    return list_installed(db, ctx.workspace_id)


def asset_headers() -> Dict[str, str]:
    """The headers every skin asset carries: no sniffing, private and immutable (the client adds ?v=<version>)."""
    return {
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": f"private, max-age={config.SKIN_ASSET_CACHE_SECONDS}, immutable",
    }


@router.get("/{skin_id}/assets/{asset_path:path}")
def get_skin_asset(
    skin_id: str,
    asset_path: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Response:
    """One file of a skin this workspace may see, when its stored manifest names it; else 404."""
    row = visible_skin(db, ctx.workspace_id, UUID(skin_id)) if SKIN_ID_PATTERN.fullmatch(skin_id) else None
    entry = manifest_of(row).get(asset_path) if row is not None else None
    if not isinstance(entry, dict) or entry.get("type") not in ASSET_TYPES:
        raise HTTPException(status_code=404, detail=ASSET_NOT_FOUND)
    try:
        data = read_skin_file(row.asset_prefix, asset_path)
    except Exception as exc:
        logger.exception("[Skins] Reading %s of skin %s failed", asset_path, skin_id)
        raise HTTPException(status_code=404, detail=ASSET_NOT_FOUND) from exc
    return Response(content=data, media_type=entry["type"], headers=asset_headers())
