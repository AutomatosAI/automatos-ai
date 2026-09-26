"""Workspace brand kit routes (PRD-167 S4 → PRD-242 S3).

Sub-router included into ``api.document_generation.router`` (prefix
``/api/documents``) — not a new mount in ``main.py``. Carries the brand kit
read/update that PRD-167 shipped, plus what a non-technical user needed to
actually brand a document:

* ``POST/GET/DELETE /brand-kit/logo`` — upload a PNG/JPEG logo into platform
  storage (the renderers inline it; see ``modules.documents.brand_logo``),
  stream it back for the UI, remove it;
* ``GET /brand-kit/suggestions`` — prefill candidates from what the workspace
  already knows about itself (workspace name, the onboarding business profile,
  the signed-in user) so the kit starts filled rather than blank.

PRD-251 D5 (S1.3) extends the kit in place, same GET/PUT: a heading font, social
handles and a brand voice on the PUT, plus the stored files a social render
inlines, each with routes that mirror the logo's:

* ``POST/GET/DELETE /brand-kit/logo-mark`` — the square mark, separate from the
  wordmark;
* ``POST /brand-kit/fonts`` (a woff2 file plus the face it provides),
  ``GET/DELETE /brand-kit/fonts/{font_id}`` — the brand's font files
  (``modules.documents.brand_fonts``).

The body of the PUT, the kit's one writer and the suggestions live in
``modules.documents.brand_kit``: the agent tools ``platform_get_brand_kit`` and
``platform_update_brand_kit`` (PRD-251 US-115) call the same functions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.principal import resolve_user_pk
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from modules.documents.brand_kit import BrandKitPatch

logger = logging.getLogger(__name__)
router = APIRouter(tags=["document-generation"])

_MANAGE = Depends(require_workspace_permission("workspace:manage"))


def _workspace_or_404(db: Session, workspace_id):
    from core.models.workspaces import Workspace

    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return ws


# ------------------------------------------------------------------
# Kit read / update
# ------------------------------------------------------------------


@router.get("/brand-kit")
async def get_brand_kit_endpoint(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Return the workspace brand kit (defaults merged in)."""
    from modules.documents.brand_kit import get_brand_kit

    ws = _workspace_or_404(db, ctx.workspace_id)
    return get_brand_kit(ws.settings)


@router.put("/brand-kit", dependencies=[_MANAGE])
async def update_brand_kit_endpoint(
    body: BrandKitPatch,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Update the workspace brand kit (validated, persisted on workspace.settings)."""
    from pydantic import ValidationError

    from modules.documents.brand_kit import brand_kit_errors, update_brand_kit

    ws = _workspace_or_404(db, ctx.workspace_id)
    try:
        return update_brand_kit(db, ws, body.model_dump())
    except ValidationError as e:
        raise HTTPException(status_code=422, detail={"message": "Invalid brand kit", "errors": brand_kit_errors(e)})


# ------------------------------------------------------------------
# Suggestions — reuse what the platform already knows (PRD-242 S3)
# ------------------------------------------------------------------


@router.get("/brand-kit/suggestions")
async def get_brand_kit_suggestions(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Prefill candidates from the workspace, its business profile and the signed-in user."""
    from core.models.core import User
    from modules.documents.brand_kit import brand_kit_suggestions

    ws = _workspace_or_404(db, ctx.workspace_id)
    user_pk = resolve_user_pk(db, ctx)
    user = db.query(User).filter(User.id == user_pk).first() if user_pk is not None else None
    return {"suggestions": brand_kit_suggestions(db, ws, user)}


# ------------------------------------------------------------------
# Logo and logo mark upload / stream / delete (PRD-242 S3, PRD-251 D5)
# ------------------------------------------------------------------

# kit field of the stored file -> the external-URL field an upload supersedes
_LOGO_FIELDS = {"logo_path": "logo_url", "logo_mark_path": "logo_mark_url"}


async def _store_logo_upload(file: UploadFile, ctx: RequestContext, db: Session, path_field: str) -> Dict[str, Any]:
    """Store an uploaded logo or logo mark and point the kit at it; the old file goes."""
    from modules.documents.brand_kit import get_brand_kit, save_brand_kit
    from modules.documents.brand_logo import (
        MAX_LOGO_BYTES,
        BrandLogoError,
        delete_brand_logo,
        save_brand_logo,
        save_brand_logo_mark,
    )

    save = save_brand_logo_mark if path_field == "logo_mark_path" else save_brand_logo
    ws = _workspace_or_404(db, ctx.workspace_id)
    data = await file.read(MAX_LOGO_BYTES + 1)
    try:
        stored_path = save(ctx.workspace_id, data)
    except BrandLogoError as e:
        raise HTTPException(status_code=422, detail=str(e))

    kit = get_brand_kit(ws.settings)
    previous = kit.get(path_field) or ""
    if previous and previous != stored_path:
        delete_brand_logo(previous)
    # An uploaded file supersedes any external URL the kit carried.
    new_kit = {**kit, path_field: stored_path, _LOGO_FIELDS[path_field]: ""}
    save_brand_kit(db, ws, new_kit)
    logger.info("[BrandKit] %s uploaded for workspace %s (%d bytes)", path_field, ctx.workspace_id, len(data))
    return new_kit


def _stream_logo(ctx: RequestContext, db: Session, path_field: str, missing: str) -> Response:
    """Stream a stored logo or logo mark (local file, then object storage)."""
    from modules.documents.brand_kit import get_brand_kit
    from modules.documents.brand_logo import load_brand_logo, logo_mime

    ws = _workspace_or_404(db, ctx.workspace_id)
    stored_path = get_brand_kit(ws.settings).get(path_field) or ""
    data = load_brand_logo(stored_path) if stored_path else None
    if not data:
        raise HTTPException(status_code=404, detail=missing)
    return Response(
        content=data,
        media_type=logo_mime(stored_path),
        headers={"Cache-Control": "private, max-age=300"},
    )


def _remove_logo(ctx: RequestContext, db: Session, path_field: str) -> Dict[str, Any]:
    """Remove a stored logo or logo mark and clear it from the kit."""
    from modules.documents.brand_kit import get_brand_kit, save_brand_kit
    from modules.documents.brand_logo import delete_brand_logo

    ws = _workspace_or_404(db, ctx.workspace_id)
    kit = get_brand_kit(ws.settings)
    if kit.get(path_field):
        delete_brand_logo(kit[path_field])
    new_kit = {**kit, path_field: ""}
    save_brand_kit(db, ws, new_kit)
    return new_kit


@router.post("/brand-kit/logo", dependencies=[_MANAGE])
async def upload_brand_logo(
    file: UploadFile = File(...),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Store a PNG/JPEG logo for the workspace and point the brand kit at it."""
    from modules.documents.brand_logo import BRAND_LOGO_ROUTE

    new_kit = await _store_logo_upload(file, ctx, db, "logo_path")
    return {**new_kit, "logo_route": BRAND_LOGO_ROUTE}


@router.get("/brand-kit/logo")
async def stream_brand_logo(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Stream the stored logo (local file, then object storage)."""
    return _stream_logo(ctx, db, "logo_path", "No logo uploaded")


@router.delete("/brand-kit/logo", dependencies=[_MANAGE])
async def delete_brand_logo_endpoint(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Remove the stored logo and clear it from the kit."""
    return _remove_logo(ctx, db, "logo_path")


@router.post("/brand-kit/logo-mark", dependencies=[_MANAGE])
async def upload_brand_logo_mark(
    file: UploadFile = File(...),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Store a square PNG/JPEG logo mark (PRD-251 D5) and point the brand kit at it."""
    from modules.documents.brand_logo import BRAND_LOGO_MARK_ROUTE

    new_kit = await _store_logo_upload(file, ctx, db, "logo_mark_path")
    return {**new_kit, "logo_mark_route": BRAND_LOGO_MARK_ROUTE}


@router.get("/brand-kit/logo-mark")
async def stream_brand_logo_mark(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Stream the stored logo mark (local file, then object storage)."""
    return _stream_logo(ctx, db, "logo_mark_path", "No logo mark uploaded")


@router.delete("/brand-kit/logo-mark", dependencies=[_MANAGE])
async def delete_brand_logo_mark_endpoint(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Remove the stored logo mark and clear it from the kit."""
    return _remove_logo(ctx, db, "logo_mark_path")


# ------------------------------------------------------------------
# Font files: upload / stream / delete (PRD-251 D5)
# ------------------------------------------------------------------


@router.post("/brand-kit/fonts", dependencies=[_MANAGE])
async def upload_brand_font(
    file: UploadFile = File(...),
    family: str = Form(...),
    weight: int = Form(400),
    style: str = Form("normal"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Store a woff2 font file for the face it provides; the same face uploaded again replaces it."""
    from modules.documents.brand_fonts import MAX_FONT_BYTES, BrandFontError, add_brand_font
    from modules.documents.brand_kit import BrandKit, get_brand_kit, save_brand_kit

    ws = _workspace_or_404(db, ctx.workspace_id)
    data = await file.read(MAX_FONT_BYTES + 1)
    kit = get_brand_kit(ws.settings)
    try:
        fonts = add_brand_font(
            ctx.workspace_id,
            kit["font_files"],
            data,
            family=family,
            weight=weight,
            style=style,
            file_name=file.filename or "",
        )
    except BrandFontError as e:
        raise HTTPException(status_code=422, detail=str(e))
    new_kit = BrandKit.model_validate({**kit, "font_files": fonts}).model_dump()
    save_brand_kit(db, ws, new_kit)
    logger.info("[BrandKit] font %r uploaded for workspace %s (%d bytes)", family, ctx.workspace_id, len(data))
    return new_kit


@router.get("/brand-kit/fonts/{font_id}")
async def stream_brand_font(
    font_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Stream a stored font file (local file, then object storage)."""
    from modules.documents.brand_fonts import WOFF2_MIME, find_brand_font, load_brand_font
    from modules.documents.brand_kit import get_brand_kit

    ws = _workspace_or_404(db, ctx.workspace_id)
    font = find_brand_font(get_brand_kit(ws.settings)["font_files"], font_id)
    data = load_brand_font(font) if font else None
    if not data:
        raise HTTPException(status_code=404, detail="No such font file")
    return Response(content=data, media_type=WOFF2_MIME, headers={"Cache-Control": "private, max-age=300"})


@router.delete("/brand-kit/fonts/{font_id}", dependencies=[_MANAGE])
async def delete_brand_font(
    font_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Remove a stored font file and take it out of the kit."""
    from modules.documents.brand_fonts import remove_brand_font
    from modules.documents.brand_kit import get_brand_kit, save_brand_kit

    ws = _workspace_or_404(db, ctx.workspace_id)
    kit = get_brand_kit(ws.settings)
    fonts = remove_brand_font(kit["font_files"], font_id)
    if fonts is None:
        raise HTTPException(status_code=404, detail="No such font file")
    new_kit = {**kit, "font_files": fonts}
    save_brand_kit(db, ws, new_kit)
    return new_kit


__all__ = ["router"]
