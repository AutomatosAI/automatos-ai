"""Workspace brand kit routes (PRD-167 S4 → PRD-242 S3).

Sub-router included into ``api.document_generation.router`` (prefix
``/api/documents``) — not a new mount in ``main.py``. Carries the brand kit
read/update that PRD-167 shipped, plus what a non-technical user needed to
actually brand a document:

* ``POST/GET/DELETE /brand-kit/logo`` — upload a PNG/JPEG logo into platform
  storage (the renderers inline it; see ``modules.documents.brand_logo``),
  stream it back for the UI, remove it;
* ``GET /brand-kit/board?format=pdf|png`` — the brand board (PRD-255 US-010),
  printed from the caller's kit in a child process (``modules.documents.brand_board_render``);
* ``GET /brand-kit/suggestions`` — prefill candidates from what the workspace
  already knows about itself (workspace name, the onboarding business profile,
  the signed-in user) so the kit starts filled rather than blank.

PRD-255 (Brand Kit v2) answers GET and PUT with the kit's effective colour roles
(``palette``: stored, else derived) and ``palette_source`` (each role ``set`` or
``derived``), and the PUT refuses a palette whose text does not read on its page,
saying what the colour sits on ("on the page (paper, white)"). GET's answer PUT
back unchanged changes nothing (F366): a derived role sent at its colour stays
derived, and ``palette_source`` is read (``{"palette_source": {"accent":
"derived"}}`` resets a role, ``{"palette_source": "derived"}`` every role).
It adds the logo's variants, each with routes that mirror the logo's (FR-9: a
variant is uploaded by the owner, never generated):

* ``POST/GET/DELETE /brand-kit/logo-dark`` — the logo for dark backgrounds;
* ``POST/GET/DELETE /brand-kit/logo-mono`` — the one-colour logo.

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
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.principal import resolve_user_pk
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from modules.documents.brand_board_render import BOARD_MEDIA_TYPES, BOARD_PDF, render_board_isolated
from modules.documents.brand_kit import BrandKitPatch
from modules.documents.brand_system import brand_kit_view
from modules.documents.thumbnails.render import ThumbnailError

logger = logging.getLogger(__name__)
router = APIRouter(tags=["document-generation"])

_MANAGE = Depends(require_workspace_permission("workspace:manage"))
SOCIAL_HANDLES_FIELD = "social_handles"
# The board is printed live from the kit: never kept by a cache, so a save shows at once.
BOARD_CACHE_CONTROL = "private, no-store"
BOARD_FILE_STEM = "brand-board"
BOARD_RENDER_FAILED = "The brand board could not be drawn. Try again in a moment."
# F372: a Brand kit page left open never saves over a change made elsewhere.
KIT_CHANGED_ELSEWHERE = "The brand kit changed since this page loaded; reload to see it"


class BrandKitPut(BrandKitPatch):
    """The PUT's body: a kit patch and, optionally, the kit's ``updated_at`` as the caller loaded it (F372).

    Sent (the Brand kit page sends it with every save), a stamp that is no longer the
    stored one is a 409 and nothing is saved: checked on the workspace row locked for the
    write (``brand_kit.update_brand_kit``), so two saves loaded at one stamp never both
    pass. Left out (the agent tool, the designer's save, an API caller), the PUT saves as
    it always did. Not a kit field.
    """

    if_updated_at: Optional[str] = None


def _handles_hidden(workspace) -> bool:
    """PRD-251B US-B106 (B3, off means invisible): while Socials is off for the workspace
    the kit is answered without its social handles, and a PUT leaves the stored handles
    exactly as they are, whether the payload lacks them or carries them: the route never
    erases, or changes, what the UI could not show."""
    from modules.socials.settings import socials_actions_hidden

    return socials_actions_hidden(workspace)


def _shown(workspace, kit: Dict[str, Any]) -> Dict[str, Any]:
    """``kit`` as the workspace sees it: without the social handles while they are hidden."""
    if not _handles_hidden(workspace):
        return kit
    return {key: value for key, value in kit.items() if key != SOCIAL_HANDLES_FIELD}


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
def get_brand_kit_endpoint(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Return the workspace brand kit (defaults merged in, every colour role and its source;
    no social handles while Socials is off)."""
    from modules.documents.brand_kit import get_brand_kit

    ws = _workspace_or_404(db, ctx.workspace_id)
    return _shown(ws, brand_kit_view(get_brand_kit(ws.settings)))


@router.put("/brand-kit", dependencies=[_MANAGE])
def update_brand_kit_endpoint(
    body: BrandKitPut,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Update the workspace brand kit (validated, persisted on workspace.settings).

    While Socials is off for the workspace the social handles are not part of the
    change (PRD-251B US-B106): the stored ones stay, and the answer leaves them out.
    ``if_updated_at`` other than the stored stamp is a 409 that saves nothing (F372).
    """
    from pydantic import ValidationError

    from modules.documents.brand_kit import BrandKitChanged, brand_kit_errors, update_brand_kit

    ws = _workspace_or_404(db, ctx.workspace_id)
    patch = body.model_dump(exclude={"if_updated_at"})
    if _handles_hidden(ws):
        patch = {key: value for key, value in patch.items() if key != SOCIAL_HANDLES_FIELD}
    try:
        kit = update_brand_kit(db, ws, patch, if_updated_at=body.if_updated_at)
    except BrandKitChanged:
        raise HTTPException(status_code=409, detail=KIT_CHANGED_ELSEWHERE) from None
    except ValidationError as e:
        detail = {"message": "Invalid brand kit", "errors": brand_kit_errors(e)}
        raise HTTPException(status_code=422, detail=detail) from None
    return _shown(ws, brand_kit_view(kit))


# ------------------------------------------------------------------
# The brand board: the kit on one page (PRD-255 US-010)
# ------------------------------------------------------------------


@router.get("/brand-kit/board")
def get_brand_board(
    fmt: str = Query(BOARD_PDF, alias="format"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Response:
    """The brand board printed from the caller's own kit, as a PDF or a PNG (page 1).

    A plain ``def``: the kit and its stored files are read synchronously, and the
    print itself runs in a child process under a time limit (``render_board_isolated``),
    never on the API process.
    """
    from modules.documents.brand_fonts import brand_kit_for_media_render
    from modules.documents.brand_kit import get_brand_kit

    if fmt not in BOARD_MEDIA_TYPES:
        raise HTTPException(status_code=422, detail=f"format must be one of: {', '.join(BOARD_MEDIA_TYPES)}")
    if not ctx.workspace_id:
        raise HTTPException(status_code=404, detail="Workspace not found")
    ws = _workspace_or_404(db, ctx.workspace_id)
    kit = brand_kit_for_media_render(get_brand_kit(ws.settings))
    try:
        data = render_board_isolated(kit, fmt)
    except ThumbnailError:
        logger.exception("[BrandKit] the brand board (%s) could not be drawn for workspace %s", fmt, ws.id)
        raise HTTPException(status_code=500, detail=BOARD_RENDER_FAILED) from None
    disposition = f'inline; filename="{BOARD_FILE_STEM}.{fmt}"'
    return Response(
        content=data,
        media_type=BOARD_MEDIA_TYPES[fmt],
        headers={"Cache-Control": BOARD_CACHE_CONTROL, "Content-Disposition": disposition},
    )


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

# kit field of the stored file -> the external-URL field an upload supersedes (the
# logo's variants have none: they are uploads only)
_LOGO_FIELDS = {"logo_path": "logo_url", "logo_mark_path": "logo_mark_url"}
LOGO_FIELD, LOGO_MARK_FIELD = "logo_path", "logo_mark_path"
LOGO_DARK_FIELD, LOGO_MONO_FIELD = "logo_dark_path", "logo_mono_path"


def _save_logo_file(workspace_id, data: bytes, path_field: str) -> str:
    """Validate and store the upload for ``path_field``; its storage-relative path."""
    from modules.documents import brand_logo as bl

    savers = {
        LOGO_FIELD: lambda: bl.save_brand_logo(workspace_id, data),
        LOGO_MARK_FIELD: lambda: bl.save_brand_logo_mark(workspace_id, data),
        LOGO_DARK_FIELD: lambda: bl.save_brand_logo(workspace_id, data, bl.LOGO_DARK_STEM),
        LOGO_MONO_FIELD: lambda: bl.save_brand_logo(workspace_id, data, bl.LOGO_MONO_STEM),
    }
    if path_field not in savers:
        raise ValueError(f"no stored logo file is kept at {path_field!r}")
    return savers[path_field]()


async def _store_logo_upload(file: UploadFile, ctx: RequestContext, db: Session, path_field: str) -> Dict[str, Any]:
    """Store an uploaded logo, logo variant or logo mark and point the kit at it; the old file goes."""
    from modules.documents.brand_kit import get_brand_kit, lock_brand_kit, save_brand_kit
    from modules.documents.brand_logo import MAX_LOGO_BYTES, BrandLogoError, delete_brand_logo

    ws = _workspace_or_404(db, ctx.workspace_id)
    data = await file.read(MAX_LOGO_BYTES + 1)
    try:
        stored_path = _save_logo_file(ctx.workspace_id, data, path_field)
    except BrandLogoError as e:
        raise HTTPException(status_code=422, detail=str(e))

    kit = get_brand_kit(lock_brand_kit(db, ws).settings)  # after the last await: held to the save's commit
    previous = kit.get(path_field) or ""
    if previous and previous != stored_path:
        delete_brand_logo(previous)
    # An uploaded file supersedes any external URL the kit carried.
    superseded = {_LOGO_FIELDS[path_field]: ""} if path_field in _LOGO_FIELDS else {}
    new_kit = {**kit, path_field: stored_path, **superseded}
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
    from modules.documents.brand_kit import get_brand_kit, lock_brand_kit, save_brand_kit
    from modules.documents.brand_logo import delete_brand_logo

    ws = _workspace_or_404(db, ctx.workspace_id)
    kit = get_brand_kit(lock_brand_kit(db, ws).settings)
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
# The logo's variants: for dark backgrounds, and one colour (PRD-255 FR-9)
# ------------------------------------------------------------------


@router.post("/brand-kit/logo-dark", dependencies=[_MANAGE])
async def upload_brand_logo_dark(
    file: UploadFile = File(...),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Store the PNG/JPEG logo for dark backgrounds and point the brand kit at it."""
    from modules.documents.brand_logo import BRAND_LOGO_DARK_ROUTE

    new_kit = await _store_logo_upload(file, ctx, db, LOGO_DARK_FIELD)
    return {**new_kit, "logo_dark_route": BRAND_LOGO_DARK_ROUTE}


@router.get("/brand-kit/logo-dark")
def stream_brand_logo_dark(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Stream the stored logo for dark backgrounds (local file, then object storage)."""
    return _stream_logo(ctx, db, LOGO_DARK_FIELD, "No logo for dark backgrounds uploaded")


@router.delete("/brand-kit/logo-dark", dependencies=[_MANAGE])
def delete_brand_logo_dark_endpoint(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Remove the stored logo for dark backgrounds and clear it from the kit."""
    return _remove_logo(ctx, db, LOGO_DARK_FIELD)


@router.post("/brand-kit/logo-mono", dependencies=[_MANAGE])
async def upload_brand_logo_mono(
    file: UploadFile = File(...),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Store the one-colour PNG/JPEG logo and point the brand kit at it."""
    from modules.documents.brand_logo import BRAND_LOGO_MONO_ROUTE

    new_kit = await _store_logo_upload(file, ctx, db, LOGO_MONO_FIELD)
    return {**new_kit, "logo_mono_route": BRAND_LOGO_MONO_ROUTE}


@router.get("/brand-kit/logo-mono")
def stream_brand_logo_mono(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Stream the stored one-colour logo (local file, then object storage)."""
    return _stream_logo(ctx, db, LOGO_MONO_FIELD, "No one-colour logo uploaded")


@router.delete("/brand-kit/logo-mono", dependencies=[_MANAGE])
def delete_brand_logo_mono_endpoint(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Remove the stored one-colour logo and clear it from the kit."""
    return _remove_logo(ctx, db, LOGO_MONO_FIELD)


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
    from modules.documents.brand_kit import BrandKit, get_brand_kit, lock_brand_kit, save_brand_kit

    ws = _workspace_or_404(db, ctx.workspace_id)
    data = await file.read(MAX_FONT_BYTES + 1)
    kit = get_brand_kit(lock_brand_kit(db, ws).settings)  # after the last await: held to the save's commit
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
    from modules.documents.brand_kit import get_brand_kit, lock_brand_kit, save_brand_kit

    ws = _workspace_or_404(db, ctx.workspace_id)
    kit = get_brand_kit(lock_brand_kit(db, ws).settings)
    fonts = remove_brand_font(kit["font_files"], font_id)
    if fonts is None:
        raise HTTPException(status_code=404, detail="No such font file")
    new_kit = {**kit, "font_files": fonts}
    save_brand_kit(db, ws, new_kit)
    return new_kit


# PRD-251B (US-B302, US-B303): the style references and the style profile, under the same prefix.
from api.document_brand_references import router as references_router  # noqa: E402

router.include_router(references_router)

__all__ = ["router"]
