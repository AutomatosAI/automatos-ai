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
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.principal import resolve_user_pk
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db

logger = logging.getLogger(__name__)
router = APIRouter(tags=["document-generation"])

_MANAGE = Depends(require_workspace_permission("workspace:manage"))


class BrandKitUpdateRequest(BaseModel):
    name: Optional[str] = None
    tagline: Optional[str] = None
    logo_url: Optional[str] = None
    primary_color: Optional[str] = None
    secondary_color: Optional[str] = None
    accent_color: Optional[str] = None
    text_color: Optional[str] = None
    font_family: Optional[str] = None
    company: Optional[dict] = None


def _workspace_or_404(db: Session, workspace_id):
    from core.models.workspaces import Workspace

    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not ws:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return ws


def _persist_kit(db: Session, ws, new_kit: Dict[str, Any]) -> None:
    from modules.documents.brand_kit import BRAND_KIT_SETTINGS_KEY

    # Reassign settings (not in-place mutate) so SQLAlchemy tracks the JSONB change.
    ws.settings = {**(ws.settings or {}), BRAND_KIT_SETTINGS_KEY: new_kit}
    db.commit()


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
    body: BrandKitUpdateRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Update the workspace brand kit (validated, persisted on workspace.settings)."""
    from pydantic import ValidationError

    from modules.documents.brand_kit import BRAND_KIT_SETTINGS_KEY, validate_brand_kit

    ws = _workspace_or_404(db, ctx.workspace_id)
    existing = (ws.settings or {}).get(BRAND_KIT_SETTINGS_KEY)
    patch = {k: v for k, v in body.model_dump().items() if v is not None}
    try:
        new_kit = validate_brand_kit(patch, existing)
    except ValidationError as e:
        raise HTTPException(status_code=422, detail={"message": "Invalid brand kit", "errors": e.errors()})
    _persist_kit(db, ws, new_kit)
    return new_kit


# ------------------------------------------------------------------
# Suggestions — reuse what the platform already knows (PRD-242 S3)
# ------------------------------------------------------------------


def build_brand_suggestions(workspace, business_profile, user) -> Dict[str, Dict[str, str]]:
    """Prefill candidates ``field -> {value, source}``; only fields with a value. Pure."""
    out: Dict[str, Dict[str, str]] = {}

    def put(field: str, value: Any, source: str) -> None:
        if field in out:
            return
        text = str(value).strip() if value is not None else ""
        if text:
            out[field] = {"value": text, "source": source}

    if business_profile is not None:
        put("name", getattr(business_profile, "company_name", None), "business_profile")
        put("company_name", getattr(business_profile, "company_name", None), "business_profile")
        domain = getattr(business_profile, "domain", None)
        if domain:
            website = domain if str(domain).startswith(("http://", "https://")) else f"https://{domain}"
            put("website", website, "business_profile")
        brands = getattr(business_profile, "brands", None) or []
        for brand in brands if isinstance(brands, list) else []:
            if isinstance(brand, dict) and brand.get("logo_url"):
                put("logo_url", brand["logo_url"], "business_profile")
                break
        voice = getattr(business_profile, "voice_notes", None)
        if voice:
            put("tagline", str(voice).strip().splitlines()[0][:120], "business_profile")
    if workspace is not None:
        put("name", getattr(workspace, "name", None), "workspace")
        put("company_name", getattr(workspace, "name", None), "workspace")
    if user is not None:
        put("email", getattr(user, "email", None), "user")
    return out


@router.get("/brand-kit/suggestions")
async def brand_kit_suggestions(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Prefill candidates from the workspace, its business profile and the signed-in user."""
    from core.models.business_profiles import BusinessProfile
    from core.models.core import User

    ws = _workspace_or_404(db, ctx.workspace_id)
    profile = (
        db.query(BusinessProfile)
        .filter(BusinessProfile.workspace_id == ctx.workspace_id)
        .order_by(BusinessProfile.created_at.desc())
        .first()
    )
    user_pk = resolve_user_pk(db, ctx)
    user = db.query(User).filter(User.id == user_pk).first() if user_pk is not None else None
    return {"suggestions": build_brand_suggestions(ws, profile, user)}


# ------------------------------------------------------------------
# Logo upload / stream / delete (PRD-242 S3)
# ------------------------------------------------------------------


@router.post("/brand-kit/logo", dependencies=[_MANAGE])
async def upload_brand_logo(
    file: UploadFile = File(...),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Store a PNG/JPEG logo for the workspace and point the brand kit at it."""
    from modules.documents.brand_kit import BRAND_KIT_SETTINGS_KEY, get_brand_kit
    from modules.documents.brand_logo import (
        BRAND_LOGO_ROUTE,
        MAX_LOGO_BYTES,
        BrandLogoError,
        delete_brand_logo,
        save_brand_logo,
    )

    ws = _workspace_or_404(db, ctx.workspace_id)
    data = await file.read(MAX_LOGO_BYTES + 1)
    try:
        logo_path = save_brand_logo(ctx.workspace_id, data)
    except BrandLogoError as e:
        raise HTTPException(status_code=422, detail=str(e))

    kit = get_brand_kit(ws.settings)
    previous = kit.get("logo_path") or ""
    if previous and previous != logo_path:
        delete_brand_logo(previous)
    # An uploaded logo supersedes any external URL the kit carried.
    new_kit = {**kit, "logo_path": logo_path, "logo_url": ""}
    _persist_kit(db, ws, new_kit)
    logger.info("[BrandKit] logo uploaded for workspace %s (%d bytes)", ctx.workspace_id, len(data))
    return {**new_kit, "logo_route": BRAND_LOGO_ROUTE}


@router.get("/brand-kit/logo")
async def stream_brand_logo(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Stream the stored logo (local file, then object storage)."""
    from modules.documents.brand_kit import get_brand_kit
    from modules.documents.brand_logo import load_brand_logo, logo_mime

    ws = _workspace_or_404(db, ctx.workspace_id)
    logo_path = get_brand_kit(ws.settings).get("logo_path") or ""
    data = load_brand_logo(logo_path) if logo_path else None
    if not data:
        raise HTTPException(status_code=404, detail="No logo uploaded")
    return Response(
        content=data,
        media_type=logo_mime(logo_path),
        headers={"Cache-Control": "private, max-age=300"},
    )


@router.delete("/brand-kit/logo", dependencies=[_MANAGE])
async def delete_brand_logo_endpoint(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Remove the stored logo and clear it from the kit."""
    from modules.documents.brand_kit import get_brand_kit
    from modules.documents.brand_logo import delete_brand_logo

    ws = _workspace_or_404(db, ctx.workspace_id)
    kit = get_brand_kit(ws.settings)
    if kit.get("logo_path"):
        delete_brand_logo(kit["logo_path"])
    new_kit = {**kit, "logo_path": ""}
    _persist_kit(db, ws, new_kit)
    return new_kit


__all__ = ["router", "build_brand_suggestions"]
