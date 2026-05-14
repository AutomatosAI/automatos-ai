"""
Sites API (PRD-008-A Phase 2).

Per-workspace CRUD for the universal Sites resource. Backs the
``/admin/sites`` dashboard hub.

Routes
------
GET    /api/sites                       list workspace's sites
POST   /api/sites                       create
GET    /api/sites/{site_id}             fetch one (404 if not owned)
PATCH  /api/sites/{site_id}             update display_name / status
PATCH  /api/sites/{site_id}/settings    shallow-merge settings JSONB

Telemetry endpoint and DELETE are out of scope until Phase 4
(``widget_event_log`` table) and the dashboard's disconnect flow.

Auth: every endpoint scoped to ``ctx.workspace_id`` from the hybrid
auth dependency. Sites owned by other workspaces are not-found
(no existence leak).
"""

from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.sites import SITE_TYPES

from services.sites import (
    USER_SETTABLE_STATUSES,
    create_site,
    get_site,
    list_sites,
    public_site_dict,
    update_site_meta,
    update_site_settings,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sites", tags=["sites"])


# ---------------------------------------------------------------------------
# Request/response shapes — never expose ``secrets``
# ---------------------------------------------------------------------------

class CreateSiteRequest(BaseModel):
    type: str = Field(..., description=f"One of {list(SITE_TYPES)}")
    display_name: str = Field(..., min_length=1, max_length=255)
    external_id: Optional[str] = Field(default=None, max_length=255)
    settings: Optional[dict] = None


class UpdateSiteMetaRequest(BaseModel):
    display_name: Optional[str] = Field(default=None, min_length=1, max_length=255)
    status: Optional[str] = Field(
        default=None,
        description=f"One of {list(USER_SETTABLE_STATUSES)}",
    )


class UpdateSiteSettingsRequest(BaseModel):
    settings: dict = Field(..., description="Top-level keys merge into site.settings")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@router.get("")
async def list_sites_route(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    rows = list_sites(db, ctx.workspace_id)
    return {"sites": [public_site_dict(s) for s in rows]}


@router.post("", status_code=201)
async def create_site_route(
    body: CreateSiteRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    try:
        site = create_site(
            db,
            workspace_id=ctx.workspace_id,
            type=body.type,
            display_name=body.display_name,
            external_id=body.external_id,
            settings=body.settings,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return public_site_dict(site)


@router.get("/{site_id}")
async def get_site_route(
    site_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    site = get_site(db, ctx.workspace_id, site_id)
    if site is None:
        raise HTTPException(status_code=404, detail="Site not found")
    return public_site_dict(site)


@router.patch("/{site_id}")
async def patch_site_meta_route(
    site_id: UUID,
    body: UpdateSiteMetaRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    if body.display_name is None and body.status is None:
        raise HTTPException(status_code=400, detail="No fields to update")
    try:
        site = update_site_meta(
            db,
            workspace_id=ctx.workspace_id,
            site_id=site_id,
            display_name=body.display_name,
            status=body.status,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if site is None:
        raise HTTPException(status_code=404, detail="Site not found")
    return public_site_dict(site)


@router.patch("/{site_id}/settings")
async def patch_site_settings_route(
    site_id: UUID,
    body: UpdateSiteSettingsRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    site = update_site_settings(
        db,
        workspace_id=ctx.workspace_id,
        site_id=site_id,
        settings_patch=body.settings,
    )
    if site is None:
        raise HTTPException(status_code=404, detail="Site not found")
    return public_site_dict(site)
