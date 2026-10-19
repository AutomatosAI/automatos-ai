"""PRD-261 C-04a — the Marketplace's Themes tab (Q11): listings and install by reference.

Routes (plan §2.2; C1-wire registers the router and lists the routes in
``reports/route-manifest.json``):

======================================================  ==========================================
``GET  /api/marketplace/themes``                        member
``GET  /api/marketplace/themes/{listingId}``            member
``POST /api/marketplace/themes/{listingId}/install``    manage (+ who installs, step-up), bucket (a)
======================================================  ==========================================

A listing shows approved versions only (``services/themes/listings.py``): the card, the
detail (previews per preset, publisher, version, ``uses``, the versions map) and, for the
caller's workspace, the install it already has. Install takes an optional
``{packageId}`` (a version from the versions map; default: the newest one this host runs)
and writes an install row that references the approved package (201), or answers the same
install again (200); update, compare and rollback of that install are A-07b's routes.

Gates, in order: the workspace permission first (``members:read`` for reads,
``workspace:manage`` for install), the Marketplace switch ``THEMES_MARKETPLACE``, then for
install A-07e's "who may install" and A-07c's step-up, so a caller refused anyway is never
asked to reverify. Both editions serve these routes; in the local edition step-up passes.
Every handler is a plain ``def`` over the synchronous session (AGENTS.md).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Optional, TypeVar
from uuid import UUID

from fastapi import APIRouter, Body, Depends, HTTPException, Response
from pydantic import BaseModel, ConfigDict
from sqlalchemy.orm import Session

from config import config
from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.auth.principal import resolve_user_pk
from core.auth.step_up import require_step_up
from core.auth.workspace_permission import require_workspace_permission
from core.database.database import get_db
from services.themes import listings
from services.themes.choices import NOT_FOUND, ThemeInstallError
from services.themes.settings import require_install_role
from services.themes.updates import install_row

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/marketplace/themes", tags=["Theme Marketplace"])

MEMBER = require_workspace_permission("members:read")
MANAGE = require_workspace_permission("workspace:manage")
INSTALL_ROLE = require_install_role()
STEP_UP = require_step_up(config.STEP_UP_MAX_AGE_MINUTES)
CALLER = get_request_context_hybrid
CREATED, OK = 201, 200
NO_WORKSPACE = "Choose a workspace first."

T = TypeVar("T")


class InstallBody(BaseModel):
    """``POST …/install``: the version to install; omitted, the newest one this host runs."""

    model_config = ConfigDict(extra="forbid")
    packageId: Optional[UUID] = None  # noqa: N815 — the wire name of plan §2.2


def require_marketplace() -> None:
    """403 with a plain sentence while ``THEMES_MARKETPLACE`` is off."""
    _answer(listings.require_marketplace_on)


def _answer(work: Callable[[], T]) -> T:
    """Run a read; a refusal becomes its own status and plain message."""
    try:
        return work()
    except ThemeInstallError as refusal:
        raise HTTPException(status_code=refusal.status_code, detail=refusal.message) from refusal


def _workspace_of(ctx: RequestContext) -> UUID:
    workspace_id = getattr(ctx, "workspace_id", None)
    if workspace_id is None:
        raise HTTPException(status_code=NOT_FOUND, detail=NO_WORKSPACE)
    return workspace_id if isinstance(workspace_id, UUID) else UUID(str(workspace_id))


@router.get("", dependencies=[Depends(MEMBER), Depends(require_marketplace)])
def list_theme_listings(ctx: RequestContext = Depends(CALLER), db: Session = Depends(get_db)) -> Dict[str, Any]:
    """Every approved theme, newest compatible version first in each card."""
    return {"listings": listings.list_listings(db, _workspace_of(ctx))}


@router.get("/{listing_id}", dependencies=[Depends(MEMBER), Depends(require_marketplace)])
def get_theme_listing(
    listing_id: str, ctx: RequestContext = Depends(CALLER), db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """One listing: previews per preset, publisher, version, ``uses`` and the versions map."""
    workspace_id = _workspace_of(ctx)
    return _answer(lambda: listings.listing_detail(db, workspace_id, listing_id))


@router.post(
    "/{listing_id}/install",
    status_code=CREATED,
    dependencies=[Depends(MANAGE), Depends(require_marketplace), Depends(INSTALL_ROLE), Depends(STEP_UP)],
)
def install_theme_listing(
    listing_id: str,
    response: Response,
    body: Optional[InstallBody] = Body(default=None),
    ctx: RequestContext = Depends(CALLER),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """Add an approved theme to this workspace by reference (201), or answer the install it has (200)."""
    workspace_id, user_pk = _workspace_of(ctx), resolve_user_pk(db, ctx)
    package_id = None if body is None else body.packageId
    try:
        install, created = listings.install_listing(db, workspace_id, user_pk, listing_id, package_id)
        if created:
            db.commit()
    except ThemeInstallError as refusal:
        db.rollback()
        raise HTTPException(status_code=refusal.status_code, detail=refusal.message) from refusal
    if created:
        logger.info("theme %s installed from its listing in workspace %s", install.theme_id, workspace_id)
    else:
        response.status_code = OK
    return {"created": created, "install": install_row(db, workspace_id, user_pk, install)}
