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
from core.auth.workspace_permission import require_workspace_permission
from core.auth.hybrid import get_request_context_hybrid
from core.database.database import get_db
from core.models.channels import ChannelConnection
from core.models.sites import SITE_TYPES
from services.callback import new_request_id
from services.destinations.base import CALLBACK_PLATFORMS, CallbackPayload, DESTINATION_TYPES
from services.destinations.dispatcher import dispatch_callback_for_site

from services.sites import (
    USER_SETTABLE_STATUSES,
    create_site,
    ensure_shopify_site_for_workspace,
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
    # Self-heal: a Shopify-connected workspace that ended up with a
    # mis-typed Site (legacy install path didn't create one) gets upgraded
    # to type=shopify on the next dashboard load. Idempotent — returns
    # immediately when the Site is already correct.
    from core.models.workspaces import Workspace
    workspace = db.query(Workspace).get(ctx.workspace_id)
    if workspace is not None:
        try:
            ensure_shopify_site_for_workspace(db, workspace)
        except Exception as e:  # noqa: BLE001 — never fail list on heal error
            logger.warning("ensure_shopify_site_for_workspace failed: %s", e)

    rows = list_sites(db, ctx.workspace_id)
    return {"sites": [public_site_dict(s) for s in rows]}


@router.post("", status_code=201, dependencies=[Depends(require_workspace_permission("workspace:manage"))])
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


@router.patch("/{site_id}", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
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


def _validate_callback_destinations(
    destinations: list, *, db: Session, workspace_id: UUID
) -> None:
    """Reject malformed destinations early so the dispatcher never has
    to deal with bad data.

    Heartbeat-style platform shape::

        {"platform": "telegram"}                            # auto-resolves chat
        {"platform": "slack",   "channel_id":  "C01ABC..."}
        {"platform": "webhook", "webhook_url": "https://…"}

    For Telegram / Slack / WhatsApp we also confirm a matching active
    ``ChannelConnection`` exists in the workspace — same prereq the
    heartbeat "Report To" dropdown enforces by only listing connected
    platforms.
    """
    if not isinstance(destinations, list):
        raise HTTPException(
            status_code=400,
            detail="callback.destinations must be a list",
        )
    for idx, dest in enumerate(destinations):
        if not isinstance(dest, dict):
            raise HTTPException(
                status_code=400,
                detail=f"callback.destinations[{idx}] must be an object",
            )
        platform = str(dest.get("platform") or "").strip().lower()
        if not platform:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"callback.destinations[{idx}].platform is required "
                    f"(one of {list(CALLBACK_PLATFORMS)})"
                ),
            )
        if platform not in CALLBACK_PLATFORMS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"callback.destinations[{idx}].platform must be one of "
                    f"{list(CALLBACK_PLATFORMS)}; got {platform!r}"
                ),
            )

        if platform == "webhook":
            url = str(dest.get("webhook_url") or "").strip()
            if not url:
                raise HTTPException(
                    status_code=400,
                    detail=f"callback.destinations[{idx}].webhook_url is required for webhook",
                )
            if not (url.startswith("https://") or url.startswith("http://")):
                raise HTTPException(
                    status_code=400,
                    detail=f"callback.destinations[{idx}].webhook_url must start with http(s)://",
                )
            continue

        # Channel-backed platforms (telegram/slack/whatsapp) — require a
        # ChannelConnection of this platform to exist in the workspace.
        # Status is NOT checked here: an ``inactive`` channel is the
        # normal pre-/start state for a freshly connected Telegram bot,
        # and the heartbeat "Report To" pattern doesn't filter on status
        # either. The dispatcher will surface a precise runtime error
        # if the adapter actually fails to deliver.
        connection_exists = (
            db.query(ChannelConnection.id)
            .filter(
                ChannelConnection.workspace_id == workspace_id,
                ChannelConnection.platform == platform,
            )
            .first()
        )
        if connection_exists is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"No {platform} channel connection in this workspace — "
                    f"connect one under Settings → Channels first."
                ),
            )


@router.patch("/{site_id}/settings", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
async def patch_site_settings_route(
    site_id: UUID,
    body: UpdateSiteSettingsRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    callback_patch = body.settings.get("callback") if isinstance(body.settings, dict) else None
    if isinstance(callback_patch, dict) and "destinations" in callback_patch:
        _validate_callback_destinations(
            callback_patch.get("destinations") or [],
            db=db,
            workspace_id=ctx.workspace_id,
        )

    site = update_site_settings(
        db,
        workspace_id=ctx.workspace_id,
        site_id=site_id,
        settings_patch=body.settings,
    )
    if site is None:
        raise HTTPException(status_code=404, detail="Site not found")
    return public_site_dict(site)


# ---------------------------------------------------------------------------
# POST /api/sites/{site_id}/callback/test
#
# Fires a synthetic callback through every configured destination and waits
# for the dispatcher to report per-destination success/failure. Used by the
# dashboard's CallbackPanel "Send test" button so a merchant can prove their
# Telegram / Slack / WhatsApp wiring works without needing a real shopper
# submission.
#
# Phone is a fixed obvious placeholder so anyone receiving the message in
# the destination channel can tell it's a test — never a real lead.
# ---------------------------------------------------------------------------

class CallbackTestDestinationResult(BaseModel):
    destination_type: str
    success: bool
    latency_ms: int
    error: Optional[str] = None
    platform: Optional[str] = None
    target: Optional[str] = None


class CallbackTestResponse(BaseModel):
    request_id: str
    destinations_attempted: int
    results: list[CallbackTestDestinationResult]


_CALLBACK_TEST_PHONE = "+10000000000"  # Reserved test-only number, never dialed.
_CALLBACK_TEST_NAME = "Dashboard test"
_CALLBACK_TEST_TOPIC = "Test callback from the dashboard — no action required."


@router.post("/{site_id}/callback/test", response_model=CallbackTestResponse, dependencies=[Depends(require_workspace_permission("workspace:manage"))])
async def send_callback_test_route(
    site_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> CallbackTestResponse:
    site = get_site(db, ctx.workspace_id, site_id)
    if site is None:
        raise HTTPException(status_code=404, detail="Site not found")

    callback_settings = (site.settings or {}).get("callback") or {}
    destinations = callback_settings.get("destinations") or []
    if not destinations:
        raise HTTPException(
            status_code=400,
            detail=(
                "No destinations configured. Add one under "
                "Settings → Widgets SDK → Callback before testing."
            ),
        )

    request_id = new_request_id()
    payload = CallbackPayload(
        request_id=request_id,
        name=_CALLBACK_TEST_NAME,
        phone=_CALLBACK_TEST_PHONE,
        product_context=_CALLBACK_TEST_TOPIC,
        urgency=None,
        preferred_time=None,
        site_display_name=site.display_name,
        site_external_id=site.external_id,
    )

    results = await dispatch_callback_for_site(
        site_id=site.id,
        workspace_id=site.workspace_id,
        session_id=f"dashboard-test-{request_id}",
        request_id=request_id,
        payload=payload,
        destinations=destinations,
    )

    logger.info(
        "callback test %s dispatched for site=%s (%d destination(s), %d success)",
        request_id, site.id, len(destinations),
        sum(1 for r in results if r.success),
    )

    return CallbackTestResponse(
        request_id=request_id,
        destinations_attempted=len(destinations),
        results=[
            CallbackTestDestinationResult(
                destination_type=r.destination_type,
                success=r.success,
                latency_ms=r.latency_ms,
                error=r.error,
                platform=(r.extra or {}).get("platform") if isinstance(r.extra, dict) else None,
                target=(r.extra or {}).get("target") if isinstance(r.extra, dict) else None,
            )
            for r in results
        ],
    )
