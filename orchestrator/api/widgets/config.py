"""
Widget Public Config Endpoint
==============================

Exposes the public slice of ``workspace.settings`` to browser-side widgets so
they can configure runtime behaviour (e.g. PRD-007 proactive engagement)
without needing a server-key JWT exchange.

Public-key (``ak_pub_*``) widgets call this endpoint once on init. Server-key
flows already receive the same payload via ``SessionTokenResponse.widget_config``.

    GET /api/widgets/config
"""

from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from core.database.database import get_db
from core.models.sites import Site
from core.models.workspaces import Workspace

from api.widgets.auth import WidgetAuthContext, widget_auth

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Widget Config"])


# Keys exposed to browser widgets. Anything not listed here stays
# server-side. Add new feature keys as PRDs land.
PUBLIC_WIDGET_CONFIG_KEYS: tuple[str, ...] = ("widget_proactive",)


def _project_public_keys(settings: Optional[dict]) -> Optional[dict]:
    """Filter a settings dict to the public-key whitelist."""
    if not settings:
        return None
    public = {k: settings[k] for k in PUBLIC_WIDGET_CONFIG_KEYS if k in settings}
    return public or None


def build_widget_config(workspace: Optional[Workspace]) -> Optional[dict]:
    """Project the public widget-config slice from ``workspace.settings``.

    Retained for backward compatibility — PRD-007 callers still pass a
    Workspace. Endpoints now go through ``resolve_widget_settings_dict``
    which prefers a Site and falls back to Workspace settings during the
    transition window.

    Returns ``None`` when no public keys are configured so existing clients
    that don't expect the field aren't pushed an empty object.
    """
    if workspace is None:
        return None
    return _project_public_keys(workspace.settings)


def resolve_widget_settings_dict(
    db: Session, workspace_id: UUID
) -> Optional[dict]:
    """Resolve the settings dict for the given workspace.

    PRD-008-A: prefer the workspace's default Site (oldest first). During
    the transition window we fall back to ``workspace.settings`` so that
    PRD-007 widgets keep working on workspaces whose migration hasn't
    populated a Site yet — e.g. a freshly-restored backup, or a workspace
    that pre-dates the backfill. Remove the fallback once the migration
    is verified live everywhere.
    """
    site = (
        db.query(Site)
        .filter(Site.workspace_id == workspace_id)
        .order_by(Site.created_at.asc())
        .first()
    )
    if site and site.settings:
        return site.settings

    workspace = (
        db.query(Workspace).filter(Workspace.id == workspace_id).first()
    )
    return workspace.settings if workspace else None


def resolve_widget_config(db: Session, workspace_id: UUID) -> Optional[dict]:
    """Public-projected widget config — what gets sent to the browser."""
    return _project_public_keys(resolve_widget_settings_dict(db, workspace_id))


class WidgetConfigResponse(BaseModel):
    """Public widget runtime config returned to the browser SDK."""

    workspace_id: str
    config: dict


@router.get("/config", response_model=WidgetConfigResponse)
async def get_widget_config(
    auth: WidgetAuthContext = Depends(widget_auth),
    db: Session = Depends(get_db),
) -> WidgetConfigResponse:
    """Return the public widget config for the authenticated workspace.

    Works for both public keys (raw API key) and server-key JWTs since both
    route through ``widget_auth``.
    """
    config = resolve_widget_config(db, auth.workspace_id) or {}
    return WidgetConfigResponse(
        workspace_id=str(auth.workspace_id),
        config=config,
    )
