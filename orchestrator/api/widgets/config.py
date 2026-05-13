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

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from core.database.database import get_db
from core.models.workspaces import Workspace

from api.widgets.auth import WidgetAuthContext, widget_auth

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Widget Config"])


# Keys from ``workspace.settings`` exposed to browser widgets. Anything not
# listed here stays server-side. Add new feature keys as PRDs land.
PUBLIC_WIDGET_CONFIG_KEYS: tuple[str, ...] = ("widget_proactive",)


def build_widget_config(workspace: Optional[Workspace]) -> Optional[dict]:
    """Project the public widget-config slice from ``workspace.settings``.

    Returns ``None`` when no public keys are configured so existing clients
    that don't expect the field aren't pushed an empty object.
    """
    if workspace is None or not workspace.settings:
        return None
    settings = workspace.settings or {}
    public = {k: settings[k] for k in PUBLIC_WIDGET_CONFIG_KEYS if k in settings}
    return public or None


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
    workspace = (
        db.query(Workspace)
        .filter(Workspace.id == auth.workspace_id)
        .first()
    )
    config = build_widget_config(workspace) or {}
    return WidgetConfigResponse(
        workspace_id=str(auth.workspace_id),
        config=config,
    )
