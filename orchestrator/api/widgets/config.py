"""
Widget Public Config Endpoint
==============================

Exposes the public slice of ``workspace.settings`` to browser-side widgets so
they can configure runtime behaviour (e.g. PRD-007 proactive engagement)
without needing a server-key JWT exchange.

Public-key (``ak_pub_*``) widgets call this endpoint once on init. Server-key
flows already receive the same payload via ``SessionTokenResponse.widget_config``.

PRD-264 EUAI-02a: the public config always carries ``ai_disclosure``, the AI
notice the widget shows before its first answer. The server builds it every
time (``modules.policy.ai_disclosure``); it is never copied from settings, and
it is there even when the workspace has no widget settings at all.

    GET /api/widgets/config
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping, Optional
from uuid import UUID

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from core.database.database import get_db
from core.models.sites import Site
from core.models.workspaces import Workspace
from modules.policy.ai_disclosure import SETTINGS_KEY as AI_DISCLOSURE_KEY
from modules.policy.ai_disclosure import AIDisclosure, build_disclosure

from api.widgets.auth import WidgetAuthContext, widget_auth

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Widget Config"])


# Keys copied to browser widgets as the deployer stored them. Anything not
# listed here stays server-side. Add new feature keys as PRDs land.
#
# - widget_proactive: PRD-007 product-page opener config (delays, triggers)
# - cart_idle:        PRD-008-A C1 cart-idle popup config (enabled flag,
#                     idle_seconds, greeting, frequency cap). The SDK
#                     reads payload.cart_idle in bootstrapCartIdle; if
#                     not whitelisted here, the SDK silently no-ops and
#                     the popup never arms even when the merchant has
#                     toggled it on in the dashboard.
# - callback:         PRD-008-A B/C1 callback-form config (enabled flag,
#                     destinations, form fields). Same whitelist gate.
STORED_PUBLIC_WIDGET_CONFIG_KEYS: tuple[str, ...] = (
    "widget_proactive",
    "cart_idle",
    "callback",
)

# Every key the browser receives. ``ai_disclosure`` (PRD-264 EUAI-02a) is
# built by the server, never projected raw: a deployer may store a reworded
# ``line`` under it, which is validated on save and again on every build.
PUBLIC_WIDGET_CONFIG_KEYS: tuple[str, ...] = (
    *STORED_PUBLIC_WIDGET_CONFIG_KEYS,
    AI_DISCLOSURE_KEY,
)


@dataclass(frozen=True)
class WidgetSettingsSource:
    """The settings a widget is configured from, and the Site name it speaks for.

    ``name`` is the Site's ``display_name`` or ``None``; never the workspace's
    name (PRD-264 FR-15).
    """

    settings: Optional[dict]
    name: Optional[str]


def project_public_config(settings: Optional[Mapping[str, Any]], site_name: Any) -> dict:
    """The public widget config: the stored public keys plus ``ai_disclosure``.

    ``ai_disclosure`` is always present, so a widget never starts unlabelled.
    ``site_name`` is a Site's ``display_name`` or ``None``; callers never pass
    a workspace name (PRD-264 FR-15).
    """
    stored = settings if isinstance(settings, Mapping) else {}
    public = {key: stored[key] for key in STORED_PUBLIC_WIDGET_CONFIG_KEYS if key in stored}
    return {**public, AI_DISCLOSURE_KEY: build_disclosure(stored, site_name).as_dict()}


def build_widget_config(workspace: Optional[Workspace]) -> dict:
    """Project the public widget-config slice from ``workspace.settings``.

    PRD-007 callers pass a Workspace; endpoints go through
    ``resolve_widget_config``, which prefers a Site. The AI notice here is
    the unnamed line: a workspace's name may carry its owner's name or email
    (PRD-264 FR-15), so it never reaches public widget output.
    """
    settings = workspace.settings if workspace is not None else None
    return project_public_config(settings, None)


def resolve_widget_source(db: Session, workspace_id: UUID) -> WidgetSettingsSource:
    """Resolve the settings and the name a workspace's widget speaks for.

    PRD-008-A: prefer the workspace's default Site (oldest first). During
    the transition window we fall back to ``workspace.settings`` so that
    PRD-007 widgets keep working on workspaces whose migration hasn't
    populated a Site yet — e.g. a freshly-restored backup, or a workspace
    that pre-dates the backfill. Remove the fallback once the migration
    is verified live everywhere.

    The name is only ever the Site's ``display_name`` (PRD-264 FR-15). The
    workspace's name is never used, even when its settings are: an
    auto-created workspace is named "{name or email}'s Workspace".
    """
    site = (
        db.query(Site)
        .filter(Site.workspace_id == workspace_id)
        .order_by(Site.created_at.asc())
        .first()
    )
    site_settings = site.settings if site is not None else None
    site_name = site.display_name if site is not None else None
    if site_settings:
        return WidgetSettingsSource(settings=site_settings, name=site_name)

    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    return WidgetSettingsSource(
        settings=workspace.settings if workspace is not None else None,
        name=site_name,
    )


def resolve_widget_settings_dict(db: Session, workspace_id: UUID) -> Optional[dict]:
    """Resolve the settings dict for the given workspace (Site preferred)."""
    return resolve_widget_source(db, workspace_id).settings


def resolve_widget_config(db: Session, workspace_id: UUID) -> dict:
    """Public-projected widget config — what gets sent to the browser."""
    source = resolve_widget_source(db, workspace_id)
    return project_public_config(source.settings, source.name)


def resolve_ai_disclosure(db: Session, workspace_id: UUID) -> AIDisclosure:
    """The AI notice for a workspace's widget (PRD-264 EUAI-02a)."""
    source = resolve_widget_source(db, workspace_id)
    return build_disclosure(source.settings, source.name)


class WidgetConfigResponse(BaseModel):
    """Public widget runtime config returned to the browser SDK."""

    workspace_id: str
    config: dict


@router.get("/config", response_model=WidgetConfigResponse)
def get_widget_config(
    auth: WidgetAuthContext = Depends(widget_auth),
    db: Session = Depends(get_db),
) -> WidgetConfigResponse:
    """Return the public widget config for the authenticated workspace.

    Works for both public keys (raw API key) and server-key JWTs since both
    route through ``widget_auth``. The config always carries ``ai_disclosure``.
    A plain ``def``: the reads are synchronous, so FastAPI runs it in its
    threadpool rather than on the event loop.
    """
    return WidgetConfigResponse(
        workspace_id=str(auth.workspace_id),
        config=resolve_widget_config(db, auth.workspace_id),
    )
