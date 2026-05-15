"""
Sites service layer (PRD-008-A Phase 2).

Thin wrappers around the ``Site`` ORM model for the dashboard API.

Authorization invariant: every function takes ``workspace_id`` and
scopes the query to it. A Site that belongs to a different workspace
is treated as not-found (no existence leak).
"""

from __future__ import annotations

import copy
from typing import Optional
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.sites import Site, SITE_TYPES, derive_default_capabilities


# Mutable status values a merchant can set via PATCH. Backend-managed
# transitions (e.g. ``error``) are NOT in this list.
USER_SETTABLE_STATUSES: tuple[str, ...] = ("active", "paused", "disconnected")


def list_sites(db: Session, workspace_id: UUID) -> list[Site]:
    """List all Sites owned by a workspace, newest first."""
    return (
        db.query(Site)
        .filter(Site.workspace_id == workspace_id)
        .order_by(Site.created_at.desc())
        .all()
    )


def get_default_site(db: Session, workspace_id: UUID) -> Optional[Site]:
    """Resolve the default Site for a workspace — the oldest one.

    Widget endpoints authenticate via API key → workspace_id, but settings
    live on Sites. For PRD-008-A v1 we assume 1 workspace = 1 Site (the
    common case); multi-Site agencies will get explicit site_id pinning on
    public keys in a follow-up.

    Returns None if the workspace has no Sites — i.e. backfill migration
    has not run yet. Callers should treat that as a 503 condition.
    """
    return (
        db.query(Site)
        .filter(Site.workspace_id == workspace_id)
        .order_by(Site.created_at.asc())
        .first()
    )


def get_site(db: Session, workspace_id: UUID, site_id: UUID) -> Optional[Site]:
    """Fetch a single Site iff it belongs to the given workspace.

    Returns None if not found or owned by another workspace — callers
    map that to 404. Don't leak existence with 403.
    """
    return (
        db.query(Site)
        .filter(Site.id == site_id, Site.workspace_id == workspace_id)
        .one_or_none()
    )


def create_site(
    db: Session,
    workspace_id: UUID,
    type: str,
    display_name: str,
    external_id: Optional[str] = None,
    settings: Optional[dict] = None,
) -> Site:
    """Create a Site with capability defaults derived from its type.

    Raises ``ValueError`` for an unknown type.
    """
    if type not in SITE_TYPES:
        raise ValueError(f"unknown site type: {type!r}")

    site = Site(
        workspace_id=workspace_id,
        type=type,
        external_id=external_id,
        display_name=display_name,
        settings=copy.deepcopy(settings) if settings else {},
        capabilities=derive_default_capabilities(type),
    )
    db.add(site)
    db.commit()
    db.refresh(site)
    return site


def update_site_meta(
    db: Session,
    workspace_id: UUID,
    site_id: UUID,
    *,
    display_name: Optional[str] = None,
    status: Optional[str] = None,
) -> Optional[Site]:
    """Update the surface-level metadata of a Site.

    Does not touch settings, capabilities, secrets, type, or external_id —
    those changes have different blast radius and go through dedicated
    endpoints (or are immutable).
    """
    site = get_site(db, workspace_id, site_id)
    if site is None:
        return None

    if display_name is not None:
        site.display_name = display_name

    if status is not None:
        if status not in USER_SETTABLE_STATUSES:
            raise ValueError(
                f"status must be one of {USER_SETTABLE_STATUSES!r}, got {status!r}"
            )
        site.status = status

    db.commit()
    db.refresh(site)
    return site


def update_site_settings(
    db: Session,
    workspace_id: UUID,
    site_id: UUID,
    settings_patch: dict,
) -> Optional[Site]:
    """Shallow-merge a partial settings update into ``site.settings``.

    Top-level keys in ``settings_patch`` overwrite the same keys in
    ``site.settings``; other top-level keys are preserved. Nested keys
    inside a top-level block are replaced wholesale — callers wanting
    deeper merges should fetch, mutate, and PATCH the full block.

    Why shallow: PRD-008-A's settings are organised by feature
    (``widget_proactive``, ``callback``, ``cart_idle``). The dashboard
    sends one block at a time; the merchant edits the whole block.
    Shallow merge is the natural unit.
    """
    if not isinstance(settings_patch, dict):
        raise ValueError("settings_patch must be a dict")

    site = get_site(db, workspace_id, site_id)
    if site is None:
        return None

    current = dict(site.settings or {})
    current.update(copy.deepcopy(settings_patch))
    site.settings = current

    db.commit()
    db.refresh(site)
    return site


def public_site_dict(site: Site) -> dict:
    """Project a Site row into the public dict shape returned by the API.

    Excludes ``secrets`` — those are server-side credentials and must
    never leave the orchestrator.
    """
    return {
        "id": str(site.id),
        "workspace_id": str(site.workspace_id),
        "type": site.type,
        "external_id": site.external_id,
        "display_name": site.display_name,
        "status": site.status,
        "settings": site.settings or {},
        "capabilities": site.effective_capabilities,
        "created_at": site.created_at.isoformat() if site.created_at else None,
        "updated_at": site.updated_at.isoformat() if site.updated_at else None,
    }
