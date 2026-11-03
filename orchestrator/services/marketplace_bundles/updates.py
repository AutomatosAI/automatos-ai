"""PRD-270 W2 US-270-14a (§5.6, K17, K18) — which of a workspace's Cloud items have moved on Automatos Cloud.

``GET /api/marketplace/cloud-sync/updates`` answers ``updates(db, workspace_id)`` for the CALLER's workspace:
``{updates: [{kind, cloudId, name, installed [{id, name, version}], available {version, revision, syncedAt}}]}``.

* An agent or a Playbook: a Cloud row (a ``marketplace_cloud_items`` mapping points at it) of which this
  workspace holds at least one copy (``cloned_from_id`` the row, owner 'workspace', this workspace) and NO copy
  at the row's version. A row's version is ``local_version`` (Automatos Cloud's and the revision's first 12),
  which every clone copies, so a copy at it was made from the current bundle. ``installed`` lists the
  workspace's copies; reviewing the new version installs a NEW copy beside them through the same route, and
  the old ones stay the workspace's (K17).
* A plugin: a Cloud plugin this workspace enabled before the plugin last moved (``enabled_at`` earlier than its
  mapping's ``synced_at``, which plugin_import.py moves only with the plugin's version: a changed revision at
  one version is refused). Enablement is by reference, so the workspace already runs the new version (K15):
  ``installed`` is empty, and ``revision`` is what a viewer's dismissal is kept by (K30).

``available`` is Automatos Cloud's version, the bundle's revision and when the Sync wrote it; ``cloudId`` the
Cloud row's key (an integer id, a plugin's uuid as a string). Only this workspace's copies and enablements are
read, never another's; local rows only, no network.
"""
from __future__ import annotations

from datetime import datetime
from itertools import groupby
from typing import Any, Dict, List, Optional, Sequence, Type
from uuid import UUID

from sqlalchemy.orm import Session, aliased

from core.models.core import Agent, WorkflowTemplate
from core.models.marketplace_cloud import CLOUD_ITEM_KEYS, MarketplaceCloudItem
from core.models.marketplace_plugins import MarketplacePlugin, WorkspaceEnabledPlugin

AGENT, PLAYBOOK, PLUGIN = "agent", "playbook", "plugin"
WORKSPACE_OWNER = "workspace"  # a workspace's own row: an installed copy


def updates(db: Session, workspace_id: UUID) -> Dict[str, List[Dict[str, Any]]]:
    """``{updates}``: the workspace's agent and Playbook copies with a newer Cloud version, then the Cloud
    plugins it enabled that have moved since."""
    return {"updates": [
        *_copy_updates(db, workspace_id, AGENT, Agent),
        *_copy_updates(db, workspace_id, PLAYBOOK, WorkflowTemplate),
        *_plugin_updates(db, workspace_id),
    ]}


def _copy_updates(db: Session, workspace_id: UUID, kind: str, model: Type[Any]) -> List[Dict[str, Any]]:
    """Each Cloud row of ``kind`` the workspace holds copies of, none of them at the row's version."""
    copy = aliased(model)
    mapped_id = getattr(MarketplaceCloudItem, CLOUD_ITEM_KEYS[kind])
    rows = (
        db.query(
            model.id.label("row_id"), model.name.label("row_name"), model.version.label("row_version"),
            MarketplaceCloudItem.version.label("cloud_version"), MarketplaceCloudItem.revision,
            MarketplaceCloudItem.synced_at, copy.id.label("copy_id"), copy.name.label("copy_name"),
            copy.version.label("copy_version"),
        )
        .join(MarketplaceCloudItem, mapped_id == model.id)
        .join(copy, copy.cloned_from_id == model.id)
        .filter(
            MarketplaceCloudItem.item_type == kind, copy.owner_type == WORKSPACE_OWNER,
            copy.workspace_id == workspace_id,
        )
        .order_by(model.name, model.id, copy.id)
        .all()
    )
    groups = [list(group) for _row_id, group in groupby(rows, key=lambda row: row.row_id)]
    return [_copy_notice(kind, group) for group in groups if all(row.copy_version != row.row_version for row in group)]


def _copy_notice(kind: str, copies: Sequence[Any]) -> Dict[str, Any]:
    first = copies[0]
    return {
        "kind": kind, "cloudId": first.row_id, "name": first.row_name,
        "installed": [{"id": row.copy_id, "name": row.copy_name, "version": row.copy_version} for row in copies],
        "available": available(first.cloud_version, first.revision, first.synced_at),
    }


def _plugin_updates(db: Session, workspace_id: UUID) -> List[Dict[str, Any]]:
    """Each Cloud plugin the workspace enabled before its mapping last moved (its last update)."""
    rows = (
        db.query(
            MarketplacePlugin.id, MarketplacePlugin.name, MarketplaceCloudItem.version,
            MarketplaceCloudItem.revision, MarketplaceCloudItem.synced_at,
        )
        .join(MarketplaceCloudItem, MarketplaceCloudItem.plugin_id == MarketplacePlugin.id)
        .join(WorkspaceEnabledPlugin, WorkspaceEnabledPlugin.plugin_id == MarketplacePlugin.id)
        .filter(
            MarketplaceCloudItem.item_type == PLUGIN, WorkspaceEnabledPlugin.workspace_id == workspace_id,
            # enabled_at (no zone: server now() in the session's TimeZone) is read in that same zone against the
            # zoned synced_at: right while the app's sessions share one TimeZone, as one database's do.
            WorkspaceEnabledPlugin.enabled_at < MarketplaceCloudItem.synced_at,
        )
        .order_by(MarketplacePlugin.name, MarketplacePlugin.id)
        .all()
    )
    return [
        {
            "kind": PLUGIN, "cloudId": str(row.id), "name": row.name, "installed": [],
            "available": available(row.version, row.revision, row.synced_at),
        }
        for row in rows
    ]


def available(version: str, revision: str, synced_at: Optional[datetime]) -> Dict[str, Any]:
    """Automatos Cloud's version, the bundle's revision and when the Sync wrote it."""
    return {"version": version, "revision": revision, "syncedAt": synced_at.isoformat() if synced_at else None}


__all__ = ["AGENT", "PLAYBOOK", "PLUGIN", "WORKSPACE_OWNER", "available", "updates"]
