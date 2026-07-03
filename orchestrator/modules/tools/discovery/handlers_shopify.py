"""Shopify sync + freshness handlers for PlatformActionExecutor (PRD-183 S3, F088).

Closes the tool-surface parity gap: the only catalog refresh used to be the
bare ``POST /api/shopify/sync/products/start`` HTTP route, so Auto could
neither run the sync nor check when the graph last synced. These handlers
promote both to first-class platform tools:

  * ``shopify_sync_catalog`` — run a catalog → knowledge-graph re-sync and
    report what changed (node/edge/community counts). This is the
    "Auto, refresh the catalog and tell me what changed" moment.
  * ``shopify_sync_status`` — read graph freshness from the stored
    ``workspace.settings.product_sync`` block.

Both are workspace-scoped: ``workspace_id`` is threaded from the executor
context (the authenticated RequestContext), never from params — an agent
cannot sync or inspect another tenant's catalog.

All handlers use the standard ``(db, workspace_id, params)`` signature.
"""
from __future__ import annotations

import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy.orm import Session

# Bound at module level so it is unit-patchable and the dependency is explicit.
from api.shopify import _product_sync_impl

logger = logging.getLogger(__name__)


async def shopify_sync_catalog(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Re-sync the Shopify catalog into the commerce graph for this workspace.

    Runs the Bulk-Op → ``map_shopify_catalog`` → ``import_graph`` pipeline and
    returns the resulting graph shape so the agent can report what changed.
    The workspace is the executor's authenticated workspace, never a param.
    """
    try:
        resp = await _product_sync_impl(str(workspace_id), db)
        data = resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
        return {
            "success": True,
            "status": data.get("status", "complete"),
            "node_count": data.get("node_count"),
            "edge_count": data.get("edge_count"),
            "community_count": data.get("community_count"),
            "object_count": data.get("object_count"),
            "duration_seconds": data.get("duration_seconds"),
        }
    except Exception as e:  # noqa: BLE001 — surface a clean tool error
        logger.error("[shopify] shopify_sync_catalog failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}


async def shopify_sync_status(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Report catalog-graph freshness: last sync status, timestamp, counts.

    Reads the ``product_sync`` block written by ``_product_sync_impl`` onto
    ``workspace.settings``. Returns ``never_synced`` (not an error) when the
    workspace has never run a sync.
    """
    try:
        from core.models.workspaces import Workspace

        ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
        if ws is None:
            return {"success": False, "error": "Workspace not found"}

        state = (ws.settings or {}).get("product_sync") or {"status": "never_synced"}
        return {"success": True, **state}
    except Exception as e:  # noqa: BLE001
        logger.error("[shopify] shopify_sync_status failed: %s", e, exc_info=True)
        return {"success": False, "error": str(e)}
