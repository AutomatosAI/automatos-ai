"""Channel handlers for PlatformActionExecutor (PRD-143 S10).

Auto's hands on the messaging-channel surface (``channel_connections`` +
``channels/drivers/`` + ``ChannelManager``) — the same DB/service layer the
``/api/channels`` router uses. Connect delegates to the router-extracted
``api.channels.connect_channel_for_workspace`` so the driver-mediated
verify/install flow has exactly one implementation. ``workspace_id`` always
comes from the executor context, never the params.
"""

import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def _row_dict(r, effective_status: str) -> Dict[str, Any]:
    return {
        "id": str(r.id),
        "platform": r.platform,
        "status": effective_status,
        "mode": r.mode or "webhook",
        "webhook_url": r.webhook_url,
        "last_verified": r.last_verified.isoformat() if r.last_verified else None,
        "last_error": r.last_error,
        "default_agent_id": r.default_agent_id,
        "message_count": r.message_count or 0,
        "last_activity_at": r.last_activity_at.isoformat() if r.last_activity_at else None,
        "created_at": r.created_at.isoformat() if r.created_at else None,
    }


async def list_channels(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """List the workspace's channel connections, reconciled against live adapter state."""
    try:
        rows = db.execute(
            text("""
                SELECT id, platform, status, mode, webhook_url, last_verified, last_error,
                       metadata, default_agent_id, message_count, last_activity_at, created_at
                FROM channel_connections
                WHERE workspace_id = :ws_id
                ORDER BY created_at DESC
            """),
            {"ws_id": str(workspace_id)},
        ).fetchall()

        try:
            from channels.manager import get_channel_manager
            manager = get_channel_manager()
        except Exception:
            manager = None

        channels = []
        for r in rows:
            effective_status = r.status
            if manager is not None and manager.is_running(str(r.id)):
                effective_status = "active"
            channels.append(_row_dict(r, effective_status))

        return {"success": True, "channels": channels, "count": len(channels)}
    except Exception as exc:
        logger.error("[channels] list_channels failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}


async def connect_channel(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Connect a messaging channel via the canonical router flow (verify + install)."""
    platform = params.get("platform")
    if not platform:
        return {"success": False, "error": "Missing required parameter: platform"}
    config = params.get("config") or {}

    try:
        from api.channels import connect_channel_for_workspace

        result = await connect_channel_for_workspace(
            db,
            workspace_id=str(workspace_id),
            platform=platform,
            config=config,
            default_agent_id=params.get("default_agent_id"),
            mode=params.get("mode"),
        )
        return {"success": True, **result}
    except ValueError as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        logger.error("[channels] connect_channel failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}


async def configure_channel(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Update a channel connection's config and/or default agent (mirrors PUT /api/channels/{id})."""
    channel_id = params.get("channel_id")
    if not channel_id:
        return {"success": False, "error": "Missing required parameter: channel_id"}

    try:
        row = db.execute(
            text("SELECT id FROM channel_connections WHERE id = :id AND workspace_id = :ws_id"),
            {"id": str(channel_id), "ws_id": str(workspace_id)},
        ).fetchone()
        if not row:
            return {"success": False, "error": "Channel connection not found"}

        import json as _json

        updates = []
        bind: Dict[str, Any] = {"id": str(channel_id)}
        if "config" in params and params["config"] is not None:
            updates.append("config = :config")
            bind["config"] = _json.dumps(params["config"])
        if "default_agent_id" in params:
            updates.append("default_agent_id = :agent_id")
            bind["agent_id"] = params["default_agent_id"]

        if not updates:
            return {"success": False, "error": "Nothing to update — provide config and/or default_agent_id"}

        updates.append("updated_at = NOW()")
        db.execute(
            text(f"UPDATE channel_connections SET {', '.join(updates)} WHERE id = :id"),
            bind,
        )
        db.commit()
        return {"success": True, "status": "updated", "channel_id": str(channel_id)}
    except Exception as exc:
        db.rollback()
        logger.error("[channels] configure_channel failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}


async def start_channel(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Start the adapter for a channel connection (mirrors POST /api/channels/{id}/start)."""
    channel_id = params.get("channel_id")
    if not channel_id:
        return {"success": False, "error": "Missing required parameter: channel_id"}

    try:
        row = db.execute(
            text("SELECT id, platform, config FROM channel_connections WHERE id = :id AND workspace_id = :ws_id"),
            {"id": str(channel_id), "ws_id": str(workspace_id)},
        ).fetchone()
        if not row:
            return {"success": False, "error": "Channel connection not found"}

        from channels.manager import get_channel_manager
        manager = get_channel_manager()
        await manager.start_adapter(str(channel_id), str(workspace_id), row.platform, row.config or {})

        db.execute(
            text("UPDATE channel_connections SET status = 'active', updated_at = NOW() WHERE id = :id"),
            {"id": str(channel_id)},
        )
        db.commit()
        return {"success": True, "status": "started", "channel_id": str(channel_id)}
    except Exception as exc:
        db.rollback()
        logger.error("[channels] start_channel failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}


async def stop_channel(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Stop the adapter for a channel connection (mirrors POST /api/channels/{id}/stop)."""
    channel_id = params.get("channel_id")
    if not channel_id:
        return {"success": False, "error": "Missing required parameter: channel_id"}

    try:
        row = db.execute(
            text("SELECT id FROM channel_connections WHERE id = :id AND workspace_id = :ws_id"),
            {"id": str(channel_id), "ws_id": str(workspace_id)},
        ).fetchone()
        if not row:
            return {"success": False, "error": "Channel connection not found"}

        from channels.manager import get_channel_manager
        manager = get_channel_manager()
        await manager.stop_adapter(str(channel_id))

        db.execute(
            text("UPDATE channel_connections SET status = 'inactive', updated_at = NOW() WHERE id = :id"),
            {"id": str(channel_id)},
        )
        db.commit()
        return {"success": True, "status": "stopped", "channel_id": str(channel_id)}
    except Exception as exc:
        db.rollback()
        logger.error("[channels] stop_channel failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}
