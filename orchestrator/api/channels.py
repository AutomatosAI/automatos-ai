"""
Channel Management API (PRD-55 US-023)
=======================================

CRUD endpoints for managing channel connections (Telegram, Slack, Discord).
"""

import logging
from typing import Any, Dict, List
from uuid import uuid4

from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from core.database.database import get_db
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/channels", tags=["channels"])

_SUPPORTED_PLATFORMS = {"telegram", "slack", "discord"}

_REQUIRED_CONFIG = {
    "telegram": ["bot_token"],
    "slack": ["bot_token", "signing_secret"],
    "discord": ["bot_token"],
}


@router.get("")
async def list_channels(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List all channel connections for the current workspace."""
    rows = db.execute(
        text("""
            SELECT id, platform, status, metadata, default_agent_id, message_count, last_activity_at, created_at
            FROM channel_connections
            WHERE workspace_id = :ws_id
            ORDER BY created_at DESC
        """),
        {"ws_id": str(ctx.workspace_id)},
    ).fetchall()

    return [
        {
            "id": str(r.id),
            "platform": r.platform,
            "status": r.status,
            "metadata": r.metadata or {},
            "default_agent_id": r.default_agent_id,
            "message_count": r.message_count or 0,
            "last_activity_at": r.last_activity_at.isoformat() if r.last_activity_at else None,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


@router.post("")
async def create_channel(
    payload: Dict[str, Any] = Body(...),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Create a new channel connection."""
    platform = payload.get("platform", "").lower()
    config = payload.get("config", {})
    default_agent_id = payload.get("default_agent_id")

    if platform not in _SUPPORTED_PLATFORMS:
        raise HTTPException(400, f"Platform must be one of {_SUPPORTED_PLATFORMS}")

    # Validate required config fields
    required = _REQUIRED_CONFIG.get(platform, [])
    missing = [f for f in required if not config.get(f)]
    if missing:
        raise HTTPException(400, f"Missing required config fields for {platform}: {missing}")

    conn_id = uuid4()
    db.execute(
        text("""
            INSERT INTO channel_connections (id, workspace_id, platform, config, status, default_agent_id, created_at, updated_at)
            VALUES (:id, :ws_id, :platform, :config, 'inactive', :agent_id, NOW(), NOW())
        """),
        {
            "id": str(conn_id),
            "ws_id": str(ctx.workspace_id),
            "platform": platform,
            "config": __import__('json').dumps(config),
            "agent_id": default_agent_id,
        },
    )
    db.commit()

    logger.info("Created channel connection %s (%s) for workspace %s", conn_id, platform, ctx.workspace_id)
    return {"id": str(conn_id), "platform": platform, "status": "inactive"}


@router.put("/{channel_id}")
async def update_channel(
    channel_id: str,
    payload: Dict[str, Any] = Body(...),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Update a channel connection's config."""
    row = db.execute(
        text("SELECT id FROM channel_connections WHERE id = :id AND workspace_id = :ws_id"),
        {"id": channel_id, "ws_id": str(ctx.workspace_id)},
    ).fetchone()

    if not row:
        raise HTTPException(404, "Channel connection not found")

    updates = []
    params: Dict[str, Any] = {"id": channel_id}

    if "config" in payload:
        updates.append("config = :config")
        params["config"] = __import__('json').dumps(payload["config"])
    if "default_agent_id" in payload:
        updates.append("default_agent_id = :agent_id")
        params["agent_id"] = payload["default_agent_id"]

    if updates:
        updates.append("updated_at = NOW()")
        db.execute(
            text(f"UPDATE channel_connections SET {', '.join(updates)} WHERE id = :id"),
            params,
        )
        db.commit()

    return {"status": "updated"}


@router.delete("/{channel_id}")
async def delete_channel(
    channel_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Delete a channel connection (stops adapter if running)."""
    row = db.execute(
        text("SELECT id, platform FROM channel_connections WHERE id = :id AND workspace_id = :ws_id"),
        {"id": channel_id, "ws_id": str(ctx.workspace_id)},
    ).fetchone()

    if not row:
        raise HTTPException(404, "Channel connection not found")

    # Stop adapter if running
    try:
        from channels.manager import get_channel_manager
        manager = get_channel_manager()
        await manager.stop_adapter(channel_id)
    except Exception:
        pass

    db.execute(
        text("DELETE FROM channel_connections WHERE id = :id"),
        {"id": channel_id},
    )
    db.commit()

    logger.info("Deleted channel connection %s", channel_id)
    return {"status": "deleted"}


@router.post("/{channel_id}/test")
async def test_channel(
    channel_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Test a channel connection by pinging the platform API."""
    row = db.execute(
        text("SELECT platform, config FROM channel_connections WHERE id = :id AND workspace_id = :ws_id"),
        {"id": channel_id, "ws_id": str(ctx.workspace_id)},
    ).fetchone()

    if not row:
        raise HTTPException(404, "Channel connection not found")

    platform = row.platform
    config = row.config or {}

    try:
        if platform == "telegram":
            import requests
            token = config.get("bot_token", "")
            resp = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10)
            if resp.status_code == 200:
                bot_info = resp.json().get("result", {})
                return {"status": "connected", "bot_name": bot_info.get("username")}
            return {"status": "error", "detail": f"Telegram API returned {resp.status_code}"}

        elif platform == "slack":
            import requests
            token = config.get("bot_token", "")
            resp = requests.post(
                "https://slack.com/api/auth.test",
                headers={"Authorization": f"Bearer {token}"},
                timeout=10,
            )
            data = resp.json()
            if data.get("ok"):
                return {"status": "connected", "team": data.get("team"), "bot_user": data.get("user")}
            return {"status": "error", "detail": data.get("error", "Unknown error")}

        elif platform == "discord":
            import requests
            token = config.get("bot_token", "")
            resp = requests.get(
                "https://discord.com/api/v10/users/@me",
                headers={"Authorization": f"Bot {token}"},
                timeout=10,
            )
            if resp.status_code == 200:
                user = resp.json()
                return {"status": "connected", "bot_name": user.get("username")}
            return {"status": "error", "detail": f"Discord API returned {resp.status_code}"}

        return {"status": "error", "detail": f"Unknown platform: {platform}"}

    except Exception as e:
        return {"status": "error", "detail": str(e)}


@router.post("/{channel_id}/start")
async def start_channel(
    channel_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Start the adapter for this channel connection."""
    row = db.execute(
        text("SELECT id, platform, config FROM channel_connections WHERE id = :id AND workspace_id = :ws_id"),
        {"id": channel_id, "ws_id": str(ctx.workspace_id)},
    ).fetchone()

    if not row:
        raise HTTPException(404, "Channel connection not found")

    try:
        from channels.manager import get_channel_manager
        manager = get_channel_manager()
        await manager.start_adapter(channel_id, str(ctx.workspace_id), row.platform, row.config or {})

        db.execute(
            text("UPDATE channel_connections SET status = 'active', updated_at = NOW() WHERE id = :id"),
            {"id": channel_id},
        )
        db.commit()

        return {"status": "started"}
    except Exception as e:
        logger.error("Failed to start channel %s: %s", channel_id, e)
        raise HTTPException(500, f"Failed to start adapter: {str(e)}")


@router.post("/{channel_id}/stop")
async def stop_channel(
    channel_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Stop the adapter for this channel connection."""
    row = db.execute(
        text("SELECT id FROM channel_connections WHERE id = :id AND workspace_id = :ws_id"),
        {"id": channel_id, "ws_id": str(ctx.workspace_id)},
    ).fetchone()

    if not row:
        raise HTTPException(404, "Channel connection not found")

    try:
        from channels.manager import get_channel_manager
        manager = get_channel_manager()
        await manager.stop_adapter(channel_id)

        db.execute(
            text("UPDATE channel_connections SET status = 'inactive', updated_at = NOW() WHERE id = :id"),
            {"id": channel_id},
        )
        db.commit()

        return {"status": "stopped"}
    except Exception as e:
        logger.error("Failed to stop channel %s: %s", channel_id, e)
        raise HTTPException(500, f"Failed to stop adapter: {str(e)}")


# ── Channel Analytics ──────────────────────────────────────────────

@router.get("/analytics")
async def get_channel_analytics(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Return channel message stats for analytics dashboard (US-026)."""
    from datetime import datetime, timedelta

    ws_id = str(ctx.workspace_id)
    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    try:
        # Messages by source from routing_decisions
        by_source = db.execute(
            text("""
                SELECT source_channel, COUNT(*) as count
                FROM routing_decisions
                WHERE workspace_id = :ws_id AND created_at >= :start
                GROUP BY source_channel
            """),
            {"ws_id": ws_id, "start": today_start},
        ).fetchall()

        # Total messages from channel connections
        channel_stats = db.execute(
            text("""
                SELECT platform, SUM(message_count) as total, MAX(last_activity_at) as last_activity
                FROM channel_connections
                WHERE workspace_id = :ws_id
                GROUP BY platform
            """),
            {"ws_id": ws_id},
        ).fetchall()

        return {
            "today_by_source": {r.source_channel: r.count for r in by_source} if by_source else {},
            "channels": [
                {
                    "platform": r.platform,
                    "total_messages": r.total or 0,
                    "last_activity": r.last_activity.isoformat() if r.last_activity else None,
                }
                for r in channel_stats
            ] if channel_stats else [],
        }
    except Exception as e:
        logger.error("Failed to get channel analytics: %s", e)
        return {"today_by_source": {}, "channels": []}
