"""
Channel Connection API endpoints (PRD-55: Autonomous Assistant Platform).

CRUD for channel connections (Telegram, Slack, Discord) plus adapter
lifecycle management and messaging analytics.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import text

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/channels", tags=["Channels"])


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class ChannelConnectionCreate(BaseModel):
    workspace_id: str
    platform: str = Field(..., description="telegram | slack | discord | whatsapp | teams | google_chat | signal | imessage | irc | matrix | line")
    config: Dict[str, Any] = Field(default_factory=dict)
    default_agent_id: Optional[int] = None


class ChannelConnectionUpdate(BaseModel):
    config: Optional[Dict[str, Any]] = None
    default_agent_id: Optional[int] = None
    status: Optional[str] = None


class ChannelConnectionResponse(BaseModel):
    id: str
    workspace_id: str
    platform: str
    config: Dict[str, Any]
    status: str
    default_agent_id: Optional[int]
    message_count: int
    last_activity_at: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]


def _row_to_response(conn: Any) -> Dict[str, Any]:
    return {
        "id": str(conn.id),
        "workspace_id": str(conn.workspace_id),
        "platform": conn.platform,
        "config": _sanitize_config(conn.config or {}),
        "status": conn.status,
        "default_agent_id": conn.default_agent_id,
        "message_count": conn.message_count or 0,
        "last_activity_at": conn.last_activity_at.isoformat() if conn.last_activity_at else None,
        "created_at": conn.created_at.isoformat() if conn.created_at else None,
        "updated_at": conn.updated_at.isoformat() if conn.updated_at else None,
    }


def _sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Mask sensitive tokens in the config for API responses."""
    _SENSITIVE_KEYS = (
        "bot_token", "signing_secret", "app_token", "access_token",
        "app_password", "password", "channel_access_token", "channel_secret",
        "service_account_json",
    )
    masked = dict(config)
    for key in _SENSITIVE_KEYS:
        if key in masked and masked[key]:
            val = str(masked[key])
            masked[key] = val[:6] + "..." + val[-4:] if len(val) > 10 else "***"
    return masked


# ---------------------------------------------------------------------------
# Platform-specific config validation
# ---------------------------------------------------------------------------

_REQUIRED_CONFIG_KEYS = {
    "telegram": ["bot_token"],
    "slack": ["bot_token", "app_token"],
    "discord": ["bot_token"],
    "whatsapp": ["access_token", "phone_number_id"],
    "teams": ["app_id", "app_password"],
    "google_chat": ["service_account_json"],
    "signal": ["phone_number"],
    "imessage": ["server_url", "password"],
    "irc": ["server", "nickname"],
    "matrix": ["homeserver", "user_id", "access_token"],
    "line": ["channel_access_token"],
}


def _validate_config(platform: str, config: Dict[str, Any]) -> None:
    required = _REQUIRED_CONFIG_KEYS.get(platform, [])
    missing = [k for k in required if not config.get(k)]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required config keys for {platform}: {missing}",
        )


# ---------------------------------------------------------------------------
# CRUD endpoints
# ---------------------------------------------------------------------------


@router.get("/")
async def list_channels(
    workspace_id: str = Query(..., description="Workspace UUID"),
) -> Dict[str, Any]:
    """List all channel connections for a workspace."""
    from core.database.database import SessionLocal
    from core.models.channels import ChannelConnection

    db = SessionLocal()
    try:
        connections = (
            db.query(ChannelConnection)
            .filter(ChannelConnection.workspace_id == workspace_id)
            .order_by(ChannelConnection.created_at.desc())
            .all()
        )
        return {
            "ok": True,
            "connections": [_row_to_response(c) for c in connections],
        }
    finally:
        db.close()


@router.post("/")
async def create_channel(body: ChannelConnectionCreate) -> Dict[str, Any]:
    """Create a new channel connection."""
    platform = body.platform.lower()
    if platform not in _REQUIRED_CONFIG_KEYS:
        supported = ", ".join(sorted(_REQUIRED_CONFIG_KEYS.keys()))
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported platform '{body.platform}'. Must be one of: {supported}",
        )

    _validate_config(platform, body.config)

    from core.database.database import SessionLocal
    from core.models.channels import ChannelConnection

    db = SessionLocal()
    try:
        conn = ChannelConnection(
            id=uuid4(),
            workspace_id=body.workspace_id,
            platform=platform,
            config=body.config,
            status="disconnected",
            default_agent_id=body.default_agent_id,
        )
        db.add(conn)
        db.commit()
        db.refresh(conn)
        return {"ok": True, "connection": _row_to_response(conn)}
    finally:
        db.close()


@router.put("/{channel_id}")
async def update_channel(
    channel_id: str, body: ChannelConnectionUpdate
) -> Dict[str, Any]:
    """Update a channel connection."""
    from core.database.database import SessionLocal
    from core.models.channels import ChannelConnection

    db = SessionLocal()
    try:
        conn = (
            db.query(ChannelConnection)
            .filter(ChannelConnection.id == channel_id)
            .first()
        )
        if not conn:
            raise HTTPException(status_code=404, detail="Channel connection not found")

        if body.config is not None:
            _validate_config(conn.platform, body.config)
            conn.config = body.config
        if body.default_agent_id is not None:
            conn.default_agent_id = body.default_agent_id
        if body.status is not None:
            conn.status = body.status

        db.commit()
        db.refresh(conn)
        return {"ok": True, "connection": _row_to_response(conn)}
    finally:
        db.close()


@router.delete("/{channel_id}")
async def delete_channel(channel_id: str) -> Dict[str, Any]:
    """Delete a channel connection (stops adapter first if running)."""
    from core.database.database import SessionLocal
    from core.models.channels import ChannelConnection

    # Stop adapter if running
    try:
        from channels.manager import get_channel_manager

        mgr = get_channel_manager()
        await mgr.stop_adapter(channel_id)
    except Exception:
        pass

    db = SessionLocal()
    try:
        conn = (
            db.query(ChannelConnection)
            .filter(ChannelConnection.id == channel_id)
            .first()
        )
        if not conn:
            raise HTTPException(status_code=404, detail="Channel connection not found")

        db.delete(conn)
        db.commit()
        return {"ok": True, "deleted": channel_id}
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Adapter lifecycle
# ---------------------------------------------------------------------------


@router.post("/{channel_id}/test")
async def test_channel(channel_id: str) -> Dict[str, Any]:
    """Test a channel connection without starting the full adapter."""
    from core.database.database import SessionLocal
    from core.models.channels import ChannelConnection
    from channels.manager import ChannelManager

    db = SessionLocal()
    try:
        conn = (
            db.query(ChannelConnection)
            .filter(ChannelConnection.id == channel_id)
            .first()
        )
        if not conn:
            raise HTTPException(status_code=404, detail="Channel connection not found")

        adapter = ChannelManager._create_adapter(conn)
        try:
            info = await adapter.test_connection()
            return {"ok": True, "connection_info": info}
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc))
    finally:
        db.close()


@router.post("/{channel_id}/start")
async def start_channel(channel_id: str) -> Dict[str, Any]:
    """Start the adapter for a channel connection."""
    from channels.manager import get_channel_manager

    mgr = get_channel_manager()
    try:
        await mgr.start_adapter(channel_id)
        return {"ok": True, "status": "started"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/{channel_id}/stop")
async def stop_channel(channel_id: str) -> Dict[str, Any]:
    """Stop the adapter for a channel connection."""
    from channels.manager import get_channel_manager

    mgr = get_channel_manager()
    try:
        await mgr.stop_adapter(channel_id)
        return {"ok": True, "status": "stopped"}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------


@router.get("/analytics")
async def get_channel_analytics(
    workspace_id: str = Query(..., description="Workspace UUID"),
) -> Dict[str, Any]:
    """Message counts by source from routing_decisions table."""
    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        rows = db.execute(
            text(
                "SELECT source, COUNT(*) as count "
                "FROM routing_decisions "
                "WHERE workspace_id = :wid::uuid "
                "GROUP BY source "
                "ORDER BY count DESC"
            ),
            {"wid": workspace_id},
        ).fetchall()

        by_source = {r[0]: r[1] for r in rows}

        # Also get channel_connections stats
        channel_rows = db.execute(
            text(
                "SELECT platform, SUM(message_count) as total "
                "FROM channel_connections "
                "WHERE workspace_id = :wid::uuid "
                "GROUP BY platform"
            ),
            {"wid": workspace_id},
        ).fetchall()

        by_platform = {r[0]: int(r[1] or 0) for r in channel_rows}

        return {
            "ok": True,
            "by_source": by_source,
            "by_platform": by_platform,
        }
    finally:
        db.close()
