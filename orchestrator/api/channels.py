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

_SUPPORTED_PLATFORMS = {
    "telegram", "slack", "discord", "teams", "google_chat",
    "signal", "imessage", "irc", "matrix", "line", "whatsapp",
    "webhook",
}


async def _ping_platform(platform: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Hit the platform's identity endpoint to verify the stored
    credentials work. Returns the same shape ``/test`` returns to the
    dashboard so callers can pass it through.

    Pure verification — no adapter lifecycle side effects.
    """
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            if platform == "telegram":
                token = config.get("bot_token", "")
                resp = await client.get(f"https://api.telegram.org/bot{token}/getMe")
                if resp.status_code == 200:
                    bot_info = resp.json().get("result", {})
                    return {"status": "connected", "bot_name": bot_info.get("username")}
                # 404 is the symptom of a bot_token missing the "<bot_id>:" prefix —
                # surface the likely cause so the dashboard error is actionable.
                if resp.status_code == 404:
                    return {
                        "status": "error",
                        "detail": (
                            "Telegram returned 404 — the bot_token is likely missing the "
                            "leading '<bot_id>:' prefix. Paste the full token from "
                            "@BotFather, e.g. '1234567890:AAF…'."
                        ),
                    }
                return {"status": "error", "detail": f"Telegram API returned {resp.status_code}"}

            if platform == "slack":
                token = config.get("bot_token", "")
                resp = await client.post(
                    "https://slack.com/api/auth.test",
                    headers={"Authorization": f"Bearer {token}"},
                )
                data = resp.json()
                if data.get("ok"):
                    return {"status": "connected", "team": data.get("team"), "bot_user": data.get("user")}
                return {"status": "error", "detail": data.get("error", "Unknown error")}

            if platform == "discord":
                token = config.get("bot_token", "")
                resp = await client.get(
                    "https://discord.com/api/v10/users/@me",
                    headers={"Authorization": f"Bot {token}"},
                )
                if resp.status_code == 200:
                    user = resp.json()
                    return {"status": "connected", "bot_name": user.get("username")}
                return {"status": "error", "detail": f"Discord API returned {resp.status_code}"}

        return {"status": "error", "detail": f"No verifier for platform {platform!r}"}
    except Exception as exc:
        logger.error("Platform ping failed for %s: %s", platform, exc)
        return {"status": "error", "detail": "Connection test failed"}


def _mark_active(db: Session, channel_id: str) -> None:
    """Flip the row to ``status='active'``. Best-effort — never raises."""
    try:
        db.execute(
            text("UPDATE channel_connections SET status = 'active', updated_at = NOW() WHERE id = :id"),
            {"id": channel_id},
        )
        db.commit()
    except Exception:
        logger.exception("Failed to mark channel %s active", channel_id)
        db.rollback()

_REQUIRED_CONFIG = {
    "telegram": ["bot_token"],
    "slack": ["bot_token", "signing_secret"],
    "discord": ["bot_token"],
    "teams": ["app_id", "app_password"],
    "google_chat": ["service_account_key"],
    "signal": ["phone_number"],
    "imessage": ["apple_id"],
    "irc": ["server", "channel", "nickname"],
    "matrix": ["homeserver_url", "access_token"],
    "line": ["channel_access_token", "channel_secret"],
    "whatsapp": ["phone_number_id", "access_token"],
}


@router.get("")
async def list_channels(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """List all channel connections for the current workspace.

    The ``status`` column is reconciled against ``ChannelManager`` at
    read time — a row that reads ``inactive`` in the DB but whose
    adapter is actually loaded and running (common after orchestrator
    restarts, or for older rows that pre-date the auto-status-update
    code) is reported as ``active`` here so the dashboard reflects
    reality. The DB itself is also corrected so subsequent reads are
    cheap.
    """
    rows = db.execute(
        text("""
            SELECT id, platform, status, metadata, default_agent_id, message_count, last_activity_at, created_at
            FROM channel_connections
            WHERE workspace_id = :ws_id
            ORDER BY created_at DESC
        """),
        {"ws_id": str(ctx.workspace_id)},
    ).fetchall()

    try:
        from channels.manager import get_channel_manager
        manager = get_channel_manager()
    except Exception:
        manager = None

    out: List[Dict[str, Any]] = []
    rows_to_repair: List[str] = []
    for r in rows:
        effective_status = r.status
        if manager is not None and manager.is_running(str(r.id)):
            if effective_status != "active":
                rows_to_repair.append(str(r.id))
            effective_status = "active"
        out.append(
            {
                "id": str(r.id),
                "platform": r.platform,
                "status": effective_status,
                "metadata": r.metadata or {},
                "default_agent_id": r.default_agent_id,
                "message_count": r.message_count or 0,
                "last_activity_at": r.last_activity_at.isoformat() if r.last_activity_at else None,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
        )

    if rows_to_repair:
        # Drift-repair: silently update the DB to match the running
        # state. Best-effort — list endpoints must not fail on a write.
        try:
            db.execute(
                text(
                    "UPDATE channel_connections SET status = 'active', updated_at = NOW() "
                    "WHERE id = ANY(:ids)"
                ),
                {"ids": rows_to_repair},
            )
            db.commit()
            logger.info(
                "list_channels: repaired stale status on %d row(s) (ws=%s)",
                len(rows_to_repair), ctx.workspace_id,
            )
        except Exception:
            logger.exception("list_channels: failed to repair stale status rows")
            db.rollback()

    return out


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

    # Verify the credentials with a platform-API ping. If the token is
    # valid, mark the row ``active`` — independent of whether we can
    # start a polling adapter on top. Many users run their bot in
    # webhook mode and don't want a polling adapter at all; previously
    # those rows stayed ``inactive`` forever.
    ping = await _ping_platform(platform, config)
    if ping.get("status") == "connected":
        _mark_active(db, str(conn_id))

    # Best-effort polling-adapter start, for users who actually want
    # polling. Doesn't gate the response status — that's now driven by
    # the ping result. Failures are logged and swallowed.
    try:
        from channels.manager import get_channel_manager
        manager = get_channel_manager()
        await manager.start_adapter(str(conn_id), str(ctx.workspace_id), platform, config)
    except Exception as exc:
        logger.warning("Polling adapter failed to start for new channel %s: %s", conn_id, exc)

    final_status = "active" if ping.get("status") == "connected" else "inactive"
    return {
        "id": str(conn_id),
        "platform": platform,
        "status": final_status,
        "test": ping,
    }


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
    except Exception as exc:
        logger.warning("Could not stop adapter for channel %s: %s", channel_id, exc)

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
    """Test a channel connection by pinging the platform API.

    On success the row is marked ``active`` so the dashboard reflects
    the merchant's confirmation that the credentials work. Does NOT
    start a polling adapter — that would conflict with a bot already
    running in webhook mode. The separate ``/start`` endpoint handles
    polling-adapter lifecycle when that's what the merchant wants.
    """
    row = db.execute(
        text("SELECT platform, config FROM channel_connections WHERE id = :id AND workspace_id = :ws_id"),
        {"id": channel_id, "ws_id": str(ctx.workspace_id)},
    ).fetchone()

    if not row:
        raise HTTPException(404, "Channel connection not found")

    result = await _ping_platform(row.platform, row.config or {})
    if result.get("status") == "connected":
        _mark_active(db, channel_id)
    return result


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
        raise HTTPException(500, "Internal server error")


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
        raise HTTPException(500, "Internal server error")


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
        # Messages by source from routing_decisions (normalize to lowercase)
        by_source = db.execute(
            text("""
                SELECT LOWER(source) as source, COUNT(*) as count
                FROM routing_decisions
                WHERE workspace_id = :ws_id AND created_at >= :start
                GROUP BY LOWER(source)
            """),
            {"ws_id": ws_id, "start": today_start},
        ).fetchall()

        # Total messages from channel connections (normalize platform to lowercase)
        channel_stats = db.execute(
            text("""
                SELECT LOWER(platform) as platform, SUM(message_count) as total, MAX(last_activity_at) as last_activity
                FROM channel_connections
                WHERE workspace_id = :ws_id
                GROUP BY LOWER(platform)
            """),
            {"ws_id": ws_id},
        ).fetchall()

        return {
            "today_by_source": {r.source: r.count for r in by_source} if by_source else {},
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
