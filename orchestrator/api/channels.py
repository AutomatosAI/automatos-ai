"""
Channel Management API (PRD-55 US-023, PRD-008-A.4)
====================================================

CRUD endpoints for managing channel connections. Per-platform behaviour
lives in ``channels/drivers/``; this module is platform-agnostic.

The driver tells us what config the merchant must paste, what modes it
supports (webhook / polling), and how to verify, send, install/uninstall
webhook, and start/stop polling.
"""

import json as _json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from config import config
from core.database.database import get_db
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext
from core.auth.workspace_permission import require_workspace_permission

from channels.drivers import (
    ConnectivityMode,
    DriverNotConfigured,
    UnknownPlatform,
    VerifyResult,
    get_driver,
    list_platforms,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/channels", tags=["channels"])

# Public-API host used to build inbound webhook URLs. Override via env
# PUBLIC_API_HOST (e.g. ``api.staging.automatos.app``) — default matches
# production. (PRD-142 W3-S5 / G7 — env via config.)
_PUBLIC_API_HOST = config.PUBLIC_API_HOST

# Kept so callers passing a platform not in the driver registry still
# get a clear "unsupported" error rather than a 500. The driver
# registry is the source of truth for what's wireable end-to-end.
_SUPPORTED_PLATFORMS = {
    "telegram", "slack", "discord", "teams", "google_chat",
    "signal", "imessage", "irc", "matrix", "line", "whatsapp",
    "webhook",
}


# ---------------------------------------------------------------------------
# Driver-mediated helpers
# ---------------------------------------------------------------------------

def _webhook_url_for(db: Session, workspace_id) -> Optional[str]:
    """Build the inbound webhook URL the platform should POST to. Uses
    ``workspaces.webhook_key`` (URL-as-secret) so the URL itself
    authenticates the inbound request. None if no key is provisioned —
    caller decides whether to fail or skip install."""
    row = db.execute(
        text("SELECT webhook_key FROM workspaces WHERE id = :ws"),
        {"ws": str(workspace_id)},
    ).fetchone()
    if not row or not row.webhook_key:
        return None
    return f"https://{_PUBLIC_API_HOST}/api/webhooks/ws/{row.webhook_key}"


def _verify_result_dict(result: VerifyResult) -> Dict[str, Any]:
    """Marshal a ``VerifyResult`` into the dict shape the dashboard
    /test handler consumes."""
    if result.ok:
        return {
            "status": "connected",
            "identity": result.identity,
            "bot_name": (result.metadata or {}).get("username"),
            "team": (result.metadata or {}).get("team"),
            "metadata": dict(result.metadata or {}),
        }
    return {"status": "error", "detail": result.error or "Unknown error"}


def _save_verify_outcome(
    db: Session,
    channel_id: str,
    result: VerifyResult,
    *,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist the outcome of a verify on the row: status,
    last_verified, last_error, and any metadata the driver returned.
    Best-effort — never raises."""
    try:
        now = datetime.now(timezone.utc)
        new_status = "active" if result.ok else "error"
        new_error = None if result.ok else (result.error or "verify failed")

        # Merge new metadata into existing
        meta_row = db.execute(
            text("SELECT metadata FROM channel_connections WHERE id = :id"),
            {"id": channel_id},
        ).fetchone()
        existing_meta: Dict[str, Any] = {}
        if meta_row and meta_row.metadata:
            existing_meta = meta_row.metadata if isinstance(meta_row.metadata, dict) else {}
        if result.metadata:
            existing_meta.update(dict(result.metadata))
        if extra_metadata:
            existing_meta.update(extra_metadata)

        db.execute(
            text(
                """
                UPDATE channel_connections
                SET status = :status,
                    last_verified = CASE WHEN :ok THEN :now ELSE last_verified END,
                    last_error = :err,
                    metadata = CAST(:meta AS JSON),
                    updated_at = NOW()
                WHERE id = :id
                """
            ),
            {
                "status": new_status,
                "ok": result.ok,
                "now": now,
                "err": new_error,
                "meta": _json.dumps(existing_meta),
                "id": channel_id,
            },
        )
        db.commit()
    except Exception:
        logger.exception("Failed to persist verify outcome for channel %s", channel_id)
        db.rollback()


async def _ping_platform_legacy(platform: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Legacy inline pinger — retained ONLY for the route handlers that
    haven't been ported to the driver interface yet (none should
    remain). New code: ``get_driver(platform)().verify(...)``.
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
            SELECT id, platform, status, mode, webhook_url, last_verified, last_error,
                   metadata, default_agent_id, message_count, last_activity_at, created_at
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
                "mode": r.mode or "webhook",
                "webhook_url": r.webhook_url,
                "last_verified": r.last_verified.isoformat() if r.last_verified else None,
                "last_error": r.last_error,
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


@router.post("", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
async def create_channel(
    payload: Dict[str, Any] = Body(...),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Create a new channel connection."""
    try:
        return await connect_channel_for_workspace(
            db,
            workspace_id=str(ctx.workspace_id),
            platform=payload.get("platform", ""),
            config=payload.get("config", {}),
            default_agent_id=payload.get("default_agent_id"),
            mode=payload.get("mode"),
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc))


async def connect_channel_for_workspace(
    db: Session,
    workspace_id: str,
    platform: str,
    config: Dict[str, Any],
    default_agent_id: Optional[str] = None,
    mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Create, verify and activate a channel connection — the single connect flow.

    Shared by ``POST /api/channels`` and the ``platform_connect_channel`` tool
    (PRD-143 S10), so the driver-mediated verify + install_webhook/start_polling
    behaviour cannot drift between the dashboard and Auto. Raises ``ValueError``
    on an unsupported platform or missing required config fields — the router
    maps that to HTTP 400.
    """
    platform = (platform or "").lower()
    if platform not in _SUPPORTED_PLATFORMS:
        raise ValueError(f"Platform must be one of {_SUPPORTED_PLATFORMS}")

    # Validate required config fields
    required = _REQUIRED_CONFIG.get(platform, [])
    missing = [f for f in required if not config.get(f)]
    if missing:
        raise ValueError(f"Missing required config fields for {platform}: {missing}")

    conn_id = uuid4()
    db.execute(
        text("""
            INSERT INTO channel_connections (id, workspace_id, platform, config, status, default_agent_id, created_at, updated_at)
            VALUES (:id, :ws_id, :platform, :config, 'inactive', :agent_id, NOW(), NOW())
        """),
        {
            "id": str(conn_id),
            "ws_id": str(workspace_id),
            "platform": platform,
            "config": _json.dumps(config),
            "agent_id": default_agent_id,
        },
    )
    db.commit()

    logger.info("Created channel connection %s (%s) for workspace %s", conn_id, platform, workspace_id)

    # ------------------------------------------------------------------
    # PRD-008-A.4 — driver-mediated verify + (install_webhook | start_polling)
    # ------------------------------------------------------------------
    try:
        driver = get_driver(platform)()
    except UnknownPlatform:
        # Platform is in _SUPPORTED_PLATFORMS but has no driver yet —
        # row is saved, but caller gets a clear note.
        return {
            "id": str(conn_id),
            "platform": platform,
            "status": "inactive",
            "test": {
                "status": "error",
                "detail": f"No driver registered for {platform!r} yet — row saved",
            },
        }

    requested_mode = str(mode or driver.default_mode().value).lower()
    if not driver.supports(ConnectivityMode(requested_mode) if requested_mode in {"webhook", "polling"} else driver.default_mode()):
        # Fall back to the driver's preferred mode rather than erroring.
        requested_mode = driver.default_mode().value

    verify_result = await driver.verify(workspace_id=str(workspace_id), config=config)

    # Persist the mode on the row now that we know which one we'll use.
    db.execute(
        text("UPDATE channel_connections SET mode = :mode WHERE id = :id"),
        {"mode": requested_mode, "id": str(conn_id)},
    )
    db.commit()

    # If verify failed, persist the error and bail — no point installing
    # webhooks or starting polling against creds we know don't work.
    _save_verify_outcome(db, str(conn_id), verify_result)
    if not verify_result.ok:
        return {
            "id": str(conn_id),
            "platform": platform,
            "status": "error",
            "mode": requested_mode,
            "test": _verify_result_dict(verify_result),
        }

    webhook_url: Optional[str] = None
    install_result: Optional[VerifyResult] = None

    if requested_mode == "webhook":
        webhook_url = _webhook_url_for(db, workspace_id)
        if webhook_url:
            try:
                install_result = await driver.install_webhook(
                    workspace_id=str(workspace_id),
                    config=config,
                    webhook_url=webhook_url,
                )
            except NotImplementedError:
                install_result = VerifyResult(ok=True, identity=webhook_url)
            except Exception as exc:  # noqa: BLE001
                logger.warning("install_webhook failed for %s: %s", platform, exc)
                install_result = VerifyResult(ok=False, error=str(exc))
            if install_result and install_result.ok:
                db.execute(
                    text(
                        "UPDATE channel_connections SET webhook_url = :url, updated_at = NOW() "
                        "WHERE id = :id"
                    ),
                    {"url": webhook_url, "id": str(conn_id)},
                )
                db.commit()
            elif install_result:
                # Webhook install failed — keep status=active (verify was ok)
                # but surface the install error so the merchant knows
                # they need to set the URL manually.
                _save_verify_outcome(
                    db, str(conn_id),
                    VerifyResult(
                        ok=False,
                        error=f"verify ok but webhook install failed: {install_result.error}",
                    ),
                )
    elif requested_mode == "polling":
        try:
            started = await driver.start_polling(
                connection_id=str(conn_id),
                workspace_id=str(workspace_id),
                config=config,
            )
            if not started:
                logger.info(
                    "Polling not started for %s — likely optional dep missing", platform,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("start_polling failed for %s: %s", platform, exc)

    return {
        "id": str(conn_id),
        "platform": platform,
        "status": "active",
        "mode": requested_mode,
        "webhook_url": webhook_url,
        "test": _verify_result_dict(verify_result),
        "install": (
            _verify_result_dict(install_result)
            if install_result is not None else None
        ),
    }


@router.put("/{channel_id}", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
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


@router.delete("/{channel_id}", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
async def delete_channel(
    channel_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Delete a channel connection.

    Calls the driver's ``uninstall_webhook`` first (so the platform
    stops POSTing to us) and stops any running polling adapter, then
    removes the row. Both side-effects are best-effort — the row is
    always deleted even if the driver call fails.
    """
    row = db.execute(
        text(
            "SELECT id, platform, mode, config FROM channel_connections "
            "WHERE id = :id AND workspace_id = :ws_id"
        ),
        {"id": channel_id, "ws_id": str(ctx.workspace_id)},
    ).fetchone()

    if not row:
        raise HTTPException(404, "Channel connection not found")

    config = row.config or {}
    if isinstance(config, str):
        try:
            config = _json.loads(config)
        except Exception:
            config = {}

    try:
        driver = get_driver(row.platform)()
    except UnknownPlatform:
        driver = None

    if driver is not None:
        if (row.mode or "webhook") == "webhook":
            try:
                await driver.uninstall_webhook(
                    workspace_id=str(ctx.workspace_id), config=config,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("uninstall_webhook failed for %s: %s", row.platform, exc)
        else:
            try:
                await driver.stop_polling(connection_id=channel_id)
            except Exception as exc:  # noqa: BLE001
                logger.warning("stop_polling failed for %s: %s", row.platform, exc)

    # Belt-and-braces: legacy adapter map may still hold a reference.
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


# ---------------------------------------------------------------------------
# Driver introspection — dashboard reads this to render the connect form.
# ---------------------------------------------------------------------------

@router.get("/platforms")
async def list_supported_platforms() -> List[Dict[str, Any]]:
    """Return every registered driver with the config fields the connect
    form needs to collect and the connectivity modes it supports."""
    out: List[Dict[str, Any]] = []
    for platform in list_platforms():
        try:
            driver = get_driver(platform)()
        except UnknownPlatform:
            continue
        out.append({
            "platform": platform,
            "display_name": driver.display_name or platform.title(),
            "modes": [m.value for m in driver.supported_modes],
            "default_mode": driver.default_mode().value,
            "required_config": [
                {"key": k, "label": label, "placeholder": placeholder}
                for k, label, placeholder in driver.required_config
            ],
            "optional_config": [
                {"key": k, "label": label, "placeholder": placeholder}
                for k, label, placeholder in driver.optional_config
            ],
        })
    return out


@router.post("/{channel_id}/test", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
async def test_channel(
    channel_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Re-verify a channel via its driver.

    Updates ``status`` / ``last_verified`` / ``last_error`` so the
    dashboard reflects the most recent outcome without a page reload.
    Never starts polling — see ``/start``.
    """
    row = db.execute(
        text("SELECT platform, config FROM channel_connections WHERE id = :id AND workspace_id = :ws_id"),
        {"id": channel_id, "ws_id": str(ctx.workspace_id)},
    ).fetchone()
    if not row:
        raise HTTPException(404, "Channel connection not found")

    try:
        driver = get_driver(row.platform)()
    except UnknownPlatform:
        return {"status": "error", "detail": f"No driver for platform {row.platform!r}"}

    config = row.config or {}
    if isinstance(config, str):
        try:
            config = _json.loads(config)
        except Exception:
            config = {}
    try:
        verify_result = await driver.verify(workspace_id=str(ctx.workspace_id), config=config)
    except Exception as exc:  # noqa: BLE001
        logger.exception("test_channel: driver.verify raised for %s", row.platform)
        verify_result = VerifyResult(ok=False, error=f"driver raised: {exc}")

    _save_verify_outcome(db, channel_id, verify_result)
    return _verify_result_dict(verify_result)


@router.post("/{channel_id}/start", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
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


@router.post("/{channel_id}/stop", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
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
