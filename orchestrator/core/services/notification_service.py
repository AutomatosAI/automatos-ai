"""
Lightweight workspace notification service.

Sends messages to the workspace owner's configured notification channel
(Telegram, Slack, webhook). Reuses channel_connections and workspace
settings infrastructure from heartbeat.
"""

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)


async def send_workspace_notification(
    workspace_id: str,
    message: str,
    channel: Optional[str] = None,
) -> bool:
    """Send a notification to the workspace's configured channel.

    Args:
        workspace_id: UUID string of the workspace.
        message: Text message to send.
        channel: Override channel ("telegram", "slack", "webhook").
                 If None, uses workspace default_notification_channel setting.

    Returns:
        True if sent successfully, False otherwise.
    """
    from core.database.database import SessionLocal
    from core.models.workspaces import Workspace

    db = SessionLocal()
    try:
        ws = db.query(Workspace).get(workspace_id)
        if not ws:
            logger.warning("[Notify] Workspace %s not found", workspace_id)
            return False

        settings = ws.settings or {}
        integrations = settings.get("integrations", {})

        # Determine channel
        notify_channel = channel or settings.get("default_notification_channel", "direct")
        if notify_channel in ("orchestrator", "direct", "in_app"):
            # No external notification needed
            return True

        # Load channel_connections for bot tokens
        from sqlalchemy import text as sql_text
        channel_config = {}
        try:
            row = db.execute(
                sql_text(
                    "SELECT config FROM channel_connections "
                    "WHERE workspace_id = :ws AND platform = :plat "
                    "ORDER BY created_at DESC LIMIT 1"
                ),
                {"ws": workspace_id, "plat": notify_channel},
            ).fetchone()
            if row and row.config:
                channel_config = row.config if isinstance(row.config, dict) else json.loads(row.config)
        except Exception as e:
            logger.debug("[Notify] Could not load channel_connections: %s", e)

        if notify_channel == "telegram":
            return await _send_telegram(workspace_id, message, integrations, channel_config)
        elif notify_channel == "slack":
            return await _send_slack(workspace_id, message, integrations, channel_config)
        elif notify_channel == "webhook":
            webhook_url = integrations.get("webhook_url") or channel_config.get("webhook_url")
            if webhook_url:
                return await _send_webhook(webhook_url, message)
            logger.warning("[Notify] No webhook_url configured for ws=%s", workspace_id)
            return False
        else:
            logger.debug("[Notify] Unknown channel '%s', skipping", notify_channel)
            return False

    except Exception as e:
        logger.error("[Notify] Failed to send notification: %s", e, exc_info=True)
        return False
    finally:
        db.close()


async def _send_telegram(
    workspace_id: str, message: str, integrations: dict, channel_config: dict
) -> bool:
    token = integrations.get("telegram_bot_token") or channel_config.get("bot_token")
    if not token:
        logger.warning("[Notify] No telegram bot token for ws=%s", workspace_id)
        return False

    chat_id = (
        integrations.get("telegram_default_chat_id")
        or channel_config.get("default_chat_id")
    )
    if not chat_id:
        # Try to resolve from Telegram API
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"https://api.telegram.org/bot{token}/getUpdates?limit=1")
                data = resp.json()
                if data.get("ok") and data.get("result"):
                    chat_id = str(data["result"][0]["message"]["chat"]["id"])
        except Exception:
            pass

    if not chat_id:
        logger.warning("[Notify] No telegram chat_id for ws=%s", workspace_id)
        return False

    from api.webhooks import _send_telegram_reply
    ok = await _send_telegram_reply(int(chat_id), message, token)
    if ok:
        logger.info("[Notify] Telegram sent to chat %s", chat_id)
    return ok


async def _send_slack(
    workspace_id: str, message: str, integrations: dict, channel_config: dict
) -> bool:
    token = integrations.get("slack_bot_token") or channel_config.get("bot_token")
    channel = integrations.get("slack_default_channel") or channel_config.get("default_channel")

    if not token:
        logger.warning("[Notify] No slack bot token for ws=%s", workspace_id)
        return False
    if not channel:
        logger.warning("[Notify] No slack channel for ws=%s", workspace_id)
        return False

    try:
        import httpx
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                "https://slack.com/api/chat.postMessage",
                headers={"Authorization": f"Bearer {token}"},
                json={"channel": channel, "text": message},
            )
            data = resp.json()
            if data.get("ok"):
                logger.info("[Notify] Slack message sent to %s", channel)
                return True
            logger.warning("[Notify] Slack API error: %s", data.get("error"))
            return False
    except Exception as e:
        logger.error("[Notify] Slack send failed: %s", e)
        return False


async def _send_webhook(webhook_url: str, message: str) -> bool:
    try:
        import httpx
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                webhook_url,
                json={"text": message, "message": message},
                headers={"Content-Type": "application/json"},
            )
            ok = resp.status_code < 400
            if ok:
                logger.info("[Notify] Webhook sent to %s", webhook_url)
            return ok
    except Exception as e:
        logger.error("[Notify] Webhook send failed: %s", e)
        return False
