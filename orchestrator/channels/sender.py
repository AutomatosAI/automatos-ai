"""Unified channel sender.

One entry point: ``send_to_channel(workspace_id, platform, text,
target=None)``. Replaces ``_send_telegram_reply``,
``send_workspace_notification``, and the per-platform branches in
``services/destinations/dispatcher.py``.

What it does
------------
1. Look up the active ``ChannelConnection`` for ``(workspace_id, platform)``.
   On miss, fall back to the legacy ``workspace.settings.integrations``
   bag for one release so existing setups keep working during the
   migration window.
2. Resolve the target if the caller didn't pass one (Telegram default
   chat captured by the bot's ``/start`` handler; Slack default channel
   from row metadata).
3. Hand off to the platform driver's ``send`` — every per-platform
   detail lives there.

The sender never raises — it returns a ``SendResult`` so callers (chat
endpoints, callback dispatcher, heartbeat notifier) can log + bubble
errors uniformly.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Mapping, Optional, Tuple
from uuid import UUID

from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session

from .drivers import (
    SendResult,
    UnknownPlatform,
    get_driver,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resolution helpers
# ---------------------------------------------------------------------------

def _load_connection(
    db: Session, workspace_id: str, platform: str,
) -> Optional[Tuple[str, dict, dict]]:
    """Return ``(connection_id, config, metadata)`` for the active
    ``channel_connections`` row matching the workspace+platform, or
    None if there isn't one. Returns the first row by ``created_at`` —
    workspaces typically have one row per platform, but this keeps
    behaviour deterministic if multiple exist."""
    row = db.execute(
        sql_text(
            "SELECT id, config, metadata FROM channel_connections "
            "WHERE workspace_id = :ws AND platform = :plat "
            "ORDER BY created_at ASC LIMIT 1"
        ),
        {"ws": str(workspace_id), "plat": platform},
    ).fetchone()
    if not row:
        return None
    cfg = row.config or {}
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg)
        except Exception:
            cfg = {}
    meta = row.metadata or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    return str(row.id), cfg, meta


def _legacy_integration_config(db: Session, workspace_id: str, platform: str) -> Optional[dict]:
    """Fallback to ``workspace.settings.integrations.*`` for workspaces
    that haven't yet been migrated to a ``channel_connections`` row.
    Only knows about Telegram + Slack — the two platforms the legacy
    bag ever stored. Returns a config dict shaped like the new row's
    ``config`` JSONB."""
    row = db.execute(
        sql_text("SELECT settings FROM workspaces WHERE id = :ws"),
        {"ws": str(workspace_id)},
    ).fetchone()
    if not row or not row.settings:
        return None
    settings = row.settings if isinstance(row.settings, dict) else {}
    integrations = settings.get("integrations") or {}
    if platform == "telegram":
        token = integrations.get("telegram_bot_token")
        if not token:
            return None
        return {"bot_token": token}
    if platform == "slack":
        token = integrations.get("slack_bot_token")
        if not token:
            return None
        return {
            "bot_token": token,
            "default_channel": integrations.get("slack_default_channel"),
        }
    return None


def _resolve_target(
    *,
    db: Session,
    workspace_id: str,
    platform: str,
    metadata: Mapping[str, Any],
    explicit_target: Optional[str],
) -> Optional[str]:
    """Pick the send target when the caller didn't pass one.

    Order of precedence:
    1. ``explicit_target`` if provided.
    2. ``metadata.default_target`` saved on the channel row (set by
       the driver's verify on first run).
    3. Platform-specific legacy fallback in
       ``workspace.settings.integrations`` — ``telegram_default_chat_id``
       is the one that matters in practice.
    """
    if explicit_target:
        return explicit_target
    if isinstance(metadata, dict):
        cached = metadata.get("default_target") or metadata.get("default_chat_id")
        if cached:
            return str(cached)

    row = db.execute(
        sql_text("SELECT settings FROM workspaces WHERE id = :ws"),
        {"ws": str(workspace_id)},
    ).fetchone()
    if not row or not row.settings:
        return None
    settings = row.settings if isinstance(row.settings, dict) else {}
    integrations = settings.get("integrations") or {}

    if platform == "telegram":
        return integrations.get("telegram_default_chat_id")
    if platform == "slack":
        return integrations.get("slack_default_channel")
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def send_to_channel(
    *,
    db: Session,
    workspace_id: str | UUID,
    platform: str,
    text: str,
    target: Optional[str] = None,
) -> SendResult:
    """Send ``text`` to the workspace's connected ``platform``.

    Never raises — failures come back as ``SendResult(ok=False, ...)``.

    ``target`` overrides the row's default destination. Most callers
    leave it None and let the sender pick from row metadata / legacy
    integrations.
    """
    ws_str = str(workspace_id)

    try:
        driver_cls = get_driver(platform)
    except UnknownPlatform as exc:
        return SendResult(ok=False, latency_ms=0, error=str(exc), retryable=False)
    driver = driver_cls()

    loaded = _load_connection(db, ws_str, platform)
    if loaded is None:
        legacy = _legacy_integration_config(db, ws_str, platform)
        if legacy is None:
            return SendResult(
                ok=False,
                latency_ms=0,
                error=(
                    f"No {platform} channel connected for this workspace. "
                    f"Connect one under Settings → Channels."
                ),
                retryable=False,
            )
        config: dict = legacy
        metadata: dict = {}
    else:
        _, config, metadata = loaded

    resolved_target = _resolve_target(
        db=db,
        workspace_id=ws_str,
        platform=platform,
        metadata=metadata,
        explicit_target=target,
    )

    try:
        return await driver.send(
            workspace_id=ws_str,
            config=config,
            target=resolved_target,
            text=text,
        )
    except Exception as exc:  # noqa: BLE001 — never raise to caller
        logger.exception(
            "channel sender: driver %s.send raised for ws=%s", platform, ws_str,
        )
        return SendResult(
            ok=False, latency_ms=0,
            error=f"driver {platform!r} raised: {exc}", retryable=True,
        )
