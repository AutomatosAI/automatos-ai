"""
Lightweight workspace notification service.

Thin façade over ``channels.sender.send_to_channel`` so legacy callers
(heartbeat, auto_reporting, board-task notifiers) don't need to know
about the driver layer. The sender does all the per-platform work and
reads creds from ``channel_connections`` (with a legacy
``workspace.settings.integrations`` fallback during the migration
window).

Backwards-compatible signature: ``await send_workspace_notification(
workspace_id, message, channel="telegram")`` keeps working exactly as
before. ``channel`` is the platform name (``"telegram"``, ``"slack"``,
``"webhook"``). ``"orchestrator"`` / ``"direct"`` / ``"in_app"`` /
``None`` short-circuit to a no-op success (this path was historically
used to mean "don't send anything externally"), preserving the
previous behaviour.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# Channel names that mean "no external delivery" — preserved from the
# previous implementation so callers that pass these don't need to
# special-case the platform.
_IN_PROCESS_CHANNELS = frozenset({"orchestrator", "direct", "in_app", "silent"})


async def send_workspace_notification(
    workspace_id: str,
    message: str,
    channel: Optional[str] = None,
) -> bool:
    """Send ``message`` to the workspace's connected ``channel``.

    Returns True on success, False on failure / no-op. Never raises —
    failures are logged and swallowed so notification delivery never
    breaks the caller's primary work (heartbeats, auto-reporting, etc.).
    """
    notify_channel = (channel or "telegram").strip().lower()

    # Legacy alias for the workspace-default channel — read it from
    # workspace settings if the caller didn't specify one.
    if notify_channel == "default":
        notify_channel = _resolve_default_channel(workspace_id) or "telegram"

    if not notify_channel or notify_channel in _IN_PROCESS_CHANNELS:
        return True

    from channels.sender import send_to_channel
    from core.database.database import SessionLocal

    db = SessionLocal()
    try:
        result = await send_to_channel(
            db=db,
            workspace_id=workspace_id,
            platform=notify_channel,
            text=message,
        )
    except Exception:
        logger.exception("[Notify] send_to_channel raised for ws=%s ch=%s", workspace_id, notify_channel)
        return False
    finally:
        try:
            db.close()
        except Exception:
            pass

    if result.ok:
        logger.info(
            "[Notify] sent ws=%s ch=%s latency=%dms",
            workspace_id, notify_channel, result.latency_ms,
        )
        return True

    logger.warning(
        "[Notify] failed ws=%s ch=%s: %s",
        workspace_id, notify_channel, result.error,
    )
    return False


def _resolve_default_channel(workspace_id: str) -> Optional[str]:
    """Read ``workspace.settings.default_notification_channel`` if set.

    Best-effort, returns None if anything goes wrong — caller falls
    back to a sensible default.
    """
    try:
        from core.database.database import SessionLocal
        from core.models.workspaces import Workspace

        db = SessionLocal()
        try:
            ws = db.query(Workspace).get(workspace_id)
            if not ws:
                return None
            settings = ws.settings or {}
            value = settings.get("default_notification_channel")
            return str(value).strip().lower() if value else None
        finally:
            db.close()
    except Exception:
        return None
