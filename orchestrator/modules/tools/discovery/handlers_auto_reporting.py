"""Auto reporting handlers — Wave 2."""

from __future__ import annotations

import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


async def get_auto_reporting_prefs(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Return effective auto_reporting settings (defaults merged with overrides)."""
    from core.services.auto_reporting import load_auto_reporting_settings

    try:
        settings = load_auto_reporting_settings(db, workspace_id)
        return {"success": True, "data": settings}
    except Exception as exc:
        logger.error("[auto_reporting] get prefs failed: %s", exc, exc_info=True)
        return {"success": False, "error": str(exc)}


async def update_auto_reporting_prefs(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge ``params`` into ``workspace.settings.auto_reporting`` and commit."""
    from core.services.auto_reporting import update_auto_reporting_settings

    # Drop bookkeeping params that the executor injects (_agent_id, etc.)
    partial = {k: v for k, v in params.items() if not k.startswith("_")}
    if not partial:
        return {"success": False, "error": "no settings keys provided"}

    try:
        merged = update_auto_reporting_settings(db, workspace_id, partial)
        db.commit()
        return {"success": True, "data": merged}
    except ValueError as exc:
        return {"success": False, "error": str(exc)}
    except Exception as exc:
        logger.error("[auto_reporting] update prefs failed: %s", exc, exc_info=True)
        db.rollback()
        return {"success": False, "error": str(exc)}


_VALID_SEVERITIES: frozenset[str] = frozenset(
    {"info", "task", "approval", "urgent", "security"}
)
_VALID_STATUSES: frozenset[str] = frozenset({"ok", "warning", "error", "info"})


async def send_notification(
    db: Session, workspace_id: UUID, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Fire an event through the unified NotificationDispatcher."""
    from core.services.notification_dispatcher import (
        NotificationDispatcher,
        VALID_EVENT_TYPES,
    )

    event_type = params.get("event_type")
    title = params.get("title")
    if not event_type or not title:
        return {"success": False, "error": "event_type and title are required"}
    if event_type not in VALID_EVENT_TYPES:
        return {
            "success": False,
            "error": f"event_type must be one of: {', '.join(sorted(VALID_EVENT_TYPES))}",
        }

    severity = params.get("severity")
    if severity is not None and severity not in _VALID_SEVERITIES:
        return {
            "success": False,
            "error": f"severity must be one of: {', '.join(sorted(_VALID_SEVERITIES))}",
        }

    status = params.get("status", "ok")
    if status not in _VALID_STATUSES:
        return {
            "success": False,
            "error": f"status must be one of: {', '.join(sorted(_VALID_STATUSES))}",
        }

    dispatcher = NotificationDispatcher(db, workspace_id)
    try:
        result = await dispatcher.dispatch(
            event_type=event_type,
            title=title,
            message=params.get("message"),
            link_type=params.get("link_type"),
            link_id=params.get("link_id"),
            agent_id=params.get("_agent_id"),
            agent_name=params.get("_agent_name"),
            status=status,
            severity=severity,
        )
        db.commit()
        return {"success": True, "data": result}
    except Exception as exc:
        logger.error("[auto_reporting] send_notification failed: %s", exc, exc_info=True)
        db.rollback()
        return {"success": False, "error": str(exc)}
