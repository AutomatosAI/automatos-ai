"""Owner notification handler — Auto's escalation channel."""

import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


_VALID_URGENCIES = {"low", "normal", "high", "urgent"}


async def notify_owner(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Send a message to the workspace owner via their configured channel.

    Resolves the channel in this order:
      1. params.channel (explicit override)
      2. workspace.settings.orchestrator.preferred_channel
      3. workspace.settings.default_notification_channel
      4. "in_app" (no external delivery)

    Always also creates a BoardTask so the request has a persistent
    record the owner can find on the board if they miss the channel
    message.
    """
    message = params.get("message")
    if not message or not isinstance(message, str):
        return {"success": False, "error": "message is required"}

    subject = params.get("subject") or "Auto needs your input"
    urgency = (params.get("urgency") or "normal").lower()
    if urgency not in _VALID_URGENCIES:
        return {"success": False, "error": f"urgency must be one of {sorted(_VALID_URGENCIES)}"}

    create_task = params.get("create_task", True)

    # Channel resolution
    from core.models.workspaces import Workspace
    workspace = db.query(Workspace).get(workspace_id)
    if not workspace:
        return {"success": False, "error": "Workspace not found"}

    settings = workspace.settings or {}
    orchestrator_settings = settings.get("orchestrator", {})

    channel = (
        params.get("channel")
        or orchestrator_settings.get("preferred_channel")
        or settings.get("default_notification_channel")
        or "in_app"
    )

    # Format the message with subject + urgency prefix so the channel-side
    # rendering carries the same context the board task would have.
    urgency_marker = {
        "urgent": "🚨 URGENT",
        "high": "⚠️ Decision needed",
        "normal": "💬 Auto",
        "low": "ℹ️ Auto",
    }.get(urgency, "💬 Auto")

    formatted = f"{urgency_marker} — {subject}\n\n{message}"

    # Send via the channel
    delivered = False
    delivery_error = None
    try:
        from core.services.notification_service import send_workspace_notification
        delivered = await send_workspace_notification(
            workspace_id=str(workspace_id),
            message=formatted,
            channel=channel if channel not in ("orchestrator", "direct") else None,
        )
    except Exception as e:
        delivery_error = str(e)
        logger.warning("[notify_owner] send failed for ws=%s: %s", workspace_id, e, exc_info=True)

    # Persistent backup: BoardTask assigned to Auto for review queue
    board_task_id = None
    if create_task:
        try:
            from core.models import Agent, BoardTask
            from sqlalchemy import and_
            auto = db.query(Agent).filter(
                and_(
                    Agent.workspace_id == workspace_id,
                    Agent.is_system_agent.is_(True),
                    Agent.name == "Auto",
                )
            ).first()

            priority_map = {"urgent": "urgent", "high": "high", "normal": "medium", "low": "low"}
            agent_id = params.get("_agent_id")

            task = BoardTask(
                workspace_id=workspace_id,
                title=f"[Auto] {subject}",
                description=message,
                status="assigned" if auto else "inbox",
                priority=priority_map.get(urgency, "medium"),
                assigned_agent_id=auto.id if auto else None,
                created_by_type="agent",
                created_by_id=str(agent_id) if agent_id else None,
                source_type="notification",
                tags=["auto-escalation", f"urgency:{urgency}"],
            )
            db.add(task)
            db.flush()
            board_task_id = task.id
        except Exception as e:
            logger.warning("[notify_owner] BoardTask creation failed: %s", e, exc_info=True)

    if not delivered and not board_task_id:
        return {
            "success": False,
            "error": delivery_error or f"Could not deliver via {channel} and BoardTask creation failed",
        }

    return {
        "success": True,
        "delivered_via": channel if delivered else "board_task_only",
        "channel_delivered": delivered,
        "board_task_id": board_task_id,
        "delivery_error": delivery_error,
    }
