"""
PRD-37: Workspace context endpoints.

The frontend expects `GET /api/workspaces/current` to return the active workspace.
In this codebase, most resources are filtered by `workspace_id`, and the auth
dependency (`get_request_context_hybrid`) provides a request-scoped workspace UUID.
"""

import logging
import os
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from sqlalchemy import func, desc
from sqlalchemy.orm import Session

from core.database.database import get_db
from core.models import Agent
from core.models.workspaces import Workspace

from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/workspaces", tags=["workspaces"])

# Keys that are allowed in workspace.settings.integrations
_ALLOWED_INTEGRATION_KEYS = {
    "telegram_bot_token",
    "slack_bot_token",
    "whatsapp_phone_number_id",
    "whatsapp_access_token",
}


@router.get("/current")
async def get_current_workspace(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Return the currently active workspace.

    The auth dependency auto-provisions a personal workspace for new Clerk users,
    so ctx.workspace_id should always point to a valid workspace.

    Returns `is_new_workspace: true` when the workspace has no agents yet,
    signalling the frontend to trigger the onboarding flow.
    """
    workspace = db.query(Workspace).get(ctx.workspace_id)

    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    # Detect brand-new workspace: no agents created yet
    agent_count = db.query(Agent).filter(Agent.workspace_id == workspace.id).count()

    # Auto-generate webhook_key if missing (for workspaces created before migration)
    if not workspace.webhook_key:
        workspace.webhook_key = uuid4().hex
        db.commit()

    # Compute webhook URL
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    webhook_url = f"{backend_url}/api/webhooks/ws/{workspace.webhook_key}" if workspace.webhook_key else None

    # Mask sensitive integration tokens for the GET response
    settings = dict(workspace.settings or {})
    integrations = dict(settings.get("integrations", {}))
    masked = {}
    for k, v in integrations.items():
        if isinstance(v, str) and len(v) > 8 and "token" in k:
            masked[k] = v[:4] + "..." + v[-4:]
        else:
            masked[k] = v
    settings["integrations"] = masked

    return {
        "id": str(workspace.id),
        "name": workspace.name,
        "slug": workspace.slug,
        "plan": workspace.plan,
        "role": ctx.user.role,
        "plan_limits": workspace.plan_limits or {},
        "is_new_workspace": agent_count == 0,
        "webhook_url": webhook_url,
        "webhook_key": workspace.webhook_key,
        "settings": settings,
    }


@router.get("/current/integrations")
async def get_integrations(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Return which integrations are configured (without exposing full tokens)."""
    workspace = db.query(Workspace).get(ctx.workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    integrations = (workspace.settings or {}).get("integrations", {})
    result = {}
    for key in _ALLOWED_INTEGRATION_KEYS:
        val = integrations.get(key, "")
        if isinstance(val, str) and val:
            result[key] = {"configured": True, "masked": val[:4] + "..." + val[-4:] if len(val) > 8 else "****"}
        else:
            result[key] = {"configured": False}

    return result


@router.put("/current/integrations")
async def save_integrations(
    payload: Dict[str, Any] = Body(...),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Save platform integration credentials.

    Body: { "telegram_bot_token": "123:ABC...", "slack_bot_token": "xoxb-..." }

    Only keys in _ALLOWED_INTEGRATION_KEYS are accepted.
    Values set to empty string or null will remove the integration.
    """
    workspace = db.query(Workspace).get(ctx.workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    settings = dict(workspace.settings or {})
    integrations = dict(settings.get("integrations", {}))

    for key, value in payload.items():
        if key not in _ALLOWED_INTEGRATION_KEYS:
            continue
        if value and isinstance(value, str) and value.strip():
            integrations[key] = value.strip()
        else:
            integrations.pop(key, None)

    settings["integrations"] = integrations
    workspace.settings = settings
    # Force SQLAlchemy to detect JSONB mutation
    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(workspace, "settings")
    db.commit()

    logger.info("Updated integrations for workspace %s: keys=%s", workspace.id, list(integrations.keys()))

    return {"status": "saved", "configured": list(integrations.keys())}


# ── BYOK Preferences ──────────────────────────────────────────────────

_ALLOWED_PROVIDERS = {"openai", "anthropic", "google", "openrouter", "azure", "grok"}


@router.get("/current/byok-preferences")
async def get_byok_preferences(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Return per-provider BYOK override preferences for the workspace."""
    workspace = db.query(Workspace).get(ctx.workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    byok_overrides = (workspace.settings or {}).get("byok_overrides", {})
    return {"byok_overrides": byok_overrides}


@router.put("/current/byok-preferences")
async def save_byok_preferences(
    payload: Dict[str, Any] = Body(...),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Save per-provider BYOK override preferences.

    Body: { "byok_overrides": { "openrouter": true, "openai": false } }
    """
    workspace = db.query(Workspace).get(ctx.workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    incoming = payload.get("byok_overrides", {})
    if not isinstance(incoming, dict):
        raise HTTPException(400, "byok_overrides must be an object")

    settings = dict(workspace.settings or {})
    overrides = dict(settings.get("byok_overrides", {}))

    for provider, enabled in incoming.items():
        if provider not in _ALLOWED_PROVIDERS:
            continue
        overrides[provider] = bool(enabled)

    settings["byok_overrides"] = overrides
    workspace.settings = settings

    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(workspace, "settings")
    db.commit()

    logger.info("Updated BYOK preferences for workspace %s: %s", workspace.id, overrides)
    return {"status": "saved", "byok_overrides": overrides}


# ── Orchestrator Soul & Personality ──────────────────────────────────

_VALID_PERSONALITY_MODES = {"friendly", "professional", "technical", "custom"}
_VALID_COMMUNICATION_STYLES = {"concise", "balanced", "detailed"}
_VALID_PROACTIVE_LEVELS = {"silent", "notify", "act_notify", "autonomous"}
_VALID_THINKING_LEVELS = {"off", "minimal", "low", "medium", "high"}
_VALID_HEARTBEAT_INTERVALS = {15, 30, 60, 120}
_VALID_NOTIFICATION_CHANNELS = {"in_app", "email", "webhook"}

_ORCHESTRATOR_DEFAULTS = {
    "personality_mode": "friendly",
    "custom_soul": "",
    "communication_style": "balanced",
    "proactive_level": "notify",
    "thinking_level": "medium",
    "heartbeat": {
        "enabled": False,
        "interval_minutes": 30,
        "active_hours_start": "08:00",
        "active_hours_end": "20:00",
        "timezone": "UTC",
        "checklist": "- Check agent health status\n- Review pending webhook failures\n- Summarize today's activity",
        "notification_channel": "in_app",
    },
}


@router.get("/current/orchestrator")
async def get_orchestrator_settings(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Return orchestrator soul, personality, and heartbeat configuration."""
    workspace = db.query(Workspace).get(ctx.workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    settings = workspace.settings or {}
    orchestrator = settings.get("orchestrator", {})

    # Merge with defaults so frontend always gets a complete object
    result = {**_ORCHESTRATOR_DEFAULTS, **orchestrator}
    result["heartbeat"] = {**_ORCHESTRATOR_DEFAULTS["heartbeat"], **orchestrator.get("heartbeat", {})}

    return result


@router.put("/current/orchestrator")
async def save_orchestrator_settings(
    payload: Dict[str, Any] = Body(...),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Save orchestrator soul, personality, and heartbeat configuration.

    Body: {
        "personality_mode": "friendly",
        "custom_soul": "You are a warm assistant...",
        "communication_style": "balanced",
        "proactive_level": "notify",
        "thinking_level": "medium",
        "heartbeat": {
            "enabled": true,
            "interval_minutes": 30,
            "active_hours_start": "08:00",
            "active_hours_end": "20:00",
            "timezone": "America/New_York",
            "checklist": "- Check agent health...",
            "notification_channel": "in_app"
        }
    }
    """
    workspace = db.query(Workspace).get(ctx.workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    # Validate top-level fields
    if "personality_mode" in payload and payload["personality_mode"] not in _VALID_PERSONALITY_MODES:
        raise HTTPException(400, f"personality_mode must be one of {_VALID_PERSONALITY_MODES}")
    if "communication_style" in payload and payload["communication_style"] not in _VALID_COMMUNICATION_STYLES:
        raise HTTPException(400, f"communication_style must be one of {_VALID_COMMUNICATION_STYLES}")
    if "proactive_level" in payload and payload["proactive_level"] not in _VALID_PROACTIVE_LEVELS:
        raise HTTPException(400, f"proactive_level must be one of {_VALID_PROACTIVE_LEVELS}")
    if "thinking_level" in payload and payload["thinking_level"] not in _VALID_THINKING_LEVELS:
        raise HTTPException(400, f"thinking_level must be one of {_VALID_THINKING_LEVELS}")

    # Validate heartbeat
    hb = payload.get("heartbeat")
    if hb and isinstance(hb, dict):
        if "interval_minutes" in hb and hb["interval_minutes"] not in _VALID_HEARTBEAT_INTERVALS:
            raise HTTPException(400, f"heartbeat.interval_minutes must be one of {_VALID_HEARTBEAT_INTERVALS}")
        if "notification_channel" in hb and hb["notification_channel"] not in _VALID_NOTIFICATION_CHANNELS:
            raise HTTPException(400, f"heartbeat.notification_channel must be one of {_VALID_NOTIFICATION_CHANNELS}")
        # Validate time format (HH:MM)
        for field in ("active_hours_start", "active_hours_end"):
            val = hb.get(field)
            if val:
                try:
                    parts = val.split(":")
                    assert len(parts) == 2
                    assert 0 <= int(parts[0]) <= 23
                    assert 0 <= int(parts[1]) <= 59
                except (ValueError, AssertionError):
                    raise HTTPException(400, f"heartbeat.{field} must be HH:MM format")

    # Merge into workspace settings
    settings = dict(workspace.settings or {})
    existing_orch = dict(settings.get("orchestrator", {}))

    # Update top-level orchestrator fields
    for key in ("personality_mode", "custom_soul", "communication_style", "proactive_level", "thinking_level"):
        if key in payload:
            existing_orch[key] = payload[key]

    # Update heartbeat (merge, don't replace)
    if hb and isinstance(hb, dict):
        existing_hb = dict(existing_orch.get("heartbeat", {}))
        existing_hb.update(hb)
        existing_orch["heartbeat"] = existing_hb

    settings["orchestrator"] = existing_orch
    workspace.settings = settings

    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(workspace, "settings")
    db.commit()

    logger.info("Updated orchestrator settings for workspace %s", workspace.id)

    # Return merged result with defaults
    result = {**_ORCHESTRATOR_DEFAULTS, **existing_orch}
    result["heartbeat"] = {**_ORCHESTRATOR_DEFAULTS["heartbeat"], **existing_orch.get("heartbeat", {})}
    return {"status": "saved", "orchestrator": result}


# ── What Automatos Knows (Memory Stats) ─────────────────────────────

@router.get("/current/memory-stats")
async def get_memory_stats(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Return memory statistics for the current workspace.

    Used by the "What Automatos Knows" section in the Orchestrator tab.
    Queries the memory_items table scoped to this workspace.
    """
    from datetime import datetime, timedelta

    workspace_id = ctx.workspace_id

    try:
        from modules.memory.storage.knowledge_system import MemoryItem
    except ImportError:
        # Graceful fallback if the model isn't available
        return {"total_memories": 0, "by_type": {}, "by_level": {}, "recent": []}

    try:
        base = db.query(MemoryItem).filter(MemoryItem.workspace_id == workspace_id)

        total = base.count()

        by_type = dict(
            db.query(MemoryItem.memory_type, func.count(MemoryItem.id))
            .filter(MemoryItem.workspace_id == workspace_id)
            .group_by(MemoryItem.memory_type)
            .all()
        )

        by_level = dict(
            db.query(MemoryItem.memory_level, func.count(MemoryItem.id))
            .filter(MemoryItem.workspace_id == workspace_id)
            .group_by(MemoryItem.memory_level)
            .all()
        )

        agents_with_memories = (
            db.query(func.count(func.distinct(MemoryItem.agent_id)))
            .filter(MemoryItem.workspace_id == workspace_id)
            .scalar() or 0
        )

        yesterday = datetime.utcnow() - timedelta(hours=24)
        recent_24h = (
            base.filter(MemoryItem.created_at >= yesterday).count()
        )

        # 5 most recent memories (content preview only)
        recent_rows = (
            base.order_by(desc(MemoryItem.created_at))
            .limit(5)
            .all()
        )
        recent = []
        for m in recent_rows:
            content_preview = str(m.content)[:120] if m.content else ""
            recent.append({
                "id": str(m.id),
                "type": m.memory_type,
                "level": m.memory_level,
                "preview": content_preview,
                "created_at": m.created_at.isoformat() if m.created_at else None,
            })

        return {
            "total_memories": total,
            "by_type": by_type,
            "by_level": by_level,
            "agents_with_memories": agents_with_memories,
            "recent_24h": recent_24h,
            "recent": recent,
        }
    except Exception as e:
        logger.error("Failed to get memory stats for workspace %s: %s", workspace_id, e)
        return {"total_memories": 0, "by_type": {}, "by_level": {}, "recent": [], "error": str(e)}

