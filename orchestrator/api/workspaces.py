"""
PRD-37: Workspace context endpoints.

The frontend expects `GET /api/workspaces/current` to return the active workspace.
In this codebase, most resources are filtered by `workspace_id`, and the auth
dependency (`get_request_context_hybrid`) provides a request-scoped workspace UUID.
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List
from uuid import UUID, uuid4

from fastapi import APIRouter, Body, Depends, HTTPException
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


# ── PRD-55: Orchestrator Settings ──────────────────────────────────────

@router.put("/current/settings")
async def save_workspace_settings(
    payload: Dict[str, Any] = Body(...),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Save workspace settings (orchestrator soul, heartbeat, etc.).

    Body: { "orchestrator": { "personality_mode": "friendly", ... } }
    Merges into existing workspace.settings.
    """
    workspace = db.query(Workspace).get(ctx.workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    settings = dict(workspace.settings or {})

    # Deep merge orchestrator settings
    if "orchestrator" in payload:
        orch = dict(settings.get("orchestrator", {}))
        incoming_orch = payload["orchestrator"]
        if isinstance(incoming_orch, dict):
            # Handle nested heartbeat merge
            if "heartbeat" in incoming_orch and isinstance(incoming_orch["heartbeat"], dict):
                hb = dict(orch.get("heartbeat", {}))
                hb.update(incoming_orch["heartbeat"])
                orch["heartbeat"] = hb
                incoming_orch = {k: v for k, v in incoming_orch.items() if k != "heartbeat"}
            orch.update(incoming_orch)
        settings["orchestrator"] = orch

    workspace.settings = settings
    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(workspace, "settings")
    db.commit()

    logger.info("Updated settings for workspace %s", workspace.id)
    return {"status": "saved", "settings": settings}


@router.get("/current/settings")
async def get_workspace_settings(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Return full workspace settings with orchestrator defaults."""
    workspace = db.query(Workspace).get(ctx.workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    settings = dict(workspace.settings or {})

    # Inject orchestrator defaults for missing fields
    orch_defaults = {
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
            "checklist": "- Check pending tasks\n- Review unrouted events\n- Summarize today's activity",
        },
    }
    orch = settings.get("orchestrator", {})
    if not isinstance(orch, dict):
        orch = {}
    for k, v in orch_defaults.items():
        if k not in orch:
            orch[k] = v
    settings["orchestrator"] = orch

    return {"settings": settings}


# ── PRD-55: Memory Stats ──────────────────────────────────────────────

@router.get("/current/memory-stats")
async def get_memory_stats(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Return memory statistics for the workspace."""
    try:
        from modules.memory.integrations.mem0_client import Mem0Client
        import asyncio

        client = Mem0Client()
        ws_id = str(ctx.workspace_id)
        global_user_id = f"ws_{ws_id}"

        loop = asyncio.get_event_loop()
        memories = await loop.run_in_executor(
            None,
            lambda: client.get_all(user_id=global_user_id),
        )
        memories = memories or []

        now = datetime.utcnow()
        day_ago = now - timedelta(hours=24)

        recent_24h = 0
        recent = []
        for m in memories:
            meta = m.get("metadata", {}) or {}
            ts_str = meta.get("timestamp", "")
            if ts_str:
                try:
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00").replace("+00:00", ""))
                    if ts > day_ago:
                        recent_24h += 1
                except (ValueError, TypeError):
                    pass
            recent.append({
                "memory": (m.get("memory", "") or "")[:120],
                "preview": (m.get("memory", "") or "")[:80],
            })

        return {
            "total": len(memories),
            "agents": len(set(
                (m.get("metadata", {}) or {}).get("agent_id", "")
                for m in memories if (m.get("metadata", {}) or {}).get("agent_id")
            )),
            "recent24h": recent_24h,
            "recent": recent[:5],
        }
    except Exception as e:
        logger.warning(f"Memory stats failed: {e}")
        return {"total": 0, "agents": 0, "recent24h": 0, "recent": []}

