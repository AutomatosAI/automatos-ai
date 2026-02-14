"""
PRD-37: Workspace context endpoints.

The frontend expects `GET /api/workspaces/current` to return the active workspace.
In this codebase, most resources are filtered by `workspace_id`, and the auth
dependency (`get_request_context_hybrid`) provides a request-scoped workspace UUID.
"""

import logging
import os
from typing import Any, Dict
from uuid import UUID, uuid4

from fastapi import APIRouter, Body, Depends, HTTPException
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

