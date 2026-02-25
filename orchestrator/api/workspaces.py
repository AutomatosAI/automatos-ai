"""
PRD-37: Workspace context endpoints.
US-004: Workspace CRUD API.
US-005: Workspace templates API (GET /templates, POST /from-template).
US-012: Workspace sharing API (POST /share, DELETE /share, GET /shared-with-me).

The frontend expects `GET /api/workspaces/current` to return the active workspace.
In this codebase, most resources are filtered by `workspace_id`, and the auth
dependency (`get_request_context_hybrid`) provides a request-scoped workspace UUID.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID, uuid4

from fastapi import APIRouter, Body, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field
from sqlalchemy import func, desc, or_
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from core.database.database import get_db
from core.models import Agent
from core.models.core import User as UserModel
from core.models.workspaces import Workspace
from core.models.workspace_shares import WorkspaceShare

from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/workspaces", tags=["workspaces"])


# ── US-004: Pydantic schemas for Workspace CRUD ─────────────────────

class WorkspaceCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    layout: Optional[Dict[str, Any]] = None
    layout_mode: Optional[str] = Field(None, pattern=r"^(grid|freeform)$")
    widgets: Optional[List[Dict[str, Any]]] = None


class WorkspaceUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    layout: Optional[Dict[str, Any]] = None
    layout_mode: Optional[str] = Field(None, pattern=r"^(grid|freeform)$")
    widgets: Optional[List[Dict[str, Any]]] = None
    visibility: Optional[str] = Field(None, pattern=r"^(private|shared|public)$")


class WorkspaceListItem(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    layout_mode: str
    visibility: str
    updated_at: Optional[str] = None

    class Config:
        from_attributes = True


class WorkspaceDetail(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    layout_mode: str
    visibility: str
    layout: Optional[Dict[str, Any]] = None
    widgets: Optional[List[Dict[str, Any]]] = None
    updated_at: Optional[str] = None
    created_at: Optional[str] = None

    class Config:
        from_attributes = True


# ── US-004: Helper — resolve DB user id from RequestContext ──────────

def _resolve_user_db_id(db: Session, ctx: RequestContext) -> Optional[int]:
    """
    Resolve the integer ``users.id`` from the Clerk user identifier
    stored in ``ctx.user.clerk_user_id`` (or ``ctx.user.id``).
    Returns None when the user cannot be resolved (e.g. anonymous/api_key).
    """
    clerk_uid = ctx.user.clerk_user_id or ctx.user.id
    if not clerk_uid:
        return None
    user = (
        db.query(UserModel)
        .filter(UserModel.clerk_user_id == str(clerk_uid))
        .first()
    )
    return user.id if user else None


def _has_share_permission(
    db: Session,
    workspace_id: UUID,
    user_db_id: Optional[int],
    required: str = "view",
) -> bool:
    """
    Check whether the user has at least ``required`` permission on a workspace
    via the workspace_shares table.

    Permission hierarchy: view < edit < admin
    """
    if user_db_id is None:
        return False
    _PERMISSION_RANK = {"view": 0, "edit": 1, "admin": 2}
    required_rank = _PERMISSION_RANK.get(required, 0)
    share = (
        db.query(WorkspaceShare)
        .filter(
            WorkspaceShare.workspace_id == workspace_id,
            WorkspaceShare.user_id == str(user_db_id),
        )
        .first()
    )
    if not share:
        return False
    return _PERMISSION_RANK.get(share.permission, 0) >= required_rank


def _is_owner(workspace: Workspace, user_db_id: Optional[int]) -> bool:
    """Return True if user_db_id matches workspace.owner_id."""
    if user_db_id is None or workspace.owner_id is None:
        return False
    return int(workspace.owner_id) == int(user_db_id)


def _ws_to_list_item(ws: Workspace) -> Dict[str, Any]:
    return {
        "id": str(ws.id),
        "name": ws.name,
        "description": ws.description,
        "layout_mode": ws.layout_mode or "grid",
        "visibility": ws.visibility or "private",
        "updated_at": ws.updated_at.isoformat() if ws.updated_at else None,
    }


def _ws_to_detail(ws: Workspace) -> Dict[str, Any]:
    return {
        "id": str(ws.id),
        "name": ws.name,
        "description": ws.description,
        "layout_mode": ws.layout_mode or "grid",
        "visibility": ws.visibility or "private",
        "layout": ws.layout,
        "widgets": ws.widgets,
        "updated_at": ws.updated_at.isoformat() if ws.updated_at else None,
        "created_at": ws.created_at.isoformat() if ws.created_at else None,
    }

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


# ══════════════════════════════════════════════════════════════════════
# US-005: Workspace Template Endpoints
# ══════════════════════════════════════════════════════════════════════


class WorkspaceTemplateItem(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    icon: Optional[str] = None
    category: Optional[str] = None
    layout_mode: str
    widgets: Optional[List[Dict[str, Any]]] = None

    class Config:
        from_attributes = True


class CloneFromTemplateRequest(BaseModel):
    name: Optional[str] = Field(None, max_length=255, description="Override name for the cloned workspace")


@router.get("/templates", response_model=List[WorkspaceTemplateItem])
async def list_workspace_templates(
    category: Optional[str] = Query(None, description="Filter by template category"),
    db: Session = Depends(get_db),
):
    """
    Return all workspace templates (is_template=True).

    No authentication required -- templates are public catalogue items.
    """
    q = db.query(Workspace).filter(
        Workspace.is_template == True,
        Workspace.is_active == True,
    )
    if category:
        q = q.filter(Workspace.template_category == category)

    templates = q.order_by(Workspace.name).all()

    return [
        WorkspaceTemplateItem(
            id=str(t.id),
            name=t.name,
            description=t.description,
            icon=t.template_icon,
            category=t.template_category,
            layout_mode=t.layout_mode or "grid",
            widgets=t.widgets,
        )
        for t in templates
    ]


@router.post("/from-template/{template_id}", status_code=201)
async def clone_from_template(
    template_id: UUID,
    body: CloneFromTemplateRequest = Body(default=CloneFromTemplateRequest()),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Clone a template workspace into a new user-owned workspace.

    Copies: layout, layout_mode, widgets, description.
    Sets: is_template=False, owner_id=current user.
    """
    template = db.query(Workspace).filter(
        Workspace.id == template_id,
        Workspace.is_template == True,
        Workspace.is_active == True,
    ).first()

    if not template:
        raise HTTPException(status_code=404, detail="Template not found")

    user_db_id = _resolve_user_db_id(db, ctx)
    if user_db_id is None:
        raise HTTPException(status_code=403, detail="Cannot resolve user for workspace creation")

    clone_name = body.name or f"{template.name} (Copy)"

    ws = Workspace(
        id=uuid4(),
        name=clone_name,
        description=template.description,
        layout=template.layout,
        layout_mode=template.layout_mode or "grid",
        widgets=template.widgets or [],
        owner_id=user_db_id,
        is_active=True,
        is_template=False,
        visibility="private",
    )
    db.add(ws)
    db.commit()
    db.refresh(ws)

    logger.info(
        "Cloned template %s -> workspace %s (%s) for user %s",
        template_id, ws.id, ws.name, user_db_id,
    )
    return _ws_to_detail(ws)


# ══════════════════════════════════════════════════════════════════════
# US-012: Workspace Sharing Endpoints
# ══════════════════════════════════════════════════════════════════════


_VALID_SHARE_PERMISSIONS = {"view", "edit", "admin"}


class ShareRequest(BaseModel):
    user_id: str = Field(..., min_length=1, description="ID of the user to share with")
    permission: str = Field(
        ...,
        description="Permission level: view, edit, or admin",
    )


class SharedWorkspaceItem(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    layout_mode: str
    visibility: str
    updated_at: Optional[str] = None
    permission: str

    class Config:
        from_attributes = True


@router.get("/shared-with-me", response_model=List[SharedWorkspaceItem])
async def list_shared_with_me(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Return all workspaces shared with the current user.

    Joins workspace_shares to workspaces and returns workspace info
    along with the permission level granted.
    """
    user_db_id = _resolve_user_db_id(db, ctx)
    if user_db_id is None:
        return []

    rows = (
        db.query(Workspace, WorkspaceShare.permission)
        .join(WorkspaceShare, WorkspaceShare.workspace_id == Workspace.id)
        .filter(
            WorkspaceShare.user_id == str(user_db_id),
            Workspace.is_active == True,
        )
        .order_by(Workspace.updated_at.desc())
        .all()
    )

    return [
        SharedWorkspaceItem(
            id=str(ws.id),
            name=ws.name,
            description=ws.description,
            layout_mode=ws.layout_mode or "grid",
            visibility=ws.visibility or "private",
            updated_at=ws.updated_at.isoformat() if ws.updated_at else None,
            permission=perm,
        )
        for ws, perm in rows
    ]


class ShareEntry(BaseModel):
    id: str
    workspace_id: str
    user_id: str
    permission: str
    user_email: Optional[str] = None
    user_name: Optional[str] = None
    created_at: Optional[str] = None


@router.get("/{workspace_id}/shares", response_model=List[ShareEntry])
async def list_workspace_shares(
    workspace_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Return all share entries for a workspace.

    Only the workspace owner may list shares.
    """
    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not ws or not ws.is_active:
        raise HTTPException(status_code=404, detail="Workspace not found")

    user_db_id = _resolve_user_db_id(db, ctx)
    if not _is_owner(ws, user_db_id):
        raise HTTPException(status_code=403, detail="Only the workspace owner can view shares")

    shares = (
        db.query(WorkspaceShare)
        .filter(WorkspaceShare.workspace_id == workspace_id)
        .all()
    )

    result = []
    for s in shares:
        # Resolve user info
        user = db.query(UserModel).filter(UserModel.id == int(s.user_id)).first()
        result.append(ShareEntry(
            id=str(s.id),
            workspace_id=str(s.workspace_id),
            user_id=str(s.user_id),
            permission=s.permission,
            user_email=user.email if user else None,
            user_name=user.name if user else None,
            created_at=s.created_at.isoformat() if s.created_at else None,
        ))

    return result


@router.post("/{workspace_id}/share", status_code=201)
async def share_workspace(
    workspace_id: UUID,
    body: ShareRequest,
    response: Response,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Share a workspace with another user.

    Creates a new workspace_shares entry or updates the permission on an
    existing one.  Only the workspace owner may share.

    Returns 201 on create, 200 on update.
    """
    if body.permission not in _VALID_SHARE_PERMISSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"permission must be one of {_VALID_SHARE_PERMISSIONS}",
        )

    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not ws or not ws.is_active:
        raise HTTPException(status_code=404, detail="Workspace not found")

    user_db_id = _resolve_user_db_id(db, ctx)
    if not _is_owner(ws, user_db_id):
        raise HTTPException(status_code=403, detail="Only the workspace owner can share it")

    # Prevent owner from sharing with themselves
    if str(user_db_id) == body.user_id:
        raise HTTPException(status_code=400, detail="Cannot share a workspace with yourself")

    existing = (
        db.query(WorkspaceShare)
        .filter(
            WorkspaceShare.workspace_id == workspace_id,
            WorkspaceShare.user_id == body.user_id,
        )
        .first()
    )

    if existing:
        existing.permission = body.permission
        db.commit()
        db.refresh(existing)
        response.status_code = 200
        logger.info(
            "Updated share on workspace %s for user %s -> %s",
            workspace_id, body.user_id, body.permission,
        )
        return {
            "id": str(existing.id),
            "workspace_id": str(existing.workspace_id),
            "user_id": str(existing.user_id),
            "permission": existing.permission,
            "created_at": existing.created_at.isoformat() if existing.created_at else None,
        }

    share = WorkspaceShare(
        id=uuid4(),
        workspace_id=workspace_id,
        user_id=body.user_id,
        permission=body.permission,
    )
    db.add(share)
    db.commit()
    db.refresh(share)

    logger.info(
        "Created share on workspace %s for user %s (%s)",
        workspace_id, body.user_id, body.permission,
    )
    return {
        "id": str(share.id),
        "workspace_id": str(share.workspace_id),
        "user_id": str(share.user_id),
        "permission": share.permission,
        "created_at": share.created_at.isoformat() if share.created_at else None,
    }


@router.delete("/{workspace_id}/share/{user_id}", status_code=204)
async def unshare_workspace(
    workspace_id: UUID,
    user_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Remove a share entry for a workspace.

    Only the workspace owner may revoke shares.
    """
    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not ws or not ws.is_active:
        raise HTTPException(status_code=404, detail="Workspace not found")

    user_db_id = _resolve_user_db_id(db, ctx)
    if not _is_owner(ws, user_db_id):
        raise HTTPException(status_code=403, detail="Only the workspace owner can unshare it")

    share = (
        db.query(WorkspaceShare)
        .filter(
            WorkspaceShare.workspace_id == workspace_id,
            WorkspaceShare.user_id == user_id,
        )
        .first()
    )
    if not share:
        raise HTTPException(status_code=404, detail="Share not found")

    db.delete(share)
    db.commit()

    logger.info("Removed share on workspace %s for user %s", workspace_id, user_id)
    return None


# ══════════════════════════════════════════════════════════════════════
# US-004: Workspace CRUD Endpoints
# ══════════════════════════════════════════════════════════════════════
#
# These routes use path parameters (/{workspace_id}) and MUST be
# registered AFTER all /current/* routes to avoid path conflicts.
# ──────────────────────────────────────────────────────────────────────


@router.get("", response_model=List[WorkspaceListItem])
async def list_workspaces(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Return all workspaces the authenticated user can access.

    Includes:
    - Workspaces owned by the user (workspaces.owner_id == user.id)
    - Workspaces shared with the user (via workspace_shares)
    """
    user_db_id = _resolve_user_db_id(db, ctx)

    if user_db_id is None:
        # API-key / anonymous callers: fall back to current workspace only
        ws = db.query(Workspace).get(ctx.workspace_id)
        return [_ws_to_list_item(ws)] if ws else []

    # Owned workspaces
    owned = (
        db.query(Workspace)
        .filter(Workspace.owner_id == user_db_id, Workspace.is_active == True)
        .all()
    )

    owned_ids = {ws.id for ws in owned}

    # Shared workspaces (via workspace_shares join)
    shared = (
        db.query(Workspace)
        .join(WorkspaceShare, WorkspaceShare.workspace_id == Workspace.id)
        .filter(
            WorkspaceShare.user_id == str(user_db_id),
            Workspace.is_active == True,
        )
        .all()
    )

    # Merge, dedup, sort by updated_at desc
    all_ws = list(owned)
    for ws in shared:
        if ws.id not in owned_ids:
            all_ws.append(ws)

    all_ws.sort(key=lambda w: w.updated_at or w.created_at, reverse=True)

    return [_ws_to_list_item(ws) for ws in all_ws]


@router.post("", status_code=201)
async def create_workspace(
    body: WorkspaceCreateRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Create a new workspace.

    The authenticated user becomes the owner.
    """
    user_db_id = _resolve_user_db_id(db, ctx)
    if user_db_id is None:
        raise HTTPException(status_code=403, detail="Cannot resolve user for workspace creation")

    ws = Workspace(
        id=uuid4(),
        name=body.name,
        description=body.description,
        layout=body.layout or {"columns": 12, "rowHeight": 100},
        layout_mode=body.layout_mode or "grid",
        widgets=body.widgets or [],
        owner_id=user_db_id,
        is_active=True,
        visibility="private",
    )
    db.add(ws)
    db.commit()
    db.refresh(ws)

    logger.info("Created workspace %s (%s) for user %s", ws.id, ws.name, user_db_id)
    return _ws_to_detail(ws)


@router.get("/{workspace_id}")
async def get_workspace(
    workspace_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Return full workspace detail including layout and widgets.

    Also updates ``last_opened_at`` on each access.
    """
    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not ws or not ws.is_active:
        raise HTTPException(status_code=404, detail="Workspace not found")

    # Access check: owner or shared
    user_db_id = _resolve_user_db_id(db, ctx)
    if not _is_owner(ws, user_db_id) and not _has_share_permission(db, workspace_id, user_db_id, "view"):
        raise HTTPException(status_code=403, detail="Access denied")

    # Update last_opened_at
    ws.last_opened_at = datetime.now(timezone.utc)
    db.commit()

    return _ws_to_detail(ws)


@router.put("/{workspace_id}")
async def update_workspace(
    workspace_id: UUID,
    body: WorkspaceUpdateRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Update a workspace.

    Requires ownership or ``edit``/``admin`` share permission.
    """
    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not ws or not ws.is_active:
        raise HTTPException(status_code=404, detail="Workspace not found")

    user_db_id = _resolve_user_db_id(db, ctx)
    if not _is_owner(ws, user_db_id) and not _has_share_permission(db, workspace_id, user_db_id, "edit"):
        raise HTTPException(status_code=403, detail="Access denied")

    # Apply partial updates
    update_data = body.model_dump(exclude_unset=True)
    for field, value in update_data.items():
        setattr(ws, field, value)

    # Flag JSONB columns so SQLAlchemy detects mutations
    if "layout" in update_data:
        flag_modified(ws, "layout")
    if "widgets" in update_data:
        flag_modified(ws, "widgets")

    db.commit()
    db.refresh(ws)

    logger.info("Updated workspace %s by user %s", workspace_id, user_db_id)
    return _ws_to_detail(ws)


@router.delete("/{workspace_id}", status_code=204)
async def delete_workspace(
    workspace_id: UUID,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """
    Delete (soft-deactivate) a workspace.

    Only the owner may delete a workspace.
    """
    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if not ws or not ws.is_active:
        raise HTTPException(status_code=404, detail="Workspace not found")

    user_db_id = _resolve_user_db_id(db, ctx)
    if not _is_owner(ws, user_db_id):
        raise HTTPException(status_code=403, detail="Only the workspace owner can delete it")

    ws.is_active = False
    db.commit()

    logger.info("Deleted workspace %s by user %s", workspace_id, user_db_id)
    return None

