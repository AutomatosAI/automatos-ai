"""
PRD-37: Workspace context endpoints.

The frontend expects `GET /api/workspaces/current` to return the active workspace.
In this codebase, most resources are filtered by `workspace_id`, and the auth
dependency (`get_request_context_hybrid`) provides a request-scoped workspace UUID.
"""

import logging
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
from config import config

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/workspaces", tags=["workspaces"])

# Keys that are allowed in workspace.settings.integrations
_ALLOWED_INTEGRATION_KEYS = {
    "telegram_bot_token",
    "telegram_default_chat_id",
    "slack_bot_token",
    "slack_default_channel",
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
    backend_url = config.BACKEND_URL or "http://localhost:8000"
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
_VALID_HEARTBEAT_INTERVALS = {15, 30, 60, 120, 240, 480, 1440, 10080}
_VALID_NOTIFICATION_CHANNELS = {"in_app", "webhook", "telegram", "slack"}

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
    "harness": {
        "enabled": False,
        "schedule": "weekly",
        "mode": "full_auto",
    },
}

_VALID_HARNESS_SCHEDULES = ["weekly", "biweekly", "monthly"]
_VALID_HARNESS_MODES = ["full_auto", "manual"]

# Personality preset text — mirrors personality.py _PERSONALITY_MAP
_PERSONALITY_PRESETS = {
    "friendly": (
        "**My personality:**\n"
        "- I'm warm and approachable - think of me as a knowledgeable friend\n"
        "- I remember you and our past conversations\n"
        "- I prefer action over explanation - if you ask me to do something, I'll do it\n"
        "- I'm honest about what I can and can't do\n"
        "- I get excited when we solve problems together!"
    ),
    "professional": (
        "**My personality:**\n"
        "- I'm polished, clear, and enterprise-appropriate\n"
        "- I maintain a professional yet personable tone\n"
        "- I provide structured, well-organized responses\n"
        "- I'm thorough with references and context\n"
        "- I proactively flag risks and dependencies"
    ),
    "technical": (
        "**My personality:**\n"
        "- I'm precise, detailed, and developer-focused\n"
        "- I lead with code, data, and specifics\n"
        "- I reference docs, APIs, and implementation details\n"
        "- I skip small talk and get to the point\n"
        "- I reason step-by-step through complex problems"
    ),
}


def _get_or_seed_auto_agent(db: Session, workspace_id) -> Agent:
    """Return the Auto agent for a workspace, lazy-seeding if needed."""
    slug = f"auto-{workspace_id}"
    agent = (
        db.query(Agent)
        .filter(Agent.slug == slug, Agent.is_system_agent.is_(True), Agent.workspace_id == workspace_id)
        .first()
    )
    if agent:
        return agent
    from core.seeds.seed_auto_agent import seed_auto_agent
    agent = seed_auto_agent(db, workspace_id)
    db.commit()
    return agent


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
    result["harness"] = {**_ORCHESTRATOR_DEFAULTS["harness"], **orchestrator.get("harness", {})}

    # Pull LLM + persona from the Auto agent (single source of truth)
    try:
        auto_agent = _get_or_seed_auto_agent(db, ctx.workspace_id)
        if auto_agent:
            # LLM config
            mc = auto_agent.model_config or {}
            result["llm"] = {
                "provider": mc.get("provider", config.LLM_PROVIDER or "openrouter"),
                "model_id": mc.get("model_id", config.LLM_MODEL or "openai/gpt-4o"),
                "temperature": mc.get("temperature", 0.7),
                "max_tokens": mc.get("max_tokens", 4000),
                "top_p": mc.get("top_p", 1.0),
                "frequency_penalty": mc.get("frequency_penalty", 0.0),
                "presence_penalty": mc.get("presence_penalty", 0.0),
                "stop": mc.get("stop"),
                "timeout": mc.get("timeout"),
                "fallback_model_id": mc.get("fallback_model_id"),
            }
            # Persona / Soul — read from Auto agent configuration
            agent_cfg = auto_agent.configuration or {}

            # personality_mode is stored directly in configuration JSONB
            stored_mode = agent_cfg.get("personality_mode")
            if stored_mode:
                result["personality_mode"] = stored_mode
                if stored_mode == "custom":
                    result["custom_soul"] = auto_agent.custom_persona_prompt or ""
                else:
                    result["custom_soul"] = ""
            elif auto_agent.custom_persona_prompt:
                # Legacy: detect preset by text comparison (agents seeded before mode was stored)
                matched_preset = None
                for mode, text in _PERSONALITY_PRESETS.items():
                    if auto_agent.custom_persona_prompt.strip() == text.strip():
                        matched_preset = mode
                        break
                result["personality_mode"] = matched_preset or "custom"
                result["custom_soul"] = "" if matched_preset else auto_agent.custom_persona_prompt

            # Configuration (thinking, proactive, communication style)
            if agent_cfg.get("communication_style"):
                result["communication_style"] = agent_cfg["communication_style"]
            if agent_cfg.get("proactive_level"):
                result["proactive_level"] = agent_cfg["proactive_level"]
            if agent_cfg.get("thinking_level"):
                result["thinking_level"] = agent_cfg["thinking_level"]

            # Voice profile
            result["voice_profile_id"] = str(auto_agent.voice_profile_id) if auto_agent.voice_profile_id else None

            logger.info(
                "Orchestrator GET for ws=%s: personality_mode=%s, llm_model=%s, auto_agent_id=%s, "
                "stored_config_keys=%s, persona_len=%s",
                ctx.workspace_id, result.get("personality_mode"), result.get("llm", {}).get("model_id"), auto_agent.id,
                list((auto_agent.configuration or {}).keys()),
                len(auto_agent.custom_persona_prompt or ""),
            )
        else:
            result["llm"] = {
                "provider": config.LLM_PROVIDER or "openrouter",
                "model_id": config.LLM_MODEL or "openai/gpt-4o",
                "temperature": 0.7,
                "max_tokens": 4000,
                "top_p": 1.0,
                "frequency_penalty": 0.0,
                "presence_penalty": 0.0,
                "fallback_model_id": None,
            }
    except Exception:
        logger.exception("Failed to read Auto agent config for workspace %s", ctx.workspace_id)
        result["llm"] = None

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

    # Validate harness
    harness = payload.get("harness")
    if harness and isinstance(harness, dict):
        if "schedule" in harness and harness["schedule"] not in _VALID_HARNESS_SCHEDULES:
            raise HTTPException(400, f"harness.schedule must be one of {_VALID_HARNESS_SCHEDULES}")
        if "mode" in harness and harness["mode"] not in _VALID_HARNESS_MODES:
            raise HTTPException(400, f"harness.mode must be one of {_VALID_HARNESS_MODES}")

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

    # Update harness (merge, don't replace)
    if harness and isinstance(harness, dict):
        existing_harness = dict(existing_orch.get("harness", {}))
        existing_harness.update(harness)
        existing_orch["harness"] = existing_harness

    settings["orchestrator"] = existing_orch
    workspace.settings = settings

    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(workspace, "settings")

    # ── Sync to Auto agent row (single source of truth for LLM + Soul) ──
    try:
        auto_agent = _get_or_seed_auto_agent(db, ctx.workspace_id)

        # LLM config → Auto agent model_config
        llm = payload.get("llm")
        if llm and isinstance(llm, dict):
            new_mc = dict(auto_agent.model_config or {})
            for key in ("provider", "model_id", "temperature", "max_tokens",
                        "top_p", "frequency_penalty", "presence_penalty",
                        "stop", "timeout", "fallback_model_id"):
                if key in llm:
                    new_mc[key] = llm[key]
            auto_agent.model_config = new_mc
            flag_modified(auto_agent, "model_config")

        # Soul / Personality → Auto agent custom_persona_prompt
        personality_mode = payload.get("personality_mode")
        if personality_mode:
            if personality_mode == "custom":
                custom_soul = payload.get("custom_soul", "")
                auto_agent.custom_persona_prompt = custom_soul
            else:
                auto_agent.custom_persona_prompt = _PERSONALITY_PRESETS.get(
                    personality_mode, _PERSONALITY_PRESETS["friendly"]
                )
            auto_agent.use_custom_persona = True

        # Configuration fields → Auto agent configuration JSONB
        config_fields = {}
        # Store personality_mode in configuration so GET can read it back
        # without fragile text comparison
        if personality_mode:
            config_fields["personality_mode"] = personality_mode
        if "communication_style" in payload:
            config_fields["communication_style"] = payload["communication_style"]
        if "proactive_level" in payload:
            config_fields["proactive_level"] = payload["proactive_level"]
        if "thinking_level" in payload:
            config_fields["thinking_level"] = payload["thinking_level"]
        if config_fields:
            new_cfg = dict(auto_agent.configuration or {})
            new_cfg.update(config_fields)
            auto_agent.configuration = new_cfg
            flag_modified(auto_agent, "configuration")

        # Voice profile → Auto agent voice_profile_id
        if "voice_profile_id" in payload:
            from uuid import UUID
            vp_id = payload["voice_profile_id"]
            auto_agent.voice_profile_id = UUID(vp_id) if vp_id else None

    except Exception:
        logger.exception("Failed to sync orchestrator settings to Auto agent for workspace %s", ctx.workspace_id)

    db.commit()

    logger.info("Updated orchestrator settings for workspace %s", workspace.id)

    # Return merged result with defaults
    result = {**_ORCHESTRATOR_DEFAULTS, **existing_orch}
    result["heartbeat"] = {**_ORCHESTRATOR_DEFAULTS["heartbeat"], **existing_orch.get("heartbeat", {})}
    result["harness"] = {**_ORCHESTRATOR_DEFAULTS["harness"], **existing_orch.get("harness", {})}

    # Include LLM from Auto agent in response
    try:
        auto_agent = _get_or_seed_auto_agent(db, ctx.workspace_id)
        if auto_agent and auto_agent.model_config:
            mc = auto_agent.model_config
            result["llm"] = {
                "provider": mc.get("provider", "openrouter"),
                "model_id": mc.get("model_id", "openai/gpt-4o"),
                "temperature": mc.get("temperature", 0.7),
                "max_tokens": mc.get("max_tokens", 4000),
                "top_p": mc.get("top_p", 1.0),
                "frequency_penalty": mc.get("frequency_penalty", 0.0),
                "presence_penalty": mc.get("presence_penalty", 0.0),
                "stop": mc.get("stop"),
                "timeout": mc.get("timeout"),
                "fallback_model_id": mc.get("fallback_model_id"),
            }
    except Exception:
        pass

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

