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
from pydantic import BaseModel
from sqlalchemy import func, desc
from sqlalchemy.orm import Session

from core.database.database import get_db
from core.models import Agent
from core.models.workspaces import Workspace
from services.onboarding_state import public_snapshot

from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext
from core.auth.workspace_permission import require_workspace_permission, workspace_permission_granted
from core.llm.defaults import DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL, get_default_model_config
from core.seeds.seed_auto_agent import compose_persona_with_doctrine
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

    The response carries the server-side onboarding snapshot (`onboarding`:
    {stage, trial}); the frontend drives the Auto-led flow off `onboarding.stage`.
    """
    workspace = db.query(Workspace).get(ctx.workspace_id)

    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

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

    # PRD-195 S8 (C.1): this used to return the SYSTEM-role twin
    # (ctx.user.role) while the frontend typed it as a workspace role — the
    # exact twin confusion the dossier flagged. Return the real per-tenant
    # role: member row (owner fallback) via the S2 resolver; the trusted
    # single-user lanes (local/anonymous) and the super-admin operator render
    # owner affordances; every other non-member renders read-only — matching
    # what the write gates now enforce (G5 UI honesty).
    if ctx.auth_type == "anonymous" or getattr(ctx.user, "system_role", None) == "super_admin":
        member_role = "owner"
    else:
        from core.auth.workspace_permission import resolve_workspace_role

        member_role = resolve_workspace_role(db, ctx) or "viewer"

    from services.plan_tiers import exposure_for_plan

    return {
        "id": str(workspace.id),
        "name": workspace.name,
        "slug": workspace.slug,
        "plan": workspace.plan,
        "role": member_role,
        "plan_limits": workspace.plan_limits or {},
        # PRD-222 W2·S1b (US-024): exposure profile derived from PLAN_TIERS for
        # this workspace's plan — nav visibility, capability families,
        # marketplace depth + tier display info. Field addition only, same route
        # (route-manifest unchanged). Hidden ≠ deleted (D5): the client trims
        # nav/marketplace labels; no route or data is removed.
        "exposure": exposure_for_plan(workspace.plan or "basic"),
        # PRD-222 W1S2: server-side onboarding stage + trial snapshot ({stage,
        # trial}). Field addition only — no new route (route-manifest unchanged).
        # PRD-222 W2·S6 (US-022) retired the legacy new-workspace boolean — its
        # only consumers (the first-login guard + tour) are gone; the frontend
        # detects a new workspace from onboarding.stage now.
        "onboarding": public_snapshot(workspace),
        "webhook_url": webhook_url,
        "webhook_key": workspace.webhook_key,
        "settings": settings,
    }


# ── Onboarding reset (PRD-222 W1·S10 / D9) — DEV/OPS ONLY, TEMPORARY ──────────


def _require_admin(ctx: RequestContext, db) -> None:
    """Workspace-admin gate for the dev reset.

    FIX (2026-08-28 test round 1, su-lock class 3rd sighting): the original check
    read the PLATFORM ``system_role`` only, so a normal user could never reset a
    workspace they own/administer — Gerard's alias got 403 on its own workspace.
    Correct contract: the caller holds ``workspace:manage`` on the CURRENT
    workspace (roles matrix), with platform admins passing through."""
    if ctx.user and getattr(ctx.user, "system_role", None) in ("admin", "super_admin"):
        return
    if ctx.user and workspace_permission_granted(db, ctx, "workspace:manage"):
        return
    raise HTTPException(status_code=403, detail="Admin access required")


class OnboardingResetRequest(BaseModel):
    reset_trial: bool = False
    wipe_built: bool = False
    wipe_credentials: bool = False


@router.post("/current/onboarding/reset")
async def reset_current_onboarding(
    payload: OnboardingResetRequest = Body(default_factory=OnboardingResetRequest),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Rewind the current workspace's onboarding so it can be re-run in place.

    DEV/OPS ONLY (PRD-222 W1·S10, decision D9). Gated on
    ``config.ONBOARDING_RESET_ENABLED``: when off the route 404s — it is not
    advertised (deliberately NOT 403, which would confirm it exists). When on,
    it is workspace-admin only (403 otherwise). Returns counts of everything
    reset/wiped. The reset itself lives in ``services.onboarding_state`` — the
    one sanctioned backward writer of the onboarding document.
    """
    if not config.ONBOARDING_RESET_ENABLED:
        raise HTTPException(status_code=404, detail="Not Found")
    _require_admin(ctx, db)

    workspace = db.query(Workspace).get(ctx.workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    from services.onboarding_state import reset_onboarding

    report = reset_onboarding(
        db,
        workspace,
        reset_trial=payload.reset_trial,
        wipe_built=payload.wipe_built,
        wipe_credentials=payload.wipe_credentials,
    )
    logger.warning(
        "Onboarding reset: workspace=%s admin=%s flags=reset_trial:%s/wipe_built:%s/wipe_credentials:%s resets=%s",
        ctx.workspace_id, getattr(ctx.user, "id", None),
        payload.reset_trial, payload.wipe_built, payload.wipe_credentials,
        report.get("resets"),
    )
    return report


# ── Post-setup checklist (PRD-222 W2·S4 / US-020) ────────────────────────────


def _workspace_checklist_counts(db: Session, workspace: Workspace) -> dict[str, int]:
    """Gather the LIVE counts the checklist derives completion from.

    All workspace-scoped, from the stores the platform already keeps:
    active Composio connections, missions (``orchestration_runs``), and active
    team members. No new bookkeeping.
    """
    from core.composio.entity_manager import EntityManager
    from core.models.orchestration import OrchestrationRun
    from core.workspaces.models import WorkspaceMember

    connections_count = len(EntityManager(db).get_connected_apps(workspace.id))
    missions_count = (
        db.query(OrchestrationRun)
        .filter(OrchestrationRun.workspace_id == workspace.id)
        .count()
    )
    members_count = (
        db.query(WorkspaceMember)
        .filter(
            WorkspaceMember.workspace_id == workspace.id,
            WorkspaceMember.is_active == True,  # noqa: E712 — SQLAlchemy column compare
        )
        .count()
    )
    return {
        "connections_count": connections_count,
        "missions_count": missions_count,
        "members_count": members_count,
    }


@router.get("/current/onboarding/checklist")
async def get_current_onboarding_checklist(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """The post-setup checklist for the current workspace (PRD-222 US-020).

    Item completion is DERIVED from live workspace counts on every read (never a
    stored tick); only the dismissal flags live in ``onboarding.checklist``. The
    invite item is omitted on single-seat plans.
    """
    from services.onboarding_state import build_checklist, get_onboarding

    workspace = db.query(Workspace).get(ctx.workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    counts = _workspace_checklist_counts(db, workspace)
    onboarding = get_onboarding(workspace)
    plan_limits = workspace.plan_limits or {}
    plan_seats = int(plan_limits.get("max_members") or 1)
    return build_checklist(
        connections_count=counts["connections_count"],
        missions_count=counts["missions_count"],
        members_count=counts["members_count"],
        plan_seats=plan_seats,
        comfort=(onboarding.get("segment") or {}).get("comfort"),
        stored=onboarding.get("checklist"),
    )


class ChecklistUpdateRequest(BaseModel):
    dismissed: Optional[bool] = None
    academy_done: Optional[bool] = None


@router.patch(
    "/current/onboarding/checklist",
    dependencies=[Depends(require_workspace_permission("workspace:manage"))],
)
async def update_current_onboarding_checklist(
    payload: ChecklistUpdateRequest = Body(default_factory=ChecklistUpdateRequest),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Persist a checklist dismissal (the card, or the manual Academy item).

    Server-side record (D8 — never localStorage). Writes only the two dismissal
    flags via ``update_checklist`` (full-JSONB reassignment); item completion
    stays derived. Returns the fresh checklist so the caller reflects the change.
    """
    from services.onboarding_state import build_checklist, get_onboarding, update_checklist

    workspace = db.query(Workspace).get(ctx.workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    update_checklist(
        db, workspace, dismissed=payload.dismissed, academy_done=payload.academy_done
    )

    counts = _workspace_checklist_counts(db, workspace)
    onboarding = get_onboarding(workspace)
    plan_limits = workspace.plan_limits or {}
    plan_seats = int(plan_limits.get("max_members") or 1)
    return build_checklist(
        connections_count=counts["connections_count"],
        missions_count=counts["missions_count"],
        members_count=counts["members_count"],
        plan_seats=plan_seats,
        comfort=(onboarding.get("segment") or {}).get("comfort"),
        stored=onboarding.get("checklist"),
    )


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


@router.put("/current/integrations", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
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


@router.put("/current/byok-preferences", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
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


@router.put("/current/voice-live", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
async def save_voice_live_settings(
    payload: Dict[str, Any] = Body(...),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """PRD-207 S4/S7: the workspace's own Auto Live gate.

    Body: ``{"voice_live": {"enabled": bool, "monthly_cap_minutes"?: int,
    "retell_voice_id"?: str}}`` — validated by the SAME fail-closed rules as
    the platform tool (one whitelist, two doors), merged never replace-blind.
    """
    from modules.voice.live_settings import validate_voice_live_update

    workspace = db.query(Workspace).get(ctx.workspace_id)
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")

    try:
        normalized = validate_voice_live_update(payload.get("voice_live"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    settings = dict(workspace.settings or {})
    merged = dict(settings.get("voice_live", {}))
    merged.update(normalized)
    settings["voice_live"] = merged
    workspace.settings = settings

    from sqlalchemy.orm.attributes import flag_modified
    flag_modified(workspace, "settings")
    db.commit()

    logger.info("Updated voice_live settings for workspace %s: %s", workspace.id, merged)
    return {"status": "saved", "voice_live": merged}


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
    "preferred_channel": "in_app",
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

# Personality preset base voices — mirror personality.py _PERSONALITY_MAP. These
# are the doctrine-FREE tone strings; legacy GET detection matches against them.
_PERSONALITY_BASE_VOICES = {
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

# What a personality-mode save actually writes to Auto's custom_persona_prompt:
# the base voice PLUS the always-on Manager's Doctrine (PRD-226 P226-RVW-4),
# composed through the SAME builder as the seed default. Two guarantees fall out:
# (1) switching personality mode never strips the doctrine; (2) the written text
# always carries the doctrine, so it can never hash-match the doctrine-free
# entries in _KNOWN_SEED_PERSONA_HASHES — the collision (friendly preset ==
# _ALEMBIC_BACKFILL_PERSONA) that made the doctrine flip-flop out on every save
# and back in on the next deploy's backfill.
_PERSONALITY_PRESETS = {
    mode: compose_persona_with_doctrine(voice)
    for mode, voice in _PERSONALITY_BASE_VOICES.items()
}


def _resolve_persona_for_mode(personality_mode: str, custom_soul: Optional[str]) -> Optional[str]:
    """Resolve the custom_persona_prompt text to write for a personality mode.

    PRD-226 (P226-RVW-4): non-custom modes write the doctrine-carrying preset so a
    settings save never strips the Manager's Doctrine. 'custom' with a non-empty
    soul writes that soul; 'custom' with an empty/absent soul returns ``None`` =
    leave the existing persona untouched (the partial-payload guard that once
    "cost us a 4k-char Irish CTO every night"). Pure and side-effect-free so the
    save behaviour is unit-testable without a DB.
    """
    if personality_mode == "custom":
        return custom_soul if custom_soul else None
    return _PERSONALITY_PRESETS.get(personality_mode, _PERSONALITY_PRESETS["friendly"])


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
                "provider": mc.get("provider", DEFAULT_LLM_PROVIDER),
                "model_id": mc.get("model_id", DEFAULT_LLM_MODEL),
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
                # Legacy: detect preset by text comparison (agents seeded before mode was stored).
                # Match against the doctrine-FREE base voices — that is what those
                # pre-doctrine rows actually hold.
                matched_preset = None
                for mode, text in _PERSONALITY_BASE_VOICES.items():
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

            logger.info(
                "Orchestrator GET for ws=%s: personality_mode=%s, llm_model=%s, auto_agent_id=%s, "
                "stored_config_keys=%s, persona_len=%s",
                ctx.workspace_id, result.get("personality_mode"), result.get("llm", {}).get("model_id"), auto_agent.id,
                list((auto_agent.configuration or {}).keys()),
                len(auto_agent.custom_persona_prompt or ""),
            )
        else:
            mc = get_default_model_config()
            mc["max_tokens"] = 4000
            result["llm"] = mc
    except Exception:
        logger.exception("Failed to read Auto agent config for workspace %s", ctx.workspace_id)
        result["llm"] = None

    return result


@router.put("/current/orchestrator", dependencies=[Depends(require_workspace_permission("workspace:manage"))])
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
    if "preferred_channel" in payload and payload["preferred_channel"] is not None:
        # Allow stored channel keys (e.g. "channel:<uuid>") in addition to platform names.
        pc = payload["preferred_channel"]
        if not isinstance(pc, str):
            raise HTTPException(400, "preferred_channel must be a string")

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

    # PRD-223 S0.1: validate the orchestrator model. This route was the
    # platform's only model-write path with no validation at all — the door
    # the 2026-07-31 quarantine incident walked through. Unknown model → 422;
    # model quarantined for the orchestrator seat → 422 with the policy reason.
    llm_payload = payload.get("llm")
    if llm_payload and isinstance(llm_payload, dict) and llm_payload.get("model_id"):
        requested_model = str(llm_payload["model_id"]).strip()
        from api.llm_marketplace import _get_or_create_from_cache
        from core.llm.model_policy import check_model_for_agent

        if not requested_model or _get_or_create_from_cache(db, requested_model) is None:
            raise HTTPException(
                422,
                f"Unknown model '{requested_model}' — not found in the model catalog. "
                "Sync the OpenRouter catalog or pick a listed model.",
            )
        allowed, reason = check_model_for_agent(
            db, ctx.workspace_id, requested_model, orchestrator_seat=True,
        )
        if not allowed:
            raise HTTPException(
                422,
                f"Model rejected for the orchestrator (Auto) role: {reason}. "
                "See Settings → System → model_policy.",
            )

    # Merge into workspace settings
    settings = dict(workspace.settings or {})
    existing_orch = dict(settings.get("orchestrator", {}))

    # Update top-level orchestrator fields
    for key in ("personality_mode", "custom_soul", "communication_style", "proactive_level", "thinking_level", "preferred_channel"):
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

            # Sync to system_settings so config.LLM_PROVIDER/LLM_MODEL stay in sync
            # (used by internal orchestrator operations: agent_factory, chatbot, etc.)
            try:
                from core.models.system_settings import SystemSetting
                for ss_key, llm_key in [("provider", "provider"), ("model", "model_id")]:
                    if llm_key in llm:
                        ss = db.query(SystemSetting).filter(
                            SystemSetting.category == "orchestrator_llm",
                            SystemSetting.key == ss_key,
                        ).first()
                        if ss:
                            ss.value = llm[llm_key]
                        else:
                            db.add(SystemSetting(
                                category="orchestrator_llm",
                                key=ss_key,
                                value=llm[llm_key],
                                default_value=llm[llm_key],
                                value_type="string",
                                is_required=True,
                            ))
            except Exception:
                logger.warning("Failed to sync LLM settings to system_settings table")
            flag_modified(auto_agent, "model_config")

        # Soul / Personality → Auto agent custom_persona_prompt
        # Defensive: only overwrite custom_persona_prompt when the caller
        # explicitly sends a non-empty value. Partial payloads (API smoke
        # tests, scripts that PATCH only the mode) must NOT wipe an existing
        # persona — that bug cost us a 4k-char Irish CTO every night at 02:00 UTC.
        personality_mode = payload.get("personality_mode")
        if personality_mode:
            resolved_persona = _resolve_persona_for_mode(
                personality_mode, payload.get("custom_soul")
            )
            if resolved_persona is not None:
                auto_agent.custom_persona_prompt = resolved_persona
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
                "provider": mc.get("provider", DEFAULT_LLM_PROVIDER),
                "model_id": mc.get("model_id", DEFAULT_LLM_MODEL),
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
    Queries the memory_short_term table scoped to this workspace.
    """
    from datetime import datetime, timedelta

    workspace_id = ctx.workspace_id

    try:
        from modules.memory.models import MemoryShortTerm
    except ImportError:
        # Graceful fallback if the model isn't available
        return {"total_memories": 0, "by_type": {}, "by_level": {}, "recent": []}

    try:
        # PRD-187 S5: reads the REAL L2 store (the relic memory_items table
        # this used to read held 0 rows, lifetime — the section always showed
        # zeros by construction).
        base = db.query(MemoryShortTerm).filter(MemoryShortTerm.workspace_id == workspace_id)

        total = base.count()

        by_type = dict(
            db.query(MemoryShortTerm.content_type, func.count(MemoryShortTerm.id))
            .filter(MemoryShortTerm.workspace_id == workspace_id)
            .group_by(MemoryShortTerm.content_type)
            .all()
        )

        promoted = (
            db.query(func.count(MemoryShortTerm.id))
            .filter(
                MemoryShortTerm.workspace_id == workspace_id,
                MemoryShortTerm.promoted_to_l3.is_(True),
            )
            .scalar() or 0
        )
        by_level = {"short_term": total - promoted, "promoted_to_durable": promoted}

        agents_with_memories = (
            db.query(func.count(func.distinct(MemoryShortTerm.agent_id)))
            .filter(
                MemoryShortTerm.workspace_id == workspace_id,
                MemoryShortTerm.agent_id.isnot(None),
            )
            .scalar() or 0
        )

        yesterday = datetime.utcnow() - timedelta(hours=24)
        recent_24h = (
            base.filter(MemoryShortTerm.created_at >= yesterday).count()
        )

        # 5 most recent memories (content preview only)
        recent_rows = (
            base.order_by(desc(MemoryShortTerm.created_at))
            .limit(5)
            .all()
        )
        recent = []
        for m in recent_rows:
            content_preview = str(m.content)[:120] if m.content else ""
            recent.append({
                "id": str(m.id),
                "type": m.content_type,
                "level": "promoted_to_durable" if m.promoted_to_l3 else "short_term",
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

