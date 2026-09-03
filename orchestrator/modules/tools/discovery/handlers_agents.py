"""Agent CRUD handlers for PlatformActionExecutor."""

import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


async def list_agents(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from core.models import Agent, agent_skills
    from core.models.composio_cache import AgentAppAssignment

    query = db.query(Agent).filter(Agent.workspace_id == workspace_id)

    status_filter = params.get("status_filter", "all")
    if status_filter != "all":
        query = query.filter(Agent.status == status_filter)

    agents = query.order_by(Agent.id).all()
    agent_ids = [a.id for a in agents]

    # Batch-load tool counts (active assignments only) and skill counts in two grouped queries.
    tool_counts: Dict[int, int] = {}
    skill_counts: Dict[int, int] = {}
    try:
        tool_rows = (
            db.query(
                AgentAppAssignment.agent_id,
                func.count(AgentAppAssignment.id).label("cnt"),
            )
            .filter(
                AgentAppAssignment.agent_id.in_(agent_ids) if agent_ids else False,
                AgentAppAssignment.is_active == True,
            )
            .group_by(AgentAppAssignment.agent_id)
            .all()
        )
        tool_counts = {r.agent_id: r.cnt for r in tool_rows}
    except Exception:
        pass
    try:
        if agent_ids:
            skill_rows = (
                db.query(
                    agent_skills.c.agent_id,
                    func.count(agent_skills.c.skill_id).label("cnt"),
                )
                .filter(agent_skills.c.agent_id.in_(agent_ids))
                .group_by(agent_skills.c.agent_id)
                .all()
            )
            skill_counts = {r.agent_id: r.cnt for r in skill_rows}
    except Exception:
        pass

    agent_list = []
    for a in agents:
        mc = a.model_config or {}
        cfg = a.configuration or {}
        hb = (cfg.get("heartbeat") or {}) if isinstance(cfg, dict) else {}
        agent_list.append({
            "id": a.id,
            "name": a.name,
            "type": a.agent_type,
            "status": a.status,
            "description": (a.description or "")[:200],
            "model_id": mc.get("model_id") or cfg.get("model") or cfg.get("llm_model"),
            "provider": mc.get("provider") or cfg.get("provider"),
            "temperature": mc.get("temperature"),
            "tools_count": tool_counts.get(a.id, 0),
            "skills_count": skill_counts.get(a.id, 0),
            "heartbeat_enabled": bool(hb.get("enabled")),
            "has_persona": bool(a.custom_persona_prompt),
            "tags": a.tags or [],
            "team": getattr(a, "team", None),
            "job_title": getattr(a, "job_title", None),
            "created_at": a.created_at.isoformat() if a.created_at else None,
        })

    return {
        "success": True,
        "agents": agent_list,
        "count": len(agent_list),
    }


async def get_agent(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from core.models import Agent, Skill, agent_skills
    from core.models.composio_cache import AgentAppAssignment

    agent_id = params.get("agent_id")
    agent_name = params.get("agent_name")

    query = db.query(Agent).filter(Agent.workspace_id == workspace_id)
    if agent_id:
        query = query.filter(Agent.id == agent_id)
    elif agent_name:
        query = query.filter(Agent.name.ilike(f"%{agent_name}%"))
    else:
        return {"success": False, "error": "Provide agent_name or agent_id"}

    agent = query.first()
    if not agent:
        return {"success": False, "error": "Agent not found"}

    # Full assigned tool list — names, app type, active state, dates, priority.
    tool_list = []
    try:
        tool_rows = (
            db.query(AgentAppAssignment)
            .filter(AgentAppAssignment.agent_id == agent.id)
            .order_by(AgentAppAssignment.priority.desc(), AgentAppAssignment.assigned_at.desc())
            .all()
        )
        tool_list = [
            {
                "id": t.id,
                "name": t.app_name,
                "app_type": t.app_type,
                "is_active": bool(t.is_active),
                "priority": t.priority or 0,
                "assigned_at": t.assigned_at.isoformat() if t.assigned_at else None,
            }
            for t in tool_rows
        ]
    except Exception as e:
        logger.warning("[get_agent] Failed to load tool assignments: %s", e)

    # Full assigned skill list — joins agent_skills → skills for name, description,
    # category, version, origin (marketplace vs workspace), active state.
    skill_list = []
    try:
        skill_rows = (
            db.query(Skill)
            .join(agent_skills, agent_skills.c.skill_id == Skill.id)
            .filter(agent_skills.c.agent_id == agent.id)
            .order_by(Skill.name)
            .all()
        )
        for s in skill_rows:
            metadata = s.skill_metadata if isinstance(s.skill_metadata, dict) else {}
            skill_list.append({
                "id": s.id,
                "name": s.name,
                "description": (s.description or "")[:200],
                "category": s.category,
                "skill_version": s.skill_version,
                "skill_source": s.skill_source,
                "origin": "workspace" if s.workspace_id is not None else "marketplace",
                "forked_from_skill_id": metadata.get("forked_from_skill_id"),
                "is_active": bool(s.is_active),
            })
    except Exception as e:
        logger.warning("[get_agent] Failed to load skill assignments: %s", e)

    mc = agent.model_config or {}
    config = agent.configuration or {}
    heartbeat = config.get("heartbeat") if isinstance(config, dict) else None

    return {
        "success": True,
        "agent": {
            "id": agent.id,
            "name": agent.name,
            "type": agent.agent_type,
            "status": agent.status,
            "description": agent.description,
            "model_id": mc.get("model_id") or config.get("model") or config.get("llm_model"),
            "provider": mc.get("provider") or config.get("provider") or config.get("llm_provider"),
            "temperature": mc.get("temperature"),
            "max_tokens": mc.get("max_tokens"),
            "has_system_prompt": bool(agent.custom_persona_prompt),
            "system_prompt_preview": (agent.custom_persona_prompt or "")[:200] or None,
            # Full lists (callers that just want counts can read len()):
            "assigned_tools": tool_list,
            "assigned_tools_count": len(tool_list),
            "active_tools_count": sum(1 for t in tool_list if t["is_active"]),
            "assigned_skills": skill_list,
            "assigned_skills_count": len(skill_list),
            "heartbeat": heartbeat,
            "tags": agent.tags or [],
            "team": getattr(agent, "team", None),
            "job_title": getattr(agent, "job_title", None),
            "reports_to_id": getattr(agent, "reports_to_id", None),
            "created_at": agent.created_at.isoformat() if agent.created_at else None,
            "updated_at": agent.updated_at.isoformat() if agent.updated_at else None,
        },
    }


async def create_agent(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from core.models import Agent

    name = params.get("name")
    if not name:
        return {"success": False, "error": "Missing required parameter: name"}

    agent_type = params.get("agent_type", "chatbot")
    description = params.get("description", "")
    model_id = params.get("model_id") or params.get("model")  # back-compat
    system_prompt = params.get("system_prompt")
    temperature = params.get("temperature")
    tags = params.get("tags")

    # Build model_config from shared defaults (core.llm.defaults is the single source)
    from core.llm.defaults import get_default_model_config
    model_config: Dict[str, Any] = get_default_model_config()
    model_note = None
    if model_id:
        # PRD-223 W1: this chat tool was an unvalidated model-write path —
        # any string became an agent's brain, provider guessed by substring.
        # The registry is now the authority for existence AND provider, and
        # the policy gate runs like every other writer.
        from api.llm_marketplace import _get_or_create_from_cache
        from core.llm.model_policy import check_model_for_agent

        resolved = _get_or_create_from_cache(db, model_id)
        if resolved is None:
            # Prod 2026-09-02 (post-#672): told "omit model_id to use the default",
            # the model retried the SAME unknown id twice and the build stalled.
            # An unknown id means the governed default in practice (PRD-223: the
            # registry decides, never the caller's string) — use it and SAY so,
            # in the result and in the log. The agent is still created.
            default_id = model_config.get("model_id")
            logger.warning(
                "[create_agent] unknown model %r — using the workspace default %r", model_id, default_id
            )
            model_note = (
                f"Model '{model_id}' is not in the catalog — the agent uses the workspace "
                f"default ({default_id}) instead."
            )
        else:
            allowed, reason = check_model_for_agent(
                db, workspace_id, model_id, orchestrator_seat=False,
            )
            if not allowed:
                return {"success": False, "error": f"Model rejected: {reason}"}
            model_config["model_id"] = model_id
            model_config["provider"] = resolved.provider
    if temperature is not None:
        model_config["temperature"] = max(0.0, min(2.0, float(temperature)))

    agent = Agent(
        name=name,
        agent_type=agent_type,
        description=description,
        status="active",
        configuration={},
        model_config=model_config,
        workspace_id=workspace_id,
        created_by="platform",
        owner_type="workspace",
        owner_id=str(workspace_id),
    )

    # System prompt -> custom_persona_prompt
    if system_prompt:
        agent.custom_persona_prompt = system_prompt
        agent.use_custom_persona = True

    # Tags
    if tags:
        agent.tags = tags

    # Org fields
    if params.get("team"):
        agent.team = params["team"]
    if params.get("job_title"):
        agent.job_title = params["job_title"]
    if params.get("reports_to_id") is not None:
        agent.reports_to_id = int(params["reports_to_id"])

    db.add(agent)
    db.flush()  # Get the ID without committing (caller commits)

    logger.info(f"[PlatformExecutor] Created agent '{name}' (id={agent.id}) in workspace {workspace_id}")

    return {
        "success": True,
        "agent": {
            "id": agent.id,
            "name": agent.name,
            "type": agent.agent_type,
            "status": agent.status,
            "description": agent.description,
            "model_id": model_config["model_id"],
            "provider": model_config["provider"],
            "temperature": model_config["temperature"],
            "has_system_prompt": bool(system_prompt),
            "tags": agent.tags or [],
        },
        "message": f"Agent '{name}' created successfully with ID {agent.id}."
        + (f" {model_note}" if model_note else ""),
        "model_note": model_note,
    }


async def update_agent(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from core.models import Agent

    agent_id = params.get("agent_id")
    agent_name = params.get("agent_name")

    query = db.query(Agent).filter(Agent.workspace_id == workspace_id)
    if agent_id:
        query = query.filter(Agent.id == agent_id)
    elif agent_name:
        query = query.filter(Agent.name.ilike(f"%{agent_name}%"))
    else:
        return {"success": False, "error": "Provide agent_name or agent_id"}

    agent = query.first()
    if not agent:
        return {"success": False, "error": "Agent not found"}

    changes = []

    # Basic fields
    if params.get("new_name"):
        agent.name = params["new_name"]
        changes.append(f"name -> '{params['new_name']}'")
    if params.get("description") is not None:
        agent.description = params["description"]
        changes.append("description updated")
    if params.get("status"):
        agent.status = params["status"]
        changes.append(f"status -> '{params['status']}'")

    # Model configuration
    model_id = params.get("model_id")
    temperature = params.get("temperature")
    if model_id or temperature is not None:
        mc = dict(agent.model_config or {})
        if model_id:
            mc["model_id"] = model_id
            # Infer provider
            if "claude" in model_id.lower() or "anthropic" in model_id.lower():
                mc["provider"] = "anthropic"
            elif "gemini" in model_id.lower():
                mc["provider"] = "google"
            elif "llama" in model_id.lower() or "mixtral" in model_id.lower():
                mc["provider"] = "groq"
            elif "/" in model_id:
                mc["provider"] = "openrouter"
            else:
                mc["provider"] = "openrouter"
            changes.append(f"model -> '{model_id}'")
        if temperature is not None:
            mc["temperature"] = max(0.0, min(2.0, float(temperature)))
            changes.append(f"temperature -> {mc['temperature']}")
        agent.model_config = mc

    # System prompt
    system_prompt = params.get("system_prompt")
    if system_prompt is not None:
        agent.custom_persona_prompt = system_prompt
        agent.use_custom_persona = True
        changes.append("system prompt updated")

    # Org fields
    team = params.get("team")
    if team is not None:
        agent.team = team
        changes.append(f"team -> '{team}'")
    job_title = params.get("job_title")
    if job_title is not None:
        agent.job_title = job_title
        changes.append(f"job_title -> '{job_title}'")
    reports_to_id = params.get("reports_to_id")
    if reports_to_id is not None:
        agent.reports_to_id = int(reports_to_id) if reports_to_id else None
        changes.append(f"reports_to_id -> {reports_to_id}")

    # Tags
    tags = params.get("tags")
    if tags is not None:
        agent.tags = tags
        changes.append(f"tags -> {tags}")

    if not changes:
        return {"success": True, "message": "No changes specified", "agent_id": agent.id}

    db.flush()
    logger.info(f"[PlatformExecutor] Updated agent {agent.id}: {', '.join(changes)}")

    return {
        "success": True,
        "agent_id": agent.id,
        "changes": changes,
        "message": f"Agent '{agent.name}' updated: {', '.join(changes)}",
    }


async def delete_agent(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Delete an agent. Requires confirmation (handled by execute())."""
    from core.models import Agent

    agent_id = params.get("agent_id")
    agent_name = params.get("agent_name")

    query = db.query(Agent).filter(Agent.workspace_id == workspace_id)
    if agent_id:
        query = query.filter(Agent.id == agent_id)
    elif agent_name:
        query = query.filter(Agent.name.ilike(f"%{agent_name}%"))
    else:
        return {"success": False, "error": "Provide agent_name or agent_id"}

    agent = query.first()
    if not agent:
        return {"success": False, "error": "Agent not found"}

    agent_info = {"id": agent.id, "name": agent.name}
    db.delete(agent)
    db.flush()

    logger.info(f"[PlatformExecutor] Deleted agent {agent_info}")

    return {
        "success": True,
        "deleted_agent": agent_info,
        "message": f"Agent '{agent_info['name']}' (ID {agent_info['id']}) has been deleted.",
    }


# ---------------------------------------------------------------------------
# Focused agent visibility handlers — heartbeat read, unassign tool, unassign skill
# ---------------------------------------------------------------------------


def _resolve_agent(db, workspace_id, params):
    """Resolve an agent by id or name within the workspace. Returns (agent, error_dict)."""
    from core.models import Agent

    agent_id = params.get("agent_id")
    agent_name = params.get("agent_name")

    query = db.query(Agent).filter(Agent.workspace_id == workspace_id)
    if agent_id:
        query = query.filter(Agent.id == agent_id)
    elif agent_name:
        query = query.filter(Agent.name.ilike(f"%{agent_name}%"))
    else:
        return None, {"success": False, "error": "Provide agent_name or agent_id"}

    agent = query.first()
    if not agent:
        return None, {"success": False, "error": "Agent not found"}
    return agent, None


async def get_agent_heartbeat(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Read the heartbeat configuration for an agent.

    Heartbeat config lives in agent.configuration['heartbeat']. Surfaces every
    field the configure handler accepts so an editing agent can read current
    state, decide what to change, and submit the diff via
    platform_configure_agent_heartbeat.
    """
    agent, err = _resolve_agent(db, workspace_id, params)
    if err:
        return err

    config = agent.configuration if isinstance(agent.configuration, dict) else {}
    hb = config.get("heartbeat") if isinstance(config.get("heartbeat"), dict) else {}

    return {
        "success": True,
        "agent": {"id": agent.id, "name": agent.name},
        "heartbeat": {
            "enabled": bool(hb.get("enabled", False)),
            "interval_minutes": hb.get("interval_minutes"),
            "prompt": hb.get("prompt"),
            "checklist": hb.get("checklist"),
            "auto_act": hb.get("auto_act"),
            "active_hours_start": hb.get("active_hours_start"),
            "active_hours_end": hb.get("active_hours_end"),
            "proactive_level": hb.get("proactive_level"),
            "notification_channel": hb.get("notification_channel"),
        },
        "is_configured": bool(hb),
    }


async def unassign_skill_from_agent(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Remove a skill assignment from an agent. Idempotent — no-op if not assigned."""
    from sqlalchemy import and_
    from core.models import Skill, agent_skills

    agent, err = _resolve_agent(db, workspace_id, params)
    if err:
        return err

    skill_id = params.get("skill_id")
    skill_name = params.get("skill_name")
    if not skill_id and not skill_name:
        return {"success": False, "error": "Provide skill_id or skill_name"}

    if skill_name and not skill_id:
        skill = (
            db.query(Skill)
            .filter(
                Skill.name.ilike(f"%{skill_name}%"),
                (Skill.workspace_id.is_(None)) | (Skill.workspace_id == workspace_id),
            )
            .first()
        )
        if not skill:
            return {"success": False, "error": f"Skill '{skill_name}' not found in this workspace"}
        skill_id = skill.id

    result = db.execute(
        agent_skills.delete().where(
            and_(
                agent_skills.c.agent_id == agent.id,
                agent_skills.c.skill_id == skill_id,
            )
        )
    )
    removed = result.rowcount or 0
    db.flush()

    logger.info(
        "[unassign_skill_from_agent] agent=%s skill_id=%s removed=%d",
        agent.name, skill_id, removed,
    )

    return {
        "success": True,
        "agent": {"id": agent.id, "name": agent.name},
        "skill_id": skill_id,
        "removed": removed,
        "message": (
            f"Removed skill {skill_id} from agent '{agent.name}'."
            if removed else
            f"Skill {skill_id} was not assigned to agent '{agent.name}' (no-op)."
        ),
    }


async def unassign_tool_from_agent(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """Remove a tool/app assignment from an agent.

    Default behaviour deactivates (sets is_active=False) so the audit trail
    is preserved. Pass hard_delete=true to drop the row entirely.
    """
    from core.models.composio_cache import AgentAppAssignment

    agent, err = _resolve_agent(db, workspace_id, params)
    if err:
        return err

    app_name = params.get("app_name") or params.get("tool_name")
    assignment_id = params.get("assignment_id")
    hard_delete = bool(params.get("hard_delete", False))

    if not app_name and not assignment_id:
        return {"success": False, "error": "Provide app_name or assignment_id"}

    query = db.query(AgentAppAssignment).filter(AgentAppAssignment.agent_id == agent.id)
    if assignment_id:
        query = query.filter(AgentAppAssignment.id == assignment_id)
    else:
        query = query.filter(AgentAppAssignment.app_name == str(app_name).upper())

    assignment = query.first()
    if not assignment:
        return {
            "success": False,
            "error": f"No assignment found on agent '{agent.name}' for {app_name or assignment_id}",
        }

    target_label = assignment.app_name
    if hard_delete:
        db.delete(assignment)
        action = "deleted"
    else:
        assignment.is_active = False
        action = "deactivated"
    db.flush()

    logger.info(
        "[unassign_tool_from_agent] agent=%s app=%s action=%s",
        agent.name, target_label, action,
    )

    return {
        "success": True,
        "agent": {"id": agent.id, "name": agent.name},
        "tool": target_label,
        "action": action,
        "message": f"Tool '{target_label}' {action} on agent '{agent.name}'.",
    }
