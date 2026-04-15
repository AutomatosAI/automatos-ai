"""Agent CRUD handlers for PlatformActionExecutor."""

import logging
from typing import Any, Dict
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


async def list_agents(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    from core.models import Agent
    from core.models.composio_cache import AgentAppAssignment

    query = db.query(Agent).filter(Agent.workspace_id == workspace_id)

    status_filter = params.get("status_filter", "all")
    if status_filter != "all":
        query = query.filter(Agent.status == status_filter)

    agents = query.order_by(Agent.id).all()

    # Batch-load tool counts for all agents in one query
    tool_counts = {}
    try:
        rows = (
            db.query(
                AgentAppAssignment.agent_id,
                func.count(AgentAppAssignment.id).label("cnt"),
            )
            .filter(AgentAppAssignment.is_active == True)
            .group_by(AgentAppAssignment.agent_id)
            .all()
        )
        tool_counts = {r.agent_id: r.cnt for r in rows}
    except Exception:
        pass

    agent_list = []
    for a in agents:
        mc = a.model_config or {}
        cfg = a.configuration or {}
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
            "has_persona": bool(a.custom_persona_prompt),
            "tags": a.tags or [],
            "created_at": a.created_at.isoformat() if a.created_at else None,
        })

    return {
        "success": True,
        "agents": agent_list,
        "count": len(agent_list),
    }


async def get_agent(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
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

    # Get assigned tools count
    tool_count = 0
    try:
        from core.models.composio_cache import AgentAppAssignment
        tool_count = (
            db.query(AgentAppAssignment)
            .filter(AgentAppAssignment.agent_id == agent.id, AgentAppAssignment.is_active == True)
            .count()
        )
    except Exception:
        pass

    mc = agent.model_config or {}
    config = agent.configuration or {}
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
            "has_system_prompt": bool(agent.custom_persona_prompt),
            "system_prompt_preview": (agent.custom_persona_prompt or "")[:200] or None,
            "assigned_tools": tool_count,
            "tags": agent.tags or [],
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
    if model_id:
        model_config["model_id"] = model_id
        # Infer provider from model name — slash-format = OpenRouter
        if "/" in model_id:
            model_config["provider"] = "openrouter"
        elif "claude" in model_id.lower() or "anthropic" in model_id.lower():
            model_config["provider"] = "anthropic"
        elif "gemini" in model_id.lower():
            model_config["provider"] = "google"
        elif "llama" in model_id.lower() or "mixtral" in model_id.lower():
            model_config["provider"] = "groq"
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
        "message": f"Agent '{name}' created successfully with ID {agent.id}.",
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
