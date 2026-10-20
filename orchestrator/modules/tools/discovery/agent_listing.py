"""PRD-256 N17-8 (F440): what platform_list_agents and platform_get_agent say an agent runs on.

Night 16: WRITER, a Claude Code agent (configuration provider 'claude', model 'sonnet'), was
said to run "on gpt-4o through OpenRouter", to wait for a host serving Anthropic, and to be "on
an older Claude Sonnet build". The listing named the model route (``model_config``) first, and a
session agent never uses it: the CLI host runs ``configuration.provider`` and ``.model``. A
session agent's entry now names its CLI and the CLI's model; an API agent's is as it was.

The two handlers' reads (tool and skill counts, the assigned tools and skills) live here too, so
each handler stays short enough to change.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Tuple

from core.cli_runtime import CONFIG_MODEL_KEY, RUNTIME_CLI, runtime_kind_of, session_cli

logger = logging.getLogger(__name__)

DESCRIPTION_CHARS = 200


def runtime_of(cfg: Any) -> str:
    """'cli' (the owner's own CLI session on their machine) or 'api' (a platform-run model)."""
    return runtime_kind_of(cfg if isinstance(cfg, dict) else {})


def runs_on(cfg: Any, api_model: Any, api_provider: Any) -> Dict[str, Any]:
    """An agent's ``model_id`` and ``provider`` as the agent tools name them: a session agent's
    CLI (``claude`` when unset) and the CLI's model, never its model route, which does nothing
    for it; an API agent's ``api_model`` and ``api_provider``, as given."""
    if runtime_of(cfg) == RUNTIME_CLI:
        return {"model_id": cfg.get(CONFIG_MODEL_KEY) or None, "provider": session_cli(cfg)}
    return {"model_id": api_model, "provider": api_provider}


def tool_and_skill_counts(db: Any, agent_ids: List[int]) -> Tuple[Dict[int, int], Dict[int, int]]:
    """Each agent's active tool assignments and its skills, in two grouped queries. A count
    that can't be read is left out: the listing stands either way."""
    if not agent_ids:
        return {}, {}
    return _counted(db, agent_ids, _tool_rows, "tool"), _counted(db, agent_ids, _skill_rows, "skill")


def _counted(db: Any, agent_ids: List[int], rows: Callable[[Any, List[int]], List[Any]],
             what: str) -> Dict[int, int]:
    try:
        return {row.agent_id: row.cnt for row in rows(db, agent_ids)}
    except Exception:  # noqa: BLE001 — a count never fails the listing
        logger.exception("[list_agents] could not count the %ss of agents %s", what, agent_ids)
        return {}


def _tool_rows(db: Any, agent_ids: List[int]) -> List[Any]:
    from sqlalchemy import func

    from core.models.composio_cache import AgentAppAssignment

    return (db.query(AgentAppAssignment.agent_id, func.count(AgentAppAssignment.id).label("cnt"))
            .filter(AgentAppAssignment.agent_id.in_(agent_ids), AgentAppAssignment.is_active.is_(True))
            .group_by(AgentAppAssignment.agent_id).all())


def _skill_rows(db: Any, agent_ids: List[int]) -> List[Any]:
    from sqlalchemy import func

    from core.models import agent_skills

    return (db.query(agent_skills.c.agent_id, func.count(agent_skills.c.skill_id).label("cnt"))
            .filter(agent_skills.c.agent_id.in_(agent_ids))
            .group_by(agent_skills.c.agent_id).all())


def assigned_tools(db: Any, agent_id: int) -> List[Dict[str, Any]]:
    """The agent's assigned tools: names, app type, active state, priority and date."""
    from core.models.composio_cache import AgentAppAssignment

    try:
        rows = (db.query(AgentAppAssignment).filter(AgentAppAssignment.agent_id == agent_id)
                .order_by(AgentAppAssignment.priority.desc(), AgentAppAssignment.assigned_at.desc()).all())
    except Exception:  # noqa: BLE001 — the agent is described without them
        logger.exception("[get_agent] could not load the tool assignments of agent %s", agent_id)
        return []
    return [{"id": t.id, "name": t.app_name, "app_type": t.app_type, "is_active": bool(t.is_active),
             "priority": t.priority or 0, "assigned_at": t.assigned_at.isoformat() if t.assigned_at else None}
            for t in rows]


def assigned_skills(db: Any, agent_id: int) -> List[Dict[str, Any]]:
    """The agent's assigned skills (agent_skills → skills): name, description, category,
    version, origin (marketplace vs workspace) and active state."""
    from core.models import Skill, agent_skills

    try:
        rows = (db.query(Skill).join(agent_skills, agent_skills.c.skill_id == Skill.id)
                .filter(agent_skills.c.agent_id == agent_id).order_by(Skill.name).all())
    except Exception:  # noqa: BLE001 — the agent is described without them
        logger.exception("[get_agent] could not load the skill assignments of agent %s", agent_id)
        return []
    return [_skill(s) for s in rows]


def _skill(skill: Any) -> Dict[str, Any]:
    metadata = skill.skill_metadata if isinstance(skill.skill_metadata, dict) else {}
    return {
        "id": skill.id,
        "name": skill.name,
        "description": (skill.description or "")[:DESCRIPTION_CHARS],
        "category": skill.category,
        "skill_version": skill.skill_version,
        "skill_source": skill.skill_source,
        "origin": "workspace" if skill.workspace_id is not None else "marketplace",
        "forked_from_skill_id": metadata.get("forked_from_skill_id"),
        "is_active": bool(skill.is_active),
    }


__all__ = ["assigned_skills", "assigned_tools", "runs_on", "runtime_of", "tool_and_skill_counts"]
