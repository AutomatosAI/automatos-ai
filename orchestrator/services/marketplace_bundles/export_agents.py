"""PRD-270 W2 US-270-12a (K9, K10) — Automatos Cloud's approved community agents, as bundles.

The rows: ``owner_type='marketplace'``, approved, active, no slug (a slug marks an agent every local
edition seeds itself, seeded.py), not a system agent, no required role. Each lineage (lineage.py)
exports its newest row. The query names its columns, so an agent's configuration (runtime-, CLI- or
machine-bound, or free-form), owner, workspace, ``created_by`` and install count are never read.

The bundle carries the agent's prompt (its custom persona prompt and whether it is used), the slug of
its persona only when that persona is shared (``cascade_installer._shared_persona_id``'s rule: a
workspace's own persona never crosses), its model's id and provider from ``model_config``, its global
skills by name and plugin (a workspace's own skill is left out), its enabled tool assignments as
Composio app names, upper-cased (what the clone assigns), and its assigned plugins by slug.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from sqlalchemy import bindparam, func, text
from sqlalchemy.orm import Session

from core.models.core import Agent, Skill, agent_skills
from core.models.marketplace_plugins import AgentAssignedPlugin, MarketplacePlugin
from core.models.personas import Persona
from services.marketplace_bundles.exported import (
    MARKETPLACE_OWNER, ExportedItem, envelope, exported_item, newest_per_lineage, publisher_name,
)
from services.marketplace_bundles.seeded import is_seeded_agent
from services.marketplace_public.publishers import publisher_names

ITEM_TYPE = "agent"
ACTIVE = "active"
SHARED_PERSONA_SCOPE = "global"
WORKSPACE_PERSONA_SCOPE = "workspace"

_COLUMNS = (
    Agent.id, Agent.public_id, Agent.cloned_from_id, Agent.original_creator_id, Agent.slug, Agent.name,
    Agent.description, Agent.agent_type, Agent.tags, Agent.marketplace_category, Agent.marketplace_icon,
    Agent.version, Agent.custom_persona_prompt, Agent.use_custom_persona, Agent.persona_id, Agent.model_config,
)
# An agent's apps, as the clone reads them (cascade_installer._copy_tool_assignments): enabled only.
_ENABLED_APPS = text(
    "SELECT agent_id, tool_id FROM agent_tool_assignments WHERE enabled IS TRUE AND agent_id IN :ids"
).bindparams(bindparam("ids", expanding=True))


@dataclass(frozen=True)
class _Parts:
    """What the exported agents' bundles take from other tables, by agent id (personas by persona id)."""

    skills: Mapping[int, List[Dict[str, Any]]]
    apps: Mapping[int, List[str]]
    plugins: Mapping[int, List[str]]
    personas: Mapping[Any, str]


def exported_agents(db: Session) -> List[ExportedItem]:
    """One ExportedItem per lineage of Automatos Cloud's exportable community agents, newest row first;
    an agent whose bundle validate_bundle refuses is logged and left out."""
    lineages = newest_per_lineage(ITEM_TYPE, _exportable_rows(db))
    if not lineages:
        return []
    rows = [lineage.newest for lineage in lineages]
    ids = [row.id for row in rows]
    parts = _Parts(
        skills=_skills(db, ids), apps=_apps(db, ids), plugins=_plugins(db, ids),
        personas=_shared_personas(db, (row.persona_id for row in rows)),
    )
    names = publisher_names(db, (row.original_creator_id for row in rows))
    items = (
        exported_item(ITEM_TYPE, lineage, envelope(
            ITEM_TYPE, lineage, publisher_name(names, lineage.newest.original_creator_id),
            _body(lineage.newest, parts),
        ))
        for lineage in lineages
    )
    return [item for item in items if item is not None]


def _exportable_rows(db: Session) -> List[Any]:
    """K9's agents, newest first (the id breaks a tie), less the ones a local edition seeds."""
    rows = (
        db.query(*_COLUMNS)
        .filter(
            Agent.owner_type == MARKETPLACE_OWNER, Agent.is_approved.is_(True), Agent.status == ACTIVE,
            Agent.is_system_agent.isnot(True), Agent.required_role.is_(None),
        )
        .order_by(Agent.created_at.desc().nullslast(), Agent.id.desc())
        .all()
    )
    return [row for row in rows if not is_seeded_agent(row.slug)]


def _body(row: Any, parts: _Parts) -> Dict[str, Any]:
    """The agent object of ``row``'s bundle."""
    model = row.model_config if isinstance(row.model_config, dict) else {}
    return {
        "agentType": row.agent_type,
        "prompt": {"custom": bool(row.use_custom_persona), "text": row.custom_persona_prompt or ""},
        "persona": parts.personas.get(row.persona_id),
        "model": {"modelId": _text_or_none(model.get("model_id")), "provider": _text_or_none(model.get("provider"))},
        "skills": parts.skills.get(row.id, []),
        "apps": parts.apps.get(row.id, []),
        "plugins": parts.plugins.get(row.id, []),
    }


def _text_or_none(value: Any) -> Optional[str]:
    return value if isinstance(value, str) and value else None


def _skills(db: Session, ids: Sequence[int]) -> Dict[int, List[Dict[str, Any]]]:
    """Each agent's global skills (workspace_id NULL), highest priority first, as {name, package}."""
    rows = (
        db.query(agent_skills.c.agent_id, Skill.name, Skill.package_slug)
        .select_from(agent_skills)
        .join(Skill, Skill.id == agent_skills.c.skill_id)
        .filter(agent_skills.c.agent_id.in_(ids), Skill.workspace_id.is_(None))
        .order_by(agent_skills.c.agent_id, agent_skills.c.priority.desc(), Skill.name, Skill.id)
        .all()
    )
    found: Dict[int, Dict[tuple, Dict[str, Any]]] = {}
    for row in rows:
        found.setdefault(row.agent_id, {}).setdefault(
            (row.name, row.package_slug), {"name": row.name, "package": row.package_slug},
        )
    return {agent_id: list(skills.values()) for agent_id, skills in found.items()}


def _apps(db: Session, ids: Sequence[int]) -> Dict[int, List[str]]:
    """Each agent's enabled tool assignments as distinct Composio app names, upper-cased and sorted."""
    found: Dict[int, set] = {}
    for agent_id, tool_id in db.execute(_ENABLED_APPS, {"ids": list(ids)}):
        if isinstance(tool_id, str) and tool_id.strip():
            found.setdefault(agent_id, set()).add(tool_id.strip().upper())
    return {agent_id: sorted(apps) for agent_id, apps in found.items()}


def _plugins(db: Session, ids: Sequence[int]) -> Dict[int, List[str]]:
    """Each agent's assigned plugins' slugs, in the assignment's priority order."""
    rows = (
        db.query(AgentAssignedPlugin.agent_id, MarketplacePlugin.slug)
        .join(MarketplacePlugin, MarketplacePlugin.id == AgentAssignedPlugin.plugin_id)
        .filter(AgentAssignedPlugin.agent_id.in_(ids))
        .order_by(AgentAssignedPlugin.agent_id, AgentAssignedPlugin.priority.asc(), MarketplacePlugin.slug)
        .all()
    )
    found: Dict[int, List[str]] = {}
    for row in rows:
        found.setdefault(row.agent_id, []).append(row.slug)
    return found


def _shared_personas(db: Session, persona_ids: Iterable[Any]) -> Dict[Any, str]:
    """persona id -> slug, for the shared (global) personas among ``persona_ids`` only."""
    ids = {persona_id for persona_id in persona_ids if persona_id is not None}
    if not ids:
        return {}
    rows = (
        db.query(Persona.id, Persona.slug)
        .filter(
            Persona.id.in_(ids), Persona.workspace_id.is_(None),
            func.coalesce(Persona.scope, SHARED_PERSONA_SCOPE) != WORKSPACE_PERSONA_SCOPE,
        )
        .all()
    )
    return {row.id: row.slug for row in rows}


__all__ = ["exported_agents"]
