"""PRD-270 US-270-07a (K10) — the public read of the Marketplace's agents.

Approved ``owner_type='marketplace'`` Agent rows, most installed first then newest (the member Agents
tab's order, ``api/marketplace.py`` ``list_items``), with the id as the last tie so pages never overlap.
The query names its columns: the row is never loaded whole, so an agent's configuration, model config,
persona prompt, owner, clone and workspace ids are never read, let alone answered. An agent's required
apps are its enabled tool assignments, as app slugs; its publisher is ``publishers.py``'s.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from core.models.core import Agent
from services.marketplace_public.fields import AGENT_FIELDS, MARKETPLACE_OWNER, display_item
from services.marketplace_public.publishers import publisher_names, publisher_of

# The app slugs of the page's agents, as the member route reads them (enabled assignments only).
_ENABLED_TOOLS = text(
    "SELECT agent_id, tool_id FROM agent_tool_assignments "
    "WHERE enabled IS TRUE AND agent_id IN :ids ORDER BY agent_id, tool_id"
).bindparams(bindparam("ids", expanding=True))


def apps_by_agent(db: Session, agent_ids: Sequence[int]) -> Dict[int, List[str]]:
    """Each listed agent's enabled tool assignments, in tool order (featured.py reads them too)."""
    found: Dict[int, List[str]] = {}
    for agent_id, tool_id in db.execute(_ENABLED_TOOLS, {"ids": list(agent_ids)}):
        found.setdefault(agent_id, []).append(tool_id)
    return found


def read_agents(db: Session, limit: int, offset: int) -> List[Dict[str, Any]]:
    """Up to ``limit`` approved Marketplace agents from ``offset``, each as its public item."""
    rows = (
        db.query(
            Agent.id, Agent.name, Agent.description, Agent.marketplace_category, Agent.marketplace_icon,
            Agent.tags, Agent.original_creator_id,
        )
        .filter(Agent.owner_type == MARKETPLACE_OWNER, Agent.is_approved.is_(True))
        .order_by(Agent.install_count.desc().nullslast(), Agent.created_at.desc().nullslast(), Agent.id.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    if not rows:
        return []
    apps = apps_by_agent(db, [row.id for row in rows])
    names = publisher_names(db, (row.original_creator_id for row in rows))
    return [
        display_item(
            AGENT_FIELDS, key=row.id, name=row.name, description=row.description,
            category=row.marketplace_category, icon=row.marketplace_icon, tags=row.tags,
            publisher=publisher_of(names, row.original_creator_id), required_apps=apps.get(row.id, ()),
        )
        for row in rows
    ]


__all__ = ["apps_by_agent", "read_agents"]
