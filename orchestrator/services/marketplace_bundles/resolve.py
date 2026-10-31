"""PRD-270 W2 US-270-13b (K14) — locally: what a bundle names, found among this install's rows.

A bundle names what an item uses by stable keys (refs.py), never by a row id of Automatos Cloud. The
import finds each one here, among the rows any workspace may use:

* a persona by its slug, only a SHARED one (``cascade_installer._shared_persona_id``'s rule: no
  workspace, scope not 'workspace'), so a Cloud agent never takes a workspace's own persona;
* a skill by its name and the slug of the plugin it comes from (Skill has no slug), only a global,
  active one (what ``install_skill`` enables);
* a plugin by its slug, a Cloud or a local one, only approved and active (what ``install_plugin``
  enables);
* a model by its model id, an ``llm_models`` row (what the disclosure calls ``here``);
* an agent by a reference: ``{slug}`` a marketplace agent with that slug (a seeded one), ``{item}``
  the Cloud agent a ``marketplace_cloud_items`` mapping holds for that externalId, ``{name}`` never a
  row. Only an approved marketplace row, with no workspace, ever answers.

What is not found is recorded, never guessed: ``unresolved`` builds the mapping's list of
``{kind, name}`` (at most ``UNRESOLVED_MAX``) that the disclosure shows, and the item is still written;
the existing installs already warn for each one they can't install. Every function only reads.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Tuple
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

from core.models.core import Agent, LLMModel, Skill
from core.models.marketplace_cloud import MarketplaceCloudItem
from core.models.marketplace_plugins import MarketplacePlugin
from core.models.personas import Persona

MARKETPLACE_OWNER = "marketplace"
PLUGIN_APPROVED = "approved"
SHARED_PERSONA_SCOPE = "global"
WORKSPACE_PERSONA_SCOPE = "workspace"
AGENT_ITEM = "agent"

# The kinds of what didn't resolve, as the mapping's `unresolved` records them.
SKILL, PERSONA, PLUGIN, MODEL, HELPER = "skill", "persona", "plugin", "model", "helper"
UNRESOLVED_MAX = 50

# A skill as a bundle names it: its name and the slug of its plugin, or None.
SkillKey = Tuple[str, Optional[str]]


class LocalAgent(NamedTuple):
    """An approved marketplace agent of this install a reference names: its id and its name."""

    id: int
    name: str


def shared_persona_by_slug(db: Session, slug: Optional[str]) -> Optional[UUID]:
    """The id of the shared (global) persona with ``slug``, or None: a workspace's own never answers."""
    if not slug:
        return None
    row = (
        db.query(Persona.id)
        .filter(
            Persona.slug == slug, Persona.workspace_id.is_(None),
            func.coalesce(Persona.scope, SHARED_PERSONA_SCOPE) != WORKSPACE_PERSONA_SCOPE,
        )
        .first()
    )
    return row.id if row is not None else None


def global_skills(db: Session, wanted: Iterable[SkillKey]) -> Dict[SkillKey, int]:
    """Each wanted (name, plugin slug or None) -> the id of the global, active skill with exactly that
    name and plugin (the lowest id when several have them); a key nothing matches is left out."""
    keys = set(wanted)
    if not keys:
        return {}
    rows = (
        db.query(Skill.id, Skill.name, Skill.package_slug)
        .filter(
            Skill.name.in_(sorted({name for name, _package in keys})),
            Skill.workspace_id.is_(None), Skill.is_active.is_(True),
        )
        .order_by(Skill.id)
        .all()
    )
    found: Dict[SkillKey, int] = {}
    for row in rows:
        key = (row.name, row.package_slug)
        if key in keys:
            found.setdefault(key, row.id)
    return found


def plugins_by_slug(db: Session, slugs: Iterable[str]) -> Dict[str, UUID]:
    """Each slug -> the id of the approved, active plugin (a Cloud or a local one) that holds it."""
    wanted = sorted(set(slugs))
    if not wanted:
        return {}
    rows = (
        db.query(MarketplacePlugin.id, MarketplacePlugin.slug)
        .filter(
            MarketplacePlugin.slug.in_(wanted), MarketplacePlugin.approval_status == PLUGIN_APPROVED,
            MarketplacePlugin.is_active.is_(True),
        )
        .all()
    )
    return {row.slug: row.id for row in rows}


def model_held(db: Session, model_id: Optional[str]) -> bool:
    """Whether an ``llm_models`` row has ``model_id``."""
    if not model_id:
        return False
    return db.query(LLMModel.id).filter(LLMModel.model_id == model_id).first() is not None


def agent_for_ref(db: Session, ref: Mapping[str, Any]) -> Optional[LocalAgent]:
    """The approved marketplace agent ``ref`` names: ``{slug}`` one with that slug, ``{item}`` the Cloud
    agent mapped to that externalId; ``{name}`` (shown to the user only) and any other kind, None."""
    query = db.query(Agent.id, Agent.name)
    if "slug" in ref:
        query = query.filter(Agent.slug == ref["slug"])
    elif "item" in ref:
        query = query.join(MarketplaceCloudItem, MarketplaceCloudItem.agent_id == Agent.id).filter(
            MarketplaceCloudItem.item_type == AGENT_ITEM, MarketplaceCloudItem.external_id == ref["item"],
        )
    else:
        return None
    row = (
        query.filter(
            Agent.owner_type == MARKETPLACE_OWNER, Agent.is_approved.is_(True), Agent.workspace_id.is_(None),
        )
        .order_by(Agent.id)
        .first()
    )
    return LocalAgent(row.id, row.name) if row is not None else None


def unresolved(entries: Iterable[Tuple[str, str]]) -> List[Dict[str, str]]:
    """The mapping's ``unresolved``: each distinct (kind, name), in order, as ``{kind, name}``, at most
    UNRESOLVED_MAX."""
    distinct = list(dict.fromkeys(entries))[:UNRESOLVED_MAX]
    return [{"kind": kind, "name": name} for kind, name in distinct]


__all__ = [
    "AGENT_ITEM", "HELPER", "MARKETPLACE_OWNER", "MODEL", "PERSONA", "PLUGIN", "SKILL", "UNRESOLVED_MAX",
    "LocalAgent", "SkillKey", "agent_for_ref", "global_skills", "model_held", "plugins_by_slug",
    "shared_persona_by_slug", "unresolved",
]
