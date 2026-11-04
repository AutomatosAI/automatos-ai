"""PRD-270 W2 US-270-13b (§5.6, K14) — locally: an Automatos Cloud agent, written as a marketplace row.

The item pass (US-270-13c's ``import_items``) hands ``upsert_agent`` each agent bundle ``items_feed``
fetched and validated, and commits the pass itself. The agent is upserted by its
``marketplace_cloud_items`` mapping (type 'agent', its externalId):

* no mapping: a NEW marketplace row (owner 'marketplace', no workspace, approved, not featured,
  active, a new public_id, no slug, no configuration) and its mapping;
* a mapping at the bundle's revision whose ``unresolved`` the bundle resolves to again: UNCHANGED, nothing
  written;
* a mapping at another revision, or at the same one while what it names resolves otherwise now (a plugin
  or a skill a later Sync brought, P270W2-RVW-1): the SAME row UPDATED in place (its name, description,
  type, tags, category, icon, prompt, persona, model, apps, skills, plugins and version) and the mapping
  moved. A local operator's approval, featuring and status of the row stand, as do its public_id and
  install count.

The row's version is ``local_version`` (Automatos Cloud's version and the revision's first 12), which
every clone copies, so a workspace copy records the bundle it was cloned from. Its apps are
``agent_tool_assignments`` rows (the raw INSERT ``submit_item`` writes, the table the clone's
``_copy_tool_assignments`` reads), its skills ``agent_skills`` rows in the bundle's order (the first the
highest priority), its plugins ``AgentAssignedPlugin`` rows in the bundle's order; on an update all three
are replaced. What doesn't resolve here (resolve.py: a shared persona, a global skill, an approved
plugin, a model) is left off the row and recorded in the mapping's ``unresolved`` (a skill with the slug
of the plugin it comes from); the item is still written.

Never touched: a row without a mapping (seeded, operator-made or submitted here) and a workspace's copy
of a Cloud row (``cloned_from_id``), which nothing here reads. A REFUSED item writes nothing: its name
held (case-insensitively, as ``cascade_installer._marketplace_agent_named`` finds a Playbook's helpers,
``.first()`` of any of one name) by another marketplace agent, which would make that lookup pick either;
or its mapping naming a row that is no longer a marketplace agent. K14 names the rows without a mapping;
another Cloud agent of the same name refuses it too (the reviews of US-270-13b), for the same reason: the
first one written keeps the name. So does a marketplace agent whose SLUG is that name (the security review of
US-270-14g): an install by reference (``package_installer._find_marketplace_agent``, a seeded package's members)
reads a slug or a name, ``.first()`` of either, so a Cloud agent named after a seeded one's slug could be
installed in its place, without its disclosure. And so does a package or a marketplace Playbook without a mapping
that names an agent so (a member's ref, a recommended agent), even while no agent here has that name: installing
it would add the Cloud agent. A rewrite refused at the revision its mapping holds (a re-resolve,
P270W2-RVW-1) leaves the row as it was: UNCHANGED. Refusals are this module's own sentences, logged with
the externalId (it passed its pattern), never a value of the bundle.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple
from uuid import UUID, uuid4

from sqlalchemy import delete, func, insert, or_, text
from sqlalchemy.orm import Session

from core.models.core import Agent, agent_skills
from core.models.marketplace_cloud import MarketplaceCloudItem
from core.models.marketplace_plugins import AgentAssignedPlugin
from services.marketplace_bundles.canonical import local_version
from services.marketplace_bundles.errors import BundleError
from services.marketplace_bundles.resolve import (
    AGENT_ITEM, MARKETPLACE_OWNER, MODEL, PERSONA, PLUGIN, SKILL, SkillKey, at_revision, global_skills,
    held_as_resolved, model_held, plugins_by_slug, shared_persona_by_slug, unresolved,
)
from services.marketplace_cloud.pull import NEW, REFUSED, UNCHANGED, UPDATED

logger = logging.getLogger(__name__)

ITEM_TYPE = AGENT_ITEM
ACTIVE = "active"  # Agent.status of a row an install may clone

NAME_HELD = "A marketplace agent this install holds already has its name, or has it as its slug."
NAMED_BY_AN_ITEM = "A Marketplace package or Playbook this install holds names an agent by its name."

# An agent member of a package no mapping points at (install_package reads its ref, else its id or slug), or an
# agent a marketplace Playbook no mapping points at recommends (cascade_installer._recommended_agent_names: its
# recommended_agents, else its template definition's), named ``name`` case-insensitively.
_NAMED_BY_AN_ITEM = text(
    "SELECT 1 FROM marketplace_packages p CROSS JOIN LATERAL jsonb_array_elements("
    "CASE WHEN jsonb_typeof(p.members) = 'array' THEN p.members ELSE '[]'::jsonb END) AS m(member) "
    "WHERE m.member ->> 'type' = 'agent' "
    "AND lower(coalesce(m.member ->> 'ref', nullif(m.member ->> 'id', ''), nullif(m.member ->> 'slug', ''))) "
    "IN (lower(:name), :lowered) "
    "AND NOT EXISTS (SELECT 1 FROM marketplace_cloud_items c WHERE c.package_id = p.id) "
    "UNION ALL "
    "SELECT 1 FROM workflow_recipes r CROSS JOIN LATERAL (VALUES (r.recommended_agents), "
    "(r.template_definition -> 'suggested_agents'), (r.template_definition -> 'recommended_agents')) AS lists(names) "
    "CROSS JOIN LATERAL jsonb_array_elements_text("
    "CASE WHEN jsonb_typeof(lists.names) = 'array' THEN lists.names ELSE '[]'::jsonb END) AS a(agent) "
    "WHERE r.owner_type = 'marketplace' AND lower(a.agent) IN (lower(:name), :lowered) "
    "AND NOT EXISTS (SELECT 1 FROM marketplace_cloud_items c WHERE c.recipe_id = r.id) "
    "LIMIT 1"
)
NOT_A_MARKETPLACE_ROW = "The agent this install held for it is gone or is no longer a marketplace agent."
BAD_VERSION = "Its version can't be recorded here."

# What submit_item writes for each tool (api/marketplace.py), the rows the clone copies.
_INSERT_APP = text(
    "INSERT INTO agent_tool_assignments (agent_id, tool_id, enabled, created_at, updated_at)"
    " VALUES (:agent_id, :tool_id, true, NOW(), NOW())"
)
_DELETE_APPS = text("DELETE FROM agent_tool_assignments WHERE agent_id = :agent_id")


class AgentParts(NamedTuple):
    """An agent bundle's references found here: its shared persona, its global skills and approved
    plugins in the bundle's order (each once), its apps (each once) and what didn't resolve."""

    persona_id: Optional[UUID]
    skill_ids: Tuple[int, ...]
    plugin_ids: Tuple[UUID, ...]
    apps: Tuple[str, ...]
    unresolved: Tuple[Dict[str, str], ...]


def upsert_agent(
    db: Session, bundle: Mapping[str, Any], revision: str, publisher: str,
) -> Tuple[str, Optional[Agent]]:
    """Write the agent ``bundle`` (validated, its canonical bytes' sha256 ``revision``) as the Cloud
    marketplace row its mapping holds, with ``publisher`` as its creator's name.

    Answers (outcome, row): NEW or UPDATED with the row written and flushed, UNCHANGED with the row held
    (nothing written: its mapping is at ``revision`` and what it names resolves as it did, or a rewrite at
    that revision was refused, so it stands as it was), or REFUSED with None (nothing written). The caller
    commits; a database error is the caller's to roll back.
    """
    external_id = bundle["externalId"]
    mapping = _mapping(db, external_id)
    row = db.get(Agent, mapping.agent_id) if mapping is not None else None
    parts = resolved_parts(db, bundle[ITEM_TYPE])
    if held_as_resolved(mapping, revision, parts.unresolved):
        return UNCHANGED, row
    refusal = _refusal(db, bundle, mapping, row)
    try:
        version = local_version(bundle["version"], revision)
    except BundleError:
        refusal = refusal or BAD_VERSION
    if refusal is not None:
        logger.warning("Automatos Cloud sync: agent %s is refused: %s", external_id, refusal)
        return (UNCHANGED, row) if at_revision(mapping, revision) else (REFUSED, None)
    fields = agent_fields(bundle, version, publisher, parts)
    outcome, row = _written_row(db, row, fields)
    _replace_links(db, row.id, parts)
    _map(db, mapping, bundle, revision, row.id, parts.unresolved)
    db.flush()
    db.expire(row, ["skills", "assigned_plugins"])
    return outcome, row


def _mapping(db: Session, external_id: str) -> Optional[MarketplaceCloudItem]:
    return (
        db.query(MarketplaceCloudItem)
        .filter(MarketplaceCloudItem.item_type == ITEM_TYPE, MarketplaceCloudItem.external_id == external_id)
        .one_or_none()
    )


def _refusal(
    db: Session, bundle: Mapping[str, Any], mapping: Optional[MarketplaceCloudItem], row: Optional[Agent],
) -> Optional[str]:
    """Why the bundle can't be written, or None: its mapped row isn't a marketplace agent any more, its
    name is held by another marketplace agent, or a package or Playbook without a mapping names its agent so."""
    if mapping is not None and not _is_marketplace_row(row):
        return NOT_A_MARKETPLACE_ROW
    if name_held(db, bundle["name"], row.id if row is not None else None):
        return NAME_HELD
    if named_by_an_item(db, bundle["name"]):
        return NAMED_BY_AN_ITEM
    return None


def _is_marketplace_row(row: Optional[Agent]) -> bool:
    return row is not None and row.owner_type == MARKETPLACE_OWNER and row.workspace_id is None


def name_held(db: Session, name: str, own_id: Optional[int] = None) -> bool:
    """Whether a marketplace agent other than ``own_id`` (the item's own Cloud row, when it has one) is
    called ``name``, or has it as its slug, case-insensitively: by Python's lower-casing, as the Playbook
    install compares, and by the database's, which differ for some non-ASCII names."""
    held_as = (func.lower(Agent.name), func.lower(Agent.slug))
    query = db.query(Agent.id).filter(
        Agent.owner_type == MARKETPLACE_OWNER,
        or_(*(column == said for column in held_as for said in (name.lower(), func.lower(name)))),
    )
    if own_id is not None:
        query = query.filter(Agent.id != own_id)
    return query.first() is not None


def named_by_an_item(db: Session, name: str) -> bool:
    """Whether a Marketplace package or Playbook no mapping points at (seeded or operator-made) names one of its
    agents ``name``: installing it would find this Cloud agent by that name, and add it without its disclosure,
    even while no agent here has the name (the security review of US-270-14g)."""
    return db.execute(_NAMED_BY_AN_ITEM, {"name": name, "lowered": name.lower()}).first() is not None


def resolved_parts(db: Session, body: Mapping[str, Any]) -> AgentParts:
    """The agent object's persona, skills and plugins found here, its apps, and what didn't resolve
    (its persona, model, skills and plugins, in that order)."""
    wanted: List[SkillKey] = [(skill["name"], skill["package"]) for skill in body["skills"]]
    persona_id = shared_persona_by_slug(db, body["persona"])
    skills = global_skills(db, wanted)
    plugins = plugins_by_slug(db, body["plugins"])
    return AgentParts(
        persona_id=persona_id,
        skill_ids=_distinct(skills[key] for key in wanted if key in skills),
        plugin_ids=_distinct(plugins[slug] for slug in body["plugins"] if slug in plugins),
        apps=_distinct(body["apps"]),
        unresolved=tuple(unresolved([
            *_missing_persona_and_model(db, body, persona_id),
            *((SKILL, name, package) for name, package in wanted if (name, package) not in skills),
            *((PLUGIN, slug) for slug in body["plugins"] if slug not in plugins),
        ])),
    )


def _missing_persona_and_model(
    db: Session, body: Mapping[str, Any], persona_id: Optional[UUID],
) -> List[Tuple[str, str]]:
    """The persona the bundle names when no shared one has its slug, and its model when no llm_models row has it."""
    missing: List[Tuple[str, str]] = []
    if body["persona"] and persona_id is None:
        missing.append((PERSONA, body["persona"]))
    model_id = body["model"]["modelId"]
    if model_id and not model_held(db, model_id):
        missing.append((MODEL, model_id))
    return missing


def _distinct(values: Iterable[Any]) -> Tuple[Any, ...]:
    return tuple(dict.fromkeys(values))


def agent_fields(bundle: Mapping[str, Any], version: str, publisher: str, parts: AgentParts) -> Dict[str, Any]:
    """The columns a Sync writes on a Cloud agent row, new or updated."""
    body = bundle[ITEM_TYPE]
    prompt = body["prompt"]
    return {
        "name": bundle["name"],
        "description": bundle["description"],
        "agent_type": body["agentType"],
        "tags": list(bundle["tags"]),
        "marketplace_category": bundle["category"] or None,
        "marketplace_icon": bundle["icon"] or None,
        "version": version,
        "model_config": model_config(body["model"]),
        "persona_id": parts.persona_id,
        "custom_persona_prompt": prompt["text"] or None,
        "use_custom_persona": bool(prompt["custom"]),
        "created_by": publisher,
    }


def model_config(model: Mapping[str, Optional[str]]) -> Optional[Dict[str, str]]:
    """``{model_id, provider}`` of what the bundle names (each only when named), or None for neither."""
    named = {"model_id": model["modelId"], "provider": model["provider"]}
    config = {key: value for key, value in named.items() if value}
    return config or None


def _written_row(db: Session, row: Optional[Agent], fields: Dict[str, Any]) -> Tuple[str, Agent]:
    """A new Cloud marketplace row with ``fields`` (flushed, so it has its id), or ``row`` moved to them."""
    if row is None:
        created = Agent(
            public_id=uuid4(), owner_type=MARKETPLACE_OWNER, owner_id=MARKETPLACE_OWNER, workspace_id=None,
            is_approved=True, is_featured=False, status=ACTIVE, slug=None, configuration={}, **fields,
        )
        db.add(created)
        db.flush()
        return NEW, created
    for column, value in fields.items():
        setattr(row, column, value)
    return UPDATED, row


def _replace_links(db: Session, agent_id: int, parts: AgentParts) -> None:
    """The row's tool assignments, skills and plugins, replaced by ``parts``' (the row's own rows only)."""
    db.execute(_DELETE_APPS, {"agent_id": agent_id})
    if parts.apps:
        db.execute(_INSERT_APP, [{"agent_id": agent_id, "tool_id": app} for app in parts.apps])
    db.execute(delete(agent_skills).where(agent_skills.c.agent_id == agent_id))
    if parts.skill_ids:
        db.execute(insert(agent_skills), _skill_rows(agent_id, parts.skill_ids))
    db.execute(delete(AgentAssignedPlugin).where(AgentAssignedPlugin.agent_id == agent_id))
    if parts.plugin_ids:
        db.execute(insert(AgentAssignedPlugin), [
            {"agent_id": agent_id, "plugin_id": plugin_id, "priority": index}
            for index, plugin_id in enumerate(parts.plugin_ids)
        ])


def _skill_rows(agent_id: int, skill_ids: Sequence[int]) -> List[Dict[str, int]]:
    """The agent's skills in order: the first has the highest priority (the agent's primary skill)."""
    count = len(skill_ids)
    return [
        {"agent_id": agent_id, "skill_id": skill_id, "priority": count - index}
        for index, skill_id in enumerate(skill_ids)
    ]


def _map(
    db: Session, mapping: Optional[MarketplaceCloudItem], bundle: Mapping[str, Any], revision: str,
    agent_id: int, missing: Sequence[Dict[str, str]],
) -> None:
    """A new mapping for a new row; for an update, the mapping moved (its synced_at bumps with it)."""
    if mapping is None:
        db.add(MarketplaceCloudItem(
            item_type=ITEM_TYPE, external_id=bundle["externalId"], version=bundle["version"],
            revision=revision, agent_id=agent_id, unresolved=list(missing),
        ))
        return
    mapping.version = bundle["version"]
    mapping.revision = revision
    mapping.unresolved = list(missing)


__all__ = [
    "AgentParts", "agent_fields", "model_config", "name_held", "named_by_an_item", "resolved_parts", "upsert_agent",
]
