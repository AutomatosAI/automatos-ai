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
  row. Only an approved marketplace row, with no workspace, ever answers;
* a Playbook by a reference (US-270-13c, a package's member): ``{templateId}`` a marketplace Playbook
  with that template id (a seeded one), ``{item}`` the Cloud Playbook mapped to that externalId; again
  only an approved marketplace row with no workspace, answered by its template id (what
  ``package_installer.install_playbook`` finds it by).

What is not found is recorded, never guessed: ``unresolved`` builds the mapping's list of
``{kind, name}`` (at most ``UNRESOLVED_MAX``) that the disclosure shows, and the item is still written;
the existing installs already warn for each one they can't install. Every function only reads.

What a later Sync can bring is resolved again once it can be (P270W2-RVW-1): a plugin, a skill a plugin
makes and a Playbook's helper that is a Cloud agent. Their entries also record where each comes from
(``SOURCE_KEYS``: a skill its plugin's slug as ``package``, a helper its agent's externalId as ``item``),
which the disclosure leaves out. ``awaited`` reads that back from a mapping, ``held_now`` finds what of it
this install holds now (or, for an agent, what the Sync's catalogue brings), and ``held_as_resolved`` tells
the writers when an item at its revision resolves as it did (nothing to write).
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, NamedTuple, Optional, Sequence, Tuple
from uuid import UUID

from sqlalchemy import func
from sqlalchemy.orm import Session

from core.models.core import Agent, LLMModel, Skill, WorkflowTemplate
from core.models.marketplace_cloud import MarketplaceCloudItem
from core.models.marketplace_plugins import MarketplacePlugin
from core.models.personas import Persona

MARKETPLACE_OWNER = "marketplace"
PLUGIN_APPROVED = "approved"
SHARED_PERSONA_SCOPE = "global"
WORKSPACE_PERSONA_SCOPE = "workspace"
AGENT_ITEM = "agent"
PLAYBOOK_ITEM = "playbook"

# The kinds of what didn't resolve, as the mapping's `unresolved` records them.
SKILL, PERSONA, PLUGIN, MODEL, HELPER = "skill", "persona", "plugin", "model", "helper"
UNRESOLVED_MAX = 50
# Where what a later Sync can bring comes from, as its unresolved entry records it (P270W2-RVW-1).
SOURCE_KEYS: Mapping[str, str] = MappingProxyType({SKILL: "package", HELPER: "item"})

# A skill as a bundle names it: its name and the slug of its plugin, or None.
SkillKey = Tuple[str, Optional[str]]


class LocalAgent(NamedTuple):
    """An approved marketplace agent of this install a reference names: its id and its name."""

    id: int
    name: str


class Awaited(NamedTuple):
    """What a later Sync can bring: plugin slugs, (skill name, plugin slug) keys and Cloud agents' externalIds."""

    plugins: FrozenSet[str] = frozenset()
    skills: FrozenSet[SkillKey] = frozenset()
    agents: FrozenSet[str] = frozenset()

    def any(self) -> bool:
        """Whether anything is awaited."""
        return bool(self.plugins or self.skills or self.agents)

    def plus(self, other: "Awaited") -> "Awaited":
        """Both awaited sets, as one."""
        return Awaited(self.plugins | other.plugins, self.skills | other.skills, self.agents | other.agents)

    def meets(self, found: "Awaited") -> bool:
        """Whether ``found`` holds any of what this awaits."""
        return bool(self.plugins & found.plugins or self.skills & found.skills or self.agents & found.agents)


NOTHING_AWAITED = Awaited()


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


def playbook_for_ref(db: Session, ref: Mapping[str, Any]) -> Optional[str]:
    """The template id of the approved marketplace Playbook ``ref`` names: ``{templateId}`` one with that
    template id, ``{item}`` the Cloud Playbook mapped to that externalId; any other kind, None."""
    query = db.query(WorkflowTemplate.template_id)
    if "templateId" in ref:
        query = query.filter(WorkflowTemplate.template_id == ref["templateId"])
    elif "item" in ref:
        query = query.join(MarketplaceCloudItem, MarketplaceCloudItem.recipe_id == WorkflowTemplate.id).filter(
            MarketplaceCloudItem.item_type == PLAYBOOK_ITEM, MarketplaceCloudItem.external_id == ref["item"],
        )
    else:
        return None
    row = (
        query.filter(
            WorkflowTemplate.owner_type == MARKETPLACE_OWNER, WorkflowTemplate.is_approved.is_(True),
            WorkflowTemplate.workspace_id.is_(None),
        )
        .first()
    )
    return row.template_id if row is not None else None


def unresolved(entries: Iterable[Sequence[Optional[str]]]) -> List[Dict[str, str]]:
    """The mapping's ``unresolved``: each distinct (kind, name), in order, as ``{kind, name}``, at most
    UNRESOLVED_MAX. An entry given as (kind, name, source) also records its source under ``SOURCE_KEYS``
    (the first one given for that kind and name), when its kind has one and the source isn't empty."""
    first: Dict[Tuple[str, str], Optional[str]] = {}
    for kind, name, *source in entries:
        first.setdefault((kind, name), source[0] if source else None)
    return [_entry(kind, name, source) for (kind, name), source in list(first.items())[:UNRESOLVED_MAX]]


def _entry(kind: str, name: str, source: Optional[str]) -> Dict[str, str]:
    key = SOURCE_KEYS.get(kind)
    return {"kind": kind, "name": name, key: source} if key is not None and source else {"kind": kind, "name": name}


def at_revision(mapping: Optional[MarketplaceCloudItem], revision: str) -> bool:
    """Whether ``mapping`` holds its item at ``revision``."""
    return mapping is not None and mapping.revision == revision


def held_as_resolved(
    mapping: Optional[MarketplaceCloudItem], revision: str, missing: Sequence[Mapping[str, str]],
) -> bool:
    """Whether ``mapping`` holds its item at ``revision`` with ``missing`` as its ``unresolved``: the item resolves
    as it did, so a Sync writes nothing (UNCHANGED). At the same revision with another resolution (a plugin, a
    skill or a helper arrived) the item is written again (P270W2-RVW-1)."""
    return at_revision(mapping, revision) and mapping.unresolved == list(missing)


def awaited(missing: Any) -> Awaited:
    """What a mapping's stored ``unresolved`` records that a later Sync can bring: its plugins, its skills that
    name their plugin and its helpers that name their Cloud agent (anything else in it left out)."""
    entries = _stored_entries(missing)
    return Awaited(
        plugins=frozenset(name for kind, name, _source in entries if kind == PLUGIN),
        skills=frozenset((name, source) for kind, name, source in entries if kind == SKILL and source),
        agents=frozenset(source for kind, _name, source in entries if kind == HELPER and source),
    )


def _stored_entries(missing: Any) -> List[Tuple[str, str, Optional[str]]]:
    """A stored ``unresolved`` as (kind, name, source); an entry that isn't ``{kind, name}`` strings is left out."""
    if not isinstance(missing, list):
        return []
    return [
        (entry["kind"], entry["name"], _source_of(entry))
        for entry in missing
        if isinstance(entry, Mapping) and isinstance(entry.get("kind"), str) and isinstance(entry.get("name"), str)
    ]


def _source_of(entry: Mapping[str, Any]) -> Optional[str]:
    source = entry.get(SOURCE_KEYS.get(entry["kind"], ""))
    return source if isinstance(source, str) and source else None


def held_now(db: Session, wanted: Awaited, arriving: FrozenSet[str] = frozenset()) -> Awaited:
    """Of ``wanted``, what resolves here now: an approved, active plugin; a global, active skill; a Cloud agent
    that is an approved marketplace row, or one of ``arriving`` (the agents whose bundles the Sync just fetched)."""
    return Awaited(
        plugins=frozenset(plugins_by_slug(db, wanted.plugins)),
        skills=frozenset(global_skills(db, wanted.skills)),
        agents=(wanted.agents & arriving) | cloud_agents(db, wanted.agents - arriving),
    )


def cloud_agents(db: Session, external_ids: Iterable[str]) -> FrozenSet[str]:
    """Of ``external_ids``, those a mapping holds as an approved marketplace agent (``agent_for_ref``'s rule)."""
    wanted = sorted(set(external_ids))
    if not wanted:
        return frozenset()
    rows = (
        db.query(MarketplaceCloudItem.external_id)
        .join(Agent, MarketplaceCloudItem.agent_id == Agent.id)
        .filter(
            MarketplaceCloudItem.item_type == AGENT_ITEM, MarketplaceCloudItem.external_id.in_(wanted),
            Agent.owner_type == MARKETPLACE_OWNER, Agent.is_approved.is_(True), Agent.workspace_id.is_(None),
        )
        .all()
    )
    return frozenset(row.external_id for row in rows)


__all__ = [
    "AGENT_ITEM", "HELPER", "MARKETPLACE_OWNER", "MODEL", "NOTHING_AWAITED", "PERSONA", "PLAYBOOK_ITEM", "PLUGIN",
    "SKILL", "SOURCE_KEYS", "UNRESOLVED_MAX", "Awaited", "LocalAgent", "SkillKey", "agent_for_ref", "at_revision",
    "awaited",
    "cloud_agents", "global_skills", "held_as_resolved", "held_now", "model_held", "playbook_for_ref",
    "plugins_by_slug", "shared_persona_by_slug", "unresolved",
]
