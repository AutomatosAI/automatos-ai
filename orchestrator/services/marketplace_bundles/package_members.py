"""PRD-270 W2 US-270-12b (K8, K9) — a package's members as bundle references.

A package row's member names its target the way ``services/package_installer.install_package`` finds it:
its ``ref`` (else its ``id`` or ``slug``) as text; an agent by its integer id, else its slug, public_id or
name, among marketplace agents; a Playbook by its template id among marketplace Playbooks, else its row
id; a skill by its id among active global skills; a plugin by its id; a model by its model id; a tool by
its app. The export resolves each the same way, reading marketplace and global rows only (a workspace's
row is never loaded), and names it by what a local install can find (K8's refs): a seeded agent by its
slug, an exported agent by its item; a seeded Playbook by its template id, an exported Playbook by its
item; a skill by its name and, when a plugin made it, that plugin's slug; a plugin by its slug; a model
by its id; a tool by its app, upper-cased. A plugin, and a plugin a skill came from, must be one the
plugin export lists (a local install has no other, and an unlisted plugin's slug is never published). A
member that resolves to none of these has no reference the format allows (``target_of`` answers None)
and its package is left out (export_packages.py).

Where install's lookup of an agent by text can match more than one row, the export takes a slug before a
public_id before a name, and the lowest id among rows of one name: a stable choice.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Dict, FrozenSet, Iterable, List, Mapping, Optional, Set

from sqlalchemy import or_
from sqlalchemy.orm import Session

from core.models.core import Agent, Skill, WorkflowTemplate as WorkflowRecipe
from services.marketplace_bundles.exported import MARKETPLACE_OWNER, ExportedItem, items_by_row
from services.marketplace_bundles.seeded import is_seeded_agent, seeded_template_ids

ROW_ID_MAX = 2 ** 31 - 1  # the agents', Playbooks' and skills' Integer ids
PLUGIN_TYPE = "plugin"
# install's lookup of an agent by text, the later column winning: a name, then a public_id, then a slug.
AGENT_TEXT_COLUMNS = ("name", "public_id", "slug")


@dataclass(frozen=True)
class Target:
    """A member's reference in the bundle, and the name its target goes by (for a member without one)."""

    ref: Mapping[str, str]
    name: str


@dataclass(frozen=True)
class _Rows:
    """Rows found by an integer id, and by a text key (an agent's slug, public_id or name; a Playbook's
    template id)."""

    by_id: Mapping[int, Any]
    by_text: Mapping[str, Any]


@dataclass(frozen=True)
class _Plugin:
    """A listed plugin, as a package member names it."""

    slug: str
    name: str


@dataclass(frozen=True)
class MemberRefs:
    """What the packages' members resolve to: the rows they name, the listed plugins by row id, and the
    row ids naming exported items."""

    agents: _Rows
    playbooks: _Rows
    skills: Mapping[int, Any]
    plugins: Mapping[Any, _Plugin]
    agent_items: Mapping[Any, str]
    playbook_items: Mapping[Any, str]

    @property
    def plugin_slugs(self) -> FrozenSet[str]:
        """The slugs of the plugins the export lists."""
        return frozenset(plugin.slug for plugin in self.plugins.values())

    def target_of(self, member_type: Any, text: Optional[str]) -> Optional[Target]:
        """The Target of a member of ``member_type`` naming ``text``; None when it has no allowed ref."""
        resolve = _TARGETS.get(member_type) if isinstance(member_type, str) else None
        return resolve(self, text) if resolve is not None and text is not None else None


def raw_ref(member: Any) -> Optional[str]:
    """A member's target as install_package reads it: its ``ref``, else its ``id`` or ``slug``, as text."""
    if not isinstance(member, dict):
        return None
    value = member.get("ref")
    if value is None:
        value = member.get("id") or member.get("slug")
    return None if value is None else str(value)


def member_refs(
    db: Session, members: Iterable[Any], agent_items: Iterable[ExportedItem], playbook_items: Iterable[ExportedItem],
    plugin_items: Iterable[ExportedItem],
) -> MemberRefs:
    """Every row ``members`` name, read in one query per kind, with the exported items' row ids and the
    listed plugins (``plugin_items``, ``exported_plugins``' items)."""
    wanted: Dict[str, Set[str]] = {}
    for member in members:
        text = raw_ref(member)
        if text is not None and isinstance(member.get("type"), str):
            wanted.setdefault(member["type"], set()).add(text)
    return MemberRefs(
        agents=_agents(db, frozenset(wanted.get("agent", ()))),
        playbooks=_playbooks(db, frozenset(wanted.get("playbook", ()))),
        skills=_skills(db, frozenset(wanted.get("skill", ()))),
        plugins=_listed_plugins(plugin_items),
        agent_items=items_by_row(agent_items, "agent"),
        playbook_items=items_by_row(playbook_items, "playbook"),
    )


def as_int(text: str) -> Optional[int]:
    """``text`` as install's ``int(ref)`` reads it, else None."""
    try:
        return int(text)
    except ValueError:
        return None


def as_uuid(text: str) -> Optional[uuid.UUID]:
    """``text`` as a UUID, else None."""
    try:
        return uuid.UUID(text)
    except ValueError:
        return None


def _row_ids(texts: Iterable[str]) -> List[int]:
    """The texts that read as integers a row id can hold, sorted."""
    return sorted({number for number in map(as_int, texts) if number is not None and 0 < number <= ROW_ID_MAX})


def _agents(db: Session, texts: FrozenSet[str]) -> _Rows:
    """The marketplace agents ``texts`` name, by id and by slug, public_id or name."""
    words = sorted(text for text in texts if as_int(text) is None)
    ids, uuids = _row_ids(texts), sorted({value for value in map(as_uuid, words) if value is not None})
    if not ids and not words:
        return _Rows({}, {})
    rows = (
        db.query(Agent.id, Agent.public_id, Agent.slug, Agent.name)
        .filter(Agent.owner_type == MARKETPLACE_OWNER, or_(
            Agent.id.in_(ids), Agent.slug.in_(words), Agent.name.in_(words), Agent.public_id.in_(uuids),
        ))
        .order_by(Agent.id.desc())
        .all()
    )
    by_text: Dict[str, Any] = {}
    for column in AGENT_TEXT_COLUMNS:
        by_text.update({str(getattr(row, column)): row for row in rows if getattr(row, column) is not None})
    return _Rows({row.id: row for row in rows}, by_text)


def _playbooks(db: Session, texts: FrozenSet[str]) -> _Rows:
    """The marketplace Playbooks ``texts`` name, by template id and by id."""
    if not texts:
        return _Rows({}, {})
    rows = (
        db.query(WorkflowRecipe.id, WorkflowRecipe.template_id, WorkflowRecipe.name)
        .filter(WorkflowRecipe.owner_type == MARKETPLACE_OWNER, or_(
            WorkflowRecipe.template_id.in_(sorted(texts)), WorkflowRecipe.id.in_(_row_ids(texts)),
        ))
        .all()
    )
    return _Rows({row.id: row for row in rows}, {row.template_id: row for row in rows})


def _skills(db: Session, texts: FrozenSet[str]) -> Dict[int, Any]:
    """The active global skills ``texts`` name by id (install_skill refuses an inactive one)."""
    ids = _row_ids(texts)
    if not ids:
        return {}
    rows = db.query(Skill.id, Skill.name, Skill.package_slug).filter(
        Skill.workspace_id.is_(None), Skill.is_active.is_(True), Skill.id.in_(ids),
    )
    return {row.id: row for row in rows.all()}


def _listed_plugins(items: Iterable[ExportedItem]) -> Dict[Any, _Plugin]:
    """Each listed plugin's row id -> its slug and name, from its bundle."""
    return {
        row_id: _Plugin(item.bundle[PLUGIN_TYPE]["slug"], item.bundle["name"])
        for item in items if item.entry.type == PLUGIN_TYPE
        for row_id in item.entry.row_ids
    }


def _agent_target(refs: MemberRefs, text: str) -> Optional[Target]:
    number = as_int(text)
    row = refs.agents.by_id.get(number) if number is not None else _agent_by_text(refs.agents, text)
    if row is None:
        return None
    if is_seeded_agent(row.slug):
        return Target({"slug": row.slug}, row.name)
    item = refs.agent_items.get(row.id)
    return Target({"item": item}, row.name) if item is not None else None


def _agent_by_text(rows: _Rows, text: str) -> Optional[Any]:
    """An agent by slug, public_id or name; a UUID in another case or form finds its public_id too."""
    found = rows.by_text.get(text)
    as_id = as_uuid(text) if found is None else None
    return rows.by_text.get(str(as_id)) if as_id is not None else found


def _playbook_target(refs: MemberRefs, text: str) -> Optional[Target]:
    number = as_int(text)
    row = refs.playbooks.by_text.get(text)
    if row is None and number is not None:
        row = refs.playbooks.by_id.get(number)
    if row is None:
        return None
    if row.template_id in seeded_template_ids():
        return Target({"templateId": row.template_id}, row.name)
    item = refs.playbook_items.get(row.id)
    return Target({"item": item}, row.name) if item is not None else None


def _skill_target(refs: MemberRefs, text: str) -> Optional[Target]:
    row = refs.skills.get(as_int(text))
    if row is None or (row.package_slug and row.package_slug not in refs.plugin_slugs):
        return None
    return Target({"name": row.name, **({"package": row.package_slug} if row.package_slug else {})}, row.name)


def _plugin_target(refs: MemberRefs, text: str) -> Optional[Target]:
    row = refs.plugins.get(as_uuid(text))
    return Target({"slug": row.slug}, row.name) if row is not None else None


def _llm_target(_refs: MemberRefs, text: str) -> Target:
    return Target({"modelId": text}, text)


def _tool_target(_refs: MemberRefs, text: str) -> Target:
    app = text.strip().upper()
    return Target({"app": app}, app)


_TARGETS: Mapping[str, Callable[[MemberRefs, str], Optional[Target]]] = MappingProxyType({
    "agent": _agent_target, "playbook": _playbook_target, "skill": _skill_target,
    "plugin": _plugin_target, "llm": _llm_target, "tool": _tool_target,
})


__all__ = ["MemberRefs", "Target", "as_int", "as_uuid", "member_refs", "raw_ref"]
