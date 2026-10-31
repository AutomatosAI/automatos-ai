"""PRD-270 W2 US-270-14a (§5.6, K18) — everything a Cloud item will use, read from this install's rows.

The web shows it before a Cloud item is added (US-270-14c's dialog) and Auto's chat install asks with it
(US-270-14g). ``disclosure(db, kind, key)`` answers one item a ``marketplace_cloud_items`` mapping points at
(kind agent|playbook|package|plugin; key its local id, slug or uuid: the key its install route takes), or
raises ``NotACloudItem`` (an unknown kind, a key not of its kind's shape, or a row no mapping points at: a
seeded or an operator-made one). Every answer has

``{kind, name, version (Automatos Cloud's), syncedAt, trust 'reviewed', publisher, unresolved}``

and its kind's part:

* agent: ``prompt`` {source 'custom'|'persona'|None, text, personaName}, the agent factory's rule
  (``AgentFactory._activation_metadata``: its own prompt when it uses one, else its persona's); ``model``
  {modelId, provider, here} (``here``: an ``llm_models`` row has it), or None when it names none; ``apps``
  [{name, needsConnection}] (``cascade_installer.check_oauth_requirements``, read-only: an app it doesn't
  know needs one); ``skills`` [{name, here}] and ``plugins`` [{slug, here}]: the row's own, then the ones
  the mapping records as not found here;
* playbook: ``steps`` [{order, helper, needsHelper, prompt}] as stored (a helper not found is named, with
  needsHelper), ``agents`` (the agents its install adds, by name) and ``apps``;
* package: ``members`` [{type, name, here}] (``here``: its local reference still answers what
  ``install_package`` installs), ``connects`` [{app, optional, note}] (its setup's required connections), and
  its agent members' prompts (``agents`` [{name, prompt}]) and Playbook members' sizes (``playbooks``
  [{name, steps}]);
* plugin: ``scan`` (Automatos Cloud's report, kept in its ``PluginSecurityScan`` row), ``localCheck`` (this
  install's static re-run, the same row), ``contents`` (its counts), ``skills`` (the names it materialised)
  and ``updatesInPlace`` (an update moves the row every workspace that enabled it runs, K15).

``unresolved`` is the mapping's list of what didn't resolve ([{kind, name}]). The publisher is what the Sync
recorded, as W1's public reads name one (``display_name``: never an address): an agent's and a Playbook's
creator name (the bundle's publisher), a package's 'Automatos' (every exported package's, export_packages.py), a
plugin's author. Every answer is
built from local rows only, with no network call; text is returned as stored, and the web renders it as text
(K30).
"""
from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Type
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.orm import Session

from core.models.core import Agent, Skill, WorkflowTemplate, agent_skills
from core.models.marketplace_cloud import CLOUD_ITEM_KEYS, MarketplaceCloudItem
from core.models.marketplace_packages import MarketplacePackage
from core.models.marketplace_plugins import AgentAssignedPlugin, MarketplacePlugin, PluginSecurityScan
from modules.tools.discovery.cascade_installer import check_oauth_requirements
from services.marketplace_bundles.resolve import (
    AGENT_ITEM, MARKETPLACE_OWNER, PLAYBOOK_ITEM, PLUGIN, PLUGIN_APPROVED, SKILL, model_held,
)  # PLUGIN and SKILL: the kinds the mapping's unresolved records
from services.marketplace_bundles.schema import PACKAGE_SLUG_PATTERN
from services.marketplace_public.fields import DEFAULT_PUBLISHER
from services.marketplace_public.publishers import display_name

TRUST = "reviewed"  # every Cloud item is Reviewed on Automatos Cloud, never Official (§8-Q3a)
PACKAGE_ITEM, PLUGIN_ITEM = "package", "plugin"
# A package member's types (marketplace_packages.MEMBER_TYPES) besides an agent and a Playbook.
SKILL_MEMBER, PLUGIN_MEMBER, LLM_MEMBER, TOOL_MEMBER = "skill", "plugin", "llm", "tool"
CUSTOM_PROMPT, PERSONA_PROMPT = "custom", "persona"
INT_KEY_MAX = 2_147_483_647  # an INTEGER id: a larger key names no row
INT_KEY_DIGITS = 10
# A Cloud package's slug passed the bundle format's pattern, so a key off it names no Cloud package.
_PACKAGE_SLUG = re.compile(PACKAGE_SLUG_PATTERN)

# The agent's apps its install copies (cascade_installer._copy_tool_assignments: the enabled ones).
_APPS = text(
    "SELECT tool_id FROM agent_tool_assignments WHERE agent_id = :agent_id AND enabled = true ORDER BY tool_id"
)


class NotACloudItem(LookupError):
    """The kind is unknown, the key isn't of its kind's shape, or no mapping points at that row."""


def disclosure(db: Session, kind: str, key: str) -> Dict[str, Any]:
    """Everything the Cloud item of ``kind`` with the local ``key`` will use, from this install's rows (K18).

    Raises NotACloudItem unless a ``marketplace_cloud_items`` mapping points at that row.
    """
    build = _BUILDERS.get(kind)
    if build is None:
        raise NotACloudItem("not a kind of Cloud item")
    return build(db, key)


# --------------------------------------------------------------------------- the row a mapping points at


def int_key(key: str) -> int:
    """An agent's or a Playbook's id from its route key; NotACloudItem for anything but an INTEGER's digits."""
    if not (key.isascii() and key.isdigit()) or len(key) > INT_KEY_DIGITS or int(key) > INT_KEY_MAX:
        raise NotACloudItem("not an id")
    return int(key)


def is_row_id(value: Any) -> bool:
    """Whether a stored reference is an INTEGER id (a bool is not)."""
    return isinstance(value, int) and not isinstance(value, bool) and 0 < value <= INT_KEY_MAX


def slug_key(key: str) -> str:
    """A package's slug from its route key; NotACloudItem for anything off the bundle format's slug pattern."""
    if _PACKAGE_SLUG.fullmatch(key) is None:
        raise NotACloudItem("not a package slug")
    return key


def uuid_key(key: str) -> UUID:
    """A plugin's id from its route key; NotACloudItem for anything but a uuid."""
    try:
        return UUID(key)
    except ValueError:
        raise NotACloudItem("not a uuid") from None


def mapped(db: Session, kind: str, model: Type[Any], local_id: Any) -> Tuple[MarketplaceCloudItem, Any]:
    """(the mapping, the row) of the ``kind`` row ``local_id`` when a mapping points at it, else NotACloudItem."""
    column = getattr(MarketplaceCloudItem, CLOUD_ITEM_KEYS[kind])
    mapping = (
        db.query(MarketplaceCloudItem)
        .filter(MarketplaceCloudItem.item_type == kind, column == local_id)
        .one_or_none()
    )
    row = db.get(model, local_id) if mapping is not None else None
    if row is None:
        raise NotACloudItem("no mapping points at it")
    return mapping, row


def header(kind: str, name: str, mapping: MarketplaceCloudItem, publisher: Optional[str]) -> Dict[str, Any]:
    """What every disclosure starts with: the item, Automatos Cloud's version, the Sync's time, the trust."""
    return {
        "kind": kind, "name": name, "version": mapping.version, "syncedAt": iso(mapping.synced_at),
        "trust": TRUST, "publisher": publisher, "unresolved": unresolved_of(mapping),
    }


def iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value is not None else None


def unresolved_of(mapping: MarketplaceCloudItem) -> List[Dict[str, str]]:
    """The mapping's ``unresolved``, each ``{kind, name}`` (anything else in it left out)."""
    return [
        {"kind": entry["kind"], "name": entry["name"]}
        for entry in dicts(mapping.unresolved)
        if isinstance(entry.get("kind"), str) and isinstance(entry.get("name"), str)
    ]


def dicts(value: Any) -> List[Mapping[str, Any]]:
    """The objects of a stored JSON list (anything else in it, or a value that isn't a list, gives none)."""
    return [item for item in value if isinstance(item, Mapping)] if isinstance(value, list) else []


def strings(value: Any) -> List[str]:
    """The strings of a stored JSON list."""
    return [item for item in value if isinstance(item, str)] if isinstance(value, list) else []


# --------------------------------------------------------------------------- an agent


def agent_disclosure(db: Session, key: str) -> Dict[str, Any]:
    """A Cloud agent's prompt, model, apps, skills and plugins."""
    mapping, agent = mapped(db, AGENT_ITEM, Agent, int_key(key))
    missing = missing_names(mapping)
    return {
        **header(AGENT_ITEM, agent.name, mapping, display_name(agent.created_by)),
        "prompt": prompt_of(agent),
        "model": model_of(db, agent.model_config),
        "apps": apps_of(db, [row.tool_id for row in db.execute(_APPS, {"agent_id": agent.id})]),
        "skills": _here("name", _agent_skills(db, agent.id), missing.get(SKILL, ())),
        "plugins": _here("slug", _agent_plugins(db, agent.id), missing.get(PLUGIN, ())),
    }


def prompt_of(agent: Agent) -> Dict[str, Optional[str]]:
    """The prompt the agent runs with, by the factory's rule: its own when it uses one, else its persona's."""
    persona = agent.persona
    persona_name = persona.name if persona is not None else None
    if agent.use_custom_persona and agent.custom_persona_prompt:
        return {"source": CUSTOM_PROMPT, "text": agent.custom_persona_prompt, "personaName": None}
    if persona is not None and persona.system_prompt:
        return {"source": PERSONA_PROMPT, "text": persona.system_prompt, "personaName": persona_name}
    return {"source": None, "text": None, "personaName": persona_name}


def model_of(db: Session, model_config: Any) -> Optional[Dict[str, Any]]:
    """``{modelId, provider, here}`` of the model the agent names, or None when it names none."""
    named = model_config if isinstance(model_config, Mapping) else {}
    model_id = named.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        return None
    return {"modelId": model_id, "provider": named.get("provider"), "here": model_held(db, model_id)}


def apps_of(db: Session, apps: Sequence[str]) -> List[Dict[str, Any]]:
    """Each app with whether it needs a connection (OAuth, or an app this install doesn't know)."""
    needs = check_oauth_requirements(db, list(apps))
    return [{"name": app, "needsConnection": needs.get(app.upper(), True)} for app in apps]


def missing_names(mapping: MarketplaceCloudItem) -> Dict[str, List[str]]:
    """The names the mapping records as not found here, by kind."""
    found: Dict[str, List[str]] = {}
    for entry in unresolved_of(mapping):
        found.setdefault(entry["kind"], []).append(entry["name"])
    return found


def _here(field: str, held: Sequence[str], missing: Sequence[str]) -> List[Dict[str, Any]]:
    return [{field: name, "here": True} for name in held] + [{field: name, "here": False} for name in missing]


def _agent_skills(db: Session, agent_id: int) -> List[str]:
    """The agent's skills' names, the highest priority (its primary skill) first."""
    rows = (
        db.query(Skill.name)
        .join(agent_skills, agent_skills.c.skill_id == Skill.id)
        .filter(agent_skills.c.agent_id == agent_id)
        .order_by(agent_skills.c.priority.desc(), Skill.id)
        .all()
    )
    return [row.name for row in rows]


def _agent_plugins(db: Session, agent_id: int) -> List[str]:
    """The agent's plugins' slugs, in their order."""
    rows = (
        db.query(MarketplacePlugin.slug)
        .join(AgentAssignedPlugin, AgentAssignedPlugin.plugin_id == MarketplacePlugin.id)
        .filter(AgentAssignedPlugin.agent_id == agent_id)
        .order_by(AgentAssignedPlugin.priority, MarketplacePlugin.slug)
        .all()
    )
    return [row.slug for row in rows]


# --------------------------------------------------------------------------- a Playbook


def playbook_disclosure(db: Session, key: str) -> Dict[str, Any]:
    """A Cloud Playbook's steps, the agents its install adds and its apps."""
    mapping, row = mapped(db, PLAYBOOK_ITEM, WorkflowTemplate, int_key(key))
    return {
        **header(PLAYBOOK_ITEM, row.name, mapping, display_name(row.created_by)),
        "steps": [step_of(step) for step in dicts(row.steps)],
        "agents": strings(row.recommended_agents),
        "apps": strings(row.required_tools),
    }


def step_of(step: Mapping[str, Any]) -> Dict[str, Any]:
    """A stored step: its order, its helper's name (an agent found here, or the one it needs), whether that
    helper is still needed, and its prompt."""
    helper = step.get("agent_name")
    prompt = step.get("prompt_template")
    return {
        "order": step.get("order"),
        "helper": helper if isinstance(helper, str) else None,
        "needsHelper": bool(step.get("needs_helper")),
        "prompt": prompt if isinstance(prompt, str) else None,
    }


# --------------------------------------------------------------------------- a package


def package_disclosure(db: Session, key: str) -> Dict[str, Any]:
    """A Cloud package's members (each still here or not), its connections, and its agents' prompts and
    Playbooks' sizes."""
    package_id = db.query(MarketplacePackage.id).filter(MarketplacePackage.slug == slug_key(key)).scalar()
    if package_id is None:
        raise NotACloudItem("no package has that slug")
    mapping, row = mapped(db, PACKAGE_ITEM, MarketplacePackage, package_id)
    members = dicts(row.members)
    agents = _member_agents(db, members)
    playbooks = _member_playbooks(db, members)
    return {
        **header(PACKAGE_ITEM, row.name, mapping, DEFAULT_PUBLISHER),
        "members": [
            {"type": member.get("type"), "name": member.get("name"), "here": member_here(db, member, agents, playbooks)}
            for member in members
        ],
        "connects": connects_of(row.setup_manifest),
        "agents": [{"name": agent.name, "prompt": prompt_of(agent)} for agent in agents.values()],
        "playbooks": [{"name": playbook.name, "steps": len(dicts(playbook.steps))} for playbook in playbooks.values()],
    }


def _refs(members: Sequence[Mapping[str, Any]], member_type: str, usable: Callable[[Any], bool]) -> List[Any]:
    """The distinct local references of the members of ``member_type``, in order, the ``usable`` ones only."""
    refs = (member.get("ref") for member in members if member.get("type") == member_type)
    return list(dict.fromkeys(ref for ref in refs if usable(ref)))


def _member_agents(db: Session, members: Sequence[Mapping[str, Any]]) -> Dict[int, Agent]:
    """The package's agent members still here (approved marketplace agents, no workspace), by id, in order."""
    ids = _refs(members, AGENT_ITEM, is_row_id)
    if not ids:
        return {}
    rows = db.query(Agent).filter(
        Agent.id.in_(ids), Agent.owner_type == MARKETPLACE_OWNER, Agent.is_approved.is_(True),
        Agent.workspace_id.is_(None),
    )
    found = {row.id: row for row in rows}
    return {agent_id: found[agent_id] for agent_id in ids if agent_id in found}


def _member_playbooks(db: Session, members: Sequence[Mapping[str, Any]]) -> Dict[str, WorkflowTemplate]:
    """The package's Playbook members still here (approved marketplace Playbooks), by template id, in order."""
    template_ids = _refs(members, PLAYBOOK_ITEM, lambda ref: isinstance(ref, str))
    if not template_ids:
        return {}
    rows = db.query(WorkflowTemplate).filter(
        WorkflowTemplate.template_id.in_(template_ids), WorkflowTemplate.owner_type == MARKETPLACE_OWNER,
        WorkflowTemplate.is_approved.is_(True), WorkflowTemplate.workspace_id.is_(None),
    )
    found = {row.template_id: row for row in rows}
    return {template_id: found[template_id] for template_id in template_ids if template_id in found}


def member_here(
    db: Session, member: Mapping[str, Any], agents: Mapping[int, Agent], playbooks: Mapping[str, WorkflowTemplate],
) -> bool:
    """Whether the member's local reference still answers what ``install_package`` installs (an app always
    does: its connection is the package's ``connects``)."""
    member_type, ref = member.get("type"), member.get("ref")
    if isinstance(ref, bool) or not isinstance(ref, (int, str)):
        return False
    if member_type == AGENT_ITEM:
        return ref in agents
    if member_type == PLAYBOOK_ITEM:
        return ref in playbooks
    check = _MEMBER_CHECKS.get(member_type)
    return check(db, ref) if check is not None else False


def _skill_here(db: Session, ref: Any) -> bool:
    if not is_row_id(ref):
        return False
    query = db.query(Skill.id).filter(Skill.id == ref, Skill.workspace_id.is_(None), Skill.is_active.is_(True))
    return query.first() is not None


def _plugin_here(db: Session, ref: Any) -> bool:
    try:
        plugin_id = UUID(str(ref))
    except ValueError:
        return False
    query = db.query(MarketplacePlugin.id).filter(
        MarketplacePlugin.id == plugin_id, MarketplacePlugin.approval_status == PLUGIN_APPROVED,
        MarketplacePlugin.is_active.is_(True),
    )
    return query.first() is not None


def _model_here(db: Session, ref: Any) -> bool:
    return isinstance(ref, str) and model_held(db, ref)


def _app_here(_db: Session, _ref: Any) -> bool:
    return True


_MEMBER_CHECKS: Mapping[str, Callable[[Session, Any], bool]] = {
    SKILL_MEMBER: _skill_here, PLUGIN_MEMBER: _plugin_here, LLM_MEMBER: _model_here, TOOL_MEMBER: _app_here,
}


def connects_of(setup: Any) -> List[Dict[str, Any]]:
    """The package setup's required connections: each app, whether it's optional, and its note."""
    connects = setup.get("required_connects") if isinstance(setup, Mapping) else None
    return [
        {"app": connect.get("app_name"), "optional": bool(connect.get("optional")), "note": connect.get("note")}
        for connect in dicts(connects)
    ]


# --------------------------------------------------------------------------- a plugin


def plugin_disclosure(db: Session, key: str) -> Dict[str, Any]:
    """A Cloud plugin's two reports (Automatos Cloud's and this install's static check), its contents and the
    skills it made here."""
    mapping, plugin = mapped(db, PLUGIN_ITEM, MarketplacePlugin, uuid_key(key))
    scan = db.get(PluginSecurityScan, plugin.security_scan_id) if plugin.security_scan_id is not None else None
    return {
        **header(PLUGIN_ITEM, plugin.name, mapping, display_name(plugin.original_author)),
        "scan": cloud_report(scan),
        "localCheck": local_check(scan),
        "contents": {
            "skills": plugin.skills_count or 0, "commands": plugin.commands_count or 0,
            "agents": plugin.agents_count or 0, "hooks": plugin.hooks_count or 0,
        },
        "skills": materialised_skills(db, plugin.materialized_skill_ids),
        "updatesInPlace": True,
    }


def cloud_report(scan: Optional[PluginSecurityScan]) -> Optional[Dict[str, Any]]:
    """Automatos Cloud's review of the plugin, as the Sync kept it: its verdict, status, risk, summary, findings."""
    if scan is None:
        return None
    return {
        "verdict": scan.overall_verdict, "status": scan.llm_scan_status, "riskScore": scan.llm_risk_score,
        "summary": scan.llm_summary,
        "findings": [
            {"severity": finding.get("severity"), "description": finding.get("description")}
            for finding in dicts(scan.llm_findings)
        ],
    }


def local_check(scan: Optional[PluginSecurityScan]) -> Optional[Dict[str, Any]]:
    """This install's static re-run over the plugin's files: its status and each finding's place and kind."""
    if scan is None:
        return None
    return {
        "status": scan.static_scan_status,
        "findings": [
            {key: finding.get(key) for key in ("type", "severity", "file", "line", "description")}
            for finding in dicts(scan.static_findings)
        ],
    }


def materialised_skills(db: Session, skill_ids: Any) -> List[str]:
    """The names of the global skills the plugin materialised here."""
    ids = [skill_id for skill_id in skill_ids if is_row_id(skill_id)] if isinstance(skill_ids, list) else []
    if not ids:
        return []
    query = db.query(Skill.name).filter(Skill.id.in_(ids), Skill.workspace_id.is_(None)).order_by(Skill.name, Skill.id)
    return [row.name for row in query]


_BUILDERS: Mapping[str, Callable[[Session, str], Dict[str, Any]]] = {
    AGENT_ITEM: agent_disclosure, PLAYBOOK_ITEM: playbook_disclosure,
    PACKAGE_ITEM: package_disclosure, PLUGIN_ITEM: plugin_disclosure,
}


__all__ = [
    "CUSTOM_PROMPT", "PERSONA_PROMPT", "TRUST", "NotACloudItem", "agent_disclosure", "apps_of", "cloud_report",
    "connects_of", "dicts", "disclosure", "header", "int_key", "is_row_id", "iso", "local_check", "mapped",
    "materialised_skills", "member_here", "missing_names", "model_of", "package_disclosure", "playbook_disclosure",
    "plugin_disclosure", "prompt_of", "slug_key", "step_of", "strings", "unresolved_of", "uuid_key",
]
