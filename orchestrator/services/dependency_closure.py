"""
PRD-230 US-004 — Dependency-closure resolver.
=============================================

D2: installing an artifact registers its FULL dependency closure. An agent brings
its LLM, skills, plugins, and (as *connect requirements*, not installs) its
connected apps; a playbook brings its member agents, recursed. Nothing
half-installed, nothing platform-dangling.

The walk is a PURE function over a ``DependencyReader`` — the graph shape lives in
``resolve_closure`` (cycle-safe, deterministic, dedup), the DB lives behind the
reader. That split makes the canonical invariant ("agent A = 3 tools + 2 skills +
1 LLM ⇒ 7") testable without Postgres, and lets the installer (US-005) drive the
exact same walk over ``DbDependencyReader``.

Closure edges (per member type):
  - agent    → LLM (``model_config``) + skills (``agent_skills``) + plugins
               (``agent_assigned_plugins``); apps (``agent_app_assignments``)
               surface as ``required_connects`` — guided steps, never installs (FR-4).
  - playbook → its member agents (recursed).
  - skill / plugin / tool / llm → leaves (self only).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, runtime_checkable

# The member-type vocabulary (mirrors core.models.marketplace_packages.MEMBER_TYPES).
AGENT, TOOL, SKILL, PLUGIN, PLAYBOOK, LLM = "agent", "tool", "skill", "plugin", "playbook", "llm"


@dataclass(frozen=True)
class TypedRef:
    """A typed artifact reference — the unit of a package member and the closure."""

    type: str
    ref: str

    @property
    def key(self) -> tuple[str, str]:
        return (self.type, self.ref)


@dataclass(frozen=True)
class RequiredConnect:
    """An app requirement derived from an agent's Composio assignments. Surfaced as
    a guided connect step (D7-②), NEVER auto-connected (FR-4)."""

    app_name: str
    app_type: str = "EXTERNAL"
    via_agent: Optional[str] = None

    @property
    def key(self) -> str:
        return self.app_name.upper()


@dataclass
class ClosureResult:
    """The resolved closure: installable ``members`` (ordered, deduped, incl. the
    root) + ``required_connects`` (distinct, not installable)."""

    members: List[TypedRef] = field(default_factory=list)
    required_connects: List[RequiredConnect] = field(default_factory=list)

    def by_type(self, t: str) -> List[TypedRef]:
        return [m for m in self.members if m.type == t]


@runtime_checkable
class DependencyReader(Protocol):
    """Reads one artifact's direct dependencies. Implementations are DB-read-only."""

    def agent_llm(self, ref: str) -> Optional[str]: ...
    def agent_skills(self, ref: str) -> List[str]: ...
    def agent_plugins(self, ref: str) -> List[str]: ...
    def agent_apps(self, ref: str) -> List[RequiredConnect]: ...
    def playbook_members(self, ref: str) -> List[TypedRef]: ...


def resolve_closure(root: TypedRef, reader: DependencyReader) -> ClosureResult:
    """Walk the dependency graph from ``root`` and return its full closure (PURE).

    Breadth-first for a deterministic, stable order; a ``visited`` set on
    ``(type, ref)`` makes it cycle-safe (a self- or mutually-referential playbook
    terminates) and idempotent (a shared dependency is registered once).
    """
    result = ClosureResult()
    seen_members: set[tuple[str, str]] = set()
    seen_connects: set[str] = set()
    visited: set[tuple[str, str]] = set()
    queue: List[TypedRef] = [root]

    while queue:
        cur = queue.pop(0)
        if cur.key in visited:
            continue
        visited.add(cur.key)

        if cur.key not in seen_members:
            result.members.append(cur)
            seen_members.add(cur.key)

        if cur.type == AGENT:
            llm = reader.agent_llm(cur.ref)
            if llm:
                queue.append(TypedRef(LLM, str(llm)))
            for skill_ref in reader.agent_skills(cur.ref):
                queue.append(TypedRef(SKILL, str(skill_ref)))
            for plugin_ref in reader.agent_plugins(cur.ref):
                queue.append(TypedRef(PLUGIN, str(plugin_ref)))
            for rc in reader.agent_apps(cur.ref):
                if rc.key not in seen_connects:
                    result.required_connects.append(rc)
                    seen_connects.add(rc.key)
        elif cur.type == PLAYBOOK:
            for member in reader.playbook_members(cur.ref):
                queue.append(member)
        # skill / plugin / tool / llm are leaves — nothing to expand.

    return result


def resolve_many(roots: List[TypedRef], reader: DependencyReader) -> ClosureResult:
    """Closure of a SET of roots (a whole package) merged into one deduped result.

    A 6-agent package ⇒ all six closures, deduped where they share an LLM/skill/
    plugin, with app requirements pooled once (D2).
    """
    merged = ClosureResult()
    seen_members: set[tuple[str, str]] = set()
    seen_connects: set[str] = set()
    for root in roots:
        one = resolve_closure(root, reader)
        for m in one.members:
            if m.key not in seen_members:
                merged.members.append(m)
                seen_members.add(m.key)
        for rc in one.required_connects:
            if rc.key not in seen_connects:
                merged.required_connects.append(rc)
                seen_connects.add(rc.key)
    return merged


# --------------------------------------------------------------------------- #
# The DB reader — DB-read-only. The pure walk above is the tested core; this maps
# the well-defined agent edges onto it. Every lookup degrades to empty on a miss
# so a partial graph never crashes the resolver.
# --------------------------------------------------------------------------- #


class DbDependencyReader:
    """Reads dependency edges from Postgres for ``resolve_closure`` (read-only)."""

    def __init__(self, db: Any):
        self.db = db

    def _agent(self, ref: str) -> Any:
        from core.models.core import Agent

        try:
            return self.db.query(Agent).get(int(ref))
        except (ValueError, TypeError):
            return self.db.query(Agent).filter(Agent.public_id == ref).one_or_none()

    def agent_llm(self, ref: str) -> Optional[str]:
        agent = self._agent(ref)
        if not agent:
            return None
        mc = agent.model_config or {}
        return mc.get("model_id") or mc.get("model") or None

    def agent_skills(self, ref: str) -> List[str]:
        agent = self._agent(ref)
        if not agent:
            return []
        return [str(s.id) for s in (agent.skills or [])]

    def agent_plugins(self, ref: str) -> List[str]:
        agent = self._agent(ref)
        if not agent:
            return []
        return [str(ap.plugin_id) for ap in (agent.assigned_plugins or [])]

    def agent_apps(self, ref: str) -> List[RequiredConnect]:
        from core.models.composio_cache import AgentAppAssignment

        try:
            agent_id = int(ref)
        except (ValueError, TypeError):
            agent = self._agent(ref)
            if not agent:
                return []
            agent_id = agent.id
        rows = (
            self.db.query(AgentAppAssignment)
            .filter(
                AgentAppAssignment.agent_id == agent_id,
                AgentAppAssignment.is_active.is_(True),
            )
            .all()
        )
        return [
            RequiredConnect(app_name=r.app_name, app_type=r.app_type or "EXTERNAL", via_agent=str(agent_id))
            for r in rows
        ]

    def playbook_members(self, ref: str) -> List[TypedRef]:
        """Member agents of a marketplace playbook (``workflow_recipes``).

        Agents are read from the 9-stage ``steps[].agent_id`` first, then the
        legacy ``template_definition.agents`` — deduped, order-preserving. The
        Playbook mechanism spans several representations (PRD-142 WS-3R); this
        reads the marketplace-recipe one and returns [] for anything else, so the
        playbook still installs as a member even when its agent linkage is absent.
        """
        from core.models.core import WorkflowTemplate

        playbook = (
            self.db.query(WorkflowTemplate)
            .filter(WorkflowTemplate.template_id == ref)
            .one_or_none()
        )
        if playbook is None:
            try:
                playbook = self.db.query(WorkflowTemplate).get(int(ref))
            except (ValueError, TypeError):
                playbook = None
        if playbook is None:
            return []

        agent_refs: List[str] = []
        seen: set[str] = set()

        def _add(value: Any) -> None:
            if value in (None, ""):
                return
            key = str(value)
            if key not in seen:
                seen.add(key)
                agent_refs.append(key)

        for step in (playbook.steps or []):
            if isinstance(step, dict):
                _add(step.get("agent_id"))
        definition = playbook.template_definition or {}
        if isinstance(definition, dict):
            for agent in definition.get("agents", []) or []:
                _add(agent.get("id") if isinstance(agent, dict) else agent)

        return [TypedRef(AGENT, r) for r in agent_refs]
