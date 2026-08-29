"""
PRD-230 US-005 — Workspace registration installer (D1/D2/D3).
============================================================

Installs a package or a single marketplace agent into a workspace with its FULL
dependency closure, workspace-owned/editable, idempotently, returning a manifest
of everything registered plus the apps that still need connecting.

REUSE, not reinvent (Hard NO: no parallel registration mechanism). Every
registration rides an EXISTING pattern:
  - agent   → ``clone_agent_to_workspace`` (owner_type='workspace' copy — D3) then
              ``cascade_agent_dependencies`` (LLM → ``workspace_models``, skills →
              ``workspace_enabled_skills``, plugins → ``workspace_enabled_plugins`` +
              ``agent_assigned_plugins``, tools → ``agent_app_assignments``).
  - playbook→ clone the ``workflow_recipes`` row (owner swap — D3) then
              ``cascade_recipe_dependencies`` (member agents + their closures).
  - skill / plugin / llm → ``install_skill`` / ``install_plugin`` / ``install_model``
              (idempotent enablement rows already carry the workspace id).
  - tool    → recorded as a workspace-available registration; connect requirements
              come from agents' app assignments, surfaced as ``required_connects``.

Idempotency (D re-install = zero dupes): the per-type installers are idempotent;
agents are guarded here — an already-cloned marketplace agent is reused, its
cascade re-run (a no-op), and reported ``already_installed``. Closure is atomic:
callers that must respect a tier quota check BEFORE calling install (US-006 D9);
this service never half-installs a closure.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional
from uuid import UUID

logger = logging.getLogger(__name__)

# cascade dep "type" → manifest member type.
_DEP_TYPE = {"model": "llm", "skill": "skill", "plugin": "plugin", "tool": "tool"}
_ADDED_STATUSES = frozenset({"installed", "reactivated", "cloned"})


class PackageInstallError(Exception):
    """Raised when a package/agent ref cannot be resolved for install."""


@dataclass(frozen=True)
class Registration:
    """One workspace registration produced by an install."""

    type: str            # agent | skill | plugin | llm | playbook | tool
    ref: str             # the workspace artifact ref (cloned id / enabled id / name)
    name: str
    status: str          # installed | already_installed | reactivated | cloned | failed
    workspace_owned: bool = True

    @property
    def key(self) -> tuple[str, str]:
        return (self.type, self.ref)

    @property
    def added(self) -> bool:
        return self.status in _ADDED_STATUSES


@dataclass
class InstallManifest:
    """Everything an install registered + the apps that still need connecting."""

    registrations: List[Registration] = field(default_factory=list)
    required_connects: List[dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add(self, reg: Registration) -> None:
        # Dedup by (type, ref): a shared dependency across a package is one row.
        if any(r.key == reg.key for r in self.registrations):
            return
        self.registrations.append(reg)

    def add_required_connect(self, app_name: str, *, app_type: str = "EXTERNAL",
                             needs_oauth: bool = True) -> None:
        key = app_name.upper()
        if any(rc["app_name"].upper() == key for rc in self.required_connects):
            return
        self.required_connects.append(
            {"app_name": app_name, "app_type": app_type, "needs_oauth": needs_oauth}
        )

    def merge(self, other: "InstallManifest") -> None:
        for reg in other.registrations:
            self.add(reg)
        for rc in other.required_connects:
            self.add_required_connect(rc["app_name"], app_type=rc.get("app_type", "EXTERNAL"),
                                      needs_oauth=rc.get("needs_oauth", True))
        self.warnings.extend(other.warnings)

    @property
    def added(self) -> List[Registration]:
        return [r for r in self.registrations if r.added]

    def by_type(self, t: str) -> List[Registration]:
        return [r for r in self.registrations if r.type == t]

    def to_dict(self) -> dict:
        return {
            "registrations": [
                {"type": r.type, "ref": r.ref, "name": r.name, "status": r.status,
                 "workspace_owned": r.workspace_owned}
                for r in self.registrations
            ],
            "required_connects": list(self.required_connects),
            "warnings": list(self.warnings),
            "added_count": len(self.added),
        }


# --------------------------------------------------------------------------- #
# Cascade → manifest
# --------------------------------------------------------------------------- #


def _absorb_cascade(manifest: InstallManifest, cascade: Any) -> None:
    """Fold a CascadeResult's installed dependencies + warnings into the manifest.

    Tool deps are ALSO surfaced as required_connects when they need OAuth — the
    guided connect step (FR-4); nothing is auto-connected.
    """
    for item in getattr(cascade, "cloned_items", []) or []:
        if item.get("type") == "agent":
            manifest.add(Registration("agent", str(item.get("id")), item.get("name") or str(item.get("id")), "cloned"))
    for dep in getattr(cascade, "installed_dependencies", []) or []:
        dtype = _DEP_TYPE.get(dep.get("type"), dep.get("type"))
        status = dep.get("status", "installed")
        status = {"assigned": "installed"}.get(status, status)
        name = dep.get("name") or ""
        manifest.add(Registration(dtype, name, name, status))
        if dep.get("type") == "tool" and dep.get("oauth_required"):
            manifest.add_required_connect(name, needs_oauth=True)
    manifest.warnings.extend(getattr(cascade, "warnings", []) or [])


# --------------------------------------------------------------------------- #
# Single marketplace agent
# --------------------------------------------------------------------------- #


def _is_uuid(value: Any) -> bool:
    """True when ``value`` parses as a UUID.

    ``Agent.public_id`` is a UUID column: comparing it to a non-UUID string
    makes Postgres raise ``InvalidTextRepresentation`` at execute time — an
    error no Python-level guard around the cast can catch, because it happens
    in the database, not in Python.
    """
    try:
        UUID(str(value))
        return True
    except (ValueError, TypeError, AttributeError):
        return False


def _find_marketplace_agent(db: Any, ref: str) -> Any:
    """Resolve a marketplace agent by integer id, public_id (UUID), slug or name.

    LIVE FAILURE (2026-08-29): package members reference agents by SLUG — the
    seed builds them that way on purpose — and this fell through the ``int()``
    guard into a filter that compared ``Agent.public_id`` (uuid) against
    "shopify-ops". Postgres raised

        invalid input syntax for type uuid: "shopify-ops"

    which escaped as an unhandled exception, so EVERY package install failed.
    The ``except (ValueError, TypeError)`` above it only ever caught the int
    cast; the UUID comparison blew up later, inside the driver.

    Each candidate column is now only compared when ``ref`` is the right shape
    for it.
    """
    from core.models.core import Agent

    q = db.query(Agent).filter(Agent.owner_type == "marketplace")

    # Integer primary key.
    try:
        return q.filter(Agent.id == int(ref)).first()
    except (ValueError, TypeError):
        pass

    # Text columns are always safe to compare; public_id only when ref IS a UUID.
    predicate = (Agent.slug == ref) | (Agent.name == ref)
    if _is_uuid(ref):
        predicate = predicate | (Agent.public_id == ref)
    return q.filter(predicate).first()


def _existing_workspace_clone(db: Any, workspace_id: UUID, marketplace_agent: Any) -> Any:
    from core.models.core import Agent

    return (
        db.query(Agent)
        .filter(
            Agent.cloned_from_id == marketplace_agent.id,
            Agent.workspace_id == workspace_id,
            Agent.owner_type == "workspace",
        )
        .first()
    )


async def install_marketplace_agent(
    db: Any, workspace_id: UUID, agent_ref: str, user_id: Optional[int] = None
) -> InstallManifest:
    """Install a marketplace agent with its full closure (D2), workspace-owned (D3).

    Idempotent: an already-installed agent is reused (not re-cloned); its cascade
    is re-run (a no-op) so any missing dependency is back-filled.
    """
    from modules.tools.discovery.cascade_installer import (
        cascade_agent_dependencies,
        clone_agent_to_workspace,
    )

    marketplace_agent = _find_marketplace_agent(db, agent_ref)
    if marketplace_agent is None:
        raise PackageInstallError(f"Marketplace agent not found: {agent_ref}")

    manifest = InstallManifest()
    existing = _existing_workspace_clone(db, workspace_id, marketplace_agent)
    if existing is not None:
        cloned_agent, agent_name, agent_status = existing, existing.name, "already_installed"
    else:
        cloned_agent, agent_name = clone_agent_to_workspace(db, workspace_id, marketplace_agent, user_id)
        agent_status = "cloned"
        marketplace_agent.install_count = (marketplace_agent.install_count or 0) + 1

    manifest.add(Registration("agent", str(cloned_agent.id), agent_name, agent_status))
    cascade = await cascade_agent_dependencies(db, workspace_id, marketplace_agent, cloned_agent)
    _absorb_cascade(manifest, cascade)
    return manifest


# --------------------------------------------------------------------------- #
# Playbook (marketplace recipe) member
# --------------------------------------------------------------------------- #


def _clone_recipe_to_workspace(db: Any, workspace_id: UUID, marketplace_recipe: Any,
                               user_id: Optional[int] = None):
    """Clone a marketplace playbook (``workflow_recipes`` row) into the workspace,
    owner swapped to the workspace (D3). Mirrors ``clone_agent_to_workspace`` and
    the recipe API's inline clone; column values are copied reflectively so new
    recipe columns are preserved without edits here.
    """
    from sqlalchemy import inspect as sa_inspect

    from core.models.core import WorkflowTemplate

    name_exists = (
        db.query(WorkflowTemplate)
        .filter(
            WorkflowTemplate.name == marketplace_recipe.name,
            WorkflowTemplate.workspace_id == workspace_id,
            WorkflowTemplate.owner_type == "workspace",
        )
        .first()
        is not None
    )
    recipe_name = f"{marketplace_recipe.name} (Copy)" if name_exists else marketplace_recipe.name

    base_template_id = (marketplace_recipe.template_id or "playbook").replace("marketplace-", "")
    template_id, counter = base_template_id, 1
    while (
        db.query(WorkflowTemplate)
        .filter(
            WorkflowTemplate.template_id == template_id,
            WorkflowTemplate.workspace_id == workspace_id,
            WorkflowTemplate.owner_type == "workspace",
        )
        .first()
    ):
        template_id, counter = f"{base_template_id}-{counter}", counter + 1

    cloned = WorkflowTemplate()
    identity = {
        "id", "template_id", "name", "workspace_id", "owner_type", "owner_id",
        "cloned_from_id", "created_by_user_id", "created_at", "updated_at",
        "install_count", "use_count",
    }
    for col in sa_inspect(WorkflowTemplate).columns.keys():
        if col not in identity:
            setattr(cloned, col, getattr(marketplace_recipe, col, None))
    cloned.template_id = template_id
    cloned.name = recipe_name
    cloned.workspace_id = workspace_id
    cloned.owner_type = "workspace"
    cloned.owner_id = str(workspace_id)
    cloned.cloned_from_id = marketplace_recipe.id
    cloned.created_by_user_id = user_id
    cloned.install_count = 0
    cloned.use_count = 0
    db.add(cloned)
    db.flush()
    return cloned, recipe_name


async def _install_playbook(db: Any, workspace_id: UUID, ref: str,
                            user_id: Optional[int] = None) -> InstallManifest:
    from modules.tools.discovery.cascade_installer import cascade_recipe_dependencies
    from core.models.core import WorkflowTemplate

    marketplace_recipe = (
        db.query(WorkflowTemplate)
        .filter(WorkflowTemplate.template_id == ref, WorkflowTemplate.owner_type == "marketplace")
        .first()
    )
    if marketplace_recipe is None:
        try:
            marketplace_recipe = db.query(WorkflowTemplate).get(int(ref))
        except (ValueError, TypeError):
            marketplace_recipe = None
    if marketplace_recipe is None:
        raise PackageInstallError(f"Marketplace playbook not found: {ref}")

    manifest = InstallManifest()
    existing = (
        db.query(WorkflowTemplate)
        .filter(
            WorkflowTemplate.cloned_from_id == marketplace_recipe.id,
            WorkflowTemplate.workspace_id == workspace_id,
            WorkflowTemplate.owner_type == "workspace",
        )
        .first()
    )
    if existing is not None:
        cloned_recipe, recipe_name, status = existing, existing.name, "already_installed"
    else:
        cloned_recipe, recipe_name = _clone_recipe_to_workspace(db, workspace_id, marketplace_recipe, user_id)
        status = "cloned"

    manifest.add(Registration("playbook", str(cloned_recipe.id), recipe_name, status))
    cascade = await cascade_recipe_dependencies(
        db=db, workspace_id=workspace_id, marketplace_recipe=marketplace_recipe,
        cloned_recipe=cloned_recipe, user_id_int=user_id,
    )
    _absorb_cascade(manifest, cascade)
    return manifest


# --------------------------------------------------------------------------- #
# Standalone skill / plugin / llm / tool members
# --------------------------------------------------------------------------- #


def _reg_from_install_result(mtype: str, ref: str, result: dict) -> Registration:
    name = str(result.get("name") or ref)
    if not result.get("success", True):
        return Registration(mtype, ref, name, "failed")
    if result.get("already_enabled") or result.get("already_installed"):
        status = "already_installed"
    elif result.get("reactivated"):
        status = "reactivated"
    else:
        status = "installed"
    return Registration(mtype, ref, name, status)


async def _install_leaf(db: Any, workspace_id: UUID, mtype: str, ref: str) -> InstallManifest:
    from modules.tools.discovery.handlers_marketplace import (
        install_model, install_plugin, install_skill,
    )

    manifest = InstallManifest()
    if mtype == "skill":
        result = await install_skill(db, workspace_id, {"skill_id": ref})
    elif mtype == "plugin":
        result = await install_plugin(db, workspace_id, {"plugin_id": ref})
    elif mtype == "llm":
        result = await install_model(db, workspace_id, {"model_id": ref})
    else:  # tool — globally available once present; connect needs ride agent apps
        manifest.add(Registration("tool", ref, ref, "installed"))
        return manifest
    manifest.add(_reg_from_install_result(mtype, ref, result))
    return manifest


# --------------------------------------------------------------------------- #
# Package
# --------------------------------------------------------------------------- #


async def _install_member(db: Any, workspace_id: UUID, mtype: str, ref: str,
                          user_id: Optional[int]) -> InstallManifest:
    if mtype == "agent":
        return await install_marketplace_agent(db, workspace_id, ref, user_id)
    if mtype == "playbook":
        return await _install_playbook(db, workspace_id, ref, user_id)
    if mtype in ("skill", "plugin", "llm", "tool"):
        return await _install_leaf(db, workspace_id, mtype, ref)
    raise PackageInstallError(f"Unknown member type: {mtype!r}")


async def install_package(
    db: Any, workspace_id: UUID, slug: str, user_id: Optional[int] = None
) -> InstallManifest:
    """Install every member of a package with its full closure (D2), workspace-owned
    (D3), idempotent, returning the combined manifest + required_connects (FR-4)."""
    from services.marketplace_packages import get_by_slug

    package = get_by_slug(db, slug)
    if package is None:
        raise PackageInstallError(f"Package not found: {slug}")

    manifest = InstallManifest()
    for member in (package.members or []):
        if not isinstance(member, dict):
            continue
        mtype = member.get("type")
        ref = member.get("ref")
        if ref is None:
            ref = member.get("id") or member.get("slug")
        if not mtype or ref is None:
            manifest.warnings.append(f"Skipped malformed member: {member!r}")
            continue
        sub = await _install_member(db, workspace_id, mtype, str(ref), user_id)
        manifest.merge(sub)

    logger.info(
        "[PackageInstaller] '%s' → %d registrations (%d new), %d connects, %d warnings",
        slug, len(manifest.registrations), len(manifest.added),
        len(manifest.required_connects), len(manifest.warnings),
    )
    return manifest
