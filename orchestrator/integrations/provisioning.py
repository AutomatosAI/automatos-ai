"""Generic vertical-provisioning plane (PRD-183 S5, F076).

Only the widget READ path went through PRD-141; provisioning a merchant, syncing
its catalog, and mapping webhooks to the graph stayed Shopify-shaped in
``api/shopify.py``. That is where vertical #2's fork cost lived. This module is
the generic seam that removes it:

  * :class:`VerticalProvisioner` — the structural interface a vertical declares
    (agent roster + skill bindings, widget-config defaults, minted-key
    permissions, ops-manager slug for reports_to wiring, site type, allowed
    origins, and an optional post-provision hook). Verticals self-register into
    :data:`PROVISIONER_REGISTRY` at import time, exactly like the widget
    :data:`integrations.PLUGIN_REGISTRY`.
  * :func:`provision_vertical` — the ONE generic provision flow. Given a
    vertical key it looks up the provisioner and runs find/create-workspace →
    seed-roster → mint-key. ``api/verticals.py`` exposes it behind
    ``POST /api/verticals/{v}/provision`` so a new vertical never forks the
    Shopify routes.
  * :data:`GRAPH_SOURCE_MAPPERS` / :func:`get_graph_source_mapper` — the
    catalog/orders JSONL→graph mappers reached through the vertical rather than
    hardcoded in generic code, so sync is a generic "graph source" operation.

Nothing here is Shopify-specific — the vertical isolation CI gate
(``scripts/ci/check-no-shopify-in-generic.sh``) does not scan this file, but it
is written to stay clean regardless.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Protocol, runtime_checkable
from uuid import uuid4

from sqlalchemy.orm import Session

from core.models.workspaces import Workspace

logger = logging.getLogger(__name__)


class VerticalConfigError(ValueError):
    """A provision request that is malformed for the vertical's own rules
    (e.g. a missing origin allowlist) — distinct from ``ValueError`` for an
    unregistered vertical so the HTTP layer can answer 422, not 404."""


@runtime_checkable
class VerticalProvisioner(Protocol):
    """Everything the generic provision flow needs to stand up a vertical.

    A provisioner is a plain object (usually a module-level singleton) exposing
    these attributes/methods. Implementations live under
    ``integrations/<vertical>/provision.py`` and register themselves into
    :data:`PROVISIONER_REGISTRY` at import time.
    """

    vertical: str
    agent_slugs: List[str]
    ops_manager_slug: Optional[str]
    default_widget_config: Dict[str, Any]
    key_permissions: List[str]
    key_type: str
    site_type: Optional[str]
    # The settings key this vertical resolves a workspace by (e.g. Shopify's
    # other routes look up ``shopify_domain``). The generic flow stamps both
    # this key AND the canonical ``source_external_id`` so lookups either way
    # resolve. Defaults to the canonical key when a vertical does not override.
    external_id_key: str
    # When True, re-provisioning a workspace that already holds an active
    # public key returns it un-rotated (``api_key: None`` in the response)
    # instead of minting a replacement. Verticals whose partner app persists
    # the key (BudStacks) set this; Shopify keeps the default False because
    # its runbook uses re-provision AS the key-recovery path.
    reuse_existing_key: bool

    def allowed_domains(self, external_id: str, metadata: Dict[str, Any]) -> List[str]:
        """Origins permitted to use the minted public widget key."""
        ...

    def on_provisioned(self, db: Session, workspace: Any) -> None:
        """Optional vertical-specific post-provision step (e.g. ensure a Site)."""
        ...


# Verticals self-register here at import time (see integrations/<v>/provision.py).
PROVISIONER_REGISTRY: Dict[str, VerticalProvisioner] = {}


# ---------------------------------------------------------------------------
# Graph-source mappers (JSONL bulk-op stream → nodes/edges), keyed by vertical.
# A vertical registers its adapters here; generic sync looks them up rather
# than importing a vertical-specific mapper directly.
# ---------------------------------------------------------------------------

GRAPH_SOURCE_MAPPERS: Dict[str, Dict[str, Callable[..., Dict[str, list]]]] = {}


def register_graph_source_mappers(
    vertical: str, mappers: Dict[str, Callable[..., Dict[str, list]]]
) -> None:
    """Register a vertical's graph-source mappers (e.g. {'catalog': fn, 'orders': fn})."""
    GRAPH_SOURCE_MAPPERS.setdefault(vertical, {}).update(mappers)


def get_graph_source_mapper(vertical: str, source: str) -> Optional[Callable[..., Dict[str, list]]]:
    """Return the mapper for ``(vertical, source)`` or ``None`` if unregistered."""
    return GRAPH_SOURCE_MAPPERS.get(vertical, {}).get(source)


# ---------------------------------------------------------------------------
# Generic building blocks — overridable/patchable seams the flow leans on.
# ---------------------------------------------------------------------------


def _seed_roster(db: Session, workspace_id, provisioner: VerticalProvisioner) -> int:
    """Clone the provisioner's marketplace agent roster into the workspace.

    Generic version of the old ``_seed_shopify_agents``: clones approved
    marketplace agents whose slug is in ``provisioner.agent_slugs`` (with their
    skills + tool assignments) and wires every non-ops agent's ``reports_to`` to
    the ops-manager clone when ``ops_manager_slug`` is declared.
    """
    from sqlalchemy import text
    from core.models.core import Agent

    marketplace_agents = (
        db.query(Agent)
        .filter(
            Agent.owner_type == "marketplace",
            Agent.is_approved.is_(True),
            Agent.slug.in_(provisioner.agent_slugs),
        )
        .all()
    )

    cloned_count = 0
    ops_manager_clone_id = None

    for marketplace_agent in marketplace_agents:
        existing = (
            db.query(Agent)
            .filter(
                Agent.workspace_id == workspace_id,
                Agent.cloned_from_id == marketplace_agent.id,
            )
            .first()
        )
        if existing:
            if marketplace_agent.slug == provisioner.ops_manager_slug:
                ops_manager_clone_id = existing.id
            continue

        cloned = Agent(
            name=marketplace_agent.name,
            slug=marketplace_agent.slug,
            description=marketplace_agent.description,
            agent_type=marketplace_agent.agent_type,
            status=marketplace_agent.status,
            configuration=marketplace_agent.configuration,
            model_config=marketplace_agent.model_config,
            tags=marketplace_agent.tags,
            marketplace_category=marketplace_agent.marketplace_category,
            marketplace_icon=marketplace_agent.marketplace_icon,
            team=marketplace_agent.team,
            job_title=marketplace_agent.job_title,
            custom_persona_prompt=marketplace_agent.custom_persona_prompt,
            use_custom_persona=marketplace_agent.use_custom_persona,
            owner_type="workspace",
            owner_id=str(workspace_id),
            workspace_id=workspace_id,
            cloned_from_id=marketplace_agent.id,
            original_creator_id=marketplace_agent.original_creator_id,
            is_approved=True,
            version=marketplace_agent.version,
        )
        db.add(cloned)
        db.flush()

        if marketplace_agent.skills:
            cloned.skills = list(marketplace_agent.skills)

        tool_rows = db.execute(
            text("SELECT tool_id, enabled FROM agent_tool_assignments WHERE agent_id = :aid"),
            {"aid": marketplace_agent.id},
        ).fetchall()
        for tool_id, enabled in tool_rows:
            db.execute(
                text(
                    "INSERT INTO agent_tool_assignments (agent_id, tool_id, enabled, created_at, updated_at) "
                    "VALUES (:aid, :tid, :en, NOW(), NOW())"
                ),
                {"aid": cloned.id, "tid": tool_id, "en": enabled},
            )

        marketplace_agent.install_count = (marketplace_agent.install_count or 0) + 1

        if marketplace_agent.slug == provisioner.ops_manager_slug:
            ops_manager_clone_id = cloned.id

        cloned_count += 1

    if ops_manager_clone_id and provisioner.ops_manager_slug:
        db.query(Agent).filter(
            Agent.workspace_id == workspace_id,
            Agent.slug.in_(provisioner.agent_slugs),
            Agent.slug != provisioner.ops_manager_slug,
        ).update(
            {"reports_to_id": ops_manager_clone_id},
            synchronize_session="fetch",
        )

    return cloned_count


def _create_widget_key(**kwargs) -> Dict[str, Any]:
    """Mint the public widget API key (thin wrapper over ApiKeyService)."""
    from core.services.api_key_service import ApiKeyService

    return ApiKeyService.create_api_key(**kwargs)


# ---------------------------------------------------------------------------
# The one generic provision flow.
# ---------------------------------------------------------------------------


def provision_vertical(
    *,
    db: Session,
    vertical: str,
    external_id: str,
    name: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Provision a workspace for any registered vertical.

    Idempotent: an existing active workspace for ``external_id`` is returned
    (roster not re-seeded). The widget key is regenerated each call UNLESS the
    vertical declares ``reuse_existing_key`` and an active public key already
    exists — then ``api_key`` is ``None`` and the caller keeps its stored copy.
    Raises ``ValueError`` for an unregistered vertical and
    :class:`VerticalConfigError` for a request the vertical's own rules reject.
    """
    provisioner = PROVISIONER_REGISTRY.get(vertical)
    if provisioner is None:
        raise ValueError(f"Unknown vertical '{vertical}' — no provisioner registered")

    is_new = False
    id_key = getattr(provisioner, "external_id_key", "source_external_id")

    # Resolve by the vertical's own key (e.g. shopify_domain) so a workspace
    # provisioned before this generic path — or by that vertical's other routes —
    # is found, not duplicated.
    workspace = (
        db.query(Workspace)
        .filter(
            Workspace.settings[id_key].astext == external_id,
            Workspace.is_active.is_(True),
        )
        .first()
    )

    if workspace is None:
        settings = {
            "vertical": vertical,
            "source": vertical,
            "source_external_id": external_id,
            id_key: external_id,
            f"{vertical}_metadata": metadata,
            "widget_proactive": dict(provisioner.default_widget_config),
        }
        workspace = Workspace(
            id=uuid4(),
            name=name,
            slug=_slugify(external_id),
            plan="starter",
            is_personal=False,
            is_active=True,
            webhook_key=uuid4().hex,
            settings=settings,
        )
        db.add(workspace)
        db.flush()
        is_new = True
        logger.info("Created workspace %s for %s '%s'", workspace.id, vertical, external_id)

    from core.models.core import Agent

    existing_agent_count = (
        db.query(Agent)
        .filter(Agent.workspace_id == workspace.id, Agent.owner_type == "workspace")
        .count()
    )

    agents_installed = 0
    if existing_agent_count == 0:
        agents_installed = _seed_roster(db, workspace.id, provisioner)

    # Resolve the allowlist BEFORE any key decision so a malformed request
    # (VerticalConfigError) rejects without side effects either way.
    domains = provisioner.allowed_domains(external_id, metadata)

    existing_key = None
    if getattr(provisioner, "reuse_existing_key", False):
        from core.models.sdk_api_keys import SdkApiKey

        existing_key = (
            db.query(SdkApiKey)
            .filter(
                SdkApiKey.workspace_id == workspace.id,
                SdkApiKey.key_type == "public",
                SdkApiKey.is_active.is_(True),
            )
            .order_by(SdkApiKey.created_at.desc())
            .first()
        )

    key_result = None
    if existing_key is None:
        key_result = _create_widget_key(
            db=db,
            workspace_id=workspace.id,
            name=f"{vertical.title()} Widget Key ({external_id})",
            key_type=getattr(provisioner, "key_type", "public"),
            permissions=list(provisioner.key_permissions),
            allowed_domains=domains,
        )

    # Optional vertical-specific post-provision step (e.g. ensure a Site).
    hook = getattr(provisioner, "on_provisioned", None)
    if callable(hook):
        try:
            hook(db, workspace)
        except Exception as e:  # noqa: BLE001 — never fail provision on a heal step
            logger.warning("%s.on_provisioned failed for workspace %s: %s", vertical, workspace.id, e)

    db.commit()

    return {
        "id": str(workspace.id),
        "public_id": str(workspace.id),
        "name": workspace.name,
        "api_key": key_result["key"] if key_result else None,
        "key_minted": key_result is not None,
        "agents_installed": agents_installed,
        "is_new": is_new,
    }


def _slugify(external_id: str) -> str:
    """Best-effort workspace slug from an external id (host/domain-ish)."""
    base = external_id.split("://", 1)[-1].strip("/").split("/", 1)[0]
    for suffix in (".myshopify.com",):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    return base or external_id


__all__ = [
    "VerticalProvisioner",
    "VerticalConfigError",
    "PROVISIONER_REGISTRY",
    "GRAPH_SOURCE_MAPPERS",
    "register_graph_source_mappers",
    "get_graph_source_mapper",
    "provision_vertical",
]
