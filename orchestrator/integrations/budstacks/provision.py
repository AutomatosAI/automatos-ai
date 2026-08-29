"""BudStacks vertical provisioner.

BudStacks (budstacks.io) is a multi-tenant cannabis-storefront SaaS — the
second vertical to provision through the generic
``POST /api/verticals/{v}/provision`` plane (PRD-183 S5). Its partner app
calls with a BudStacks tenant id as ``external_id`` and the tenant's storefront
hostnames in ``metadata.domains``; the flow stands up a workspace and mints an
``ak_pub_*`` widget key origin-locked to those hostnames.

Deliberate differences from the Shopify provisioner:

* ``reuse_existing_key = True`` — the BudStacks app persists the minted key in
  its own tenant row, so a re-provision call must NOT rotate the key out from
  under a live storefront (the Shopify app relies on the opposite behaviour as
  its key-recovery path, hence the per-vertical opt-in rather than a global
  change).
* ``allowed_domains`` comes entirely from ``metadata.domains`` (subdomain,
  custom domain, www variant) and is REQUIRED — there is no wildcard fallback
  like ``*.myshopify.com``. An empty list raises
  :class:`~integrations.provisioning.VerticalConfigError` (HTTP 422), because a
  public key without an origin allowlist is exactly what the api-keys HTTP
  layer refuses to create.
* ``agent_slugs`` is empty for now: no cannabis-storefront marketplace roster
  exists yet, and seeding the Shopify-branded agents into a dispensary
  workspace would be wrong. The BudStacks app keeps configuring its agent id
  explicitly until a roster lands (surfaced in the integration PRD as an open
  question — not silently descoped).

Importing ``integrations.budstacks`` self-registers this provisioner into
``integrations.provisioning.PROVISIONER_REGISTRY``.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


# No marketplace roster yet — see module docstring.
BUDSTACKS_AGENT_SLUGS: List[str] = []


# Proactive engagement stays off until a tenant opts in from the platform side.
DEFAULT_WIDGET_PROACTIVE_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "respect_consent": True,
}


class BudstacksProvisioner:
    """VerticalProvisioner implementation for the BudStacks vertical."""

    vertical = "budstacks"
    agent_slugs = BUDSTACKS_AGENT_SLUGS
    ops_manager_slug = None
    default_widget_config = DEFAULT_WIDGET_PROACTIVE_CONFIG
    key_permissions = ["chat"]
    key_type = "public"
    site_type = None
    # BudStacks resolves tenants by their own tenant id, stamped alongside the
    # canonical source_external_id.
    external_id_key = "budstacks_tenant_id"
    # Never rotate a live storefront's key on re-provision (see module docstring).
    reuse_existing_key = True

    def allowed_domains(self, external_id: str, metadata: Dict[str, Any]) -> List[str]:
        """Origin allowlist from ``metadata.domains`` — required, normalized, deduped.

        Accepts bare hostnames or full origins; normalizes to ``https://host``.
        """
        from integrations.provisioning import VerticalConfigError

        raw = (metadata or {}).get("domains") or []
        hosts: List[str] = []
        for entry in raw:
            if not isinstance(entry, str):
                continue
            host = entry.split("://", 1)[-1].strip().strip("/").split("/", 1)[0]
            if host:
                hosts.append(f"https://{host}")

        if not hosts:
            raise VerticalConfigError(
                "budstacks provisioning requires metadata.domains — the storefront "
                "hostnames the public widget key is origin-locked to"
            )

        seen: set[str] = set()
        return [d for d in hosts if not (d in seen or seen.add(d))]

    def on_provisioned(self, db: Session, workspace: Any) -> None:
        """No vertical-specific post-provision step for BudStacks."""


# Module-level singleton — the registered provisioner instance.
provisioner = BudstacksProvisioner()


def register() -> None:
    """Self-register the BudStacks provisioner (called from ``__init__.py``)."""
    from integrations.provisioning import PROVISIONER_REGISTRY

    PROVISIONER_REGISTRY["budstacks"] = provisioner
