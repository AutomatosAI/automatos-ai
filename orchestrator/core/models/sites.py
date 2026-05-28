"""
Site ORM model (PRD-008-A).

A Site is anywhere the Automatos widget runs — a Shopify store, a Wix site,
a WooCommerce install, or a raw <script> embed on a custom website.

Sites are the first-class home for per-merchant widget settings. They are
1:N children of Workspace: one workspace can have many Sites (e.g. an
agency managing five Shopify stores).

See ``docs/PRDS/PRD-008-A-HUMAN-HANDOFF-AND-SITES.md`` (in the
``automatos-shopify`` repo) for the full spec.
"""

from __future__ import annotations

from uuid import uuid4

from sqlalchemy import Column, DateTime, String, UniqueConstraint, Index
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.sql import func

from core.database.base import Base


# Type allow-list. Adding a new type requires a deliberate change here +
# matching adapter wiring elsewhere. Keeps the dashboard's Site-type picker
# and the capability defaults in lockstep.
SITE_TYPES: tuple[str, ...] = ("shopify", "wix", "woocommerce", "custom")


# Capability keys exposed to the dashboard. Frontend components branch on
# these (e.g. CartIdlePanel renders only if has_cart). Adding a new key
# requires a coordinated frontend + backend change — keep them in this
# canonical tuple so the trip-wire test catches drift.
CAPABILITY_KEYS: tuple[str, ...] = (
    "has_cart",
    "has_catalog",
    "has_volume_discounts",
    "has_customer_records",
    "has_working_hours_source",
    "supports_theme_override",
)


def derive_default_capabilities(site_type: str) -> dict[str, bool]:
    """Compute the capability flags for a freshly-connected Site of the
    given type. The connector may later flip individual flags based on
    runtime probes (e.g. confirmed scopes for has_volume_discounts).
    """
    if site_type not in SITE_TYPES:
        raise ValueError(f"unknown site type: {site_type!r}")

    if site_type == "shopify":
        return {
            "has_cart": True,
            "has_catalog": True,
            "has_volume_discounts": False,  # depends on merchant price rules
            "has_customer_records": True,
            "has_working_hours_source": True,  # shop.timezone available
            "supports_theme_override": True,
        }

    # wix, woocommerce, custom all default to off until their adapters ship.
    return {key: False for key in CAPABILITY_KEYS}


class Site(Base):
    __tablename__ = "sites"
    __table_args__ = (
        UniqueConstraint("workspace_id", "type", "external_id", name="uq_sites_workspace_type_external"),
        Index("idx_sites_workspace_id", "workspace_id"),
        Index("idx_sites_type_external", "type", "external_id"),
        {"extend_existing": True},
    )

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    workspace_id = Column(PGUUID(as_uuid=True), nullable=False)

    type = Column(String(20), nullable=False)
    external_id = Column(String(255), nullable=True)
    display_name = Column(String(255), nullable=False)

    status = Column(String(20), nullable=False, server_default="active")

    settings = Column(JSONB, nullable=False, default=dict, server_default="{}")
    capabilities = Column(JSONB, nullable=False, default=dict, server_default="{}")
    secrets = Column(JSONB, nullable=True)  # encrypted at rest by app layer

    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    def __repr__(self) -> str:
        return (
            f"<Site id={self.id} type={self.type!r} "
            f"external_id={self.external_id!r} workspace={self.workspace_id}>"
        )

    @property
    def effective_capabilities(self) -> dict[str, bool]:
        # Merges stored capabilities over the type's defaults so legacy
        # rows (capabilities={}) still surface the right flags to the UI
        # without depending on a one-shot data backfill.
        defaults = (
            derive_default_capabilities(self.type)
            if self.type in SITE_TYPES
            else {key: False for key in CAPABILITY_KEYS}
        )
        stored = self.capabilities or {}
        return {key: bool(stored.get(key, defaults[key])) for key in CAPABILITY_KEYS}
