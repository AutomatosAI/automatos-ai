"""
PRD-230: Marketplace Packages
=============================

A **package** is a curated, per-vertical bundle of *existing* marketplace
artifacts — agents, tools, skills, plugins, playbooks, LLMs — with matching
metadata and a setup manifest. It is DATA, not code (D4): curating a new vertical
is content work, not a deploy.

This is the wave's ONE new table (US-003). Everything a package installs rides
the EXISTING per-type registration patterns (``workspace_enabled_*`` rows,
marketplace agent copies, LLM availability) — see the closure resolver (US-004)
and installer (US-005). The package row only holds the definition:

  - ``members``        typed refs ``[{"type": agent|tool|skill|plugin|playbook|llm,
                       "ref": <id-or-slug>}]`` — what the package installs.
  - ``matching``       business-type signals used to rank the package against a
                       workspace (platforms, url patterns, vocabulary).
  - ``setup_manifest`` the guided-setup manifest: ``questions``, ``required_connects``
                       (Composio apps incl. the Shopify two-step), ``guide_steps``
                       (the D7 three-step flow), ``report_templates``.
"""

from uuid import uuid4

from sqlalchemy import Boolean, Column, DateTime, Index, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID

from core.database.base import Base

__all__ = ["MarketplacePackage", "MEMBER_TYPES"]

# Member type vocabulary (D2 closure walks these). Kept here as the single source
# of truth for the model, the resolver, the installer, and the seeds.
MEMBER_TYPES: tuple[str, ...] = ("agent", "tool", "skill", "plugin", "playbook", "llm")


class MarketplacePackage(Base):
    """A curated per-vertical bundle of existing marketplace artifacts (PRD-230)."""

    __tablename__ = "marketplace_packages"
    __table_args__ = (
        Index("idx_marketplace_packages_showcase", "showcase"),
        {"extend_existing": True},
    )

    id = Column(PGUUID(as_uuid=True), primary_key=True, default=uuid4)
    slug = Column(String(120), unique=True, nullable=False)
    name = Column(String(255), nullable=False)
    description = Column(Text)

    # Classification / matching (business-type signals). JSONB arrays/objects so a
    # new vertical is pure content (D4) — no migration to add a package.
    vertical_tags = Column(JSONB, nullable=False, default=list)   # e.g. ["shopify","ecommerce"]
    matching = Column(JSONB, nullable=False, default=dict)        # {platforms, url_patterns, vocabulary}

    # The definition: typed member refs + the guided setup manifest.
    members = Column(JSONB, nullable=False, default=list)         # [{"type": ..., "ref": ...}]
    setup_manifest = Column(JSONB, nullable=False, default=dict)  # {questions, required_connects, guide_steps, report_templates}

    # Showcase packages surface first in the marketplace Packages tab.
    showcase = Column(Boolean, nullable=False, default=False)

    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)

    def to_dict(self) -> dict:
        """API/serialisation shape (US-007 detail popup reads this)."""
        return {
            "id": str(self.id),
            "slug": self.slug,
            "name": self.name,
            "description": self.description,
            "vertical_tags": list(self.vertical_tags or []),
            "matching": dict(self.matching or {}),
            "members": list(self.members or []),
            "setup_manifest": dict(self.setup_manifest or {}),
            "showcase": bool(self.showcase),
        }

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<MarketplacePackage slug={self.slug!r} members={len(self.members or [])}>"
