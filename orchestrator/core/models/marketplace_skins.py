"""PRD-259 §5: skins in the Marketplace.

A skin is stored like the other Marketplace items: its definition is data (like
a Package), its files are a stored bundle under a prefix in the marketplace
bucket (like a Plugin), and a workspace installs it by reference (like a Widget).

* ``marketplace_skins``: one row per skin version. ``skin`` holds the validated
  skin.json plus the asset manifest; ``asset_prefix`` is where its files live
  (``services/skins/storage.py``). A Marketplace skin (``owner_type`` 'marketplace')
  goes through review (``status``); a private import (``owner_type`` 'workspace')
  belongs to one workspace and is approved on import.
* ``skin_installations``: a workspace's installs; at most one active row per skin
  (``uninstalled_at`` IS NULL).

JSON columns are ``JSON().with_variant(JSONB(), "postgresql")`` and id columns the
portable ``Uuid``, so the tables also build under SQLite in the unit tests. The
``prd259_skins`` migration creates both, and ``tests/test_prd259_skins_models.py``
holds the migration and the models together.
"""

from __future__ import annotations

from uuid import uuid4

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Uuid,
    false,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from core.database.base import Base

__all__ = [
    "MarketplaceSkin",
    "SkinInstallation",
    "SKIN_OWNER_TYPES",
    "SKIN_STATUSES",
    "SKIN_STATUS_DEFAULT",
]

SKIN_OWNER_TYPES = ("marketplace", "workspace")
SKIN_STATUSES = ("pending", "approved", "rejected")
SKIN_STATUS_DEFAULT = "pending"

# The partial unique indexes' predicates, the same on Postgres and SQLite.
WORKSPACE_OWNED = "owner_type = 'workspace'"
MARKETPLACE_OWNED = "owner_type = 'marketplace'"
ACTIVE_INSTALLATION = "uninstalled_at IS NULL"


def _in_list(column: str, values: tuple) -> str:
    quoted = ", ".join(f"'{v}'" for v in values)
    return f"{column} IN ({quoted})"


def _json_type():
    return JSON().with_variant(JSONB(), "postgresql")


def _partial_unique(name: str, predicate: str, *columns: str) -> Index:
    where = text(predicate)
    return Index(name, *columns, unique=True, postgresql_where=where, sqlite_where=where)


class MarketplaceSkin(Base):
    """One version of a skin: a Marketplace skin or one workspace's private import."""

    __tablename__ = "marketplace_skins"
    __table_args__ = (
        CheckConstraint(_in_list("owner_type", SKIN_OWNER_TYPES), name="ck_marketplace_skins_owner_type"),
        CheckConstraint(
            "owner_type <> 'workspace' OR workspace_id IS NOT NULL",
            name="ck_marketplace_skins_workspace_owner",
        ),
        CheckConstraint(_in_list("status", SKIN_STATUSES), name="ck_marketplace_skins_status"),
        Index("ix_marketplace_skins_workspace_id", "workspace_id"),
        Index("ix_marketplace_skins_status", "status"),
        _partial_unique("uq_marketplace_skins_workspace_slug", WORKSPACE_OWNED, "workspace_id", "slug"),
        _partial_unique(
            "uq_marketplace_skins_author_slug_version", MARKETPLACE_OWNED, "author_user_id", "slug", "version",
        ),
        {"extend_existing": True},
    )

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    slug = Column(String(64), nullable=False)
    name = Column(String(40), nullable=False)
    version = Column(String(32), nullable=False)
    # Integer, matching users.id (see widget_installation.user_id).
    author_user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"))
    skin = Column(_json_type(), nullable=False)
    asset_prefix = Column(String, nullable=False)
    preview_path = Column(String)
    owner_type = Column(String(16), nullable=False)
    workspace_id = Column(Uuid(as_uuid=True))
    status = Column(String(16), nullable=False, default=SKIN_STATUS_DEFAULT, server_default=SKIN_STATUS_DEFAULT)
    rejection_reason = Column(Text)
    featured = Column(Boolean, nullable=False, default=False, server_default=false())
    install_count = Column(Integer, nullable=False, default=0, server_default="0")
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    installations = relationship("SkinInstallation", back_populates="skin", passive_deletes=True)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<MarketplaceSkin slug={self.slug!r} version={self.version!r} owner={self.owner_type!r}>"


class SkinInstallation(Base):
    """A skin installed in a workspace, by reference; at most one active row per skin."""

    __tablename__ = "skin_installations"
    __table_args__ = (
        Index("ix_skin_installations_workspace_id", "workspace_id"),
        _partial_unique("uq_skin_installations_active", ACTIVE_INSTALLATION, "workspace_id", "skin_id"),
        {"extend_existing": True},
    )

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid4)
    workspace_id = Column(Uuid(as_uuid=True), nullable=False)
    skin_id = Column(Uuid(as_uuid=True), ForeignKey("marketplace_skins.id", ondelete="CASCADE"), nullable=False)
    version = Column(String(32), nullable=False)
    installed_by = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"))
    installed_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    uninstalled_at = Column(DateTime(timezone=True))

    skin = relationship("MarketplaceSkin", back_populates="installations")

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"<SkinInstallation skin_id={self.skin_id} workspace_id={self.workspace_id}>"
