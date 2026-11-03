"""PRD-270 §5.3 and §5.6 — the Sync with Automatos Cloud's status row, and the rows it wrote.

* ``MarketplaceCloudSync`` — one row (``id`` 1, a CHECK) the local edition's Sync writes:
  when it last tried, when it last succeeded, the catalogue's ``generatedAt``, the last
  error, and the last Sync's counts (new, updated, unchanged and refused listings). A
  failed Sync records only its attempt and error, so the last good Sync's facts stay.
* ``MarketplaceCloudItem`` — one row per Automatos Cloud item the Sync wrote (W2): the one
  local row it points at (a marketplace agent, a Playbook, a package or a plugin, by the key
  of its type), its Cloud identity (``external_id``, a hash of its lineage), Automatos Cloud's
  version, the bundle's revision and the refs that did not resolve locally. A local row is a
  Cloud row exactly when one of these points at it; nothing but the Sync writes one.

``MarketplaceCloudSync`` mirrors ``theme_revocation_sync`` (``core/models/themes.py``), but it
is a Marketplace table, not a theme table: the Sync is the Marketplace's. The Alembic
revisions ``prd270_w1_cloud_sync`` and ``prd270_w2_cloud_bundles`` declare the same shapes,
and ``tests/test_prd270w1_migration.py`` and ``tests/test_prd270w2_migration.py`` hold them
together.
"""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from sqlalchemy import (
    CheckConstraint, Column, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint, text,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PGUUID
from sqlalchemy.sql import func

from core.database.base import Base

__all__ = [
    "MarketplaceCloudSync", "MARKETPLACE_CLOUD_SYNC_ROW_ID",
    "MarketplaceCloudItem", "CLOUD_ITEM_TYPES", "CLOUD_ITEM_KEYS",
]

# marketplace_cloud_sync holds exactly one row, this id.
MARKETPLACE_CLOUD_SYNC_ROW_ID = 1
SINGLE_ROW = f"id = {MARKETPLACE_CLOUD_SYNC_ROW_ID}"

# The item types a Sync writes (themes are W1's, on theme_packages.source).
CLOUD_ITEM_TYPES = ("agent", "playbook", "package", "plugin")
# The column that holds each type's local row: a mapping sets exactly this one.
CLOUD_ITEM_KEYS: Mapping[str, str] = MappingProxyType({
    "agent": "agent_id", "playbook": "recipe_id", "package": "package_id", "plugin": "plugin_id",
})
# Exactly the revision's DDL text.
ITEM_TYPE_CHECK = "item_type IN ('agent', 'playbook', 'package', 'plugin')"
ONE_ROW_CHECK = (
    "num_nonnulls(agent_id, recipe_id, package_id, plugin_id) = 1"
    " AND (agent_id IS NULL OR item_type = 'agent')"
    " AND (recipe_id IS NULL OR item_type = 'playbook')"
    " AND (package_id IS NULL OR item_type = 'package')"
    " AND (plugin_id IS NULL OR item_type = 'plugin')"
)


def _count_column() -> Column:
    return Column(Integer, nullable=False, default=0, server_default=text("0"))


def _one_mapping_per(key: str) -> Index:
    """At most one mapping points at a local row (``uq_marketplace_cloud_items_<type>``)."""
    name = f"uq_marketplace_cloud_items_{key.removesuffix('_id')}"
    return Index(name, key, unique=True, postgresql_where=text(f"{key} IS NOT NULL"))


def _local_key(target: str, column_type) -> Column:
    return Column(column_type, ForeignKey(target, ondelete="CASCADE"))


class MarketplaceCloudSync(Base):
    """The local edition's Sync with Automatos Cloud: its last attempt and success, the catalogue's time, the counts."""

    __tablename__ = "marketplace_cloud_sync"
    __table_args__ = (
        CheckConstraint(SINGLE_ROW, name="ck_marketplace_cloud_sync_single_row"),
        {"extend_existing": True},
    )

    id = Column(Integer, primary_key=True, autoincrement=False, default=MARKETPLACE_CLOUD_SYNC_ROW_ID)
    last_attempt_at = Column(DateTime(timezone=True))
    last_success_at = Column(DateTime(timezone=True))
    feed_generated_at = Column(DateTime(timezone=True))  # the catalogue's generatedAt
    last_error = Column(Text)
    new_count = _count_column()
    updated_count = _count_column()
    unchanged_count = _count_column()
    refused_count = _count_column()
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())


class MarketplaceCloudItem(Base):
    """One Automatos Cloud item a Sync wrote: the local row it points at, its Cloud id, version and revision."""

    __tablename__ = "marketplace_cloud_items"
    __table_args__ = (
        CheckConstraint(ITEM_TYPE_CHECK, name="ck_marketplace_cloud_items_type"),
        CheckConstraint(ONE_ROW_CHECK, name="ck_marketplace_cloud_items_one_row"),
        UniqueConstraint("item_type", "external_id", name="uq_marketplace_cloud_items_external"),
        *(_one_mapping_per(key) for key in CLOUD_ITEM_KEYS.values()),
        {"extend_existing": True},
    )

    id = Column(Integer, primary_key=True)
    item_type = Column(String(16), nullable=False)  # one of CLOUD_ITEM_TYPES
    external_id = Column(String(32), nullable=False)  # a hash of the item's lineage on Automatos Cloud
    version = Column(String(32), nullable=False)  # Automatos Cloud's version of it
    revision = Column(String(64), nullable=False)  # the sha256 of the bundle's canonical bytes
    agent_id = _local_key("agents.id", Integer)
    recipe_id = _local_key("workflow_recipes.id", Integer)
    package_id = _local_key("marketplace_packages.id", PGUUID(as_uuid=True))
    plugin_id = _local_key("marketplace_plugins.id", PGUUID(as_uuid=True))
    unresolved = Column(JSONB, nullable=False, default=list, server_default=text("'[]'::jsonb"))  # refs not found here
    # When a Sync last wrote the mapping: the ORM bumps it on an update; a Core upsert sets it itself.
    synced_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())
