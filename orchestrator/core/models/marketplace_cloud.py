"""PRD-270 §5.3 — the Sync with Automatos Cloud's one status row.

* ``MarketplaceCloudSync`` — one row (``id`` 1, a CHECK) the local edition's Sync writes:
  when it last tried, when it last succeeded, the catalogue's ``generatedAt``, the last
  error, and the last Sync's counts (new, updated, unchanged and refused listings). A
  failed Sync records only its attempt and error, so the last good Sync's facts stay.

It mirrors ``theme_revocation_sync`` (``core/models/themes.py``), but it is a Marketplace
table, not a theme table: the Sync is the Marketplace's. The Alembic revision
``prd270_w1_cloud_sync`` declares the same shape, and ``tests/test_prd270w1_migration.py``
holds the two together.
"""
from __future__ import annotations

from sqlalchemy import CheckConstraint, Column, DateTime, Integer, Text, text
from sqlalchemy.sql import func

from core.database.base import Base

__all__ = ["MarketplaceCloudSync", "MARKETPLACE_CLOUD_SYNC_ROW_ID"]

# marketplace_cloud_sync holds exactly one row, this id.
MARKETPLACE_CLOUD_SYNC_ROW_ID = 1
SINGLE_ROW = f"id = {MARKETPLACE_CLOUD_SYNC_ROW_ID}"


def _count_column() -> Column:
    return Column(Integer, nullable=False, default=0, server_default=text("0"))


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
