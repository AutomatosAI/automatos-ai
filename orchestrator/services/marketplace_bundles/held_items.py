"""PRD-270 W2 US-270-13d (§5.5, K16) — how many Cloud rows of each item type this install holds.

W1's Sync status (``pull.sync_status``) answers it as ``items``. A local row is a Cloud row exactly when a
``marketplace_cloud_items`` mapping points at it, and a mapping goes with its row (its key cascades), so the
mappings are counted by type. A leaf: W1's ``pull.py`` imports it, and it imports neither ``pull.py`` nor the
passes that do.
"""
from __future__ import annotations

from typing import Dict

from sqlalchemy import func
from sqlalchemy.orm import Session

from core.models.marketplace_cloud import CLOUD_ITEM_TYPES, MarketplaceCloudItem


def items_held(db: Session) -> Dict[str, int]:
    """``{agent, playbook, package, plugin: the number of Cloud rows of that type held here}``."""
    rows = (
        db.query(MarketplaceCloudItem.item_type, func.count(MarketplaceCloudItem.id))
        .group_by(MarketplaceCloudItem.item_type)
        .all()
    )
    counted = {item_type: int(count) for item_type, count in rows}
    return {item_type: counted.get(item_type, 0) for item_type in CLOUD_ITEM_TYPES}


__all__ = ["items_held"]
