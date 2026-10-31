"""PRD-270 US-270-07b (K10) — the public read of the Marketplace's plugins.

Approved and active ``marketplace_plugins`` rows, as the member Plugins tab lists them
(``api/marketplace_plugins.py`` ``list_plugins``), most enabled first then newest, with the id as the
last tie. The query names its columns: a plugin's S3 bucket and path, source URL and repo, security
scan and status, approval and rejection notes, recommended models, materialized skill ids and long
description are never read, let alone answered.

The key is the plugin's id (the one its detail dialog takes); the category its category's name; the
publisher its original author as a display name (``publishers.display_name``: one that holds an ``@``,
or none, reads ``Automatos``). A plugin declares no apps, so it lists no required apps.
"""

from __future__ import annotations

from typing import Any, Dict, List

from sqlalchemy.orm import Session

from core.models.marketplace_plugins import MarketplacePlugin, PluginCategory
from services.marketplace_public.fields import PLUGIN_FIELDS, count_of, display_item, text_or_none
from services.marketplace_public.publishers import display_name

APPROVED_PLUGIN = "approved"


def _item(row: Any) -> Dict[str, Any]:
    return display_item(
        PLUGIN_FIELDS, key=row.id, name=row.name, description=row.description, category=row.category_name,
        tags=row.tags, publisher=display_name(row.original_author), version=text_or_none(row.version),
        skillsCount=count_of(row.skills_count), commandsCount=count_of(row.commands_count),
        agentsCount=count_of(row.agents_count), hooksCount=count_of(row.hooks_count),
    )


def read_plugins(db: Session, limit: int, offset: int) -> List[Dict[str, Any]]:
    """Up to ``limit`` approved, active Marketplace plugins from ``offset``, each as its public item."""
    rows = (
        db.query(
            MarketplacePlugin.id, MarketplacePlugin.name, MarketplacePlugin.version, MarketplacePlugin.description,
            MarketplacePlugin.tags, MarketplacePlugin.original_author, MarketplacePlugin.skills_count,
            MarketplacePlugin.commands_count, MarketplacePlugin.agents_count, MarketplacePlugin.hooks_count,
            PluginCategory.name.label("category_name"),
        )
        .outerjoin(PluginCategory, PluginCategory.id == MarketplacePlugin.category_id)
        .filter(MarketplacePlugin.approval_status == APPROVED_PLUGIN, MarketplacePlugin.is_active.is_(True))
        .order_by(
            MarketplacePlugin.enable_count.desc().nullslast(), MarketplacePlugin.created_at.desc(),
            MarketplacePlugin.id,
        )
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [_item(row) for row in rows]


__all__ = ["read_plugins"]
