"""PRD-270 US-270-07b (K10) — the public read of the Marketplace's applications.

The Applications tab lists the Composio app catalogue from the platform's own cache, never a live
Composio call (``api/tools.py`` ``marketplace``): ``composio_apps_cache`` rows whose status is
``ACTIVE``, less the platform's internal tools, most actions first. This reads the same rows, with the
slug as the last tie so pages never overlap. The tab's connected flag is a workspace's Composio
connections, so no connection, entity or workspace is read here; nor is an app's metadata (its trigger
configs), auth schemes, sync times or row id.

The key is the app's slug; the icon its logo, answered only as an ``https`` URL; the tags its
categories and the category the first of them. An application needs no other app, so it lists no
required apps, and the catalogue is published by Automatos.
"""

from __future__ import annotations

from typing import Any, Dict, List

from sqlalchemy.orm import Session

from api.tools import INTERNAL_APP_NAMES
from core.models.composio_cache import ComposioAppCache
from services.marketplace_public.fields import APPLICATION_FIELDS, count_of, display_item, https_url_or_none, strings

ACTIVE_APP = "ACTIVE"


def _item(row: Any) -> Dict[str, Any]:
    categories = strings(row.categories)
    return display_item(
        APPLICATION_FIELDS, key=row.app_slug, name=row.display_name, description=row.description,
        category=categories[0] if categories else None, icon=https_url_or_none(row.logo_url), tags=categories,
        actionCount=count_of(row.action_count), triggerCount=count_of(row.trigger_count),
    )


def read_applications(db: Session, limit: int, offset: int) -> List[Dict[str, Any]]:
    """Up to ``limit`` of the apps the Applications tab lists, from ``offset``, each as its public item."""
    rows = (
        db.query(
            ComposioAppCache.app_slug, ComposioAppCache.display_name, ComposioAppCache.description,
            ComposioAppCache.logo_url, ComposioAppCache.categories, ComposioAppCache.action_count,
            ComposioAppCache.trigger_count,
        )
        .filter(ComposioAppCache.status == ACTIVE_APP, ~ComposioAppCache.app_name.in_(INTERNAL_APP_NAMES))
        .order_by(ComposioAppCache.action_count.desc().nullslast(), ComposioAppCache.app_slug)
        .offset(offset)
        .limit(limit)
        .all()
    )
    return [_item(row) for row in rows]


__all__ = ["read_applications"]
