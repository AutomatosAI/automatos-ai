"""PRD-270 W2 US-270-14a (§5.6, K18) — which local rows are Automatos Cloud items, by the key each install takes.

``GET /api/marketplace/cloud-sync/items`` answers ``cloud_items(db)``: one list per type, of the rows a
``marketplace_cloud_items`` mapping points at, each ``{id, externalId, version, revision, name}`` (a package's
``slug`` in place of ``id``), by name. The key is the one its install route takes: an agent's and a Playbook's
integer id (``POST /api/marketplace/items/{id}/install``, ``POST /api/workflow-recipes/install/{id}``), a
package's slug (``POST /api/marketplace/packages/{slug}/install``) and a plugin's uuid (the ``plugin_id`` of
``POST /api/workspaces/{ws}/plugins``). ``version`` is Automatos Cloud's, ``revision`` the bundle's sha256.

The web marks these rows ('From Automatos Cloud', 'Reviewed on Automatos Cloud') and its install gate asks for
their disclosure before an add (US-270-14b, 14d). Only the mappings and the Cloud rows they point at are read,
never a workspace's copy; no network.
"""
from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Tuple, Type
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.core import Agent, WorkflowTemplate
from core.models.marketplace_cloud import CLOUD_ITEM_KEYS, MarketplaceCloudItem
from core.models.marketplace_packages import MarketplacePackage
from core.models.marketplace_plugins import MarketplacePlugin


class ListedType(NamedTuple):
    """One type's list: its name in the answer, its item type, its model and the key its install takes."""

    answer: str
    item_type: str
    model: Type[Any]
    key_field: str  # the answer's name for the key, and the model's column


LISTED: Tuple[ListedType, ...] = (
    ListedType("agents", "agent", Agent, "id"),
    ListedType("playbooks", "playbook", WorkflowTemplate, "id"),
    ListedType("packages", "package", MarketplacePackage, "slug"),
    ListedType("plugins", "plugin", MarketplacePlugin, "id"),
)


def cloud_items(db: Session) -> Dict[str, List[Dict[str, Any]]]:
    """``{agents, playbooks, packages, plugins}``: each Cloud row of that type, by name."""
    return {listed.answer: _listed(db, listed) for listed in LISTED}


def _listed(db: Session, listed: ListedType) -> List[Dict[str, Any]]:
    model = listed.model
    mapped_id = getattr(MarketplaceCloudItem, CLOUD_ITEM_KEYS[listed.item_type])
    rows = (
        db.query(
            getattr(model, listed.key_field).label("key"), model.name.label("name"),
            MarketplaceCloudItem.external_id, MarketplaceCloudItem.version, MarketplaceCloudItem.revision,
        )
        .join(MarketplaceCloudItem, mapped_id == model.id)
        .filter(MarketplaceCloudItem.item_type == listed.item_type)
        .order_by(model.name, MarketplaceCloudItem.id)
        .all()
    )
    return [
        {
            listed.key_field: plain(row.key), "externalId": row.external_id, "version": row.version,
            "revision": row.revision, "name": row.name,
        }
        for row in rows
    ]


def plain(key: Any) -> Any:
    """A key as JSON carries it: a uuid as its string, anything else as it is."""
    return str(key) if isinstance(key, UUID) else key


__all__ = ["LISTED", "ListedType", "cloud_items", "plain"]
