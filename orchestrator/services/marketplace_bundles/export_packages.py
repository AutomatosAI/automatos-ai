"""PRD-270 W2 US-270-12b (K9, K10) — Automatos Cloud's admin-made packages, as bundles.

The rows: every ``marketplace_packages`` row whose slug no local edition seeds (seeded.py). A package has
no submit route, so the rest are admin-made; its lineage is its slug (lineage.py). It has no version,
category, icon or creator either: its version is PACKAGE_VERSION, the one every community item is
submitted at (``submit_item``), and the revision versions it by content (K10); its category and icon are
none, its tags its vertical tags, and its publisher Automatos, as W1's public read says of a package.

Every member must have a reference the format allows (package_members.py): one that resolves to none,
an agent neither seeded nor exported or a plugin the plugin export does not list, say, keeps the package
out, logged, since a local install would refuse it. The matching signals keep only their three lists and the setup manifest only its four, each
object only its list's keys (schema.SETUP_OBJECT_KEYS), so a key like api_key or webhook_url in an
admin's manifest never travels. A bundle the validator refuses is logged and left out.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

from sqlalchemy.orm import Session

from core.models.marketplace_packages import MarketplacePackage
from services.marketplace_bundles.export_plugins import exported_plugins
from services.marketplace_bundles.exported import (
    ExportedItem, Lineage, envelope, exported_item, newest_per_lineage, tags_of,
)
from services.marketplace_bundles.package_members import MemberRefs, member_refs, raw_ref
from services.marketplace_bundles.schema import MATCHING_KEYS, SETUP_KEYS, SETUP_OBJECT_KEYS
from services.marketplace_bundles.seeded import seeded_package_slugs
from services.marketplace_public.fields import DEFAULT_PUBLISHER

logger = logging.getLogger(__name__)

ITEM_TYPE = "package"
PACKAGE_VERSION = "1.0.0"

_COLUMNS = (
    MarketplacePackage.id, MarketplacePackage.slug, MarketplacePackage.name, MarketplacePackage.description,
    MarketplacePackage.vertical_tags.label("tags"), MarketplacePackage.matching, MarketplacePackage.members,
    MarketplacePackage.setup_manifest,
)


@dataclass(frozen=True)
class _Package:
    """A package row with the display columns the envelope reads: no version, category or icon of its own."""

    id: Any
    slug: str
    name: str
    description: Optional[str]
    tags: Any
    matching: Any
    members: Any
    setup_manifest: Any
    version: str = PACKAGE_VERSION
    marketplace_category: Optional[str] = None
    marketplace_icon: Optional[str] = None


def exported_packages(
    db: Session, agent_items: Iterable[ExportedItem], playbook_items: Iterable[ExportedItem], *,
    plugin_items: Optional[Iterable[ExportedItem]] = None,
) -> List[ExportedItem]:
    """One ExportedItem per package no local edition seeds, by slug; ``agent_items``, ``playbook_items`` and
    ``plugin_items`` are ``exported_agents``', ``exported_playbooks``' and ``exported_plugins``' items (the
    plugins exported here when the caller has none). A package with a member of no allowed reference, or
    whose bundle validate_bundle refuses, is logged and left out."""
    lineages = newest_per_lineage(ITEM_TYPE, _rows(db))
    if not lineages:
        return []
    members = (member for lineage in lineages for member in _list(lineage.newest.members))
    plugins = exported_plugins(db) if plugin_items is None else plugin_items
    refs = member_refs(db, members, agent_items, playbook_items, plugins)
    items = (_exported(lineage, refs) for lineage in lineages)
    return [item for item in items if item is not None]


def _rows(db: Session) -> List[_Package]:
    """The packages no local edition seeds, by slug."""
    seeded = seeded_package_slugs()
    rows = db.query(*_COLUMNS).order_by(MarketplacePackage.slug).all()
    return [_Package(**row._asdict()) for row in rows if row.slug not in seeded]


def _exported(lineage: Lineage, refs: MemberRefs) -> Optional[ExportedItem]:
    """The package's ExportedItem; None, logged, when a member has no allowed reference or the bundle is refused."""
    row = lineage.newest
    members = [_member(member, refs) for member in _list(row.members)]
    missing = next((index for index, member in enumerate(members) if member is None), None)
    if missing is not None:
        logger.warning(
            "Automatos Cloud export: package row %s is left out: its member %d has no reference the bundle "
            "format allows (neither seeded nor exported, or not found)", row.id, missing,
        )
        return None
    body = {
        "slug": row.slug,
        "verticalTags": tags_of(row.tags),
        "matching": _matching(row.matching),
        "members": members,
        "setup": _setup(row.setup_manifest),
    }
    return exported_item(ITEM_TYPE, lineage, envelope(ITEM_TYPE, lineage, DEFAULT_PUBLISHER, body))


def _member(member: Any, refs: MemberRefs) -> Optional[Dict[str, Any]]:
    """A member as {type, ref, name, description}: its own name, else its target's; None without an allowed ref."""
    member_type = member.get("type") if isinstance(member, dict) else None
    target = refs.target_of(member_type, raw_ref(member))
    if target is None:
        return None
    name = member.get("name")
    return {
        "type": member_type,
        "ref": dict(target.ref),
        "name": name if isinstance(name, str) and name.strip() else target.name,
        "description": deepcopy(member.get("description")) or "",
    }


def _matching(value: Any) -> Dict[str, Any]:
    """The matching signals' three lists (none reads as an empty list); any other key is left out."""
    matching = value if isinstance(value, dict) else {}
    return {key: _listed(matching.get(key)) for key in MATCHING_KEYS}


def _setup(value: Any) -> Dict[str, Any]:
    """The setup manifest's four lists, each object with only its list's keys; any other key is left out."""
    manifest = value if isinstance(value, dict) else {}
    return {name: _setup_list(manifest.get(name), SETUP_OBJECT_KEYS[name]) for name in SETUP_KEYS}


def _setup_list(value: Any, keys: Sequence[str]) -> Any:
    """A setup list's objects with only ``keys`` (a null value left out); anything but objects is left for
    the validator to refuse."""
    if not isinstance(value, list):
        return _listed(value)
    return [
        {key: deepcopy(entry[key]) for key in keys if entry.get(key) is not None} if isinstance(entry, dict) else entry
        for entry in value
    ]


def _listed(value: Any) -> Any:
    """``value`` copied; none reads as an empty list, anything else is left for the validator to refuse."""
    return [] if value is None else deepcopy(value)


def _list(value: Any) -> List[Any]:
    return list(value) if isinstance(value, list) else []


__all__ = ["PACKAGE_VERSION", "exported_packages"]
