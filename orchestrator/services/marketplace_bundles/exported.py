"""PRD-270 W2 (K9, K10) — what every export of Automatos Cloud shares: the exported item, the newest row
of each lineage, the envelope, and the strict check every bundle passes before it is listed.

An export reads named columns only, so a row's configuration, owner, workspace or install count is never
loaded, and builds its bundle field by field from the format's own keys (schema.py). Display fields the
format caps more tightly than their columns are fitted, never refused: an icon that is no emoji or short
name (an image URL, say) reads as none, tags keep the ones that fit, and the publisher's name is cut to
the format's length. Every other field travels as it is, and a bundle the validator refuses is logged as
an error and left out, so the catalogue never lists what a local install would refuse.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional

from services.marketplace_bundles.canonical import revision_of
from services.marketplace_bundles.errors import BundleError
from services.marketplace_bundles.lineage import external_id, lineage_of
from services.marketplace_bundles.schema import (
    BUNDLE_SCHEMA, ICON_MAX, ICON_PATTERN, PUBLISHER_NAME_MAX, TAG_MAX, TAGS_MAX,
)
from services.marketplace_bundles.validate import validate_bundle
from services.marketplace_public.fields import DEFAULT_PUBLISHER
from services.marketplace_public.publishers import publisher_of

logger = logging.getLogger(__name__)

MARKETPLACE_OWNER = "marketplace"
_ICON = re.compile(ICON_PATTERN)


@dataclass(frozen=True)
class ItemEntry:
    """An exported item's identity: its type, externalId and version, the ids of its lineage's exported
    rows on Automatos Cloud, and the id of the submitter's own row the lineage was cloned from (None for a
    ``row:`` lineage): a Playbook or a package naming any of them names this item."""

    type: str
    external_id: str
    version: str
    row_ids: FrozenSet[Any]
    source_id: Optional[Any] = None


@dataclass(frozen=True)
class ExportedItem:
    """One approved item as Automatos Cloud lists it: its entry, its bundle (a schema 1 document
    validate_bundle passed; never mutated) and its revision, the sha256 of the bundle's canonical bytes."""

    entry: ItemEntry
    bundle: Dict[str, Any]
    revision: str


@dataclass(frozen=True)
class Lineage:
    """A lineage's newest exportable row, its externalId and the ids of all its exportable rows."""

    external_id: str
    newest: Any
    row_ids: FrozenSet[Any]

    @property
    def source_id(self) -> Optional[Any]:
        """The submitter's own row every row of a ``from:`` lineage was cloned from; None for ``row:``."""
        return getattr(self.newest, "cloned_from_id", None)


def newest_per_lineage(item_type: str, rows: Iterable[Any]) -> List[Lineage]:
    """One Lineage per lineage among ``rows``, which come newest first, in that order. A row nothing
    stable names (no clone source and no natural key) is left out and logged."""
    order: List[str] = []
    newest: Dict[str, Any] = {}
    ids: Dict[str, List[Any]] = {}
    for row in rows:
        lineage = lineage_of(item_type, row)
        if lineage is None:
            logger.warning("Automatos Cloud export: %s row %s has no lineage and is left out", item_type, row.id)
            continue
        if lineage not in newest:
            order.append(lineage)
            newest[lineage] = row
        ids.setdefault(lineage, []).append(row.id)
    return [Lineage(external_id(item_type, key), newest[key], frozenset(ids[key])) for key in order]


def icon_of(value: Any) -> str:
    """``value`` when it is an icon the format allows (an emoji or a short name), else no icon."""
    if isinstance(value, str) and len(value) <= ICON_MAX and _ICON.fullmatch(value) is not None:
        return value
    return ""


def tags_of(value: Any) -> List[str]:
    """The row's distinct tags that fit the format, in their order, at most TAGS_MAX."""
    if not isinstance(value, (list, tuple)):
        return []
    fitting = (tag for tag in value if isinstance(tag, str) and tag.strip() and len(tag) <= TAG_MAX)
    return list(dict.fromkeys(fitting))[:TAGS_MAX]


def publisher_name(names: Mapping[int, str], user_id: Optional[int]) -> str:
    """W1's public display name of the row's original creator (publishers.py: never an email, else
    'Automatos'), cut to the format's length."""
    name = publisher_of(names, user_id)[:PUBLISHER_NAME_MAX].rstrip()
    return name or DEFAULT_PUBLISHER


def envelope(item_type: str, lineage: Lineage, publisher: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """The bundle of the lineage's newest row (an agent's or a Playbook's columns), its type's
    ``body`` under the type's name."""
    row = lineage.newest
    return {
        "bundle": BUNDLE_SCHEMA,
        "type": item_type,
        "externalId": lineage.external_id,
        "version": row.version,
        "name": row.name,
        "description": row.description or "",
        "category": row.marketplace_category or "",
        "icon": icon_of(row.marketplace_icon),
        "tags": tags_of(row.tags),
        "publisher": {"name": publisher},
        item_type: body,
    }


def exported_item(item_type: str, lineage: Lineage, bundle: Dict[str, Any]) -> Optional[ExportedItem]:
    """The ExportedItem of ``bundle`` once validate_bundle passes it; None, logged as an error, when it
    refuses it (the message names the code and the place, never a value of the bundle)."""
    try:
        validate_bundle(bundle)
        revision = revision_of(bundle)
    except BundleError as error:
        logger.error(
            "Automatos Cloud export: %s row %s is left out: its bundle is refused (%s: %s)",
            item_type, lineage.newest.id, error.code, error.message,
        )
        return None
    entry = ItemEntry(item_type, lineage.external_id, bundle["version"], lineage.row_ids, lineage.source_id)
    return ExportedItem(entry, bundle, revision)


def items_by_row(items: Iterable[ExportedItem], item_type: str) -> Dict[Any, str]:
    """Each row id that names an exported item of ``item_type`` -> its externalId: the item's exported
    rows, and the submitter's own row its lineage was cloned from (a community Playbook's step names that
    one; its id is never answered, only the item's externalId)."""
    return {
        row_id: item.entry.external_id
        for item in items if item.entry.type == item_type
        for row_id in (*item.entry.row_ids, item.entry.source_id) if row_id is not None
    }


__all__ = [
    "MARKETPLACE_OWNER", "ExportedItem", "ItemEntry", "Lineage", "envelope", "exported_item", "icon_of",
    "items_by_row", "newest_per_lineage", "publisher_name", "tags_of",
]
