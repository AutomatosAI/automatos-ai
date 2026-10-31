"""PRD-270 W2 US-270-12c (K11, K12) — the hosted catalogue's item entries, and the bundles they name.

``with_items`` is what W1's ``catalog.catalog_snapshot`` adds to ``build_catalog``: a NEW catalogue whose
``items`` are the built catalogue's (its themes, unchanged and first), then one entry per exported item
(``item_set``: agents, then Playbooks, then packages, then plugins). The snapshot then makes the bytes, the
weak ETag and the 30-second copy, and W1's route applies its size cap, so all of them cover the entries.

An entry is METADATA ONLY, K11's keys exactly (``entry_of``): never a bundle body, since W1's route refuses
the whole catalogue, themes included, over ``MARKETPLACE_CATALOG_FEED_MAX_KB``. It names its bundle by path
(``catalog_paths.bundle_path``) and says ``trust: 'reviewed'``: nothing in a bundle is signed, so nothing is
Official (§8-Q3a). A bundle the strict validator refuses never gets here (``exported.exported_item`` logs it
as an error); one whose canonical bytes are over ``MARKETPLACE_CATALOG_BUNDLE_MAX_KB`` is left out and logged
as a warning.

Each build keeps its bundles' canonical bytes by revision, in one ``Listing`` (the newest build, in a
``BoundedCache``), so the bundle route answers from the build a catalogue listed. ``bundle_bytes_for``
answers only a revision the newest build lists: a revision a past build listed is not served once a newer
build drops it. A process whose newest build is older than ``ITEMS_KEPT_SECONDS`` (or that has none: each
worker keeps its own) builds again, at most once for every caller and one build at a time, so a request
for an unlisted revision never builds on its own while the listing is fresh. A build is kept only when no
build that started later is kept already, so a slow build never replaces a newer one. A build that fails
keeps nothing, and the next request tries again (W1's rule for the catalogue).

Never imports ``services/marketplace_cloud/catalog.py``: it imports this module.
"""
from __future__ import annotations

import logging
import threading
import time
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

from sqlalchemy.orm import Session

from config import config
from services.marketplace_bundles.canonical import capped_bytes
from services.marketplace_bundles.catalog_paths import bundle_path
from services.marketplace_bundles.errors import BundleError
from services.marketplace_bundles.export_agents import exported_agents
from services.marketplace_bundles.export_packages import exported_packages
from services.marketplace_bundles.export_playbooks import exported_playbooks
from services.marketplace_bundles.export_plugins import exported_plugins
from services.marketplace_bundles.exported import ExportedItem
from services.themes.bounded_cache import BoundedCache
from services.themes.contract import KB

logger = logging.getLogger(__name__)

ITEM_TRUST = "reviewed"
# As long as W1's catalogue snapshot is kept (catalog.CATALOG_KEPT_SECONDS, not imported: see above).
ITEMS_KEPT_SECONDS = 30
LISTING_KEY = "listing"


class Listing(NamedTuple):
    """One build: the catalogue entries, the canonical bytes of each listed bundle by revision, and when it
    started reading (monotonic seconds)."""

    entries: Tuple[Dict[str, Any], ...]
    bundles: Mapping[str, bytes]
    built_at: float


# The newest build, under LISTING_KEY; ``_building`` lets one bundle request build at a time, ``_keeping``
# keeps a build only when it is not older than the one held.
_listings: BoundedCache[Listing] = BoundedCache(1)
_building = threading.Lock()
_keeping = threading.Lock()


def bundle_max_bytes() -> int:
    """The largest bundle the hosted catalogue lists (``MARKETPLACE_CATALOG_BUNDLE_MAX_KB``)."""
    return int(config.MARKETPLACE_CATALOG_BUNDLE_MAX_KB) * KB


def forget_cached() -> None:
    """Drop the kept build (tests; a process restart does the same)."""
    _listings.clear()


def item_set(db: Session) -> List[ExportedItem]:
    """Every item Automatos Cloud exports: agents, then Playbooks, then packages, then plugins. A Playbook
    names exported agents and a package any exported item, so each export reads the ones before it."""
    agents = exported_agents(db)
    playbooks = exported_playbooks(db, agents)
    plugins = exported_plugins(db)
    packages = exported_packages(db, agents, playbooks, plugin_items=plugins)
    return [*agents, *playbooks, *packages, *plugins]


def entry_of(item: ExportedItem) -> Dict[str, Any]:
    """The item as the catalogue lists it (K11): its identity, revision and display fields, read from its
    validated bundle, ``trust: 'reviewed'`` and the path of its bundle."""
    bundle = item.bundle
    return {
        "type": item.entry.type,
        "externalId": item.entry.external_id,
        "version": item.entry.version,
        "revision": item.revision,
        "name": bundle["name"],
        "description": bundle["description"],
        "category": bundle["category"],
        "icon": bundle["icon"],
        "tags": list(bundle["tags"]),
        "publisher": {"name": bundle["publisher"]["name"]},
        "trust": ITEM_TRUST,
        "bundle": bundle_path(item.revision),
    }


def with_items(catalog: Mapping[str, Any], db: Session) -> Dict[str, Any]:
    """A new catalogue: ``catalog`` with an entry per listed item after its own items (``catalog`` is not
    changed). The build is kept for ``bundle_bytes_for``."""
    listing = _build(db)
    return {**catalog, "items": [*catalog["items"], *listing.entries]}


def bundle_bytes_for(db: Session, revision: str) -> Optional[bytes]:
    """The canonical bytes of the bundle the newest build lists under ``revision``, or None."""
    return _current(db).bundles.get(revision)


def _fresh() -> Optional[Listing]:
    held = _listings.get(LISTING_KEY)
    if held is None or time.monotonic() - held.built_at >= ITEMS_KEPT_SECONDS:
        return None
    return held


def _current(db: Session) -> Listing:
    """The kept build while it is fresh; else one built now, once for every caller."""
    held = _fresh()
    if held is not None:
        return held
    with _building:
        held = _fresh()
        return held if held is not None else _build(db)


def _build(db: Session) -> Listing:
    """Every exported item whose bundle fits the cap, as entries and bytes; kept unless a newer build is."""
    started, cap = time.monotonic(), bundle_max_bytes()
    capped = ((item, _capped(item, cap)) for item in item_set(db))
    listed = [(item, data) for item, data in capped if data is not None]
    listing = Listing(
        entries=tuple(entry_of(item) for item, _data in listed),
        bundles=MappingProxyType({item.revision: data for item, data in listed}),
        built_at=started,
    )
    _keep(listing)
    return listing


def _keep(listing: Listing) -> None:
    with _keeping:
        held = _listings.get(LISTING_KEY)
        if held is None or held.built_at <= listing.built_at:
            _listings.put(LISTING_KEY, listing)


def _capped(item: ExportedItem, cap: int) -> Optional[bytes]:
    """The bundle's canonical bytes, or None, logged as a warning, when they are over ``cap``."""
    try:
        return capped_bytes(item.bundle, cap)
    except BundleError as error:
        logger.warning(
            "Automatos Cloud catalogue: %s %s is left out (%s: %s)",
            item.entry.type, item.entry.external_id, error.code, error.message,
        )
        return None


__all__ = [
    "ITEMS_KEPT_SECONDS", "ITEM_TRUST", "Listing", "bundle_bytes_for", "bundle_max_bytes", "entry_of",
    "forget_cached", "item_set", "with_items",
]
