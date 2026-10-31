"""PRD-270 W2 US-270-12c (K11, K12) — the hosted catalogue's item entries, and the bundles they name.

``with_items`` is what W1's ``catalog.catalog_snapshot`` adds to ``build_catalog``: a NEW catalogue whose
``items`` are the built catalogue's (its themes, unchanged and first), then one entry per exported item
(``item_set``: agents, then Playbooks, then packages, then plugins). The snapshot then makes the bytes, the
weak ETag and the 30-second copy, and W1's route applies its size cap, so all of them cover the entries.

The entries never push the catalogue past what is served and read (P270W2-RVW-3): W1's route refuses the
whole catalogue, themes included, over ``MARKETPLACE_CATALOG_FEED_MAX_KB``, and every local reader over
``feed.CATALOG_MAX_ITEMS`` items. So the themes always come first, whole, and the entries after them are the
longest run of ``item_set``'s order (each export's own: agents and Playbooks newest first, packages and plugins
by slug) that keeps both within the ``Room`` the themes leave (``room_beside``: the item count, and the bytes
the route sends for the themes-only catalogue, its envelope included, with one separator byte counted per
entry). Every item after that run is left out, logged as a warning with its type and externalId once per
build: a package naming one is refused by local installs until it is listed.

An entry is METADATA ONLY, K11's keys exactly (``entry_of``): never a bundle body, since W1's route refuses
the whole catalogue, themes included, over ``MARKETPLACE_CATALOG_FEED_MAX_KB``. It names its bundle by path
(``catalog_paths.bundle_path``) and says ``trust: 'reviewed'``: nothing in a bundle is signed, so nothing is
Official (§8-Q3a). A bundle the strict validator refuses never gets here (``exported.exported_item`` logs it
as an error); one whose canonical bytes are over ``MARKETPLACE_CATALOG_BUNDLE_MAX_KB`` is left out and logged
as a warning.

Each build keeps its listed bundles' canonical bytes by revision, in one ``Listing`` (the newest build, in a
``BoundedCache``), so the bundle route answers from the build a catalogue listed: a left-out item's bundle is
not served. ``bundle_bytes_for`` answers only a revision the newest build lists: a revision a past build
listed is not served once a newer build drops it. A process whose newest build is older than
``ITEMS_KEPT_SECONDS`` (or that has none: each worker keeps its own) builds again, at most once for every
caller and one build at a time, so a request for an unlisted revision never builds on its own while the
listing is fresh. That build has no themes to measure, so it fits the room the build it replaces was fitted
to; with none held, the room of a catalogue with no theme (``_widest_room``), so it never lists less than a
catalogue does. Each process measures only its own catalogue builds: where several serve these routes, one
that last measured other themes can list a few edge items differently until it builds a catalogue. A build is
kept only when no build that started later is kept already, so a slow build never replaces a newer one. A
build that fails keeps nothing, and the next request tries again (W1's rule for the catalogue).

Never imports ``services/marketplace_cloud/catalog.py``: it imports this module.
"""
from __future__ import annotations

import logging
import threading
import time
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

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
from services.marketplace_cloud.feed import CATALOG_MAX_ITEMS
from services.themes.bounded_cache import BoundedCache
from services.themes.contract import KB
from services.themes.revocation_feed import feed_bytes

logger = logging.getLogger(__name__)

ITEM_TRUST = "reviewed"
# As long as W1's catalogue snapshot is kept (catalog.CATALOG_KEPT_SECONDS, not imported: see above).
ITEMS_KEPT_SECONDS = 30
LISTING_KEY = "listing"
# The comma a compact JSON array puts before every entry but the first, counted for every entry: never under.
ENTRY_SEPARATOR_BYTES = 1


class Room(NamedTuple):
    """What a catalogue's own items leave its item entries: how many more items every local reader takes
    (``feed.CATALOG_MAX_ITEMS``) and how many more bytes the hosted route serves
    (``MARKETPLACE_CATALOG_FEED_MAX_KB``). Either may be negative: then no entry fits."""

    items: int
    size: int


class Listing(NamedTuple):
    """One build: the catalogue entries, the canonical bytes of each listed bundle by revision, when it
    started reading (monotonic seconds), and the room it was fitted to."""

    entries: Tuple[Dict[str, Any], ...]
    bundles: Mapping[str, bytes]
    built_at: float
    room: Room


ListedBundle = Tuple[ExportedItem, bytes]


# The newest build, under LISTING_KEY; ``_building`` lets one bundle request build at a time, ``_keeping``
# keeps a build only when it is not older than the one held.
_listings: BoundedCache[Listing] = BoundedCache(1)
_building = threading.Lock()
_keeping = threading.Lock()


def bundle_max_bytes() -> int:
    """The largest bundle the hosted catalogue lists (``MARKETPLACE_CATALOG_BUNDLE_MAX_KB``)."""
    return int(config.MARKETPLACE_CATALOG_BUNDLE_MAX_KB) * KB


def catalog_cap_bytes() -> int:
    """The largest catalogue the hosted route serves (``MARKETPLACE_CATALOG_FEED_MAX_KB``, as W1's
    ``catalog.catalog_max_bytes``, which is not imported: see above)."""
    return int(config.MARKETPLACE_CATALOG_FEED_MAX_KB) * KB


def room_beside(catalog: Mapping[str, Any]) -> Room:
    """The room ``catalog``'s own items leave: the item count, and the bytes the route sends for ``catalog``
    as it is (``feed_bytes``: its envelope and items), each taken from its cap."""
    return Room(CATALOG_MAX_ITEMS - len(catalog["items"]), catalog_cap_bytes() - len(feed_bytes(catalog)))


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
    """A new catalogue: ``catalog`` with an entry per listed item after its own items, as many as fit the room
    they leave (``catalog`` is not changed). The build is kept for ``bundle_bytes_for``."""
    listing = _build(db, room_beside(catalog))
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
    """The kept build while it is fresh; else one built now, once for every caller, fitted to the room of the
    build it replaces (``_held_room``)."""
    held = _fresh()
    if held is not None:
        return held
    with _building:
        held = _fresh()
        return held if held is not None else _build(db, _held_room())


def _held_room() -> Room:
    """The room the build held, however old, was fitted to; with none held, ``_widest_room``."""
    held = _listings.get(LISTING_KEY)
    return _widest_room() if held is None else held.room


def _widest_room() -> Room:
    """The room of a catalogue with no theme, its envelope not counted: no catalogue leaves more."""
    return Room(CATALOG_MAX_ITEMS, catalog_cap_bytes())


def _build(db: Session, room: Room) -> Listing:
    """The exported items whose bundles fit the cap, as many as fit ``room`` (``_fitted``), as entries and
    bytes; kept unless a newer build is."""
    started, cap = time.monotonic(), bundle_max_bytes()
    capped = ((item, _capped(item, cap)) for item in item_set(db))
    listed, entries = _fitted([(item, data) for item, data in capped if data is not None], room)
    listing = Listing(
        entries=entries,
        bundles=MappingProxyType({item.revision: data for item, data in listed}),
        built_at=started,
        room=room,
    )
    _keep(listing)
    return listing


def _fitted(capped: Sequence[ListedBundle], room: Room) -> Tuple[Tuple[ListedBundle, ...], Tuple[Dict[str, Any], ...]]:
    """The longest run of ``capped``, in its order, whose entries fit ``room``, with those entries; each item
    after the run is logged as left out."""
    entries: List[Dict[str, Any]] = []
    size = 0
    for item, _data in capped:
        entry = entry_of(item)
        size += len(feed_bytes(entry)) + ENTRY_SEPARATOR_BYTES
        if len(entries) >= room.items or size > room.size:
            break
        entries.append(entry)
    fits = len(entries)
    for item, _data in capped[fits:]:
        logger.warning(
            "Automatos Cloud catalogue: %s %s is left out: no room after the themes and the %d entries listed "
            "(at most %d items, %d KB); a package naming it is refused by local installs until it is listed",
            item.entry.type, item.entry.external_id, fits, CATALOG_MAX_ITEMS, catalog_cap_bytes() // KB,
        )
    return tuple(capped[:fits]), tuple(entries)


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
    "ITEMS_KEPT_SECONDS", "ITEM_TRUST", "Listing", "Room", "bundle_bytes_for", "bundle_max_bytes",
    "catalog_cap_bytes", "entry_of", "forget_cached", "item_set", "room_beside", "with_items",
]
