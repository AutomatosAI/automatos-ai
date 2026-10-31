"""PRD-270 W2 US-270-13d (§5.1, §5.5, K16) — the Sync's passes after W1's themes: the plugins, then the items.

W1's ``POST /api/marketplace/cloud-sync`` runs W1's theme pass (``pull.sync_catalog``), then ``run_item_passes``
once, then W1's auto-update (``updates.apply_cloud_updates``). The catalogue is fetched once: W1's result
carries it parsed (``SyncOutcome.catalog``), and the passes read its item entries (``items_feed.Entries``).

They run only when W1's pass succeeded, in the local edition with ``AUTOMATOS_CLOUD_URL`` set
(``pull.sync_enabled``), and under W1's one-Sync-at-a-time rule: they take W1's in-process lock again
(``pull._SYNC_LOCK``, never blocking) and hold it while they run, so a Sync pressed meanwhile gets W1's "already
running" and runs no pass, and no theme pass overlaps them. ``sync_catalog`` releases that lock when it returns
(K4 leaves it so), so there is a moment between W1's pass and these when another Sync can take it first: then
this Sync's passes don't run (logged; its answer counts its themes only) and that Sync's passes bring the items.

1. The plugin pass (``plugin_import.import_plugins``, each plugin committed on its own) comes first: an agent's
   plugins, and the skills they make, must exist before the agent is written.
2. The item pass: the agent, Playbook and package entries no mapping holds at their revision have their
   bundles fetched (``items_feed.fetch_bundles``) with no transaction open, and ``import_items`` writes them
   in one transaction.

Each type is counted {new, updated, unchanged, refused}: an entry the catalogue refused (``Entries.refused``), an
unchanged entry (nothing fetched), a bundle its fetch refused, and what each pass answered. When the plugin
pass fails (its mapping read, or anything unforeseen), its error is recorded in W1's ``marketplace_cloud_sync``
row and no item pass runs (an agent written without its plugins would keep them unresolved until its next
revision): every entry of the four types then counts refused, as ``import_items`` counts a pass that failed (a
plugin committed before the failure is unchanged on the next Sync). A failed item pass records its own error;
the theme pass's rows and the plugins stand. A plugin past ``AUTOMATOS_CLOUD_PLUGINS_PER_SYNC`` is counted in
``plugins.waiting`` only.

The four counts, summed over the types, are added to that row's in one UPDATE, so its counts, and the answer's
``counts``, are the whole Sync's (W1's toast "N new, M updated from Automatos Cloud" then counts everything); if
that UPDATE fails, the row records why. Synchronous and blocking (W1's plain-``def`` route); it commits.
"""
from __future__ import annotations

import logging
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, NamedTuple, Optional, Tuple

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from core.models.marketplace_cloud import CLOUD_ITEM_TYPES, MARKETPLACE_CLOUD_SYNC_ROW_ID, MarketplaceCloudSync
from services.marketplace_bundles.import_items import ITEM_TYPES, import_items
from services.marketplace_bundles.items_feed import CatalogEntry, Entries, changed_entries, fetch_bundles
from services.marketplace_bundles.plugin_import import ITEM_TYPE as PLUGIN
from services.marketplace_bundles.plugin_import import RunAsync, import_plugins
from services.marketplace_cloud import pull
from services.marketplace_cloud.fetch import ClientFactory
from services.marketplace_cloud.pull import LAST_ERROR_MAX, REFUSED, UNCHANGED, SyncCounts, SyncOutcome, sync_row

logger = logging.getLogger(__name__)

PLUGINS_FAILED = (
    "The plugins from Automatos Cloud couldn't be read, so no plugin or item was added; the themes stand."
)
ITEMS_NOT_READ = "The items from Automatos Cloud couldn't be read, so none was added; the themes and plugins stand."
COUNTS_NOT_SAVED = "The Sync's plugin and item counts couldn't be saved; what it added stands."

NO_COUNTS: Mapping[str, SyncCounts] = MappingProxyType({item_type: SyncCounts() for item_type in CLOUD_ITEM_TYPES})
COUNT_COLUMNS: Mapping[str, Any] = MappingProxyType({
    "new": MarketplaceCloudSync.new_count, "updated": MarketplaceCloudSync.updated_count,
    "unchanged": MarketplaceCloudSync.unchanged_count, "refused": MarketplaceCloudSync.refused_count,
})


class Passes(NamedTuple):
    """The passes after W1's: the counts by type (agent, playbook, package, plugin) and the plugins waiting."""

    counts: Mapping[str, SyncCounts] = NO_COUNTS
    waiting: int = 0

    def total(self) -> SyncCounts:
        """The four counts summed over the types."""
        return _summed(self.counts.values())

    def view(self, themes: SyncCounts) -> Dict[str, Any]:
        """``{counts, items, plugins}``: the whole Sync's counts (``themes``, W1's, and every type's), each
        type's ``{new, updated, unchanged, refused}`` and ``{waiting}``."""
        return {
            "counts": _summed((themes, self.total())).view(),
            "items": {item_type: self.counts[item_type].view() for item_type in CLOUD_ITEM_TYPES},
            "plugins": {"waiting": self.waiting},
        }


def run_item_passes(
    db: Session, sync_result: SyncOutcome, client_factory: Optional[ClientFactory], run_async: RunAsync,
) -> Dict[str, Any]:
    """After W1's theme pass: the plugin pass, then the item pass, their counts added to W1's row (K16).

    Answers ``{counts, items: {agent, playbook, package, plugin: counts}, plugins: {waiting}}``; ``counts`` is
    the whole Sync's. Runs nothing (every type counted zero) unless ``sync_result`` succeeded, carries its
    catalogue and the Sync is on here, or while another Sync holds W1's lock. ``run_async`` runs one coroutine
    off the event loop (``asyncio.run`` in the plain-def route's worker thread).
    """
    if not (sync_result.ok and sync_result.catalog is not None and pull.sync_enabled()):
        return Passes().view(sync_result.counts)
    if not pull._SYNC_LOCK.acquire(blocking=False):  # W1's one-Sync lock: sync_catalog held it for the theme pass
        logger.warning("Sync with Automatos Cloud: another Sync took the lock first, so its passes bring the items")
        return Passes().view(sync_result.counts)
    try:
        passes = _passes(db, sync_result.catalog.entries, client_factory, run_async)
        _add_counts(db, passes.total())
    finally:
        pull._SYNC_LOCK.release()
    answer = passes.view(sync_result.counts)
    logger.info("Sync with Automatos Cloud items: %s, %d plugins waiting", answer["items"], passes.waiting)
    return answer


def _passes(db: Session, entries: Entries, client_factory: Optional[ClientFactory], run_async: RunAsync) -> Passes:
    """The plugin pass, then the item pass, counted by type with the catalogue's refusals."""
    catalogued = refused_by_type(entries.refused)
    try:
        plugins = import_plugins(db, entries.listed, client_factory, run_async)
    except Exception:  # its mapping read, or anything unforeseen: no item pass runs without its plugins
        logger.exception("Sync with Automatos Cloud: the plugin pass failed")
        record_error(db, PLUGINS_FAILED)
        return Passes(merged(catalogued, refused_by_type(entries.listed)))
    items = item_counts(db, entries.listed, client_factory)
    return Passes(merged(catalogued, {PLUGIN: plugins.counts}, items), plugins.waiting)


def item_counts(
    db: Session, listed: Iterable[CatalogEntry], client_factory: Optional[ClientFactory],
) -> Dict[str, SyncCounts]:
    """The item pass over the listed agent, Playbook and package entries: the unchanged ones counted, the
    others' bundles fetched and written in one transaction (``import_items``); the counts by type."""
    chosen = tuple(entry for entry in listed if entry.type in ITEM_TYPES)
    if not chosen:
        return {}
    try:
        changed = changed_entries(db, chosen)
        db.rollback()  # the mappings' read ends before any fetch
    except SQLAlchemyError:
        logger.exception("Sync with Automatos Cloud: the item mappings couldn't be read")
        record_error(db, ITEMS_NOT_READ)
        return refused_by_type(chosen)
    to_fetch = frozenset(changed)
    unchanged = counted((entry.type for entry in chosen if entry not in to_fetch), UNCHANGED)
    fetched = fetch_bundles(changed, client_factory)
    written = import_items(db, fetched.bundles).counts if fetched.bundles else {}
    return merged(unchanged, refused_by_type(fetched.refused), written)


def counted(types: Iterable[str], verdict: str = REFUSED) -> Dict[str, SyncCounts]:
    """One ``verdict`` per type named, counted by type."""
    counts: Dict[str, SyncCounts] = {}
    for item_type in types:
        counts = {**counts, item_type: counts.get(item_type, SyncCounts()).plus(verdict)}
    return counts


def refused_by_type(items: Iterable[Any]) -> Dict[str, SyncCounts]:
    """Each of ``items`` (an entry or a refusal: anything with a ``type``) counted refused, by type."""
    return counted(item.type for item in items)


def merged(*parts: Mapping[str, SyncCounts]) -> Dict[str, SyncCounts]:
    """The four types' counts summed over ``parts``; a type a part leaves out counts nothing there."""
    return {
        item_type: _summed(part[item_type] for part in parts if item_type in part)
        for item_type in CLOUD_ITEM_TYPES
    }


def _summed(counts: Iterable[SyncCounts]) -> SyncCounts:
    listed: Tuple[SyncCounts, ...] = tuple(counts)
    return SyncCounts(**{name: sum(getattr(one, name) for one in listed) for name in COUNT_COLUMNS})


def record_error(db: Session, message: str) -> None:
    """Roll the pass back and record its plain error in W1's row (the counts and the success stand)."""
    try:
        db.rollback()
        sync_row(db).last_error = message[:LAST_ERROR_MAX]
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        logger.exception("Sync with Automatos Cloud: the passes' failure couldn't be recorded")


def _add_counts(db: Session, total: SyncCounts) -> None:
    """Add ``total`` to W1's row's four counts in one UPDATE (W1's pass wrote the row); a failure is logged,
    leaves the theme pass's counts and is recorded as the row's error."""
    if total == SyncCounts():
        return
    added = {column: column + getattr(total, name) for name, column in COUNT_COLUMNS.items()}
    try:
        (
            db.query(MarketplaceCloudSync)
            .filter(MarketplaceCloudSync.id == MARKETPLACE_CLOUD_SYNC_ROW_ID)
            .update(added, synchronize_session="fetch")
        )
        db.commit()
    except SQLAlchemyError:
        logger.exception("Sync with Automatos Cloud: the passes' counts couldn't be recorded")
        record_error(db, COUNTS_NOT_SAVED)


__all__ = [
    "COUNTS_NOT_SAVED", "ITEMS_NOT_READ", "PLUGINS_FAILED", "Passes", "counted", "item_counts", "merged", "record_error",
    "refused_by_type", "run_item_passes",
]
