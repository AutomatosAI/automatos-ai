"""PRD-270 US-270-03a (§5.1, §5.3, K13, K14) — Sync with Automatos Cloud: the local edition's catalogue pull.

``sync_catalog`` runs only in the local edition with ``AUTOMATOS_CLOUD_URL`` set, when someone presses
the button (nothing schedules it, §8-Q2), one Sync at a time per process: a second press while one runs
is answered "already running", never queued. It GETs the catalogue (``fetch.py``), checks it
(``feed.py``) and writes what verified in ONE transaction:

* a theme id that any local-origin Marketplace row holds (``source`` NULL: the operator's own listing,
  whatever its status) is skipped whole and counted refused, so its installs are never offered a Cloud
  version (``listings.all_listings`` groups versions by theme id);
* an item whose digest a Marketplace row already holds is ``unchanged``; a version number a Marketplace
  row holds with other bytes is refused, and so is a row a unique index refuses because another Sync or
  a submission wrote it meanwhile (each write is its own savepoint, so the rest of the Sync goes on);
* any other item is written as a Cloud row: ``new`` when the theme had no Cloud row before this Sync,
  else ``updated`` (the counts are per version). A Cloud row is an approved, reviewed Marketplace
  version, ``source`` 'cloud', with the checked manifest, file list and signature, the Marketplace
  storage prefix and no stored file (its package is fetched on install, US-270-04a), ``scan_verdict``
  'not_run' (this install scanned nothing: the signature is the review's proof) and no submitter, plus
  one ``theme_review_events`` 'approved' row with no actor and ``detail {source: 'cloud'}``.

Nothing else is written: a private upload (``owner_type`` 'workspace') is never read for a write, a
version the catalogue no longer lists stays, and a revoked one is the revocation pull's. On success the
``marketplace_cloud_sync`` row records the attempt, the success, the catalogue's ``generatedAt`` and the
counts. Any failure (the setting, the fetch, a refused catalogue, a keys file that won't load, a database
error, anything unexpected) rolls every row back and records only ``last_attempt_at`` and ``last_error``:
the last good Sync stays. Synchronous; it ends the caller's reads before the fetch, so no transaction is
open while Automatos Cloud answers, and it commits.
"""
from __future__ import annotations

import logging
import threading
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, FrozenSet, Iterable, NamedTuple, Optional, Tuple

from sqlalchemy import distinct, func, or_
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from config import config
from core.models.marketplace_cloud import MARKETPLACE_CLOUD_SYNC_ROW_ID, MarketplaceCloudSync
from core.models.themes import THEME_SOURCE_CLOUD, ThemePackage, ThemeReviewEvent
from services.marketplace_cloud.feed import Catalog, CatalogError, CheckedItem, feed_max_bytes, parse_catalog
from services.marketplace_cloud.fetch import (
    ACCEPT_JSON, ClientFactory, CloudFetchFailed, checked_origin, fetch_bytes,
)
from services.marketplace_cloud.urls import catalog_url, configured_origin, configured_url
from services.themes import signature_verify
from services.themes.choices import MARKETPLACE_OWNER
from services.themes.revocation_feed import as_utc, iso_time
from services.themes.storage import theme_prefix

logger = logging.getLogger(__name__)

STATUS_APPROVED = "approved"
EVENT_APPROVED = "approved"
SCAN_NOT_RUN = "not_run"
EVENT_DETAIL = {"source": THEME_SOURCE_CLOUD}
STALE_AFTER = timedelta(days=7)
LAST_ERROR_MAX = 500
NEW, UPDATED, UNCHANGED, REFUSED = "new", "updated", "unchanged", "refused"
WRITTEN = frozenset({NEW, UPDATED})

SYNC_OFF = "Sync with Automatos Cloud runs only on a local install with AUTOMATOS_CLOUD_URL set."
KEYS_UNREADABLE = "This install's signing keys couldn't be read, so nothing was synced."
NOT_SAVED = "The synced listings couldn't be saved."
ALREADY_RUNNING = "A Sync with Automatos Cloud is already running. Try again when it has finished."
UNEXPECTED = "The Sync with Automatos Cloud failed; the server log has the detail."

_SYNC_LOCK = threading.Lock()


@dataclass(frozen=True)
class SyncCounts:
    """One Sync's listings: new, updated, unchanged and refused; ``skipped`` items of a type this wave doesn't list."""

    new: int = 0
    updated: int = 0
    unchanged: int = 0
    refused: int = 0
    skipped: int = 0

    def plus(self, verdict: str) -> "SyncCounts":
        """These counts with one more ``verdict``."""
        return replace(self, **{verdict: getattr(self, verdict) + 1})

    def view(self) -> Dict[str, int]:
        """``{new, updated, unchanged, refused}``, as the routes and the status answer them."""
        return {NEW: self.new, UPDATED: self.updated, UNCHANGED: self.unchanged, REFUSED: self.refused}


@dataclass(frozen=True)
class SyncOutcome:
    """One Sync: ``ok`` with its counts, else its plain ``error``: ``off`` when it can't run here and ``busy``
    while another runs (neither records anything), else the error it recorded."""

    ok: bool
    counts: SyncCounts = field(default_factory=SyncCounts)
    error: Optional[str] = None
    off: bool = False
    busy: bool = False


class Held(NamedTuple):
    """What the Marketplace held before this Sync, for the catalogue's theme ids and digests."""

    local_themes: FrozenSet[str]
    cloud_themes: FrozenSet[str]
    digests: FrozenSet[str]
    versions: FrozenSet[Tuple[str, str]]

    def with_item(self, item: CheckedItem) -> "Held":
        """This, with ``item``'s digest and version held too (the Sync wrote it)."""
        return self._replace(
            digests=self.digests | {item.digest}, versions=self.versions | {(item.theme_id, item.version)},
        )


def sync_enabled() -> bool:
    """True in the local edition with ``AUTOMATOS_CLOUD_URL`` set (the hosted edition serves the catalogue)."""
    return bool(config.IS_LOCAL_EDITION) and bool(configured_url())


# --------------------------------------------------------------------------- applying


def held_state(db: Session, items: Iterable[CheckedItem]) -> Held:
    """The Marketplace rows of the catalogue's theme ids or digests, as ``Held``."""
    listed = tuple(items)
    if not listed:
        return Held(frozenset(), frozenset(), frozenset(), frozenset())
    rows = (
        db.query(ThemePackage.theme_id, ThemePackage.version, ThemePackage.digest, ThemePackage.source)
        .filter(ThemePackage.owner_type == MARKETPLACE_OWNER)
        .filter(or_(ThemePackage.theme_id.in_(sorted({item.theme_id for item in listed})),
                    ThemePackage.digest.in_(sorted({item.digest for item in listed}))))
        .all()
    )
    return Held(
        local_themes=frozenset(row.theme_id for row in rows if row.source is None),
        cloud_themes=frozenset(row.theme_id for row in rows if row.source == THEME_SOURCE_CLOUD),
        digests=frozenset(row.digest for row in rows),
        versions=frozenset((row.theme_id, row.version) for row in rows),
    )


def classify(item: CheckedItem, held: Held) -> str:
    """What the Sync does with ``item``: ``new`` or ``updated`` (written), ``unchanged`` or ``refused``."""
    if item.theme_id in held.local_themes:
        logger.warning("Automatos Cloud: %s is this install's own listing, so the Sync leaves it", item.theme_id)
        return REFUSED
    if item.digest in held.digests:
        return UNCHANGED
    if (item.theme_id, item.version) in held.versions:
        logger.warning(
            "Automatos Cloud: %s %s is held here with other files, so it is refused", item.theme_id, item.version,
        )
        return REFUSED
    return UPDATED if item.theme_id in held.cloud_themes else NEW


def _cloud_row(item: CheckedItem) -> ThemePackage:
    return ThemePackage(
        id=uuid.uuid4(), theme_id=item.theme_id, version=item.version, owner_type=MARKETPLACE_OWNER,
        workspace_id=None, digest=item.digest, manifest=item.manifest, file_manifest=item.files,
        storage_prefix=theme_prefix(item.theme_id, item.version), status=STATUS_APPROVED, reviewed=True,
        scan_verdict=SCAN_NOT_RUN, signature=item.signature, source=THEME_SOURCE_CLOUD, submitted_by=None,
    )


def write_item(db: Session, item: CheckedItem, at: datetime) -> bool:
    """Write ``item``'s Cloud row and its 'approved' review event in a savepoint (no file is stored).

    False, with nothing written, when a unique index refuses the row: another Sync or a submission wrote
    that version or digest since ``held_state`` read. Any other database error is raised. The caller commits.
    """
    package = _cloud_row(item)
    try:
        with db.begin_nested():
            db.add(package)
            db.flush()
            db.add(ThemeReviewEvent(
                package_id=package.id, event=EVENT_APPROVED, from_status=None, to_status=STATUS_APPROVED,
                actor_user_id=None, detail=dict(EVENT_DETAIL), created_at=at,
            ))
            db.flush()
    except IntegrityError:
        logger.warning(
            "Automatos Cloud: %s %s was written here meanwhile, so it is refused", item.theme_id, item.version,
        )
        return False
    return True


def sync_row(db: Session) -> MarketplaceCloudSync:
    """The one ``marketplace_cloud_sync`` row, added when there is none yet."""
    row = db.get(MarketplaceCloudSync, MARKETPLACE_CLOUD_SYNC_ROW_ID)
    if row is None:
        row = MarketplaceCloudSync(id=MARKETPLACE_CLOUD_SYNC_ROW_ID)
        db.add(row)
    return row


def _record_success(db: Session, catalog: Catalog, counts: SyncCounts, at: datetime) -> None:
    sync = sync_row(db)
    sync.last_attempt_at = at
    sync.last_success_at = at
    sync.feed_generated_at = catalog.generated_at
    sync.last_error = None
    sync.new_count, sync.updated_count = counts.new, counts.updated
    sync.unchanged_count, sync.refused_count = counts.unchanged, counts.refused


def apply_catalog(db: Session, catalog: Catalog, at: datetime) -> SyncCounts:
    """Write every checked item as K14 says and record the success; the caller commits."""
    held = held_state(db, catalog.items)
    counts = SyncCounts(refused=len(catalog.refused), skipped=catalog.skipped)
    for item in catalog.items:
        verdict = classify(item, held)
        if verdict in WRITTEN:
            verdict = verdict if write_item(db, item, at) else REFUSED
            held = held.with_item(item)
        counts = counts.plus(verdict)
    _record_success(db, catalog, counts, at)
    return counts


def _failed(db: Session, attempted_at: datetime, message: str) -> SyncOutcome:
    """Roll every row change back and record the attempt and its error; nothing else changes."""
    db.rollback()
    logger.warning("Sync with Automatos Cloud failed: %s", message)
    try:
        sync = sync_row(db)
        sync.last_attempt_at = attempted_at
        sync.last_error = message[:LAST_ERROR_MAX]
        db.commit()
    except SQLAlchemyError:
        db.rollback()
        logger.exception("Sync with Automatos Cloud: the failure couldn't be recorded")
    return SyncOutcome(ok=False, error=message)


def _run_sync(db: Session, client_factory: Optional[ClientFactory]) -> SyncOutcome:
    attempted_at = datetime.now(timezone.utc)
    try:
        origin = checked_origin(configured_url())
        db.rollback()  # the route's reads end (nothing is written before here): none is open while Cloud answers
        data = fetch_bytes(catalog_url(origin), feed_max_bytes(), ACCEPT_JSON, client_factory)
        counts = apply_catalog(db, parse_catalog(data, db), attempted_at)
        db.commit()
    except (CloudFetchFailed, CatalogError) as refused:
        return _failed(db, attempted_at, str(refused))
    except signature_verify.SigningKeysError:
        logger.exception("Sync with Automatos Cloud: the signing keys file didn't load")
        return _failed(db, attempted_at, KEYS_UNREADABLE)
    except SQLAlchemyError:
        logger.exception("Sync with Automatos Cloud: the synced listings couldn't be saved")
        return _failed(db, attempted_at, NOT_SAVED)
    except Exception:  # anything else still rolls back and is recorded, never a half-written session
        logger.exception("Sync with Automatos Cloud failed unexpectedly")
        return _failed(db, attempted_at, UNEXPECTED)
    logger.info("Sync with Automatos Cloud: %s", counts.view())
    return SyncOutcome(ok=True, counts=counts)


def sync_catalog(db: Session, client_factory: Optional[ClientFactory] = None) -> SyncOutcome:
    """Fetch, check and write the catalogue in one transaction; any failure writes no listing.

    The Sync owns ``db``'s transaction: whatever the caller left in it is rolled back before the fetch, as on any
    failure, so the caller hands over a session that has only read.
    """
    if not sync_enabled():
        return SyncOutcome(ok=False, error=SYNC_OFF, off=True)
    if not _SYNC_LOCK.acquire(blocking=False):
        return SyncOutcome(ok=False, error=ALREADY_RUNNING, busy=True)
    try:
        return _run_sync(db, client_factory)
    finally:
        _SYNC_LOCK.release()


# --------------------------------------------------------------------------- status


def is_stale(last_success_at: Optional[datetime], now: datetime) -> bool:
    """True when no Sync succeeded in the last ``STALE_AFTER`` (or ever)."""
    return last_success_at is None or as_utc(now) - as_utc(last_success_at) > STALE_AFTER


def _stored_counts(row: Optional[MarketplaceCloudSync]) -> SyncCounts:
    if row is None:
        return SyncCounts()
    return SyncCounts(new=row.new_count or 0, updated=row.updated_count or 0,
                      unchanged=row.unchanged_count or 0, refused=row.refused_count or 0)


def sync_status(db: Session, now: Optional[datetime] = None) -> Dict[str, Any]:
    """What the Sync's status shows: on or off, the origin in use, the last attempt, success and error, the counts,
    how many themes this install holds from Automatos Cloud, and whether the last success is over a week old."""
    row = db.get(MarketplaceCloudSync, MARKETPLACE_CLOUD_SYNC_ROW_ID)
    themes = (
        db.query(func.count(distinct(ThemePackage.theme_id)))
        .filter(ThemePackage.source == THEME_SOURCE_CLOUD)
        .scalar()
    )
    last_success_at = None if row is None else row.last_success_at
    return {
        "enabled": sync_enabled(),
        "cloudUrl": configured_origin(),
        "lastAttemptAt": None if row is None else iso_time(row.last_attempt_at),
        "lastSuccessAt": iso_time(last_success_at),
        "lastError": None if row is None else row.last_error,
        "generatedAt": None if row is None else iso_time(row.feed_generated_at),
        "counts": _stored_counts(row).view(),
        "themes": int(themes or 0),
        "stale": is_stale(last_success_at, now or datetime.now(timezone.utc)),
    }


__all__ = [
    "ALREADY_RUNNING", "KEYS_UNREADABLE", "NOT_SAVED", "STALE_AFTER", "SYNC_OFF", "UNEXPECTED", "Held",
    "SyncCounts", "SyncOutcome", "apply_catalog", "classify", "held_state", "is_stale", "sync_catalog",
    "sync_enabled", "sync_row", "sync_status", "write_item",
]
