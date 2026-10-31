"""PRD-270 US-270-01 (§5.2, K8) — the hosted catalogue: every signed theme version SaaS lists, schema 1.

``build_catalog`` answers ``{schema: 1, generatedAt, items}``, themes only, built from the Themes
tab's own listing functions (``services/themes/listings.py``): ``all_listings`` (approved, reviewed
Marketplace versions that no ``disable`` covers, newest first per theme), its signature verdict, its
publisher and cover rules and ``previews_by_preset``. There is one item per listed version that

* is this edition's own row (``source`` NULL): a row a Sync wrote is never served on, and
* carries a signature an active Automatos key verifies over its file hashes.

An item carries what a local install needs to check it before listing it: the ``.sig.json``
envelope, the file manifest (``files``) and ``themeJson``, the base64 of the stored theme.json bytes.
theme.json is one of the hashed files, so the name, description and screenshot list a Sync shows are
bound by the signature too. Every URL is a catalogue path (``urls.py``). Nothing else leaves: no
package id, workspace, submitter, scan report, review, install or revocation-admin datum.

theme.json is read once per version and kept in memory, as ``serve.py`` keeps boot files: a stored
file never changes, and its first read is re-hashed against the file manifest (``serve.verified_file``).
A build does every database read first, then detaches the listed rows with their loaded columns and ends
its read transaction, so no pool connection waits while storage gives back a theme.json.
A keys file that does not load raises ``SigningKeysError``, and a theme.json storage can't give back
raises ``CatalogUnavailable``: the route answers 503 to both, never a catalogue that silently lists
less. While ``THEMES_MARKETPLACE`` is off the catalogue lists no item.

The route is public, so it never builds per request: ``catalog_snapshot`` keeps the built catalogue,
its body and its ETag for ``CATALOG_KEPT_SECONDS`` and builds one at a time, so concurrent callers share
one build. While another caller builds, a caller is answered the last snapshot this process holds,
however old, at once; when it holds none, the caller waits at most ``CATALOG_BUILD_WAIT_SECONDS`` and then
gets ``CatalogBusy`` (the route's 503 with ``Retry-After``). A build that fails keeps nothing, and the next
caller that gets the build builds again.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from sqlalchemy.orm import Session

from config import config
from core.database.read_release import release_if_read_only
from core.models.themes import ThemePackage
from services.marketplace_cloud.urls import catalog_file_path, package_path
from services.themes import signature_verify
from services.themes.bounded_cache import BoundedCache
from services.themes.contract import KB
from services.themes.listings import (
    FREE_LABEL, PHONE_PREVIEWS, PREVIEW_KINDS, SWITCH_OFF, WEB_PREVIEWS, all_listings, cover_url, listing_id_of,
    marketplace_switch, previews_by_preset, publisher_view, signature_verdict,
)
from services.themes.revocation_feed import as_utc, feed_bytes, feed_etag
from services.themes.serve import ThemeFileMissing, ThemeStorageUnavailable, module_url, read_stored_file, verified_file
from services.themes.signed_upload import ENVELOPE_ALGORITHM, ENVELOPE_FORM

logger = logging.getLogger(__name__)

CATALOG_SCHEMA = 1
ITEM_TYPE = "theme"
THEME_JSON = "theme.json"
FILE_FIELDS = ("type", "size", "sha256")
# More versions than a catalogue under its default 1024 KB cap can list (each item is a few KB).
THEME_JSON_KEPT = 512
CATALOG_KEPT_SECONDS = 30
# How long a caller with no snapshot held waits for another caller's build, and the busy 503's Retry-After.
CATALOG_BUILD_WAIT_SECONDS = 2.0
CATALOG_BUSY_RETRY_SECONDS = 5
SNAPSHOT_KEY = "catalog"

UNREADABLE = "The theme.json of a listed theme version can't be read from storage."
BUILD_BUSY = "Another caller is building the catalogue and no catalogue is held yet."


class CatalogSnapshot(NamedTuple):
    """A built catalogue, the bytes the route sends, its weak ETag, and when it was built (monotonic seconds)."""

    catalog: Dict[str, Any]
    body: bytes
    etag: str
    built_at: float


# (digest, storage prefix) -> base64 of the stored theme.json, re-hashed against the file manifest when read.
_theme_json: BoundedCache[str] = BoundedCache(THEME_JSON_KEPT)
# The one kept snapshot, under SNAPSHOT_KEY; ``_building`` lets one caller build at a time.
_snapshots: BoundedCache[CatalogSnapshot] = BoundedCache(1)
_building = threading.Lock()


class CatalogUnavailable(RuntimeError):
    """Storage could not give back a listed version's theme.json (the route's 503)."""


class CatalogBusy(RuntimeError):
    """Another caller's build held past ``CATALOG_BUILD_WAIT_SECONDS`` and no snapshot is held (the route's 503
    with ``Retry-After``)."""


def catalog_max_bytes() -> int:
    """The largest catalogue the hosted edition serves (``MARKETPLACE_CATALOG_FEED_MAX_KB``)."""
    return int(config.MARKETPLACE_CATALOG_FEED_MAX_KB) * KB


def forget_cached() -> None:
    """Drop the kept snapshot and every kept theme.json (tests; a process restart does the same)."""
    _snapshots.clear()
    _theme_json.clear()


def _fresh_snapshot() -> Optional[CatalogSnapshot]:
    held = _snapshots.get(SNAPSHOT_KEY)
    if held is None or time.monotonic() - held.built_at >= CATALOG_KEPT_SECONDS:
        return None
    return held


def _took_the_build(held: Optional[CatalogSnapshot]) -> bool:
    """True when this caller now holds ``_building``. With a snapshot held it never waits (that snapshot is its
    answer); with none, it waits at most ``CATALOG_BUILD_WAIT_SECONDS``."""
    if held is not None:
        return _building.acquire(blocking=False)
    return _building.acquire(timeout=CATALOG_BUILD_WAIT_SECONDS)


def catalog_snapshot(db: Session) -> CatalogSnapshot:
    """The catalogue with its body and ETag, built at most once per ``CATALOG_KEPT_SECONDS`` for every caller.

    While another caller builds, the last snapshot held, however old (a moment-old catalogue beats none), is
    the answer at once; with none held, ``CatalogBusy`` after at most ``CATALOG_BUILD_WAIT_SECONDS``. Neither
    reads the database or storage. Raises what ``build_catalog`` raises; a failed build keeps nothing.
    """
    held = _fresh_snapshot()
    if held is not None:
        return held
    last = _snapshots.get(SNAPSHOT_KEY)
    if not _took_the_build(last):
        if last is None:
            raise CatalogBusy(BUILD_BUSY)
        return last
    try:
        held = _fresh_snapshot()
        if held is not None:
            return held
        catalog = build_catalog(db)
        snapshot = CatalogSnapshot(catalog, feed_bytes(catalog), feed_etag(catalog), time.monotonic())
        _snapshots.put(SNAPSHOT_KEY, snapshot)
        return snapshot
    finally:
        _building.release()


def build_catalog(db: Session, now: Optional[datetime] = None) -> Dict[str, Any]:
    """The catalogue, schema 1: ``{schema, generatedAt, items}``, items by theme id, newest version first."""
    moment = as_utc(now or datetime.now(timezone.utc)).replace(microsecond=0)
    return {"schema": CATALOG_SCHEMA, "generatedAt": moment.isoformat(), "items": _items(db)}


# A catalogued version, its verdict and its publisher: everything the database gives an item.
Catalogued = Tuple[ThemePackage, signature_verify.Verdict, Dict[str, Any]]


def _items(db: Session) -> List[Dict[str, Any]]:
    """Every item: the database read first (``_catalogued``), its transaction ended, then each theme.json read."""
    if marketplace_switch() == SWITCH_OFF:
        return []
    signature_verify.load_keys()  # a keys file that doesn't load is the route's 503, never an empty catalogue
    catalogued = _catalogued(db)
    _end_reads(db, [package for package, _verdict, _publisher in catalogued])
    return [_item(package, verdict, publisher) for package, verdict, publisher in catalogued]


def _catalogued(db: Session) -> List[Catalogued]:
    """Each listed version the catalogue lists (``catalogued_verdict``), by theme id and newest first, with its
    verdict and publisher: every database read a build does."""
    rows = [row for versions in all_listings(db).values() for row in versions]
    verdicts = [(row, catalogued_verdict(row)) for row in rows]
    return [(row, verdict, publisher_view(db, row.theme_id, verdict)) for row, verdict in verdicts
            if verdict is not None]


def _end_reads(db: Session, packages: Sequence[ThemePackage]) -> None:
    """Detach ``packages`` with their loaded columns and end ``db``'s read transaction, so its pool connection is
    back before storage is read. A session holding changes keeps its rows, and a transaction
    ``release_if_read_only`` won't end (it wrote or holds a lock) is kept; the log says so."""
    if db.new or db.dirty or db.deleted:
        logger.warning("Catalogue: built inside a session holding changes, so its transaction is kept over storage")
        return
    for package in packages:
        db.expunge(package)
    if not release_if_read_only(db) and db.in_transaction():
        logger.warning("Catalogue: the build's transaction wasn't released (it wrote, locked or isn't tracked), so "
                       "it is kept over storage")


def catalogued_verdict(package: ThemePackage) -> Optional[signature_verify.Verdict]:
    """The verdict that puts a listed version in the catalogue: this edition's own row (``source`` NULL) whose
    signature an active key verifies. None for a Sync's row or one that isn't signed so (the files route's rule
    too, ``catalog_files.py``)."""
    if package.source is not None:
        return None
    verdict = signature_verdict(package)
    return verdict if verdict.verified else None


def _item(package: ThemePackage, verdict: signature_verify.Verdict, publisher: Dict[str, Any]) -> Dict[str, Any]:
    """One catalogued version as the catalogue lists it; reads its theme.json (the kept text first), never the
    database."""
    manifest = package.manifest or {}
    return {
        "type": ITEM_TYPE,
        "id": package.theme_id,
        "listingId": listing_id_of(package.theme_id),
        "version": package.version,
        "digest": package.digest,
        "name": manifest.get("name"),
        "description": manifest.get("description"),
        "publisher": publisher,
        "engines": manifest.get("engines"),
        "uses": list(manifest.get("uses") or []),
        "pricing": manifest.get("pricing") or FREE_LABEL,
        "signature": _envelope(package, verdict),
        "files": _files(package),
        "themeJson": _theme_json_text(package),
        "package": package_path(package.digest),
        **_screenshots(package),
    }


def _envelope(package: ThemePackage, verdict: signature_verify.Verdict) -> Dict[str, str]:
    """The kit's ``.sig.json`` for the version: its id, version and digest, the key that verified, the signature
    as standard base64 exactly as it encodes (the form ``signed_upload.parse_envelope`` holds it to)."""
    signature = base64.b64encode(base64.b64decode(package.signature, validate=True)).decode("ascii")
    return {
        "form": ENVELOPE_FORM, "algorithm": ENVELOPE_ALGORITHM, "keyId": verdict.key_id,
        "id": package.theme_id, "version": package.version, "digest": package.digest, "signature": signature,
    }


def _listed(package: ThemePackage) -> Mapping[str, Any]:
    return package.file_manifest if isinstance(package.file_manifest, dict) else {}


def _files(package: ThemePackage) -> Dict[str, Dict[str, Any]]:
    """The file manifest, each file's type, size and sha256 only, by path."""
    return {
        path: {field: row.get(field) for field in FILE_FIELDS}
        for path, row in sorted(_listed(package).items()) if isinstance(row, dict)
    }


def _theme_json_text(package: ThemePackage) -> str:
    """Base64 of the stored theme.json, read and checked once per version."""
    key = (package.digest, package.storage_prefix)
    held = _theme_json.get(key)
    if held is not None:
        return held
    try:
        data, _type, _sha256 = verified_file(package, THEME_JSON, read_stored_file)
    except (ThemeFileMissing, ThemeStorageUnavailable) as unreadable:
        logger.error(
            "Catalogue: the theme.json of %s %s (digest %s) could not be read (%s)",
            package.theme_id, package.version, package.digest, type(unreadable).__name__,
        )
        raise CatalogUnavailable(UNREADABLE) from unreadable
    text = base64.b64encode(data).decode("ascii")
    _theme_json.put(key, text)
    return text


def _screenshots(package: ThemePackage) -> Dict[str, Any]:
    """The cover, the phone cover and the previews per preset (``listings``' rules), each URL a catalogue path."""
    paths = {module_url(package.digest, path): catalog_file_path(package.digest, path) for path in _listed(package)}
    view = previews_by_preset(package)
    return {
        "cover": _path_of(cover_url(package, WEB_PREVIEWS), paths),
        "phoneCover": _path_of(cover_url(package, PHONE_PREVIEWS), paths),
        "presets": [_preset(preset, paths) for preset in view["presets"]],
        **{kind.other_key: [paths[url] for url in view[kind.other_key]] for kind in PREVIEW_KINDS},
    }


def _preset(preset: Mapping[str, Any], paths: Mapping[str, str]) -> Dict[str, Any]:
    shots = {
        kind.field: [{"screen": shot["screen"], "url": paths[shot["url"]]} for shot in preset[kind.field]]
        for kind in PREVIEW_KINDS
    }
    return {"id": preset["id"], "default": preset["default"], **shots}


def _path_of(url: Optional[str], paths: Mapping[str, str]) -> Optional[str]:
    return None if url is None else paths[url]


__all__ = [
    "CATALOG_BUILD_WAIT_SECONDS", "CATALOG_BUSY_RETRY_SECONDS", "CATALOG_KEPT_SECONDS", "CATALOG_SCHEMA",
    "CatalogBusy", "CatalogSnapshot", "CatalogUnavailable", "build_catalog", "catalog_max_bytes",
    "catalog_snapshot", "catalogued_verdict", "forget_cached",
]
