"""PRD-270 W2 US-270-13a (§5.3, §5.6, K13) — locally: the catalogue's item entries, and their bundles fetched.

W1's ``feed.parse_catalog`` hands each item of a bundle type (``BUNDLE_TYPES``: agent, Playbook, package,
plugin) to ``Entries.read``; a theme stays W1's, and any other type is W1's to skip, uncounted (schema 1 is
additive). An entry (``parse_item_entry``) is refused, counted and never fetched, unless:

* its ``trust`` is exactly 'reviewed': nothing from Automatos Cloud is signed, so nothing is Official;
* its type, externalId (32 hex), version (K8's), revision (64 hex) and display fields (name, description,
  category, icon, tags and publisher, at K8's caps) pass the bundle format's own checks;
* no earlier entry of the catalogue has its type and externalId.

Only those keys are read. Its ``bundle`` (a path the catalogue gives) never is: the URL is DERIVED.

``changed_entries`` leaves out each entry a ``marketplace_cloud_items`` mapping holds at the same revision
(unchanged: nothing is fetched). ``rechecked_entries`` names those of them to fetch again at their revision
(P270W2-RVW-1): the mapping records, unresolved, what a later Sync can bring (``resolve.awaited``: a plugin, a
skill a plugin makes, a Playbook's helper that is a Cloud agent), and this install now holds some of it, or
the Sync has just fetched that agent's bundle, so the item pass resolves the item anew; what hasn't arrived
fetches nothing. ``fetch_bundle`` reads one bundle from the configured origin
(``AUTOMATOS_CLOUD_URL``, held to ``fetch.checked_origin``'s rule) and ``catalog_paths.bundle_path`` of the
revision, through W1's guarded GET (``fetch.fetch_bytes``: no redirect, identity encoding, no Authorization,
Cookie or X-Workspace-ID) capped at ``AUTOMATOS_CLOUD_BUNDLE_MAX_KB``. The body's sha256 must be the
revision; it must be UTF-8 JSON, a bundle ``validate_bundle`` passes, in its canonical form (the same bytes
on both sides, K8), and of the entry's type, externalId and version. ``fetch_bundles`` does that for each
entry: one that fails is refused, and the next is fetched.

Every refusal is a plain sentence of this module's own, W1's fetch's or the validator's, none of which
echoes what it received, logged with the type and, once it passed its pattern, the externalId. A leaf for
W1's reader: it imports W1's ``fetch.py`` and ``urls.py``, never ``feed.py`` or ``pull.py`` (``feed.py``
imports it).
"""
from __future__ import annotations

import json
import logging
import re
from functools import reduce
from typing import Any, Dict, FrozenSet, Iterable, Mapping, NamedTuple, Optional, Tuple

from sqlalchemy.orm import Session

from config import config
from core.models.marketplace_cloud import MarketplaceCloudItem
from services.marketplace_bundles.canonical import canonical_bytes
from services.marketplace_bundles.catalog_paths import bundle_path
from services.marketplace_bundles.errors import BundleError
from services.marketplace_bundles.resolve import NOTHING_AWAITED, Awaited, awaited, held_now
from services.marketplace_bundles.schema import (
    BUNDLE_TYPES, CATEGORY_MAX, DESCRIPTION_MAX, EXTERNAL_ID_LENGTH, EXTERNAL_ID_PATTERN, ICON_MAX, ICON_PATTERN,
    NAME_MAX, PUBLISHER_NAME_MAX, REVISION_PATTERN, TAG_MAX, TAGS_MAX, VERSION_MAX, VERSION_PATTERN, choice, listing,
    obj, text,
)
from services.marketplace_bundles.validate import check_value, validate_bundle
from services.marketplace_cloud.fetch import ACCEPT_JSON, ClientFactory, CloudFetchFailed, checked_origin, fetch_bytes
from services.marketplace_cloud.urls import configured_url
from services.themes.contract import KB
from services.themes.digest import sha256_hex

logger = logging.getLogger(__name__)

ENTRY_TRUST = "reviewed"
ENTRY_WHERE = "entry"  # how a refusal names the entry's fields
REVISION_LENGTH = 64
# The keys of an entry this install reads (K11's, less ``bundle``, which is never read: the URL is derived).
ENTRY_SPEC = obj({
    "type": choice(BUNDLE_TYPES),
    "externalId": text(EXTERNAL_ID_LENGTH, pattern=EXTERNAL_ID_PATTERN),
    "version": text(VERSION_MAX, pattern=VERSION_PATTERN),
    "revision": text(REVISION_LENGTH, pattern=REVISION_PATTERN),
    "name": text(NAME_MAX, min_len=1),
    "description": text(DESCRIPTION_MAX),
    "category": text(CATEGORY_MAX),
    "icon": text(ICON_MAX, pattern=ICON_PATTERN),
    "tags": listing(text(TAG_MAX, min_len=1), TAGS_MAX),
    "publisher": obj({"name": text(PUBLISHER_NAME_MAX, min_len=1)}),
    "trust": choice((ENTRY_TRUST,)),
})
ENTRY_KEYS = tuple(ENTRY_SPEC["fields"])

_EXTERNAL_ID = re.compile(EXTERNAL_ID_PATTERN)
_REVISION = re.compile(REVISION_PATTERN)

NOT_A_BUNDLE_TYPE = "Its type isn't one of the item types this install reads."
NOT_REVIEWED = "It isn't marked Reviewed on Automatos Cloud (nothing from Automatos Cloud is Official)."
BAD_ENTRY = "Its catalogue entry isn't one this install reads: {message}"
LISTED_TWICE = "Another entry of the catalogue has its type and externalId."
NOT_FETCHED = "Its bundle couldn't be fetched: {reason}"
NOT_THE_REVISION = "Its bundle isn't the one the catalogue lists (its sha256 isn't the revision)."
NOT_JSON = "Its bundle isn't UTF-8 JSON."
BAD_BUNDLE = "Its bundle isn't one this install reads: {message}"
NOT_CANONICAL = "Its bundle isn't in the canonical form Automatos Cloud serves."
OTHER_ITEM = "Its bundle is for another item or version than its catalogue entry."
NO_ID = "-"


class EntryRefused(ValueError):
    """One item entry or bundle was refused; the message is plain and echoes nothing it held."""


class CatalogEntry(NamedTuple):
    """A catalogue entry that passed: what to fetch, and what the fetched bundle must say."""

    type: str
    external_id: str
    version: str
    revision: str
    name: str


class RefusedEntry(NamedTuple):
    """An item refused, by its type (for the Sync's counts), and why."""

    type: str
    reason: str


class Entries(NamedTuple):
    """The catalogue's item entries: those that passed, in its order, and those refused."""

    listed: Tuple[CatalogEntry, ...] = ()
    refused: Tuple[RefusedEntry, ...] = ()

    def read(self, item: Any) -> "Entries":
        """These entries with ``item`` read: listed when it passes, else refused and logged. An item not of a
        bundle type is not an entry: the answer is these entries, unchanged."""
        if not is_bundle_item(item):
            return self
        try:
            entry = parse_item_entry(item)
            if any((held.type, held.external_id) == (entry.type, entry.external_id) for held in self.listed):
                raise EntryRefused(LISTED_TWICE)
        except EntryRefused as refusal:
            _log_refused(item["type"], item.get("externalId"), refusal)
            return self._replace(refused=(*self.refused, RefusedEntry(item["type"], str(refusal))))
        return self._replace(listed=(*self.listed, entry))


NO_ENTRIES = Entries()


class FetchedBundle(NamedTuple):
    """A bundle fetched for ``entry``, every check passed."""

    entry: CatalogEntry
    bundle: Dict[str, Any]


class Fetched(NamedTuple):
    """What ``fetch_bundles`` fetched, in the entries' order, and the entries it refused."""

    bundles: Tuple[FetchedBundle, ...]
    refused: Tuple[RefusedEntry, ...]


def fetch_max_bytes() -> int:
    """The largest bundle a local install reads (``AUTOMATOS_CLOUD_BUNDLE_MAX_KB``)."""
    return int(config.AUTOMATOS_CLOUD_BUNDLE_MAX_KB) * KB


def is_bundle_item(item: Any) -> bool:
    """True for a catalogue item whose type is one of the bundle types (``feed.parse_catalog``'s dispatch)."""
    return isinstance(item, dict) and isinstance(item.get("type"), str) and item["type"] in BUNDLE_TYPES


def _log_refused(item_type: str, external_id: Any, refusal: EntryRefused) -> None:
    """A refusal in the log: its type, its externalId once that passed its pattern, and the plain reason."""
    shown = external_id if isinstance(external_id, str) and _EXTERNAL_ID.fullmatch(external_id) else NO_ID
    logger.warning("Automatos Cloud catalogue: %s %s refused: %s", item_type, shown, refusal)


# --------------------------------------------------------------------------- an entry


def parse_item_entry(item: Mapping[str, Any]) -> CatalogEntry:
    """One item entry, checked as K13 says; ``EntryRefused`` with a plain reason on the first check that fails."""
    if not is_bundle_item(item):
        raise EntryRefused(NOT_A_BUNDLE_TYPE)
    if item.get("trust") != ENTRY_TRUST:
        raise EntryRefused(NOT_REVIEWED)
    fields = {key: item[key] for key in ENTRY_KEYS if key in item}
    try:
        check_value(fields, ENTRY_SPEC, ENTRY_WHERE)
    except BundleError as error:
        raise EntryRefused(BAD_ENTRY.format(message=error.message)) from error
    return CatalogEntry(fields["type"], fields["externalId"], fields["version"], fields["revision"], fields["name"])


def changed_entries(db: Session, entries: Iterable[CatalogEntry]) -> Tuple[CatalogEntry, ...]:
    """``entries`` less each one a ``marketplace_cloud_items`` mapping holds at the same revision (unchanged)."""
    listed = tuple(entries)
    held = _held(db, listed)
    return tuple(entry for entry in listed if not _at_revision(held, entry))


def rechecked_entries(
    db: Session, entries: Iterable[CatalogEntry], arriving: FrozenSet[str] = frozenset(),
) -> Tuple[CatalogEntry, ...]:
    """Of ``entries``, each a mapping holds at the same revision whose ``unresolved`` awaits something that has
    arrived: a plugin or a skill this install now holds, or a Cloud agent it holds or this Sync has fetched
    (``arriving``, those agent bundles' externalIds). Their bundles are fetched again (P270W2-RVW-1); only
    reads."""
    listed = tuple(entries)
    held = _held(db, listed)
    waiting = {entry: awaited(held[_key(entry)].unresolved) for entry in listed if _at_revision(held, entry)}
    waiting = {entry: wanted for entry, wanted in waiting.items() if wanted.any()}
    if not waiting:
        return ()
    found = held_now(db, reduce(Awaited.plus, waiting.values(), NOTHING_AWAITED), arriving)
    return tuple(entry for entry, wanted in waiting.items() if wanted.meets(found))


def _key(entry: CatalogEntry) -> Tuple[str, str]:
    return entry.type, entry.external_id


def _held(db: Session, listed: Tuple[CatalogEntry, ...]) -> Dict[Tuple[str, str], Any]:
    """The mappings of ``listed`` by (type, externalId): each one's revision and ``unresolved``."""
    if not listed:
        return {}
    rows = (
        db.query(
            MarketplaceCloudItem.item_type, MarketplaceCloudItem.external_id, MarketplaceCloudItem.revision,
            MarketplaceCloudItem.unresolved,
        )
        .filter(MarketplaceCloudItem.external_id.in_(sorted({entry.external_id for entry in listed})))
        .all()
    )
    return {(row.item_type, row.external_id): row for row in rows}


def _at_revision(held: Mapping[Tuple[str, str], Any], entry: CatalogEntry) -> bool:
    """Whether a mapping holds ``entry``'s item at its revision."""
    row = held.get(_key(entry))
    return row is not None and row.revision == entry.revision


# --------------------------------------------------------------------------- a bundle


def bundle_url(origin: str, revision: str) -> str:
    """The bundle ``revision`` on ``origin``, derived from the setting and the revision (never a catalogue's URL)."""
    return f"{origin}{bundle_path(revision)}"


def _bundle_of(data: bytes) -> Dict[str, Any]:
    """The bundle these bytes hold: UTF-8 JSON, valid, and in its canonical form."""
    try:
        document = json.loads(data.decode("utf-8"))
    except (ValueError, RecursionError) as bad:  # UnicodeDecodeError is a ValueError
        raise EntryRefused(NOT_JSON) from bad
    try:
        bundle = validate_bundle(document)
        canonical = canonical_bytes(bundle)
    except BundleError as error:
        raise EntryRefused(BAD_BUNDLE.format(message=error.message)) from error
    if canonical != data:
        raise EntryRefused(NOT_CANONICAL)
    return bundle


def _origin() -> str:
    """The configured Automatos Cloud origin, else ``EntryRefused`` saying why the setting isn't one."""
    try:
        return checked_origin(configured_url())
    except CloudFetchFailed as failed:
        raise EntryRefused(NOT_FETCHED.format(reason=failed)) from failed


def _fetched_from(origin: str, entry: CatalogEntry, client_factory: Optional[ClientFactory]) -> Dict[str, Any]:
    """The bundle ``entry`` lists, fetched from ``origin`` at the path its revision derives, and checked."""
    if not isinstance(entry.revision, str) or _REVISION.fullmatch(entry.revision) is None:
        raise EntryRefused(BAD_ENTRY.format(message=f"{ENTRY_WHERE}.revision is not in the form the bundle format allows."))
    try:
        data = fetch_bytes(bundle_url(origin, entry.revision), fetch_max_bytes(), ACCEPT_JSON, client_factory)
    except CloudFetchFailed as failed:
        raise EntryRefused(NOT_FETCHED.format(reason=failed)) from failed
    if sha256_hex(data) != entry.revision:
        raise EntryRefused(NOT_THE_REVISION)
    bundle = _bundle_of(data)
    if (bundle["type"], bundle["externalId"], bundle["version"]) != (entry.type, entry.external_id, entry.version):
        raise EntryRefused(OTHER_ITEM)
    return bundle


def fetch_bundle(entry: CatalogEntry, client_factory: Optional[ClientFactory] = None) -> Dict[str, Any]:
    """The bundle ``entry`` lists, fetched from the derived URL and checked as K13 says; else ``EntryRefused``."""
    return _fetched_from(_origin(), entry, client_factory)


def fetch_bundles(entries: Iterable[CatalogEntry], client_factory: Optional[ClientFactory] = None) -> Fetched:
    """Each entry's bundle, fetched in turn; one that fails is refused and logged, and the rest are still fetched.

    The setting is read once: while it isn't an origin, every entry is refused for that one reason, logged
    once, and nothing is requested.
    """
    listed = tuple(entries)
    try:
        origin = _origin()
    except EntryRefused as refusal:
        if listed:
            logger.warning("Automatos Cloud: no bundle fetched: %s", refusal)
        return Fetched((), tuple(RefusedEntry(entry.type, str(refusal)) for entry in listed))
    bundles, refused = [], []
    for entry in listed:
        try:
            bundles.append(FetchedBundle(entry, _fetched_from(origin, entry, client_factory)))
        except EntryRefused as refusal:
            _log_refused(entry.type, entry.external_id, refusal)
            refused.append(RefusedEntry(entry.type, str(refusal)))
    return Fetched(tuple(bundles), tuple(refused))


__all__ = [
    "BUNDLE_TYPES", "ENTRY_KEYS", "ENTRY_TRUST", "NO_ENTRIES", "CatalogEntry", "Entries", "EntryRefused", "Fetched",
    "FetchedBundle", "RefusedEntry", "bundle_url", "changed_entries", "fetch_bundle", "fetch_bundles",
    "fetch_max_bytes", "is_bundle_item", "parse_item_entry", "rechecked_entries",
]
