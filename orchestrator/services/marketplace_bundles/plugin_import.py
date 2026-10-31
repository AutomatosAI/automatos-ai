"""PRD-270 W2 US-270-15a (§5.6, K15) — locally: Automatos Cloud's plugins, checked like an upload and stored.

The Sync's plugin pass (US-270-13d runs it after W1's theme pass and before the item pass, so an agent's
plugins and their skills exist first). ``import_plugins`` reads the catalogue's plugin entries: one a
``marketplace_cloud_items`` mapping holds at the same revision is unchanged. Every new or changed one has
its bundle fetched and its slug checked (steps 1 and 2) BEFORE the cap (P270W2-RVW-2), on every Sync, as the
item pass reads every new or changed item's bundle (each capped at ``AUTOMATOS_CLOUD_BUNDLE_MAX_KB``, the
catalogue at W1's ``feed.CATALOG_MAX_ITEMS``): a plugin they refuse is counted refused and takes none of the
``AUTOMATOS_CLOUD_PLUGINS_PER_SYNC`` places, which bound package downloads only. Of those that pass, the
first by name download their packages now and the rest wait for the next Sync (``plugins.waiting`` counts
only them); a plugin whose package was refused in this process at its revision comes after every one not yet
tried (``refused_before``), so a package refused once never holds a place again ahead of a plugin not yet
tried. A cap of 0 downloads none and requests nothing: every new or changed plugin waits. Each is imported
on its own, in K15's order, and nothing is stored until every check has passed:

1. its bundle (``items_feed.fetch_bundles``: the derived URL, the revision, the strict validator);
2. its slug: a plugin row without this item's mapping that holds it refuses the item (that row is never
   touched), and so does a changed revision at the version the mapped row already holds (its files would
   change under one version);
3. its package, from the DERIVED URL (the configured origin and ``catalog_paths.plugin_package_path`` of the
   bundle's slug and version, never a URL a catalogue or a bundle carries) through W1's ``fetch.fetch_bytes``,
   capped at ``PLUGIN_MAX_UPLOAD_SIZE_MB``;
4. the upload's own zip rules (``plugin_upload_service.py``, its constants named the same): at most
   ``MAX_FILE_COUNT`` files, each at most ``MAX_FILE_UNCOMPRESSED_SIZE``, at most the smaller of
   ``MAX_TOTAL_UNCOMPRESSED_SIZE`` and the S3 cap in all, compressed at most ``MAX_COMPRESSION_RATIO`` times;
   no path that ``plugin_files.is_safe_path`` refuses (absolute, '..', not normal, a backslash or a control
   character) and no file named twice. Every rule reads the zip's directory, before any file is read;
5. manifest.json at its root, naming the bundle's slug and version (``plugin_files.checked_files``);
6. this install's STATIC scan (``plugin_security_scanner.static_scan``: patterns only) over its text files,
   ``SCAN_CHUNK_LINES`` lines at a time: a critical finding refuses it, and so do a line over
   ``MAX_SCANNED_LINE`` characters (the scanner's ``.+`` patterns backtrack over a long line) and more than
   ``MAX_SCAN_FINDINGS`` findings (so memory holds a bounded report). The LLM scan never runs here: a Sync
   makes no model call;
7. the rows read again: still what step 2 read (a writer that came meanwhile refuses the item).

Then ``MarketplaceS3Service.extract_plugin`` stores the files under ``plugins/<slug>/<version>/`` (MinIO
locally, the prefix an upload writes), the prefix must then hold exactly the package's files (an object left
there before is never materialised unscanned), and ONE transaction writes the ``PluginSecurityScan`` row
(Automatos Cloud's report, this install's static result, scanned_by 'automatos-cloud'), the
``MarketplacePlugin`` row, the mapping and a ``PluginSyncHistory`` row ('cloud-sync'), and commits. A new
plugin arrives approved and active (approved_by 'automatos-cloud'): the local edition has no plugin review
page, Automatos Cloud's review is the approval, and each workspace still chooses whether to enable it. An
update moves the same row to the new version, so the workspaces that enabled it (``WorkspaceEnabledPlugin``
holds only the two ids) run the new version; it keeps the row's approval and is_active as they are (a local
deactivation stands), and once the row has moved the old version's files are removed. Any failure from the
store to the commit removes what was stored (``delete_plugin``) and writes no row, unless a committed row
names those files (another writer's, or a commit that reached the database though it answered an error).
After the commit, ``SkillMaterializer.materialize_plugin`` makes the plugin's SKILL.md files global skills
(it commits itself, so each plugin commits on its own), for a row that is approved and active only.

``run_async`` runs one coroutine to its end and answers its result: the static scan, the S3 calls and the
materialiser are async. It must run them OFF the server's event loop (``asyncio.run`` in the plain-def
route's worker thread, as the tests do): the scan is CPU work and the materialiser queries the database
synchronously. The pass holds no transaction open across a network or storage write: the session it is
handed carries nothing pending (the theme pass committed), and every read ends in a rollback.

Every refusal is a plain sentence of this module's own, W1's fetch's or ``plugin_files``', logged with the
item's externalId (it passed its pattern), never what the catalogue, the bundle or the package held. A
failure of one plugin refuses that plugin only; reading the mappings at the start is the one database read
whose failure the caller records for the whole pass.
"""
from __future__ import annotations

import io
import logging
import struct
import zipfile
import zlib
from dataclasses import replace
from datetime import datetime, timezone
from typing import (
    Any, Awaitable, Callable, Dict, FrozenSet, Iterable, Iterator, List, Mapping, NamedTuple, Optional, Tuple,
)
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from config import config
from core.models.marketplace_cloud import MarketplaceCloudItem
from core.models.marketplace_plugins import MarketplacePlugin, PluginSecurityScan, PluginSyncHistory
from core.services.marketplace_s3 import MarketplaceS3Service
from core.services.plugin_security_scanner import StaticFinding, StaticScanResult, static_scan
from core.services.plugin_upload_service import _auto_categorise
from core.services.skill_materializer import SkillMaterializer
from services.marketplace_bundles.catalog_paths import plugin_package_path
from services.marketplace_bundles.items_feed import CatalogEntry, FetchedBundle, changed_entries, fetch_bundles
from services.marketplace_bundles.plugin_files import (
    MB, PluginFilesRefused, StoredPlugin, checked_files, is_safe_path, plugin_files_max_bytes, plugin_prefix,
)
from services.marketplace_cloud.fetch import (
    ACCEPT_PACKAGE, ClientFactory, CloudFetchFailed, checked_origin, fetch_bytes,
)
from services.marketplace_cloud.pull import NEW, REFUSED, UPDATED, SyncCounts
from services.marketplace_cloud.urls import configured_url
from services.themes.bounded_cache import BoundedCache

logger = logging.getLogger(__name__)

ITEM_TYPE = "plugin"
CLOUD_ACTOR = "automatos-cloud"  # scanned_by, approved_by and performed_by
SOURCE_TYPE = "automatos-cloud"
HISTORY_ACTION = "cloud-sync"
HISTORY_DONE = "completed"
SAFE = "safe"
APPROVED = "approved"
CRITICAL = "critical"
SCAN_PASSED, SCAN_FLAGGED = "passed", "flagged"  # static_scan's two statuses
FOLDER_MARK = "/"

# The upload's own zip rules (plugin_upload_service.py's upload_plugin, the same names and values).
MAX_FILE_COUNT = 500
MAX_FILE_UNCOMPRESSED_SIZE = 10 * 1024 * 1024  # 10 MB per file
MAX_TOTAL_UNCOMPRESSED_SIZE = 50 * 1024 * 1024  # 50 MB total
MAX_COMPRESSION_RATIO = 100  # reject suspiciously high ratios (zip bomb)
# What zipfile raises for an archive or a member it can't read (bad CRC, truncated, encrypted, unsupported).
UNREADABLE_ZIP = (
    zipfile.BadZipFile, zipfile.LargeZipFile, NotImplementedError, RuntimeError, EOFError, OSError, ValueError,
    zlib.error, struct.error,
)
# The local static scan's bounds: each re.search runs over one line, holding the interpreter until it ends.
MAX_SCANNED_LINE = 8192
SCAN_CHUNK_LINES = 200
MAX_SCAN_FINDINGS = 200

NO_ORIGIN = "No plugin was imported: {reason}"
NOT_READ = "This install's plugin rows couldn't be read."
SLUG_HELD = "A plugin this install holds already has its slug."
NOT_ITS_SLUG = "The plugin this install holds for it has another slug."
SAME_VERSION = "Its bundle changed but its plugin version didn't, so its files would change under one version."
CHANGED_HERE = "This install's plugin rows changed while it was checked."
PACKAGE_NOT_FETCHED = "Its package couldn't be fetched: {reason}"
NOT_A_ZIP = "Its package isn't a zip archive this install can read."
TOO_MANY_FILES = "Its package holds more than {count} files."
NAMED_TWICE = "Its package names one file twice."
UNSAFE_PATH = "A file in its package has an unsafe path."
FILE_TOO_LARGE = "A file in its package is larger than {mb} MB."
FILES_TOO_LARGE = "Its package's files are larger than {mb} MB in all."
RATIO_TOO_HIGH = "A file in its package is compressed more than {ratio} times (a possible zip bomb)."
NOT_THE_PLUGIN = "Its package isn't the bundle's plugin: {reason}"
LINE_TOO_LONG = (
    "A text file in its package has a line longer than {count} characters, which this install doesn't scan."
)
TOO_MANY_FINDINGS = "This install's static scan found more than {count} issues in its files."
CRITICAL_FINDING = "This install's static scan found a critical issue in its files."
NOT_STORED = "Its files couldn't be stored."
OTHER_FILES_STORED = "Its storage held other files than its package's."
NOT_SAVED = "It couldn't be saved."
ROW_GONE = "The plugin this install held for it is gone or has moved."

RunAsync = Callable[[Awaitable[Any]], Any]
Identity = Tuple[str, str]  # a plugin entry's (externalId, revision)

# Each plugin this process refused once it held a place, at its revision: ordered after the plugins not yet
# tried (P270W2-RVW-2). Kept only while the entry stays new or changed; a restart forgets. W1's one-Sync lock
# serialises the passes that read and replace it.
REFUSED_KEY = "refused"
_refused_here: BoundedCache[FrozenSet[Identity]] = BoundedCache(1)


class PluginRefused(ValueError):
    """One plugin was refused; the message is plain and echoes nothing its bundle or package held."""


class PluginPass(NamedTuple):
    """The plugin pass: its counts (new, updated, unchanged, refused) and how many plugins wait."""

    counts: SyncCounts
    waiting: int


class Seams(NamedTuple):
    """What one pass reads with: the checked origin, the HTTP client factory, the coroutine runner, storage."""

    origin: str
    client_factory: Optional[ClientFactory]
    run_async: RunAsync
    s3: MarketplaceS3Service


class Target(NamedTuple):
    """What a plugin's import writes: a new row, or the mapped row (by id) moved from its version."""

    verdict: str
    plugin_id: Optional[UUID] = None
    mapping_id: Optional[int] = None
    previous_version: Optional[str] = None


class Targeted(NamedTuple):
    """A fetched plugin whose slug check passed, with what its import writes: decided before the cap."""

    fetched: FetchedBundle
    target: Target


class CheckedPlugin(NamedTuple):
    """A plugin whose bundle, slug, package, manifest and static scan all passed: ready to store."""

    fetched: FetchedBundle
    target: Target
    data: bytes  # the package as downloaded: extract_plugin stores it
    paths: Tuple[str, ...]  # its files' paths, as the prefix must then hold them
    scan: StaticScanResult

    @property
    def bundle(self) -> Dict[str, Any]:
        return self.fetched.bundle

    @property
    def slug(self) -> str:
        return self.fetched.bundle[ITEM_TYPE]["slug"]

    @property
    def version(self) -> str:
        return self.fetched.bundle["version"]


def plugins_per_sync() -> int:
    """The new or changed plugins whose packages one Sync downloads (``AUTOMATOS_CLOUD_PLUGINS_PER_SYNC``)."""
    return max(0, int(config.AUTOMATOS_CLOUD_PLUGINS_PER_SYNC))


def identity(entry: CatalogEntry) -> Identity:
    """The entry's (externalId, revision): what a refusal here is remembered by."""
    return entry.external_id, entry.revision


def refused_before() -> FrozenSet[Identity]:
    """The plugins this process refused once they held a place, each at the revision it was refused at."""
    return _refused_here.get(REFUSED_KEY) or frozenset()


def forget_refused() -> None:
    """Forget every refusal this process remembers (tests; a restart does the same)."""
    _refused_here.clear()


def remember_refused(changed: Iterable[CatalogEntry], refused: Iterable[CatalogEntry]) -> None:
    """Remember ``refused`` at their revisions, with each refusal already remembered whose entry is still in
    ``changed`` (one imported since, gone from the catalogue or at a new revision is forgotten)."""
    still = frozenset(identity(entry) for entry in changed)
    _refused_here.put(REFUSED_KEY, (refused_before() & still) | frozenset(identity(entry) for entry in refused))


def place_order(entry: CatalogEntry, refused: FrozenSet[Identity]) -> Tuple[bool, str, str]:
    """Where ``entry`` comes for a place: by name (then externalId), after every other when ``refused`` holds it
    at its revision."""
    return identity(entry) in refused, entry.name.casefold(), entry.external_id


def chosen_entries(
    entries: Iterable[CatalogEntry], top: int, refused: FrozenSet[Identity] = frozenset(),
) -> Tuple[Tuple[CatalogEntry, ...], Tuple[CatalogEntry, ...]]:
    """``entries`` in ``place_order``: the first ``top`` to import now, and the rest, which wait."""
    ordered = tuple(sorted(entries, key=lambda entry: place_order(entry, refused)))
    return ordered[:top], ordered[top:]


def chosen_plugins(
    targeted: Iterable[Targeted], top: int, refused: FrozenSet[Identity],
) -> Tuple[Tuple[Targeted, ...], Tuple[Targeted, ...]]:
    """``targeted`` in ``place_order``: the first ``top`` to download now, and the rest, which wait."""
    ordered = tuple(sorted(targeted, key=lambda one: place_order(one.fetched.entry, refused)))
    return ordered[:top], ordered[top:]


def import_plugins(
    db: Session, entries: Iterable[CatalogEntry], client_factory: Optional[ClientFactory], run_async: RunAsync,
) -> PluginPass:
    """Import the catalogue's new and changed plugins, each checked, stored and committed on its own (K15).

    ``entries`` are the catalogue's listed entries of every type; only plugins are read. Every new or changed
    plugin's bundle and slug are checked before the cap; a refusal counts that plugin refused and the next is
    imported.
    """
    plugins = tuple(entry for entry in entries if entry.type == ITEM_TYPE)
    changed = changed_entries(db, plugins)
    db.rollback()  # the mappings' read ends before any fetch
    counts, top = SyncCounts(unchanged=len(plugins) - len(changed)), plugins_per_sync()
    if not changed or top == 0:  # 0 downloads none: nothing is requested, and every changed plugin waits
        return PluginPass(counts, len(changed))
    try:
        origin = checked_origin(configured_url())
    except CloudFetchFailed as off:
        logger.warning("Automatos Cloud: %s", NO_ORIGIN.format(reason=off))
        return PluginPass(replace(counts, refused=len(changed)), 0)
    fetched = fetch_bundles(changed, client_factory)
    targeted = targeted_plugins(db, fetched.bundles)
    chosen, waiting = chosen_plugins(targeted, top, refused_before())
    seams = Seams(origin, client_factory, run_async, MarketplaceS3Service())
    verdicts = tuple((one.fetched.entry, import_plugin(db, one, seams)) for one in chosen)
    remember_refused(changed, (entry for entry, verdict in verdicts if verdict == REFUSED))
    counts = replace(counts, refused=len(fetched.refused) + len(fetched.bundles) - len(targeted))
    for _, verdict in verdicts:
        counts = counts.plus(verdict)
    logger.info("Automatos Cloud plugins: %s, %d waiting", counts.view(), len(waiting))
    return PluginPass(counts, len(waiting))


def targeted_plugins(db: Session, bundles: Iterable[FetchedBundle]) -> Tuple[Targeted, ...]:
    """Each fetched plugin whose slug check passes (K15's step 2), with its target; one it refuses is logged
    and left out, so it takes no place under the cap."""
    targeted = (_targeted(db, fetched) for fetched in bundles)
    return tuple(one for one in targeted if one is not None)


def _targeted(db: Session, fetched: FetchedBundle) -> Optional[Targeted]:
    try:
        slug, version = fetched.bundle[ITEM_TYPE]["slug"], fetched.bundle["version"]
        return Targeted(fetched, held_target(db, fetched.entry, slug, version))
    except PluginRefused as refused:
        _log_refused(fetched.entry, refused)
    except Exception:  # anything unforeseen refuses this plugin only, as in its checks: the pass goes on
        db.rollback()
        logger.exception("Automatos Cloud: plugin %s couldn't be checked", fetched.entry.external_id)
    return None


def import_plugin(db: Session, one: Targeted, seams: Seams) -> str:
    """One plugin, checked then stored and committed: ``new`` or ``updated``, else ``refused`` (logged)."""
    entry = one.fetched.entry
    try:
        checked = checked_plugin(db, one, seams)
    except PluginRefused as refused:
        _log_refused(entry, refused)
        return REFUSED
    except Exception:  # anything unforeseen refuses this plugin only: nothing is stored yet, the pass goes on
        db.rollback()
        logger.exception("Automatos Cloud: plugin %s couldn't be checked", entry.external_id)
        return REFUSED
    return _stored(db, checked, seams)


def _log_refused(entry: CatalogEntry, refused: PluginRefused) -> None:
    logger.warning("Automatos Cloud: plugin %s refused: %s", entry.external_id, refused)


# --------------------------------------------------------------------------- the checks


def checked_plugin(db: Session, one: Targeted, seams: Seams) -> CheckedPlugin:
    """K15's checks of one fetched bundle after its slug check (``targeted_plugins``), in order, before anything
    is stored; ``PluginRefused`` on the first that fails."""
    fetched, target = one
    bundle = fetched.bundle
    slug, version = bundle[ITEM_TYPE]["slug"], bundle["version"]
    data = _package(seams, slug, version)
    files = package_files(data)
    try:
        checked_files(StoredPlugin(slug, version, plugin_prefix(slug, version)), files)
    except PluginFilesRefused as refused:
        raise PluginRefused(NOT_THE_PLUGIN.format(reason=refused)) from refused
    scan = local_scan(text_files(files), seams.run_async)
    if held_target(db, fetched.entry, slug, version) != target:
        raise PluginRefused(CHANGED_HERE)
    return CheckedPlugin(fetched, target, data, tuple(sorted(files)), scan)


def held_target(db: Session, entry: CatalogEntry, slug: str, version: str) -> Target:
    """What importing ``entry`` at ``slug`` and ``version`` writes, from the rows held now; the read ends here."""
    try:
        mapping = (
            db.query(MarketplaceCloudItem.id, MarketplaceCloudItem.plugin_id)
            .filter(MarketplaceCloudItem.item_type == ITEM_TYPE)
            .filter(MarketplaceCloudItem.external_id == entry.external_id)
            .first()
        )
        row = db.query(MarketplacePlugin.id, MarketplacePlugin.version).filter(MarketplacePlugin.slug == slug).first()
    except SQLAlchemyError as failed:
        logger.exception("Automatos Cloud: plugin %s's rows couldn't be read", entry.external_id)
        raise PluginRefused(NOT_READ) from failed
    finally:
        db.rollback()
    return target_of(mapping, row, version)


def target_of(mapping: Any, row: Any, version: str) -> Target:
    """New when neither a mapping nor a row with the slug is held; the mapped row's update when that row
    holds the slug at another version; else ``PluginRefused``."""
    if mapping is None:
        if row is not None:
            raise PluginRefused(SLUG_HELD)
        return Target(NEW)
    if row is None or row.id != mapping.plugin_id:
        raise PluginRefused(NOT_ITS_SLUG)
    if row.version == version:
        raise PluginRefused(SAME_VERSION)
    return Target(UPDATED, row.id, mapping.id, row.version)


def _package(seams: Seams, slug: str, version: str) -> bytes:
    """The plugin's package from the derived URL, capped at ``PLUGIN_MAX_UPLOAD_SIZE_MB``."""
    url = f"{seams.origin}{plugin_package_path(slug, version)}"
    try:
        return fetch_bytes(url, plugin_files_max_bytes(), ACCEPT_PACKAGE, seams.client_factory)
    except CloudFetchFailed as failed:
        raise PluginRefused(PACKAGE_NOT_FETCHED.format(reason=failed)) from failed


def package_files(data: bytes) -> Dict[str, bytes]:
    """The files of a package zip by path, once its directory passes the upload's zip rules; else
    ``PluginRefused``. Folder entries are left out, as the upload and ``extract_plugin`` leave them."""
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except UNREADABLE_ZIP as unreadable:
        raise PluginRefused(NOT_A_ZIP) from unreadable
    with archive:
        members = checked_members(archive.infolist())
        return {info.filename: _member_bytes(archive, info) for info in members}


def checked_members(infos: Iterable[zipfile.ZipInfo]) -> Tuple[zipfile.ZipInfo, ...]:
    """The zip's file entries, when the count, the names, each entry and the total pass; else ``PluginRefused``."""
    members = tuple(info for info in infos if not info.filename.endswith(FOLDER_MARK))
    if len(members) > MAX_FILE_COUNT:
        raise PluginRefused(TOO_MANY_FILES.format(count=MAX_FILE_COUNT))
    if len({info.filename for info in members}) != len(members):
        raise PluginRefused(NAMED_TWICE)
    for info in members:
        check_member(info)
    cap = min(MAX_TOTAL_UNCOMPRESSED_SIZE, plugin_files_max_bytes())
    if sum(info.file_size for info in members) > cap:
        raise PluginRefused(FILES_TOO_LARGE.format(mb=cap // MB))
    return members


def check_member(info: zipfile.ZipInfo) -> None:
    """One file entry: a safe path, at most the per-file size, compressed at most the ratio (an entry that
    declares a size but no compressed bytes is over it)."""
    if not is_safe_path(info.filename):
        raise PluginRefused(UNSAFE_PATH)
    if info.file_size > MAX_FILE_UNCOMPRESSED_SIZE:
        raise PluginRefused(FILE_TOO_LARGE.format(mb=MAX_FILE_UNCOMPRESSED_SIZE // MB))
    if info.file_size > info.compress_size * MAX_COMPRESSION_RATIO:
        raise PluginRefused(RATIO_TOO_HIGH.format(ratio=MAX_COMPRESSION_RATIO))


def _member_bytes(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    """A checked entry's bytes (zipfile stops at its declared size and checks its CRC)."""
    try:
        return archive.read(info)
    except UNREADABLE_ZIP as unreadable:
        raise PluginRefused(NOT_A_ZIP) from unreadable


def text_files(files: Mapping[str, bytes]) -> Dict[str, str]:
    """The files that are UTF-8 text, as the static scan reads them (the upload skips binary files too)."""
    return {path: text for path, text in ((path, _text_of(data)) for path, data in files.items()) if text is not None}


def _text_of(data: bytes) -> Optional[str]:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return None


def local_scan(texts: Mapping[str, str], run_async: RunAsync) -> StaticScanResult:
    """This install's static scan of ``texts``, ``SCAN_CHUNK_LINES`` lines per ``static_scan`` call, each
    finding at its line in the file; ``PluginRefused`` for a line over ``MAX_SCANNED_LINE``, a critical
    finding or more than ``MAX_SCAN_FINDINGS`` findings."""
    findings: List[StaticFinding] = []
    for path in sorted(texts):
        for start, chunk in _chunks(texts[path]):
            found = run_async(static_scan({path: chunk})).findings
            if any(finding.severity == CRITICAL for finding in found):
                raise PluginRefused(CRITICAL_FINDING)
            findings.extend(finding.model_copy(update={"line": finding.line + start}) for finding in found)
            if len(findings) > MAX_SCAN_FINDINGS:
                raise PluginRefused(TOO_MANY_FINDINGS.format(count=MAX_SCAN_FINDINGS))
    return StaticScanResult(status=SCAN_FLAGGED if findings else SCAN_PASSED, findings=findings)


def _chunks(text: str) -> Iterator[Tuple[int, str]]:
    """``text`` in runs of ``SCAN_CHUNK_LINES`` lines, each with the number of lines before it."""
    lines = text.split("\n")
    if any(len(line) > MAX_SCANNED_LINE for line in lines):
        raise PluginRefused(LINE_TOO_LONG.format(count=MAX_SCANNED_LINE))
    for start in range(0, len(lines), SCAN_CHUNK_LINES):
        yield start, "\n".join(lines[start:start + SCAN_CHUNK_LINES])


# --------------------------------------------------------------------------- storing and writing


def _stored(db: Session, checked: CheckedPlugin, seams: Seams) -> str:
    """Store the files, write and commit the rows, then materialise the skills and remove the old version's
    files; refused, with what was stored removed and no row written, when the store or the commit fails."""
    committed = False
    try:
        _store_files(checked, seams)
        plugin_id = committed_rows(db, checked, seams.origin)
        committed = True
    except PluginRefused as refused:
        _log_refused(checked.fetched.entry, refused)
        return REFUSED
    finally:
        if not committed:
            _discard(db, seams, checked.slug, checked.version)
    _materialize(db, plugin_id, seams.run_async)
    if checked.target.previous_version is not None:
        _discard(db, seams, checked.slug, checked.target.previous_version)
    return checked.target.verdict


def _store_files(checked: CheckedPlugin, seams: Seams) -> None:
    """Store the package's files under its prefix; the prefix must then hold exactly those files."""
    prefix = plugin_prefix(checked.slug, checked.version)
    try:
        seams.run_async(seams.s3.extract_plugin(checked.slug, checked.version, checked.data))
        stored = seams.run_async(seams.s3.list_plugin_files(checked.slug, checked.version))
    except Exception as failed:  # storage raises many types; logged, refused, and what it stored is removed
        logger.exception("Automatos Cloud: plugin %s %s couldn't be stored", checked.slug, checked.version)
        raise PluginRefused(NOT_STORED) from failed
    if sorted(stored) != [prefix + path for path in checked.paths]:
        raise PluginRefused(OTHER_FILES_STORED)


def _discard(db: Session, seams: Seams, slug: str, version: str) -> None:
    """Remove the stored files of ``slug`` at ``version``, unless a committed plugin row names them."""
    if _files_named(db, slug, version):
        logger.warning("Automatos Cloud: plugin %s %s's files are kept: a plugin row names them", slug, version)
        return
    try:
        seams.run_async(seams.s3.delete_plugin(slug, version))
    except Exception:  # nothing else to undo: logged for the operator (no row names the files)
        logger.exception("Automatos Cloud: plugin %s %s's stored files couldn't be removed", slug, version)


def _files_named(db: Session, slug: str, version: str) -> bool:
    """Whether a committed row names ``slug``'s files at ``version``; True when that can't be read (kept, not
    deleted, while unsure)."""
    try:
        named = (
            db.query(MarketplacePlugin.id)
            .filter(MarketplacePlugin.slug == slug, MarketplacePlugin.s3_path == plugin_prefix(slug, version))
            .first()
        )
    except SQLAlchemyError:
        logger.exception("Automatos Cloud: plugin %s %s's files couldn't be checked, so they're kept", slug, version)
        return True
    finally:
        db.rollback()
    return named is not None


def committed_rows(db: Session, checked: CheckedPlugin, origin: str) -> UUID:
    """Write the scan, plugin, mapping and history rows in one transaction and commit; the plugin's id.
    ``PluginRefused`` (rolled back, logged) when the database refuses them or the mapped row is gone."""
    try:
        plugin_id = _written(db, checked, origin)
        db.commit()
    except PluginRefused:
        db.rollback()
        raise
    except Exception as failed:  # the database's errors and anything unforeseen: rolled back, logged, refused
        db.rollback()
        logger.exception("Automatos Cloud: plugin %s %s couldn't be saved", checked.slug, checked.version)
        raise PluginRefused(NOT_SAVED) from failed
    return plugin_id


def _written(db: Session, checked: CheckedPlugin, origin: str) -> UUID:
    scan = scan_row(checked)
    db.add(scan)
    db.flush()
    plugin = _plugin_row(db, checked, row_fields(checked, origin, scan.id))
    _map(db, checked, plugin.id)
    db.add(history_row(checked, plugin.id, scan.id, origin))
    db.flush()
    return plugin.id


def scan_row(checked: CheckedPlugin) -> PluginSecurityScan:
    """The scan row: Automatos Cloud's report (its verdict, LLM status, risk score, findings and summary) and
    THIS install's static result."""
    report = checked.bundle[ITEM_TYPE]["scan"]
    findings = checked.scan.findings
    return PluginSecurityScan(
        plugin_slug=checked.slug, plugin_version=checked.version,
        static_scan_status=checked.scan.status, static_findings=[finding.model_dump() for finding in findings],
        blocked_patterns_found=sorted({finding.pattern for finding in findings}),
        llm_scan_status=report["llmStatus"], llm_risk_score=report["riskScore"],
        llm_findings=[dict(finding) for finding in report["findings"]], llm_summary=report["summary"],
        llm_model_used=None, llm_tokens_used=None, overall_verdict=SAFE, scanned_by=CLOUD_ACTOR,
    )


def row_fields(checked: CheckedPlugin, origin: str, scan_id: UUID) -> Dict[str, Any]:
    """The plugin row's columns a new import sets and an update moves."""
    bundle, plugin = checked.bundle, checked.bundle[ITEM_TYPE]
    contents = plugin["contents"]
    return {
        "name": bundle["name"], "version": checked.version, "s3_path": plugin_prefix(checked.slug, checked.version),
        "description": bundle["description"], "long_description": plugin["longDescription"],
        "tags": list(bundle["tags"]), "skills_count": contents["skills"], "commands_count": contents["commands"],
        "agents_count": contents["agents"], "hooks_count": contents["hooks"], "source_type": SOURCE_TYPE,
        "source_url": origin, "original_author": plugin["author"], "license": plugin["license"],
        "recommended_models": list(plugin["recommendedModels"]), "token_estimate": plugin["tokenEstimate"],
        "security_scan_id": scan_id, "security_status": SAFE,
    }


def _now() -> datetime:
    """Now, as the plugin tables' naive UTC columns hold it."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _plugin_row(db: Session, checked: CheckedPlugin, fields: Dict[str, Any]) -> MarketplacePlugin:
    """A new row, approved and active on arrival; or the mapped row, moved to the new version."""
    category_id = _auto_categorise(checked.slug, fields["name"], fields["tags"], fields["description"], db)
    if checked.target.verdict == NEW:
        plugin = MarketplacePlugin(
            slug=checked.slug, **fields, category_id=category_id, approval_status=APPROVED,
            approved_by=CLOUD_ACTOR, approved_at=_now(), is_active=True,
        )
        db.add(plugin)
        db.flush()
        return plugin
    plugin = db.get(MarketplacePlugin, checked.target.plugin_id)
    if plugin is None or plugin.slug != checked.slug or plugin.version != checked.target.previous_version:
        raise PluginRefused(ROW_GONE)
    for column, value in {**fields, "category_id": plugin.category_id or category_id}.items():
        setattr(plugin, column, value)
    db.flush()
    return plugin


def _map(db: Session, checked: CheckedPlugin, plugin_id: UUID) -> None:
    """The item's mapping: added for a new plugin, moved to the new version and revision for an update."""
    entry = checked.fetched.entry
    if checked.target.mapping_id is None:
        db.add(MarketplaceCloudItem(
            item_type=ITEM_TYPE, external_id=entry.external_id, version=checked.version, revision=entry.revision,
            plugin_id=plugin_id, unresolved=[],
        ))
        return
    mapping = db.get(MarketplaceCloudItem, checked.target.mapping_id)
    if mapping is None or mapping.plugin_id != plugin_id:
        raise PluginRefused(ROW_GONE)
    mapping.version, mapping.revision = checked.version, entry.revision


def history_row(checked: CheckedPlugin, plugin_id: UUID, scan_id: UUID, origin: str) -> PluginSyncHistory:
    """The plugin's 'cloud-sync' history row: what arrived, from which version, and its scan."""
    contents = checked.bundle[ITEM_TYPE]["contents"]
    return PluginSyncHistory(
        action=HISTORY_ACTION, plugin_id=plugin_id, plugin_slug=checked.slug, status=HISTORY_DONE,
        completed_at=_now(), performed_by=CLOUD_ACTOR,
        details={
            "outcome": checked.target.verdict, "version": checked.version,
            "previous_version": checked.target.previous_version, "external_id": checked.fetched.entry.external_id,
            "revision": checked.fetched.entry.revision, "source_type": SOURCE_TYPE, "source_url": origin,
            "security_status": SAFE, "scan_id": str(scan_id), "files_count": len(checked.paths),
            "skills_count": contents["skills"], "commands_count": contents["commands"],
        },
    )


def _materialize(db: Session, plugin_id: UUID, run_async: RunAsync) -> None:
    """Make the committed plugin's SKILL.md files global skills (the materialiser commits) when its row is
    approved and active here (an update keeps a local deactivation, and its skills stay as they are); a
    failure is logged and leaves the plugin as committed. The transaction ends here either way."""
    try:
        plugin = db.get(MarketplacePlugin, plugin_id)
        db.refresh(plugin)  # loaded here, so the coroutine reads no expired column
        if plugin.approval_status != APPROVED or not plugin.is_active:
            logger.info("Automatos Cloud: plugin %s isn't approved and active here: no skill is made", plugin_id)
            return
        run_async(SkillMaterializer(db).materialize_plugin(plugin))
    except Exception:  # the plugin stands; its skills are logged as not made
        logger.exception("Automatos Cloud: plugin %s's skills couldn't be materialised", plugin_id)
    finally:
        db.rollback()


__all__ = [
    "CLOUD_ACTOR", "HISTORY_ACTION", "MAX_COMPRESSION_RATIO", "MAX_FILE_COUNT", "MAX_FILE_UNCOMPRESSED_SIZE",
    "MAX_SCAN_FINDINGS", "MAX_SCANNED_LINE", "MAX_TOTAL_UNCOMPRESSED_SIZE", "SCAN_CHUNK_LINES", "SOURCE_TYPE",
    "CheckedPlugin", "Identity", "PluginPass", "PluginRefused", "RunAsync", "Seams", "Target", "Targeted",
    "check_member", "checked_members", "checked_plugin", "chosen_entries", "chosen_plugins", "committed_rows",
    "forget_refused", "held_target", "history_row", "identity", "import_plugin", "import_plugins", "local_scan",
    "package_files", "place_order", "plugins_per_sync", "refused_before", "remember_refused", "row_fields",
    "scan_row", "target_of", "targeted_plugins", "text_files",
]
