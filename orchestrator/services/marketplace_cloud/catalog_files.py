"""PRD-270 US-270-02 (§5.2, K9) — what the catalogue's public files route serves: a package and its screenshots.

``GET /api/marketplace/catalog/files/{digest}/{path}`` (``api/marketplace_catalog.py``) serves a file of a
version only when the version is one the catalogue lists (``catalog.py``'s rule): an approved, reviewed
Marketplace version that no ``disable`` covers, this edition's own row (``source`` NULL), signed by an
active Automatos key, while ``THEMES_MARKETPLACE`` is on. Of such a version it serves two things:

* ``package.autotheme``: the version rebuilt from its stored files (``review_package.rebuilt_package``).
  Core stores no zip, and the digest depends on the files alone, so the rebuilt package reads
  (``package.read_theme_package``) to the row's digest and file hashes, which its stored signature
  verifies: a local install accepts it. A stored file that is gone or changed, or a file list that
  doesn't add up to the digest, serves nothing (logged).
* a screenshot theme.json lists under ``previews`` or ``phonePreviews`` that the file manifest holds
  (``listings.preview_paths``), re-hashed on every read with its listed type (``serve.verified_file``).
  Its path must also be one theme.json's screenshot patterns allow (``manifest.PREVIEW_PATTERN``,
  ``PHONE_PREVIEW_PATTERN``), so a stored manifest that names any other file serves nothing.

Everything else is :class:`CatalogFileMissing` (the route's 404): README.md, theme.json, the entry, a
source file, a version the catalogue doesn't list, a private upload's digest, a pending version's.
A keys file that doesn't load raises ``SigningKeysError``, and storage that fails raises
:class:`CatalogFilesUnavailable` (both the route's 503).

The route asks in two steps: ``served_version`` (the database only) and then ``read_catalog_file``
(storage). Between them it compares ``If-None-Match`` with ``listed_sha256``, the hash the ETag names,
known before any read, so a client that holds the bytes gets its 304 without a single storage read or a
rebuild. Blocking (S3): plain-``def`` routes only.
"""

from __future__ import annotations

import logging
from typing import FrozenSet, Mapping, NamedTuple, Optional, Pattern

from sqlalchemy.orm import Session

from core.models.themes import ThemePackage
from services.marketplace_cloud.catalog import catalogued_verdict
from services.marketplace_cloud.urls import PACKAGE_FILE
from services.themes import signature_verify
from services.themes.choices import ThemeInstallError
from services.themes.listings import (
    PHONE_PREVIEWS, SWITCH_OFF, WEB_PREVIEWS, approved_query, marketplace_switch, preview_paths,
)
from services.themes.manifest import PHONE_PREVIEW_PATTERN, PREVIEW_PATTERN
from services.themes.review_package import PACKAGE_CONTENT_TYPE, UNAVAILABLE, rebuilt_package
from services.themes.revocations import is_disabled
from services.themes.serve import (
    DIGEST_PATTERN, FileReader, ThemeFileMissing, ThemeStorageUnavailable, read_stored_file, verified_file,
)

logger = logging.getLogger(__name__)

NOT_SERVED = "That file isn't served by the Automatos Cloud catalogue."
STORAGE_DOWN = "Theme storage is unavailable."
# Each screenshot list, and the paths theme.json allows in it.
SCREENSHOT_PATTERNS: Mapping[str, Pattern[str]] = {
    WEB_PREVIEWS.field: PREVIEW_PATTERN, PHONE_PREVIEWS.field: PHONE_PREVIEW_PATTERN,
}


class CatalogFile(NamedTuple):
    """The bytes served, their type, and the hash the ETag names (a screenshot's sha256, the package's digest)."""

    data: bytes
    content_type: str
    sha256: str


class CatalogFileMissing(LookupError):
    """Not a file the catalogue serves, or one whose stored bytes no longer check out (the route's 404)."""


class CatalogFilesUnavailable(RuntimeError):
    """Object storage failed for a reason other than a missing object (the route's 503)."""


def catalogued_version(db: Session, digest: str) -> Optional[ThemePackage]:
    """The version with ``digest`` when the catalogue lists it, else None.

    Raises ``SigningKeysError`` when the keys file doesn't load: the route's 503, as the catalogue's.
    """
    if DIGEST_PATTERN.fullmatch(digest or "") is None or marketplace_switch() == SWITCH_OFF:
        return None
    signature_verify.load_keys()
    # uq_theme_packages_marketplace_digest: at most one Marketplace row holds a digest.
    package = approved_query(db).filter(ThemePackage.digest == digest).first()
    if package is None or is_disabled(db, package) or catalogued_verdict(package) is None:
        return None
    return package


def listed_screenshots(package: ThemePackage) -> FrozenSet[str]:
    """Every path theme.json lists under ``previews`` or ``phonePreviews`` that the package holds and that
    theme.json's pattern for that list allows."""
    return frozenset(
        path for kind in (WEB_PREVIEWS, PHONE_PREVIEWS) for path in preview_paths(package, kind)
        if SCREENSHOT_PATTERNS[kind.field].fullmatch(path)
    )


def served_version(db: Session, digest: str, path: str) -> ThemePackage:
    """The catalogued version ``digest`` when ``path`` is its package or a listed screenshot; else
    :class:`CatalogFileMissing`. Reads the database only."""
    package = catalogued_version(db, digest)
    if package is None or (path != PACKAGE_FILE and path not in listed_screenshots(package)):
        raise CatalogFileMissing(NOT_SERVED)
    return package


def listed_sha256(package: ThemePackage, path: str) -> Optional[str]:
    """The hash a served file's ETag names, known before any read: the package's digest, or the screenshot's
    sha256 in the file manifest (None when the manifest row has none: the read then refuses it)."""
    if path == PACKAGE_FILE:
        return package.digest
    listed = package.file_manifest if isinstance(package.file_manifest, dict) else {}
    row = listed.get(path)
    sha256 = row.get("sha256") if isinstance(row, dict) else None
    return sha256 if isinstance(sha256, str) else None


def read_catalog_file(package: ThemePackage, path: str, read: FileReader = read_stored_file) -> CatalogFile:
    """``path`` of a version ``served_version`` answered: its rebuilt package, or the listed screenshot."""
    if path == PACKAGE_FILE:
        return _package_file(package, read)
    try:
        data, content_type, sha256 = verified_file(package, path, read)
    except ThemeFileMissing as gone:
        raise CatalogFileMissing(NOT_SERVED) from gone
    except ThemeStorageUnavailable as down:
        raise CatalogFilesUnavailable(STORAGE_DOWN) from down
    return CatalogFile(data, content_type, sha256)


def _package_file(package: ThemePackage, read: FileReader) -> CatalogFile:
    """The rebuilt ``.autotheme``; a stored file gone or changed, or a broken file list, serves nothing."""
    try:
        data = rebuilt_package(package, read)
    except ThemeInstallError as refused:  # rebuilt_package's ThemeReviewError (409 or 503)
        if refused.status_code == UNAVAILABLE:
            raise CatalogFilesUnavailable(STORAGE_DOWN) from refused
        logger.error(
            "Catalogue files: the package of %s %s (digest %s) can't be rebuilt: %s",
            package.theme_id, package.version, package.digest, refused.message,
        )
        raise CatalogFileMissing(NOT_SERVED) from refused
    return CatalogFile(data, PACKAGE_CONTENT_TYPE, package.digest)


__all__ = [
    "CatalogFile", "CatalogFileMissing", "CatalogFilesUnavailable", "catalogued_version", "listed_sha256",
    "listed_screenshots", "read_catalog_file", "served_version",
]
