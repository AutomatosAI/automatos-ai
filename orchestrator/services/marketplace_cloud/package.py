"""PRD-270 US-270-04a (§5.4, K16) — a Cloud theme's package, fetched from Automatos Cloud and stored first.

A Sync writes a Cloud row (``source`` 'cloud': its checked manifest, file list and signature) and stores no
file. ``ensure_stored`` runs before an install of a listed version (``api/marketplace_themes.py``
``install_theme_listing``, for the version ``listings.install_target`` names) and before a person's update
(``api/themes_install.py`` ``update_theme_install``, through ``ensure_stored_for_update``: only a Cloud
version of the theme this workspace's install runs is fetched; every other refusal stays ``apply_update``'s):

* nothing for a row this install made (``source`` NULL), or for a Cloud row already stored: its theme.json
  is in storage and hashes to its file list (``storage.write_theme_files`` writes in path order and
  theme.json sorts after every other path a package may hold, so it is there only when every file is);
* else ONE ``GET`` of ``<AUTOMATOS_CLOUD_URL origin>/api/marketplace/catalog/files/<digest>/package.autotheme``
  through ``fetch.py``, capped at ``THEME_MAX_PACKAGE_MB``. The URL is derived from the operator's setting
  and the row's 64-hex digest (``urls.package_url``), never taken from a catalogue, so no other host is
  ever asked, and a redirect is never followed;
* the bytes are read with ``package.read_theme_package``; their manifest's id and version, their digest and
  their file list must be the row's; then ``check_upload_signature`` must verify the envelope rebuilt from
  the row (``ENVELOPE_FORM``, ``ENVELOPE_ALGORITHM``, the key id its stored signature verifies with, its
  id, version, digest and signature) over the downloaded files' hashes, so a key retired, or revoked by
  the revocation feed, since the Sync refuses it;
* only then ``storage.write_theme_files(package.storage_prefix, files)``, which removes what it wrote when
  a write fails.

Every refusal is a ``CloudPackageError``, a ``ThemeInstallError`` with a ``code``, raised before any byte
is stored, so both routers answer it as they answer any refusal: 409 for refused bytes (413 past the cap)
and while the Sync is off on this install, 502 when Automatos Cloud can't be reached or doesn't answer
200, 503 when storage or the signing keys file fails. Messages are plain and name only the row's theme and
version. One download of a version at a time per process: a second caller waits (at most ``wait_seconds``,
past the fetch's own deadline, else 503), then finds the files stored; other versions never wait for it.
Synchronous and blocking (plain-``def`` routes only); it reads the database and never writes it. The
caller's read transaction stays open while Automatos Cloud answers, which the fetch's deadline bounds.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Dict, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from core.models.themes import THEME_SOURCE_CLOUD, ThemePackage
from services.marketplace_cloud.fetch import (
    ACCEPT_PACKAGE, DEADLINE_FACTOR, ClientFactory, CloudAnswerTooLarge, CloudFetchFailed, checked_origin,
    fetch_bytes, timeout_seconds,
)
from services.marketplace_cloud.pull import sync_enabled
from services.marketplace_cloud.urls import configured_url, package_url
from services.themes import signature_verify, storage
from services.themes.choices import CONFLICT, ThemeInstallError, find_active_install
from services.themes.contract import MANIFEST_PATH, MB, echo
from services.themes.package import ThemePackage as CheckedThemePackage
from services.themes.package import ThemePackageError, cap_bytes, read_theme_package
from services.themes.revocations import is_disabled
from services.themes.serve import ThemeFileMissing, ThemeStorageUnavailable, read_stored_file, verified_file
from services.themes.signed_upload import (
    ENVELOPE_ALGORITHM, ENVELOPE_FORM, SHA256_HEX, SignatureRefused, check_upload_signature,
)

logger = logging.getLogger(__name__)

TOO_LARGE_STATUS, BAD_GATEWAY, UNAVAILABLE = 413, 502, 503
CLOUD_OFF, CLOUD_UNREACHABLE = "CLOUD_OFF", "CLOUD_UNREACHABLE"
PACKAGE_TOO_LARGE, PACKAGE_REFUSED = "PACKAGE_TOO_LARGE", "PACKAGE_REFUSED"
OTHER_THEME, DIGEST_MISMATCH = "OTHER_THEME", "DIGEST_MISMATCH"
SIGNING_KEYS_UNREADABLE, STORAGE_UNAVAILABLE = "SIGNING_KEYS_UNREADABLE", "STORAGE_UNAVAILABLE"
DOWNLOAD_BUSY = "DOWNLOAD_BUSY"

OFF = (
    "{theme} comes from Automatos Cloud, and this install doesn't sync with it now (AUTOMATOS_CLOUD_URL), "
    "so its files can't be fetched."
)
UNREACHABLE = "{theme} {version} couldn't be fetched from Automatos Cloud, so nothing was installed. {reason}"
TOO_LARGE = "Automatos Cloud's package of {theme} {version} is larger than {mb} MB, so nothing was installed."
NOT_A_PACKAGE = (
    "Automatos Cloud's package of {theme} {version} isn't a theme package this install accepts, "
    "so nothing was installed."
)
FOR_ANOTHER = "Automatos Cloud's package isn't {theme} {version}, so nothing was installed."
OTHER_FILES = (
    "Automatos Cloud sent other files for {theme} {version} than the ones listed here, so nothing was installed."
)
NOT_SIGNED = (
    "Automatos Cloud's package of {theme} {version} isn't signed by an active Automatos key, "
    "so nothing was installed."
)
KEYS_UNREADABLE = "This install's signing keys couldn't be read, so nothing was installed."
STORAGE_DOWN = "Theme files can't be stored right now, so nothing was installed. Try again in a moment."
BUSY = "{theme} {version} is still being fetched from Automatos Cloud, so nothing was installed. Try again in a moment."

# One lock per version's digest, so a second caller waits for that download alone (bounded: one per Cloud version).
_LOCKS_GUARD = threading.Lock()
_DIGEST_LOCKS: Dict[str, threading.Lock] = {}


class CloudPackageError(ThemeInstallError):
    """A Cloud theme's package that wasn't stored: ``code`` says why; a router answers its status and message."""

    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(status_code, message)
        self.code = code


def is_cloud(package: Any) -> bool:
    """True for a Marketplace row a Sync with Automatos Cloud wrote."""
    return getattr(package, "source", None) == THEME_SOURCE_CLOUD


def wait_seconds() -> float:
    """How long a caller waits for another download of the same version: past that download's own deadline."""
    return float(DEADLINE_FACTOR + 1) * timeout_seconds()


def _digest_lock(digest: str) -> threading.Lock:
    """This process's one lock for the download of the version ``digest``."""
    with _LOCKS_GUARD:
        return _DIGEST_LOCKS.setdefault(digest, threading.Lock())


def _named(template: str, package: ThemePackage, **extra: Any) -> str:
    return template.format(theme=echo(package.theme_id), version=echo(package.version), **extra)


def is_stored(package: ThemePackage) -> bool:
    """True when the row's theme.json is in storage and hashes to its file list (it is the last file written)."""
    try:
        verified_file(package, MANIFEST_PATH, read_stored_file)
    except ThemeFileMissing:
        return False
    except ThemeStorageUnavailable as down:
        raise CloudPackageError(UNAVAILABLE, STORAGE_UNAVAILABLE, STORAGE_DOWN) from down
    return True


def _download(package: ThemePackage, client_factory: Optional[ClientFactory]) -> bytes:
    """The package's bytes from the URL derived from the setting and the row's digest, capped."""
    if not sync_enabled():
        raise CloudPackageError(CONFLICT, CLOUD_OFF, _named(OFF, package))
    if not SHA256_HEX.fullmatch(package.digest or ""):
        raise CloudPackageError(CONFLICT, PACKAGE_REFUSED, _named(NOT_A_PACKAGE, package))
    limit = cap_bytes("package")
    try:
        url = package_url(checked_origin(configured_url()), package.digest)
        return fetch_bytes(url, limit, ACCEPT_PACKAGE, client_factory)
    except CloudAnswerTooLarge as large:
        logger.warning("Automatos Cloud: the package of %s %s is over the cap", package.theme_id, package.version)
        message = _named(TOO_LARGE, package, mb=limit // MB)
        raise CloudPackageError(TOO_LARGE_STATUS, PACKAGE_TOO_LARGE, message) from large
    except CloudFetchFailed as failed:
        logger.warning("Automatos Cloud: %s %s wasn't fetched: %s", package.theme_id, package.version, failed)
        message = _named(UNREACHABLE, package, reason=str(failed))
        raise CloudPackageError(BAD_GATEWAY, CLOUD_UNREACHABLE, message) from failed


def _read(package: ThemePackage, data: bytes) -> CheckedThemePackage:
    """The downloaded package, read and held to the row: its id and version, then its digest and file list."""
    try:
        checked = read_theme_package(data)
    except ThemePackageError as refused:
        codes = [error["code"] for error in refused.errors]
        logger.warning("Automatos Cloud: the package of %s %s was refused: %s", package.theme_id, package.version,
                       codes)
        raise CloudPackageError(CONFLICT, PACKAGE_REFUSED, _named(NOT_A_PACKAGE, package)) from refused
    if checked.manifest.get("id") != package.theme_id or checked.manifest.get("version") != package.version:
        logger.warning("Automatos Cloud: the package fetched for %s %s is another theme", package.theme_id,
                       package.version)
        raise CloudPackageError(CONFLICT, OTHER_THEME, _named(FOR_ANOTHER, package))
    if checked.digest != package.digest or checked.file_manifest != package.file_manifest:
        logger.warning("Automatos Cloud: the package fetched for %s %s has other files", package.theme_id,
                       package.version)
        raise CloudPackageError(CONFLICT, DIGEST_MISMATCH, _named(OTHER_FILES, package))
    return checked


def row_envelope(package: ThemePackage) -> bytes:
    """The ``.sig.json`` the row's signature is: its key id is the one the stored signature verifies with
    (none when no key does, which the envelope's form then refuses)."""
    verdict = signature_verify.package_verdict(package)
    return json.dumps({
        "form": ENVELOPE_FORM, "algorithm": ENVELOPE_ALGORITHM, "keyId": verdict.key_id, "id": package.theme_id,
        "version": package.version, "digest": package.digest, "signature": package.signature,
    }).encode("utf-8")


def _check_signature(db: Session, package: ThemePackage, checked: CheckedThemePackage) -> None:
    """``check_upload_signature`` on the downloaded files with the row's envelope, or a refusal."""
    try:
        check_upload_signature(db, checked, row_envelope(package))
    except SignatureRefused as refused:
        code = refused.error["code"]
        logger.warning("Automatos Cloud: the package of %s %s wasn't verified: %s", package.theme_id,
                       package.version, code)
        raise CloudPackageError(CONFLICT, code, _named(NOT_SIGNED, package)) from refused
    except signature_verify.SigningKeysError as bad:
        logger.exception("Automatos Cloud: the signing keys file didn't load")
        raise CloudPackageError(UNAVAILABLE, SIGNING_KEYS_UNREADABLE, KEYS_UNREADABLE) from bad


def _store(package: ThemePackage, checked: CheckedThemePackage) -> None:
    try:
        storage.write_theme_files(package.storage_prefix, checked.files)
    except Exception as failed:  # S3 or MinIO, or an unusable prefix: write_theme_files removed what it wrote
        logger.exception("Automatos Cloud: the files of %s %s couldn't be stored", package.theme_id, package.version)
        raise CloudPackageError(UNAVAILABLE, STORAGE_UNAVAILABLE, STORAGE_DOWN) from failed
    logger.info("Automatos Cloud: stored %s %s (%d files)", package.theme_id, package.version, len(checked.files))


def ensure_stored(db: Session, package: ThemePackage, client_factory: Optional[ClientFactory] = None) -> None:
    """Fetch, check and store a Cloud row's files before it installs; nothing for any other row or one stored.

    ``CloudPackageError`` with nothing stored when the bytes are refused, Cloud fails or storage fails.
    """
    if not is_cloud(package) or is_stored(package):
        return
    lock = _digest_lock(package.digest)
    if not lock.acquire(timeout=wait_seconds()):
        raise CloudPackageError(UNAVAILABLE, DOWNLOAD_BUSY, _named(BUSY, package))
    try:
        if is_stored(package):  # another request stored it while this one waited
            return
        logger.info("Automatos Cloud: fetching %s %s, not stored here yet", package.theme_id, package.version)
        checked = _read(package, _download(package, client_factory))
        _check_signature(db, package, checked)
        _store(package, checked)
    finally:
        lock.release()


def ensure_stored_for_update(db: Session, workspace_id: UUID, install_id: UUID, package_id: UUID) -> None:
    """``ensure_stored`` for the version an update names, when it is a Cloud version, no ``disable`` covers it,
    of the theme this workspace's active install ``install_id`` runs.

    A Cloud row is a Marketplace version, which every workspace may use. Anything else (no such install, another
    theme, a local row, a missing id) fetches nothing and is left to ``updates.apply_update``. Its other refusals (a
    version that isn't newer, a pin, an unapproved version) come after the fetch: they aren't copied here, so a
    change to them can never move an install to a version whose files were not stored.
    """
    install = find_active_install(db, workspace_id, install_id)
    package = db.get(ThemePackage, package_id)
    if install is None or not is_cloud(package) or package.theme_id != install.theme_id:
        return
    if not is_disabled(db, package):
        ensure_stored(db, package)


__all__ = [
    "CLOUD_OFF", "CLOUD_UNREACHABLE", "DIGEST_MISMATCH", "DOWNLOAD_BUSY", "OTHER_THEME", "PACKAGE_REFUSED",
    "PACKAGE_TOO_LARGE", "SIGNING_KEYS_UNREADABLE", "STORAGE_UNAVAILABLE", "CloudPackageError", "ensure_stored",
    "ensure_stored_for_update", "is_cloud", "is_stored", "row_envelope", "wait_seconds",
]
