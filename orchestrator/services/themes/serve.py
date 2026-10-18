"""PRD-261 A-08a §6 — which package a caller may load by digest, its checked bytes, and its boot.

``GET /api/themes/files/{digest}/{path}`` and ``GET /api/themes/boot/{installId}``
(``api/themes_files.py``) authorise by membership (``membership.py``), never by
``ctx.workspace_id``. A digest is servable to a caller when, for one ``theme_packages`` row
with that digest:

* it is an approved Marketplace package (an approved listing); or
* it is installed (an active ``theme_installations`` row whose ``package_id`` is the row) in a
  workspace where the caller is a member or owner, and the package belongs to that install:
  a Marketplace package, or a private package of **that** workspace (a private package
  referenced from another workspace's install serves nothing). A private package also needs
  the caller's opt-in to that version in that workspace (``theme_opt_ins``, Q6).

Two workspaces may hold the same digest, so the row is chosen by the digest and the caller's
authorisation, never as "the one row with that digest". A live ``disable`` revocation
(``revocations.py``) then serves nothing.

Every file read is checked against the row's ``file_manifest``: the path must be listed, its
stored bytes must hash to the listed sha256 (a mismatch is logged as tampering and serves
nothing), and the listed type is the type served. SVG is never served (the package reader
refuses it, PRD §6; refused here too).

The boot is the §2.2 ``ThemeBoot``: the theme's classes and tone policy, ``boot/prepaint.json``
checked again by ``css_value.py``, and ``integrity`` (module URL → ``sha256-<base64>``, from the
stored hex) for every JS module, which ``active`` (A-07b) carries too. A caller who has not
opted in, or a revoked version, gets the base boot: the theme's base classes and nothing else.
"""

from __future__ import annotations

import base64
import logging
import re
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Tuple
from uuid import UUID

from sqlalchemy import or_

from config import config
from core.models.themes import ThemeInstallation, ThemeOptIn, ThemePackage
from core.storage import get_s3_client
from services.themes.contract import HTML_CLASS_OWNERS, parse_json
from services.themes.css_value import PREPAINT_PATH, prepaint_errors
from services.themes.digest import sha256_hex
from services.themes.membership import Membership
from services.themes.revocations import is_disabled

logger = logging.getLogger(__name__)

FILES_ROUTE = "/api/themes/files"
DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
STORAGE_ROOT = "themes/"
MARKETPLACE = "marketplace"
PRIVATE = "workspace"
APPROVED = "approved"
# A Marketplace package keeps serving its installs once deactivated (a revocation is the kill
# switch); a pending or rejected one was never installable.
INSTALLED_MARKETPLACE_STATUSES = ("approved", "deactivated")
MODULE_SUFFIX = ".js"
MODULE_TYPE = "text/javascript"
REFUSED_SUFFIXES = (".svg", ".svgz")
REFUSED_TYPES = frozenset({"image/svg+xml"})
INTEGRITY_PREFIX = "sha256-"
MISSING_OBJECT_CODES = frozenset({"NoSuchKey", "404", "NotFound"})

FILE_CACHE_SETTING = "THEME_FILE_CACHE_SECONDS"  # named in config.py by B2-wire
FILE_CACHE_DEFAULT_SECONDS = 86400

STUDIO_BASE = "studio"
STUDIO_HTML_CLASS = "studio"
# Every class a style may put on <html> (lib/ui-style.ts STYLE_HTML_CLASS_NAMES): Studio's
# and each builtin theme's.
STYLE_HTML_CLASSES: Tuple[str, ...] = (STUDIO_HTML_CLASS, *sorted(HTML_CLASS_OWNERS))
USER_TONE = "user"
FIXED_TONES = frozenset({"light", "dark"})

# (storage prefix, package path) -> the stored bytes, or None when the object is gone.
FileReader = Callable[[str, str], Optional[bytes]]


class ThemeFileMissing(LookupError):
    """The file isn't one the package lists, is gone from storage, or its bytes don't match."""


class ThemeStorageUnavailable(RuntimeError):
    """Object storage failed for a reason other than a missing object (the route's 503)."""


class BootTarget(NamedTuple):
    """An active install the caller may see, and the package it runs."""

    install: ThemeInstallation
    package: ThemePackage


def file_cache_seconds() -> int:
    """How long a browser may keep a theme file (``THEME_FILE_CACHE_SECONDS`` once B2-wire names it)."""
    return int(getattr(config, FILE_CACHE_SETTING, FILE_CACHE_DEFAULT_SECONDS))


def _storage_error_code(exc: Exception) -> str:
    response = getattr(exc, "response", None) or {}
    return str(response.get("Error", {}).get("Code", ""))


def read_stored_file(storage_prefix: str, path: str) -> Optional[bytes]:
    """One stored theme file's bytes from the Marketplace bucket (blocking; plain-``def`` routes only)."""
    if not storage_prefix.startswith(STORAGE_ROOT) or not storage_prefix.endswith("/"):
        raise ThemeStorageUnavailable(f"Not a theme storage prefix: {storage_prefix!r}")
    key = f"{storage_prefix}{path}"
    try:
        response = get_s3_client().get_object(Bucket=config.MARKETPLACE_S3_BUCKET, Key=key)
        return response["Body"].read()
    except Exception as exc:
        if _storage_error_code(exc) in MISSING_OBJECT_CODES:
            logger.error("Theme file %s is listed in its package but missing from storage", key)
            return None
        logger.exception("Theme file %s could not be read from storage", key)
        raise ThemeStorageUnavailable("Theme storage is unavailable.") from exc


def is_private(package: ThemePackage) -> bool:
    """A workspace's own upload: code that runs only after a member's opt-in."""
    return package.owner_type == PRIVATE


def _opted_in(db: Any, workspace_id: Any, user_pk: Optional[int], package_id: Any) -> bool:
    if user_pk is None:
        return False
    row = (
        db.query(ThemeOptIn.id)
        .filter(ThemeOptIn.workspace_id == workspace_id, ThemeOptIn.user_id == user_pk, ThemeOptIn.package_id == package_id)
        .first()
    )
    return row is not None


def may_run(db: Any, membership: Membership, install: ThemeInstallation, package: ThemePackage) -> bool:
    """A member may run an installed package: a Marketplace one always, a private one after opting in to it."""
    if not is_private(package):
        return True
    return _opted_in(db, install.workspace_id, membership.user_pk, package.id)


def belongs_to_install(install: ThemeInstallation, package: ThemePackage) -> bool:
    """The package is one this install may run: a live Marketplace version, or its own workspace's upload."""
    if package.owner_type == MARKETPLACE:
        return package.status in INSTALLED_MARKETPLACE_STATUSES
    return package.workspace_id is not None and package.workspace_id == install.workspace_id


def _approved_listing(db: Any, digest: str) -> Optional[ThemePackage]:
    return (
        db.query(ThemePackage)
        .filter(ThemePackage.owner_type == MARKETPLACE, ThemePackage.digest == digest, ThemePackage.status == APPROVED)
        .first()
    )


def _installed_rows(db: Any, membership: Membership, digest: str) -> List[Tuple[ThemePackage, ThemeInstallation]]:
    """Every (package, active install) with ``digest`` in a workspace the caller belongs to."""
    if not membership.is_member_anywhere:
        return []
    query = (
        db.query(ThemePackage, ThemeInstallation)
        .join(ThemeInstallation, ThemeInstallation.package_id == ThemePackage.id)
        .filter(ThemePackage.digest == digest, ThemeInstallation.uninstalled_at.is_(None))
        .filter(or_(ThemePackage.owner_type == MARKETPLACE, ThemePackage.workspace_id == ThemeInstallation.workspace_id))
        .order_by(ThemeInstallation.installed_at)
    )
    if not membership.every_workspace:
        query = query.filter(ThemeInstallation.workspace_id.in_(sorted(membership.workspace_ids, key=str)))
    return query.all()


def authorised_package(db: Any, membership: Membership, digest: str) -> Optional[ThemePackage]:
    """The package row the caller may load files of by ``digest``, or None (the route's 404)."""
    if DIGEST_PATTERN.fullmatch(digest or "") is None:
        return None
    listing = _approved_listing(db, digest)
    if listing is not None:
        return None if is_disabled(db, listing) else listing
    for package, install in _installed_rows(db, membership, digest):
        if belongs_to_install(install, package) and may_run(db, membership, install, package):
            return None if is_disabled(db, package) else package
    return None


def _listed_file(package: ThemePackage, path: str) -> Mapping[str, Any]:
    manifest = package.file_manifest if isinstance(package.file_manifest, dict) else {}
    row = manifest.get(path)
    if not isinstance(row, dict) or not isinstance(row.get("sha256"), str) or not isinstance(row.get("type"), str):
        raise ThemeFileMissing(path)
    if path.lower().endswith(REFUSED_SUFFIXES) or row["type"].split(";")[0].strip().lower() in REFUSED_TYPES:
        raise ThemeFileMissing(path)
    return row


def verified_file(package: ThemePackage, path: str, reader: FileReader) -> Tuple[bytes, str, str]:
    """``(bytes, listed type, sha256)`` of a listed file whose stored bytes still hash to the listing.

    Raises :class:`ThemeFileMissing` otherwise; a hash mismatch is logged as tampering.
    """
    row = _listed_file(package, path)
    data = reader(package.storage_prefix, path)
    if data is None:
        raise ThemeFileMissing(path)
    if sha256_hex(data) != row["sha256"]:
        logger.error(
            "Theme file %s of package %s (digest %s) does not match its file manifest: refused as tampered",
            path, package.id, package.digest,
        )
        raise ThemeFileMissing(path)
    return data, row["type"], row["sha256"]


def integrity_value(sha256_hex_value: str) -> str:
    """The Subresource Integrity form of a stored sha256 hex: ``sha256-`` + base64 of the raw digest."""
    return INTEGRITY_PREFIX + base64.b64encode(bytes.fromhex(sha256_hex_value)).decode("ascii")


def module_url(digest: str, path: str) -> str:
    """Where the files route serves ``path`` of the package ``digest``."""
    return f"{FILES_ROUTE}/{digest}/{path}"


def module_integrity(package: ThemePackage) -> Dict[str, str]:
    """``{module URL: 'sha256-<base64>'}`` for every JS module the package lists (PA-13)."""
    manifest = package.file_manifest if isinstance(package.file_manifest, dict) else {}
    return {
        module_url(package.digest, path): integrity_value(row["sha256"])
        for path, row in sorted(manifest.items())
        if path.endswith(MODULE_SUFFIX) and isinstance(row, dict) and row.get("type") == MODULE_TYPE
    }


def boot_target(db: Any, membership: Membership, install_id: UUID) -> Optional[BootTarget]:
    """The active install ``install_id`` and its package, when the caller belongs to its workspace."""
    install = db.get(ThemeInstallation, install_id)
    if install is None or install.uninstalled_at is not None or not membership.includes(install.workspace_id):
        return None
    package = db.get(ThemePackage, install.package_id)
    if package is None or not belongs_to_install(install, package):
        return None
    return BootTarget(install=install, package=package)


def _boot(add_classes: List[str], extra_remove: Tuple[str, ...], policy: str) -> Dict[str, Any]:
    removable = sorted(set(STYLE_HTML_CLASSES + extra_remove) - set(add_classes))
    return {
        "addClasses": add_classes,
        "removeClasses": removable,
        "tone": {"policy": policy, "value": policy if policy in FIXED_TONES else None},
        "prepaint": None,
        "lookStorage": None,
        "lookGuard": None,
        "integrity": None,
    }


def _base_classes(manifest: Mapping[str, Any]) -> List[str]:
    return [STUDIO_HTML_CLASS] if manifest.get("base", STUDIO_BASE) == STUDIO_BASE else []


def base_boot(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    """The boot of a theme that does not run: its base's classes (its own class removed), the user's tone."""
    own_class = manifest.get("htmlClass")
    return _boot(_base_classes(manifest), (own_class,) if isinstance(own_class, str) and own_class else (), USER_TONE)


def theme_prepaint(package: ThemePackage, reader: FileReader) -> Optional[Dict[str, Any]]:
    """``boot/prepaint.json`` re-hashed and re-checked by ``css_value.py``; None when the package has none.

    Raises :class:`ThemeFileMissing` for a tampered, unreadable or refused table.
    """
    if PREPAINT_PATH not in (package.file_manifest or {}):
        return None
    data, _type, _sha = verified_file(package, PREPAINT_PATH, reader)
    document, error = parse_json(data, PREPAINT_PATH)
    errors = [error] if error is not None else prepaint_errors(document)
    if errors:
        logger.error("Package %s's %s fails the CSS-value check on read: %s", package.id, PREPAINT_PATH, errors[0]["message"])
        raise ThemeFileMissing(PREPAINT_PATH)
    return document


def theme_boot(package: ThemePackage, prepaint: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The boot of a theme that runs: its base's classes and its own, its tone, prepaint and integrity."""
    manifest = package.manifest or {}
    classes = [*_base_classes(manifest), str(manifest["htmlClass"])]
    boot = _boot(classes, (), str(manifest.get("tone", USER_TONE)))
    return {**boot, "prepaint": prepaint, "integrity": module_integrity(package)}


def install_boot(db: Any, membership: Membership, target: BootTarget, reader: FileReader) -> Dict[str, Any]:
    """The first-paint data of an install: the theme's own, or the base boot when it may not run."""
    package = target.package
    manifest = package.manifest or {}
    if not may_run(db, membership, target.install, package) or is_disabled(db, package):
        return base_boot(manifest)
    try:
        prepaint = theme_prepaint(package, reader)
    except ThemeFileMissing:
        return base_boot(manifest)
    return theme_boot(package, prepaint)


__all__ = [
    "BootTarget", "FileReader", "ThemeFileMissing", "ThemeStorageUnavailable", "authorised_package", "base_boot",
    "belongs_to_install", "boot_target", "file_cache_seconds", "install_boot", "integrity_value", "is_private",
    "may_run", "module_integrity", "module_url", "read_stored_file", "theme_boot", "theme_prepaint", "verified_file",
]
