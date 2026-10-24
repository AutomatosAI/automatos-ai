"""PRD-261 A-08a §6 — which package a caller may load by digest, its checked bytes, and its boot.

``GET /api/themes/files/{digest}/{path}`` and ``GET /api/themes/boot/{installId}``
(``api/themes_files.py``) authorise by membership (``membership.py``), never by
``ctx.workspace_id``. A digest is servable to a caller when, for one ``theme_packages`` row
with that digest:

* it is an approved, reviewed Marketplace package (an approved listing); or
* it is installed (an active ``theme_installations`` row whose ``package_id`` is the row) in a
  workspace where the caller is a member or owner, and the package belongs to that install:
  a Marketplace package, or a private package of **that** workspace (a private package
  referenced from another workspace's install serves nothing). Private code (``choices.is_private``:
  a Marketplace package not reviewed yet, or a workspace's own upload no active Automatos key vouches
  for (QB-24), the same rule ``active`` reports as ``optIn.required``) also needs the caller's opt-in
  to that version in that workspace (``theme_opt_ins``, Q6).

Two workspaces may hold the same digest, so the row is chosen by the digest and the caller's
authorisation, never as "the one row with that digest". A row a live ``disable`` covers
(``revocations.is_disabled``) serves nothing, and the next authorised row is tried: the approved
listing first, then the caller's installed rows. ``theme.json`` is one of the hashed files, so
every row with a digest has the same theme id and version, and a stored revocation (by digest, or
by theme id and version range) covers all of them or none: revoked bytes are never served through
a sibling row. A signature disable (C-05c) is one row's own (its stored signature doesn't check
out), so a Marketplace listing whose signature fails still leaves a workspace's own copy of the
same bytes serving to its members (private code, so after their opt-in).

Every file read is checked against the row's ``file_manifest``: the path must be listed, its
stored bytes must hash to the listed sha256 (a mismatch is logged as tampering and serves
nothing), and the listed type is the type served. SVG is never served (the package reader
refuses it, PRD §6; refused here too).

The boot is the §2.2 ``ThemeBoot``: the theme's classes and tone policy, ``boot/prepaint.json``
checked again by ``css_value.py``, and ``integrity`` (module URL → ``sha256-<base64>``, from the
stored hex) for every JS module, which ``active`` (A-07b) carries too. A caller who has not
opted in, or a revoked version, gets the base boot: the theme's base classes and nothing else.

PRD-261 W-A6b (gap G1): the boot of a theme that runs also carries ``entry``, what the host's
theme registry needs of it (``theme_entry``: its name, base, class, width, tone, pages, routes,
nav and slots, from the checked ``theme.json``), so the root layout hands the client the same
registry entry on the server render and the first client render. The base boot's is null.

PRD-261 W-A6b (gap G2): the boot of a theme that runs paints the member's stored look before first
paint. ``lookStorage`` names the browser keys the host's look store keeps
the theme's active look under (``look_storage``, the frontend's ``installedLookKeys`` format,
``tests/fixtures/themes/look-storage.json`` holds the two equal), and ``lookGuard`` is the look
header of the theme's checked ``looks/schema.json``, so only a look of this theme's format paints.
A package with no readable schema gets neither: its default look paints. The entry carries the
theme id (``themeId``), the one the host's look store and the looks API key the looks by.

PRD-261 A-19c (host 1.1.0, plan §2.2.3): the boot of a theme that runs also carries ``looks``,
``{defaultLook, presets}``, so a generic look runtime runs a theme's looks from data alone
(``boot_looks``). The presets are ``looks/presets/*.look.json``, each re-hashed and checked
against the theme's checked ``looks/schema.json``: the default look first, the rest in file-name
order, at most ``BOOT_PRESETS_MAX`` of at most ``BOOT_PRESET_MAX_BYTES`` each. A preset that is
too big, unreadable or fails the schema is dropped and logged. ``prepaint`` may carry the tone
rule (``css_value.tone_rule_errors``), checked again with the rest of the table on read.
"""

from __future__ import annotations

import base64
import logging
import re
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Tuple
from uuid import UUID

from sqlalchemy import or_

from config import config
from core.models.themes import ThemeInstallation, ThemePackage
from core.storage import get_s3_client
from services.themes.choices import is_private
from services.themes.contract import BOOT_PRESET_MAX_BYTES, BOOT_PRESETS_MAX, HTML_CLASS_OWNERS, parse_json
from services.themes.css_value import PREPAINT_PATH, prepaint_errors
from services.themes.digest import sha256_hex
from services.themes.look_schema import (
    PRESET_SUFFIX, PRESETS_DIR, SCHEMA_PATH as LOOK_SCHEMA_PATH, LookSchema, preset_paths, validate_look,
    validate_look_schema,
)
from services.themes.membership import Membership
from services.themes.optin import has_opted_in
from services.themes.revocations import is_disabled
from services.themes.storage import theme_key

logger = logging.getLogger(__name__)

FILES_ROUTE = "/api/themes/files"
DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
MARKETPLACE = "marketplace"
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

STUDIO_BASE = "studio"
STUDIO_HTML_CLASS = "studio"
# Every class a boot takes off <html> unless its style carries it: the Style axis's (frontend
# lib/ui-style.ts STYLE_HTML_CLASS_NAMES, Studio's) and each class PA-8 keeps for one theme's
# package (contract.py HTML_CLASS_OWNERS, Play's until A-20).
STYLE_HTML_CLASSES: Tuple[str, ...] = (STUDIO_HTML_CLASS, *sorted(HTML_CLASS_OWNERS))
USER_TONE = "user"
# theme.json requires minWidth; a manifest stored before it was required draws at every width.
DEFAULT_MIN_WIDTH = 0
FIXED_TONES = frozenset({"light", "dark"})
# Where the host's look store keeps an installed theme's active look in the browser (frontend
# lib/theme-host/looks/storage.ts themeLookKeys and bootLookStorage):
# `automatos-theme.<theme id>.look`, with `:<scope>` appended per account, and `.current` naming
# the scope the boot script paints.
INSTALLED_LOOK_KEY_PREFIX = "automatos-theme."
ACTIVE_LOOK_SUFFIX = ".look"
LOOK_POINTER_SUFFIX = ".current"
LOOK_SCOPE_SEPARATOR = ":"
# The boot fields theme_looks answers (``looks``: A-19c, host 1.1.0).
LOOK_BOOT_FIELDS = ("lookStorage", "lookGuard", "looks")

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
    """How long a browser may keep a theme file (``THEME_FILE_CACHE_SECONDS``)."""
    return int(config.THEME_FILE_CACHE_SECONDS)


def _storage_error_code(exc: Exception) -> str:
    response = getattr(exc, "response", None) or {}
    return str(response.get("Error", {}).get("Code", ""))


def read_stored_file(storage_prefix: str, path: str) -> Optional[bytes]:
    """One stored theme file's bytes from the Marketplace bucket, or None when the object is gone.

    The one stored-file reader (the files and boot routes, ``looks.py``): blocking, so plain-``def``
    routes only. The key comes from ``storage.theme_key``, the one prefix and path check.
    """
    try:
        key = theme_key(storage_prefix, path)
    except ValueError as exc:
        logger.error("Theme package row has an unusable storage key (%r, %r)", storage_prefix, path)
        raise ThemeStorageUnavailable("Theme storage is unavailable.") from exc
    try:
        response = get_s3_client().get_object(Bucket=config.MARKETPLACE_S3_BUCKET, Key=key)
        return response["Body"].read()
    except Exception as exc:
        if _storage_error_code(exc) in MISSING_OBJECT_CODES:
            logger.error("Theme file %r is listed in its package but missing from storage", key)
            return None
        logger.exception("Theme file %r could not be read from storage", key)
        raise ThemeStorageUnavailable("Theme storage is unavailable.") from exc


def may_run(db: Any, membership: Membership, install: ThemeInstallation, package: ThemePackage) -> bool:
    """A member may run an installed package: reviewed Marketplace code always, private code after opting in to it."""
    if not is_private(package):
        return True
    return has_opted_in(db, install.workspace_id, membership.user_pk, package.id)


def belongs_to_install(install: ThemeInstallation, package: ThemePackage) -> bool:
    """The package is one this install may run: a live Marketplace version, or its own workspace's upload."""
    if package.owner_type == MARKETPLACE:
        return package.status in INSTALLED_MARKETPLACE_STATUSES
    return package.workspace_id is not None and package.workspace_id == install.workspace_id


def _approved_listing(db: Any, digest: str) -> Optional[ThemePackage]:
    return (
        db.query(ThemePackage)
        .filter(
            ThemePackage.owner_type == MARKETPLACE, ThemePackage.digest == digest,
            ThemePackage.status == APPROVED, ThemePackage.reviewed.is_(True),
        )
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
    """The package row the caller may load files of by ``digest``, or None (the route's 404).

    The approved listing unless a ``disable`` covers it, else the first of the caller's installed
    rows that may run and no ``disable`` covers (module docstring: a stored revocation covers
    every row of the digest; a signature disable only its own row).
    """
    if DIGEST_PATTERN.fullmatch(digest or "") is None:
        return None
    listing = _approved_listing(db, digest)
    if listing is not None and not is_disabled(db, listing):
        return listing
    for package, install in _installed_rows(db, membership, digest):
        runs = belongs_to_install(install, package) and may_run(db, membership, install, package)
        if runs and not is_disabled(db, package):
            return package
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
            "Theme file %r of package %s (digest %s) does not match its file manifest: refused as tampered",
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
    """``{module URL: 'sha256-<base64>'}`` for every JS module the package lists (PA-13).

    The one integrity map: the boot route and ``active`` (``defaults.py``) both carry it. A module
    is a ``.js`` path listed as ``text/javascript``; one without a valid sha256 is left out and logged.
    """
    manifest = package.file_manifest if isinstance(package.file_manifest, dict) else {}
    pinned: Dict[str, str] = {}
    for path, row in sorted(manifest.items()):
        if not path.endswith(MODULE_SUFFIX) or not isinstance(row, dict) or row.get("type") != MODULE_TYPE:
            continue
        sha = row.get("sha256")
        if not isinstance(sha, str) or DIGEST_PATTERN.fullmatch(sha) is None:  # a sha256 hex, like a digest
            logger.error("Theme package %s has no valid sha256 for %r", package.id, path)
            continue
        pinned[module_url(package.digest, path)] = integrity_value(sha)
    return pinned


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
        "looks": None,
        "integrity": None,
        "entry": None,
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


def look_storage(theme_id: str) -> Dict[str, str]:
    """The browser keys the boot script reads a theme's stored active look from (``ThemeLookStorage``)."""
    active = f"{INSTALLED_LOOK_KEY_PREFIX}{theme_id}{ACTIVE_LOOK_SUFFIX}"
    return {
        "pointerKey": f"{active}{LOOK_POINTER_SUFFIX}",
        "scopedPrefix": f"{active}{LOOK_SCOPE_SEPARATOR}",
        "fallbackKey": active,
    }


def _theme_id(package: ThemePackage) -> str:
    return str((package.manifest or {}).get("id") or package.theme_id)


def theme_look_schema(package: ThemePackage, reader: FileReader) -> Optional[LookSchema]:
    """``looks/schema.json`` re-hashed and checked by ``look_schema.py``; None when the package has none or it fails.

    A failure is logged: the boot then paints the theme's default look, never an unguarded stored one.
    """
    if LOOK_SCHEMA_PATH not in (package.file_manifest or {}):
        return None
    try:
        data, _type, _sha = verified_file(package, LOOK_SCHEMA_PATH, reader)
    except ThemeFileMissing:
        logger.error("Package %s's %s can't be read: its boot paints the default look", package.id, LOOK_SCHEMA_PATH)
        return None
    document, error = parse_json(data, LOOK_SCHEMA_PATH)
    schema, errors = validate_look_schema(document) if error is None else (None, [error])
    if schema is None:
        logger.error("Package %s's %s fails its check on read: %s", package.id, LOOK_SCHEMA_PATH, errors[0]["message"])
    return schema


def _preset_id(path: str) -> str:
    """A preset's id: its file name (``look_schema.preset_ids``)."""
    return path[len(PRESETS_DIR):-len(PRESET_SUFFIX)]


def _boot_preset_order(package: ThemePackage) -> List[str]:
    """The package's preset paths: the manifest's default look first, the rest in file-name order."""
    default = (package.manifest or {}).get("look")
    paths = preset_paths(package.file_manifest or {})
    return [path for path in paths if _preset_id(path) == default] + [path for path in paths if _preset_id(path) != default]


def _checked_preset(schema: LookSchema, path: str, data: bytes) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """One stored preset's look document, or why it can't ride the boot."""
    if len(data) > BOOT_PRESET_MAX_BYTES:
        return None, f"it is over {BOOT_PRESET_MAX_BYTES} bytes"
    document, error = parse_json(data, path)
    errors = [error] if error is not None else validate_look(schema, document)
    if errors:
        return None, errors[0]["message"]
    if document.get("id") != _preset_id(path):
        return None, "its id is not its file name"
    return document, None


def _boot_preset(package: ThemePackage, schema: LookSchema, path: str, reader: FileReader) -> Optional[Dict[str, Any]]:
    """One preset as the boot carries it (the stored look document), or None, logged, when it can't ride."""
    listed = (package.file_manifest or {}).get(path)
    size = listed.get("size") if isinstance(listed, dict) else None
    if isinstance(size, int) and size > BOOT_PRESET_MAX_BYTES:
        logger.error("Package %s's preset %s is over %d bytes: its boot leaves it out", package.id, path, BOOT_PRESET_MAX_BYTES)
        return None
    try:
        data, _type, _sha = verified_file(package, path, reader)
    except ThemeFileMissing:
        logger.error("Package %s's preset %s can't be read: its boot leaves it out", package.id, path)
        return None
    document, problem = _checked_preset(schema, path, data)
    if problem is not None:
        logger.error("Package %s's preset %s fails its check on read (%s): its boot leaves it out", package.id, path, problem)
    return document


def boot_looks(package: ThemePackage, schema: LookSchema, reader: FileReader) -> Optional[Dict[str, Any]]:
    """The boot's ``looks``: ``{defaultLook, presets}``, or None when no preset survives its check.

    The default look is the manifest's ``look`` when its preset survives, else the first preset
    that does (``looks.package_default_look``'s rule: a theme without ``look`` starts on its first
    preset by file name). Presets past ``BOOT_PRESETS_MAX`` are left out and logged.
    """
    presets: List[Dict[str, Any]] = []
    for path in _boot_preset_order(package):
        if len(presets) == BOOT_PRESETS_MAX:
            logger.error("Package %s has more than %d presets: its boot carries the first %d", package.id, BOOT_PRESETS_MAX, BOOT_PRESETS_MAX)
            break
        preset = _boot_preset(package, schema, path, reader)
        if preset is not None:
            presets.append(preset)
    if not presets:
        return None
    return {"defaultLook": presets[0]["id"], "presets": presets}


def theme_looks(package: ThemePackage, reader: FileReader) -> Dict[str, Any]:
    """The boot's ``lookStorage``, ``lookGuard`` and ``looks``: where the stored look is, its format, the presets.

    The guard is the schema's look header; one with no interface has no guard (the boot script checks
    both), so any stored look that fits the prepaint table paints, as the keys are this theme's own.
    ``looks`` is ``boot_looks``' answer (A-19c).
    """
    schema = theme_look_schema(package, reader)
    if schema is None:
        return dict.fromkeys(LOOK_BOOT_FIELDS)
    guard = {"schema": schema.look_schema, "iface": schema.interface} if schema.interface else None
    return {"lookStorage": look_storage(_theme_id(package)), "lookGuard": guard, "looks": boot_looks(package, schema, reader)}


def theme_entry(package: ThemePackage) -> Dict[str, Any]:
    """The theme's registry entry, as the host reads it (frontend ``lib/theme-host/installed-entry.ts``).

    Every field comes from the ``theme.json`` checked on import (``manifest.py``); ``ownsPages`` lists
    the core pages the theme draws (``pages``' routes), and ``reviewed`` is the package's review, the
    rule ``choices.is_private`` reads. The frontend checks the answer again before it uses it.
    """
    manifest = package.manifest or {}
    pages = manifest.get("pages")
    return {
        "themeId": _theme_id(package),
        "name": str(manifest.get("name", "")),
        "base": str(manifest.get("base", STUDIO_BASE)),
        "htmlClass": str(manifest["htmlClass"]),
        "minWidth": manifest.get("minWidth", DEFAULT_MIN_WIDTH),
        "tone": str(manifest.get("tone", USER_TONE)),
        "reviewed": not is_private(package),
        "ownsPages": sorted(pages) if isinstance(pages, dict) else [],
        "routes": list(manifest.get("routes", [])),
        "nav": [dict(item) for item in manifest.get("nav", []) if isinstance(item, dict)],
        "slots": list(manifest.get("slots", [])),
    }


def theme_boot(
    package: ThemePackage, prepaint: Optional[Dict[str, Any]], looks: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """The boot of a theme that runs: its base's classes and its own, its tone, prepaint, looks, integrity and entry.

    ``looks`` is ``theme_looks``' answer; without it the boot paints the default look only.
    """
    manifest = package.manifest or {}
    classes = [*_base_classes(manifest), str(manifest["htmlClass"])]
    boot = _boot(classes, (), str(manifest.get("tone", USER_TONE)))
    chosen = {field: (looks or {}).get(field) for field in LOOK_BOOT_FIELDS}
    return {
        **boot, **chosen,
        "prepaint": prepaint, "integrity": module_integrity(package), "entry": theme_entry(package),
    }


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
    return theme_boot(package, prepaint, theme_looks(package, reader))


__all__ = [
    "BootTarget", "FileReader", "ThemeFileMissing", "ThemeStorageUnavailable", "authorised_package", "base_boot",
    "belongs_to_install", "boot_looks", "boot_target", "file_cache_seconds", "install_boot", "integrity_value", "is_private",
    "look_storage", "may_run", "module_integrity", "module_url", "read_stored_file", "theme_boot", "theme_entry",
    "theme_look_schema", "theme_looks", "theme_prepaint", "verified_file",
]
