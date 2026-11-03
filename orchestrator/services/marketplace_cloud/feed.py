"""PRD-270 K14 — the Automatos Cloud catalogue as a local install reads it: checked whole, then item by item.

``parse_catalog`` refuses the whole catalogue (``CatalogError``) when it is larger than
``AUTOMATOS_CLOUD_FEED_MAX_KB``, isn't UTF-8 JSON, isn't schema 1 (``{schema: 1, generatedAt, items}``)
or lists more than ``CATALOG_MAX_ITEMS`` items. Then each item is checked on its own, and one that fails
is refused with a plain reason and counted, never listed:

* its ``type`` is ``'theme'``. An item of a bundle type (an agent, a Playbook, a package or a plugin,
  PRD-270's second wave) is read as an item entry instead (``items_feed.Entries.read``, on ``entries``).
  Any other type name is SKIPPED, neither refused nor a reason to fail the Sync; an item with no
  ``type``, or one that isn't a name, is refused;
* its id, version and digest are ones a theme can have;
* its ``package`` is exactly the catalogue's files path for its digest, or an https URL under the Design
  Kit's releases (§5.4). It is only checked: a package is fetched from a URL derived from the setting
  and the digest (``urls.package_url``), never from the catalogue;
* ``files`` is a file manifest a package can have (each path a theme file, its type the one that path
  is served as, its size and sha256), holding theme.json, and ``digest_of_manifest(files)`` is the digest;
* ``themeJson`` (base64) hashes to ``files['theme.json']`` and passes ``parse_json`` and
  ``validate_manifest`` (``package._manifest``'s reading) with the item's id and version. A version whose
  ``engines`` this host doesn't meet fails it too (it could never install here), with its own reason;
* ``check_upload_signature`` verifies its envelope over that manifest, file list and digest, as an
  upload's is checked (``CheckedThemePackage(manifest, {}, files, digest)``): an active key this install
  ships, the key the envelope names, and not a key the revocation feed revoked.

A signature covers the id, version and file hashes, and theme.json is one of the hashed files, so
everything a Cloud row keeps (the manifest, the file list, the signature) is bound by it. The item's
display fields (name, cover, presets and the rest) are read from nothing here. The keys file is read once
per catalogue; one that does not load raises ``signature_verify.SigningKeysError``: the Sync fails, it
never lists less.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

from config import config
from services.marketplace_bundles.items_feed import NO_ENTRIES, Entries, is_bundle_item
from services.marketplace_cloud.urls import package_path
from services.themes import signature_verify
from services.themes.contract import KB, MANIFEST_PATH, parse_json
from services.themes.digest import digest_of_manifest, sha256_hex
from services.themes.manifest import ID_PATTERN, VERSION_PATTERN, validate_manifest
from services.themes.package import JSON_MAX_BYTES, MAX_MEMBERS, ThemePackage as CheckedThemePackage, file_kind
from services.themes.signed_upload import SHA256_HEX, SignatureRefused, check_upload_signature

logger = logging.getLogger(__name__)

CATALOG_SCHEMA = 1
THEME_TYPE = "theme"
CATALOG_MAX_ITEMS = 1000
FILE_FIELDS = frozenset({"type", "size", "sha256"})
# The one manifest refusal that means "a newer host", not "a bad manifest".
ENGINES_MISMATCH = "ENGINES_MISMATCH"
KIT_RELEASES = "https://github.com/AutomatosAI/automatos-design-kit/releases/download/"
# A release's tag and asset name: two plain segments, so no '..', query, fragment or other host.
RELEASE_ASSET = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}/[A-Za-z0-9][A-Za-z0-9._+-]{0,127}")

TOO_LARGE = "The Automatos Cloud catalogue is larger than {kb} KB."
NOT_JSON = "The Automatos Cloud catalogue isn't UTF-8 JSON."
NOT_SCHEMA = "The Automatos Cloud catalogue isn't one this install reads (schema 1)."
TOO_MANY = "The Automatos Cloud catalogue lists more than {count} items."

NOT_AN_OBJECT = "It isn't a JSON object."
NO_TYPE = "It has no type."
BAD_TYPE = "Its type isn't a name."
BAD_IDENTITY = "Its id, version or digest isn't one a theme can have."
BAD_PACKAGE = "Its package isn't on Automatos Cloud's files route or the Design Kit's releases."
BAD_FILES = "Its file list isn't one a theme package can have."
DIGEST_MISMATCH = "Its file list doesn't add up to its digest."
THEME_JSON_MISMATCH = "Its theme.json doesn't match its file list."
BAD_MANIFEST = "Its theme.json isn't a valid theme manifest."
NEEDS_NEWER_HOST = "It needs a newer Automatos than this install runs."
OTHER_THEME = "Its theme.json is for another theme or version."
BAD_ENVELOPE = "Its signature isn't a signature envelope."
NOT_SIGNED = "Its signature isn't one an active Automatos key made for these files ({code})."
REFUSED_ITEM = "Item {number}: {reason}"


class CatalogError(ValueError):
    """The whole catalogue was refused; the message is plain and echoes nothing it held."""


class ItemRefused(ValueError):
    """One item was refused; the message is plain and echoes nothing it held."""


class CheckedItem(NamedTuple):
    """A theme version whose signature verified: what a Cloud row keeps."""

    theme_id: str
    version: str
    digest: str
    manifest: Dict[str, Any]
    files: Dict[str, Dict[str, Any]]
    signature: str


class Catalog(NamedTuple):
    """A checked catalogue: its time, the items that verified, why the others were refused, how many were skipped,
    and its item entries (``items_feed``)."""

    generated_at: datetime
    items: Tuple[CheckedItem, ...]
    refused: Tuple[str, ...]
    skipped: int
    entries: Entries = NO_ENTRIES


def feed_max_bytes() -> int:
    """The largest catalogue a local install reads (``AUTOMATOS_CLOUD_FEED_MAX_KB``)."""
    return int(config.AUTOMATOS_CLOUD_FEED_MAX_KB) * KB


# --------------------------------------------------------------------------- the whole catalogue


def parse_catalog(data: bytes, db: Any) -> Catalog:
    """The catalogue's bytes, refused whole (``CatalogError``) or checked item by item.

    ``db`` gives the keys the revocation feed revoked (``check_upload_signature``).
    """
    limit = feed_max_bytes()
    if len(data) > limit:
        raise CatalogError(TOO_LARGE.format(kb=limit // KB))
    raw = _json_object(data)
    items = raw.get("items")
    if not _is_schema_one(raw.get("schema")) or not isinstance(items, list):
        raise CatalogError(NOT_SCHEMA)
    if len(items) > CATALOG_MAX_ITEMS:
        raise CatalogError(TOO_MANY.format(count=CATALOG_MAX_ITEMS))
    generated_at = _generated_at(raw.get("generatedAt"))
    keys = signature_verify.load_keys()  # once per catalogue; a keys file that doesn't load fails the Sync
    checked: List[CheckedItem] = []
    refused: List[str] = []
    entries = NO_ENTRIES
    skipped = 0
    for number, item in enumerate(items, start=1):
        if is_bundle_item(item):
            entries = entries.read(item)
            continue
        if _is_other_type(item):
            skipped += 1
            continue
        try:
            checked.append(check_item(db, item, keys))
        except ItemRefused as refusal:
            logger.warning("Automatos Cloud catalogue: item %d refused: %s", number, refusal)
            refused.append(REFUSED_ITEM.format(number=number, reason=refusal))
    return Catalog(generated_at, tuple(checked), tuple(refused), skipped, entries)


def _json_object(data: bytes) -> Dict[str, Any]:
    try:
        raw = json.loads(data.decode("utf-8"))
    except (ValueError, RecursionError) as bad:  # UnicodeDecodeError is a ValueError
        raise CatalogError(NOT_JSON) from bad
    if not isinstance(raw, dict):
        raise CatalogError(NOT_SCHEMA)
    return raw


def _is_schema_one(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == CATALOG_SCHEMA


def _generated_at(value: Any) -> datetime:
    """``generatedAt``: an ISO 8601 time with an offset, in UTC."""
    try:
        parsed = datetime.fromisoformat(value) if isinstance(value, str) else None
    except ValueError:
        parsed = None
    if parsed is None or parsed.tzinfo is None:
        raise CatalogError(NOT_SCHEMA)
    return parsed.astimezone(timezone.utc)


def _is_other_type(item: Any) -> bool:
    """True for an item named as a type this wave doesn't list (skipped); one with no type name is refused instead."""
    return isinstance(item, dict) and isinstance(item.get("type"), str) and item["type"] != THEME_TYPE


# --------------------------------------------------------------------------- one item


def check_item(db: Any, item: Any, keys: Optional[Sequence[signature_verify.SigningKey]] = None) -> CheckedItem:
    """One theme item, every K14 check in order; ``ItemRefused`` with a plain reason on the first that fails.

    ``keys`` defaults to the keys file.
    """
    if not isinstance(item, dict):
        raise ItemRefused(NOT_AN_OBJECT)
    if "type" not in item:
        raise ItemRefused(NO_TYPE)
    if item["type"] != THEME_TYPE:
        raise ItemRefused(BAD_TYPE)
    theme_id, version, digest = _identity(item)
    if not _is_package_url(item.get("package"), digest):
        raise ItemRefused(BAD_PACKAGE)
    files = _files(item.get("files"))
    if digest_of_manifest(files) != digest:
        raise ItemRefused(DIGEST_MISMATCH)
    manifest = _manifest(item.get("themeJson"), files)
    if manifest.get("id") != theme_id or manifest.get("version") != version:
        raise ItemRefused(OTHER_THEME)
    checked = CheckedThemePackage(manifest, {}, files, digest)
    signature = _verified_signature(db, item.get("signature"), checked, keys)
    return CheckedItem(theme_id, version, digest, manifest, files, signature)


def _identity(item: Mapping[str, Any]) -> Tuple[str, str, str]:
    theme_id, version, digest = item.get("id"), item.get("version"), item.get("digest")
    ok = all(isinstance(value, str) for value in (theme_id, version, digest))
    if not ok or not (ID_PATTERN.fullmatch(theme_id) and VERSION_PATTERN.fullmatch(version)
                      and SHA256_HEX.fullmatch(digest)):
        raise ItemRefused(BAD_IDENTITY)
    return theme_id, version, digest


def _is_package_url(value: Any, digest: str) -> bool:
    """The catalogue's own files path for this digest, or a Design Kit release asset (§5.4's allowlist)."""
    if not isinstance(value, str):
        return False
    if value == package_path(digest):
        return True
    return value.startswith(KIT_RELEASES) and RELEASE_ASSET.fullmatch(value[len(KIT_RELEASES):]) is not None


def _file_row(path: Any, row: Any) -> Dict[str, Any]:
    """One file's row, held to what that path is in a package: its served type, a size under its cap, a sha256."""
    kind = file_kind(path) if isinstance(path, str) else None
    if kind is None or not isinstance(row, dict) or set(row) != FILE_FIELDS:
        raise ItemRefused(BAD_FILES)
    size, sha256 = row["size"], row["sha256"]
    if row["type"] != kind.content_type or not isinstance(size, int) or isinstance(size, bool):
        raise ItemRefused(BAD_FILES)
    if not 0 <= size <= kind.cap() or not isinstance(sha256, str) or not SHA256_HEX.fullmatch(sha256):
        raise ItemRefused(BAD_FILES)
    return {"type": kind.content_type, "size": size, "sha256": sha256}


def _files(value: Any) -> Dict[str, Dict[str, Any]]:
    """The item's file manifest, checked, by path; it must hold theme.json."""
    if not isinstance(value, dict) or not 0 < len(value) <= MAX_MEMBERS or MANIFEST_PATH not in value:
        raise ItemRefused(BAD_FILES)
    return {path: _file_row(path, value[path]) for path in sorted(value)}


def _theme_json_bytes(value: Any, listed: Mapping[str, Any]) -> bytes:
    """``themeJson`` decoded, when it is exactly the theme.json the file list names."""
    if not isinstance(value, str) or len(value) > 2 * JSON_MAX_BYTES:
        raise ItemRefused(THEME_JSON_MISMATCH)
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as bad:
        raise ItemRefused(THEME_JSON_MISMATCH) from bad
    if len(raw) != listed["size"] or sha256_hex(raw) != listed["sha256"]:
        raise ItemRefused(THEME_JSON_MISMATCH)
    return raw


def _manifest(value: Any, files: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    """The checked theme.json (``package._manifest``'s reading)."""
    document, error = parse_json(_theme_json_bytes(value, files[MANIFEST_PATH]), MANIFEST_PATH)
    if error is not None:
        raise ItemRefused(BAD_MANIFEST)
    result = validate_manifest(document)
    if result.errors or result.manifest is None:
        newer_host = bool(result.errors) and all(error["code"] == ENGINES_MISMATCH for error in result.errors)
        raise ItemRefused(NEEDS_NEWER_HOST if newer_host else BAD_MANIFEST)
    return result.manifest


def _verified_signature(
    db: Any, envelope: Any, checked: CheckedThemePackage, keys: Optional[Sequence[signature_verify.SigningKey]],
) -> str:
    """The envelope's signature, once ``check_upload_signature`` verifies it for these files."""
    if not isinstance(envelope, dict):
        raise ItemRefused(BAD_ENVELOPE)
    try:
        verified = check_upload_signature(db, checked, json.dumps(envelope).encode("utf-8"), keys)
    except SignatureRefused as refused:
        raise ItemRefused(NOT_SIGNED.format(code=refused.error["code"])) from refused
    return verified.signature


__all__ = [
    "CATALOG_MAX_ITEMS", "CATALOG_SCHEMA", "KIT_RELEASES", "THEME_TYPE", "Catalog", "CatalogError", "CheckedItem",
    "ItemRefused", "check_item", "feed_max_bytes", "parse_catalog",
]
