"""PRD-270 W2 US-270-12d (§5.6, K12) — an exported plugin's stored files, as one deterministic zip.

``GET /api/marketplace/catalog/plugins/{slug}/{version}/package.zip`` (``api/marketplace_catalog_items.py``)
answers the zip of a plugin ``export_plugins.exported_plugins`` lists with that slug at that version, and of
no other plugin, in two steps: ``exported_plugin`` (the database only) and ``built_package`` (storage
only), so the route ends its read transaction before the storage reads. A local install's Sync downloads
it from the path it builds itself (``catalog_paths.plugin_package_path``) and checks it as an upload
(US-270-15a).

* ``exported_plugin``: the listed plugin's slug, version and stored prefix. The export never reads the S3
  path (so it can never reach a bundle); this reads it for the one row the export listed, and serves it
  only when it is ``MarketplaceS3Service.extract_plugin``'s prefix, ``plugins/<slug>/<version>/``.
* ``stored_plugin_files``: every object under that prefix, listed page by page as
  ``MarketplaceS3Service.list_plugin_files`` does and read RAW through ``core.storage.get_s3_client``
  (``MarketplaceS3Service.get_file`` decodes UTF-8 and would refuse a binary file). A key ending in '/' is
  a folder marker, skipped; a path that is absolute or climbs with '..' (what ``extract_plugin`` refuses),
  is not in its normal form ('./a', 'a//b') or holds a backslash or a control character refuses the
  plugin. The total is at most ``PLUGIN_MAX_UPLOAD_SIZE_MB``: the listed sizes are checked before any
  read, and no read takes more than the room left.
* ``checked_files``: manifest.json is there, and its slug and version are the row's. The files are served
  as stored, byte for byte, manifest.json included.
* ``plugin_zip``: members sorted by path, each dated (1980, 1, 1, 0, 0, 0), deflated, with fixed
  attributes, paths relative to the prefix: the same files always give the same bytes, so the route's ETag
  (the zip's sha256) holds across requests and workers.

Files that can't be served (over the cap, an unsafe path, no or another manifest, a prefix not the
plugin's) raise :class:`PluginFilesRefused`: ``built_package`` logs it and answers None, the route's 404.
Storage that fails raises :class:`PluginFilesUnavailable` (logged), the route's 503. The zip is built on
every read, like W1's rebuilt theme package (``catalog_files.py``). Blocking (S3): plain-``def`` routes only.
"""
from __future__ import annotations

import io
import json
import logging
import pathlib
import re
import zipfile
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

from sqlalchemy.orm import Session

from config import config
from core.models.marketplace_plugins import MarketplacePlugin
from core.storage import get_s3_client
from services.marketplace_bundles.export_plugins import exported_plugins
from services.themes.digest import sha256_hex

logger = logging.getLogger(__name__)

MB = 1024 * 1024
MANIFEST_FILE = "manifest.json"
FOLDER_MARK = "/"
PARENT_PART = ".."
# A backslash (a separator to some extractors) or an ASCII control character, NUL included.
_UNSAFE_CHARACTERS = re.compile(r"[\\\x00-\x1f\x7f]")
ZIP_DATE_TIME = (1980, 1, 1, 0, 0, 0)
ZIP_CREATE_SYSTEM = 3  # Unix: ZipInfo's own default depends on the platform it runs on
ZIP_FILE_ATTRIBUTES = 0o100644 << 16  # a regular file, rw-r--r--
STORAGE_DOWN = "Plugin storage is unavailable."


class PluginFilesRefused(ValueError):
    """A listed plugin's stored files can't be served as its package (the route's 404, logged)."""


class PluginFilesUnavailable(RuntimeError):
    """Storage failed while listing or reading a plugin's files (the route's 503, logged)."""


class StoredPlugin(NamedTuple):
    """The exported plugin's slug and version, and the prefix its row says its files are under."""

    slug: str
    version: str
    s3_path: Optional[str]


class PluginPackage(NamedTuple):
    """The zip of a plugin's stored files and its sha256 (the route's ETag)."""

    data: bytes
    sha256: str


def plugin_files_max_bytes() -> int:
    """The most a plugin's files may hold in all (``PLUGIN_MAX_UPLOAD_SIZE_MB``, an upload's own cap)."""
    return int(config.PLUGIN_MAX_UPLOAD_SIZE_MB) * MB


def plugin_prefix(slug: str, version: str) -> str:
    """Where ``MarketplaceS3Service.extract_plugin`` stores a plugin version's files."""
    return f"plugins/{slug}/{version}/"


def exported_plugin(db: Session, slug: str, version: str) -> Optional[StoredPlugin]:
    """The plugin ``exported_plugins`` lists as ``slug`` at ``version``, with its row's stored prefix; None
    when it lists no such plugin (pending, unsafe, inactive, unknown, or at another version)."""
    listed = [item for item in exported_plugins(db, slug=slug) if item.entry.version == version]
    if not listed:
        return None
    row = (
        db.query(MarketplacePlugin.slug, MarketplacePlugin.version, MarketplacePlugin.s3_path)
        .filter(MarketplacePlugin.id.in_(tuple(listed[0].entry.row_ids)), MarketplacePlugin.slug == slug,
                MarketplacePlugin.version == version)
        .first()
    )
    return None if row is None else StoredPlugin(row.slug, row.version, row.s3_path)


def built_package(plugin: StoredPlugin) -> Optional[PluginPackage]:
    """The zip of an exported plugin's stored files and its sha256; None when they can't be served (logged
    as an error). Reads no row. Raises PluginFilesUnavailable when storage fails."""
    try:
        files = checked_files(plugin, stored_plugin_files(plugin))
    except PluginFilesRefused as refused:
        logger.error("Automatos Cloud plugin package: %s %s is not served: %s", plugin.slug, plugin.version, refused)
        return None
    data = plugin_zip(files)
    return PluginPackage(data, sha256_hex(data))


def stored_plugin_files(plugin: StoredPlugin) -> Dict[str, bytes]:
    """Every file under the plugin's stored prefix, by its path relative to the prefix, read raw. Raises
    PluginFilesRefused for a prefix not the plugin's, an unsafe path or a total over the cap, and
    PluginFilesUnavailable when storage fails."""
    prefix, cap = _prefix_of(plugin), plugin_files_max_bytes()
    client = _client()
    files: Dict[str, bytes] = {}
    total = 0
    for key, path in _listed(client, prefix, cap):
        data = _read(client, key, cap - total)
        total += len(data)
        if total > cap:
            raise PluginFilesRefused(f"its files are over the {cap}-byte cap")
        files[path] = data
    return files


def checked_files(plugin: StoredPlugin, files: Mapping[str, bytes]) -> Mapping[str, bytes]:
    """``files`` when they hold manifest.json naming the plugin's slug and version; else PluginFilesRefused."""
    manifest = _manifest_of(files)
    if manifest.get("slug") != plugin.slug or manifest.get("version") != plugin.version:
        raise PluginFilesRefused("its manifest.json names another slug or version")
    return files


def plugin_zip(files: Mapping[str, bytes]) -> bytes:
    """A zip of ``files`` whose bytes depend on the files alone: members sorted by path, a fixed date and
    attributes, deflated."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(files):
            archive.writestr(_member(path), files[path])
    return buffer.getvalue()


def _member(path: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(path, date_time=ZIP_DATE_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = ZIP_CREATE_SYSTEM
    info.external_attr = ZIP_FILE_ATTRIBUTES
    return info


def _prefix_of(plugin: StoredPlugin) -> str:
    """The row's stored prefix, when it is the one ``extract_plugin`` writes for its slug and version."""
    prefix = plugin_prefix(plugin.slug, plugin.version)
    if plugin.s3_path != prefix:
        raise PluginFilesRefused("its row names a storage prefix that is not the plugin's")
    return prefix


def _manifest_of(files: Mapping[str, bytes]) -> Dict[str, Any]:
    if MANIFEST_FILE not in files:
        raise PluginFilesRefused("its files hold no manifest.json")
    try:
        manifest = json.loads(files[MANIFEST_FILE].decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as unreadable:
        raise PluginFilesRefused("its manifest.json is not JSON") from unreadable
    if not isinstance(manifest, dict):
        raise PluginFilesRefused("its manifest.json is not an object")
    return manifest


def _client() -> Any:
    try:
        return get_s3_client()
    except Exception as failed:
        logger.exception("Automatos Cloud plugin package: no storage client")
        raise PluginFilesUnavailable(STORAGE_DOWN) from failed


def _listed(client: Any, prefix: str, cap: int) -> List[Tuple[str, str]]:
    """Every (key, relative path) of a file under ``prefix``, page by page; the listed sizes over ``cap``
    refuse the plugin before any read."""
    found: List[Tuple[str, str]] = []
    listed_bytes, token = 0, None
    while True:
        page = _page(client, prefix, token)
        for item in page.get("Contents", []):
            path = _relative(item["Key"], prefix)
            listed_bytes += int(item.get("Size") or 0)
            if listed_bytes > cap:
                raise PluginFilesRefused(f"its files are over the {cap}-byte cap")
            if path is not None:
                found.append((item["Key"], path))
        token = page.get("NextContinuationToken") if page.get("IsTruncated") else None
        if token is None:
            return found


def _page(client: Any, prefix: str, token: Optional[str]) -> Mapping[str, Any]:
    asked: Dict[str, Any] = {"Bucket": config.MARKETPLACE_S3_BUCKET, "Prefix": prefix}
    if token:
        asked = {**asked, "ContinuationToken": token}
    try:
        return client.list_objects_v2(**asked)
    except Exception as failed:
        logger.exception("Automatos Cloud plugin package: %r could not be listed", prefix)
        raise PluginFilesUnavailable(STORAGE_DOWN) from failed


def _relative(key: str, prefix: str) -> Optional[str]:
    """``key``'s path under ``prefix``; None for a folder marker. A key off the prefix, or a path that is
    absolute, climbs with '..', is not in its normal form or holds a backslash or a control character,
    refuses the plugin."""
    if not key.startswith(prefix):
        raise PluginFilesRefused("storage listed a key outside its prefix")
    path = key[len(prefix):]
    if not path or path.endswith(FOLDER_MARK):
        return None
    if not is_safe_path(path):
        raise PluginFilesRefused("a stored file has an unsafe path")
    return path


def is_safe_path(path: str) -> bool:
    """Whether ``path`` is a plugin file's path as an upload stores it: relative, naming a file ('.' names
    none), never climbing with '..', in its normal form ('./a' and 'a//b' are not) and with no backslash or
    control character (the local Sync holds a downloaded package's files to it too, US-270-15a)."""
    parts = pathlib.PurePosixPath(path)
    return not (
        parts.is_absolute() or not parts.name or PARENT_PART in parts.parts or str(parts) != path
        or _UNSAFE_CHARACTERS.search(path)
    )


def _read(client: Any, key: str, room: int) -> bytes:
    """The object's bytes, at most ``room + 1`` of them (one more than fits tells the caller it is over)."""
    try:
        body = client.get_object(Bucket=config.MARKETPLACE_S3_BUCKET, Key=key)["Body"]
        try:
            return body.read(room + 1)
        finally:
            body.close()
    except Exception as failed:
        logger.exception("Automatos Cloud plugin package: %r could not be read", key)
        raise PluginFilesUnavailable(STORAGE_DOWN) from failed


__all__ = [
    "PluginFilesRefused", "PluginFilesUnavailable", "PluginPackage", "StoredPlugin", "built_package",
    "checked_files", "exported_plugin", "is_safe_path", "plugin_files_max_bytes", "plugin_prefix", "plugin_zip",
    "stored_plugin_files",
]
