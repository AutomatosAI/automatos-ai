"""PRD-261 §5 — where a theme package's files live in the marketplace bucket.

Started from PRD-259's skin storage (#1136, ``services/skins/storage.py:45-103``). Every client
comes from ``core.storage``: AWS S3 in SaaS, MinIO locally through the same code path.

* A Marketplace package: ``themes/<publisher>/<slug>/<version>/``.
* A private upload: ``themes/_workspace/<workspace_id>/<publisher>/<slug>/<version>/``, so two
  workspaces uploading the same theme never share or overwrite files.

These calls block (the S3 client is synchronous): call them from a plain-``def`` route, which
FastAPI runs in its threadpool, never directly on the event loop.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Mapping, Optional
from uuid import UUID

from config import config
from core.storage import ensure_bucket, get_s3_client
from services.themes.manifest import ID_PATTERN, VERSION_PATTERN
from services.themes.package import ThemeFile

logger = logging.getLogger(__name__)

THEMES_ROOT = "themes"
WORKSPACE_SEGMENT = "_workspace"


def _checked_id_version(theme_id: str, version: str) -> None:
    if not ID_PATTERN.fullmatch(theme_id or ""):
        raise ValueError(f"Not a theme id: {theme_id!r}")
    if not VERSION_PATTERN.fullmatch(version or ""):
        raise ValueError(f"Not a theme version: {version!r}")


def theme_prefix(theme_id: str, version: str, workspace_id: Optional[UUID] = None) -> str:
    """The key prefix of one package version's files: the Marketplace's, or one workspace's upload."""
    _checked_id_version(theme_id, version)
    if workspace_id is None:
        return f"{THEMES_ROOT}/{theme_id}/{version}/"
    return f"{THEMES_ROOT}/{WORKSPACE_SEGMENT}/{UUID(str(workspace_id))}/{theme_id}/{version}/"


def _is_plain_relative(path: str) -> bool:
    """A relative POSIX path in its normal form: no '..', '.', empty segment, backslash or control character."""
    if not path or "\\" in path or any(ord(ch) < 32 or ord(ch) == 127 for ch in path):
        return False
    return all(part not in ("", ".", "..") for part in path.split("/"))


def theme_key(prefix: str, path: str) -> str:
    """The object key of ``path`` (a package-relative path) under ``prefix``."""
    if not _is_plain_relative(path):
        raise ValueError(f"Unsafe theme file path: {path!r}")
    root, _sep, rest = prefix.rstrip("/").partition("/")
    if root != THEMES_ROOT or not prefix.endswith("/") or not _is_plain_relative(rest):
        raise ValueError(f"Not a theme prefix: {prefix!r}")
    return f"{prefix}{path}"


def _remove_written(client, bucket: str, keys: List[str]) -> None:
    """Undo a part-written upload. A failure here leaves only unreferenced files: logged, not raised."""
    for key in keys:
        try:
            client.delete_object(Bucket=bucket, Key=key)
        except Exception:
            logger.exception("[Themes] Removing the part-written file %s failed", key)


def write_theme_files(prefix: str, files: Mapping[str, ThemeFile]) -> List[str]:
    """Store each file under ``prefix`` (blocking); returns the keys written.

    Every path is checked before the first write, so an unsafe one writes nothing. When a write
    fails, the files this call already wrote are removed and the error is raised.
    """
    keyed = [(theme_key(prefix, path), stored) for path, stored in sorted(files.items())]
    bucket = config.MARKETPLACE_S3_BUCKET
    ensure_bucket(bucket)
    client = get_s3_client()
    written: List[str] = []
    for key, stored in keyed:
        try:
            client.put_object(Bucket=bucket, Key=key, Body=stored.data, ContentType=stored.content_type)
        except Exception:
            logger.exception("[Themes] Storing %s failed after %d of %d files", key, len(written), len(keyed))
            _remove_written(client, bucket, written)
            raise
        written = [*written, key]
    logger.info("[Themes] Stored %d theme files under s3://%s/%s", len(written), bucket, prefix)
    return written


def read_theme_file(prefix: str, path: str) -> bytes:
    """The bytes of one stored theme file (blocking)."""
    response = get_s3_client().get_object(Bucket=config.MARKETPLACE_S3_BUCKET, Key=theme_key(prefix, path))
    return response["Body"].read()


def delete_theme_files(prefix: str, paths: Iterable[str]) -> List[str]:
    """Remove the files at ``paths`` under ``prefix`` (blocking); returns the keys removed.

    Every path is checked before the first delete, like :func:`write_theme_files`.
    """
    keys = [theme_key(prefix, path) for path in paths]
    client = get_s3_client()
    for key in keys:
        client.delete_object(Bucket=config.MARKETPLACE_S3_BUCKET, Key=key)
    logger.info("[Themes] Removed %d theme files under s3://%s/%s", len(keys), config.MARKETPLACE_S3_BUCKET, prefix)
    return keys
