"""PRD-259 §5: where a skin's files live in the marketplace bucket.

Every client comes from ``core.storage`` (the calls ``core/services/marketplace_s3.py``
makes): AWS S3 in SaaS, MinIO locally through the same code path. No other client.

* A Marketplace skin: ``skins/{slug}/{version}/`` (PRD §5).
* A private import: ``skins/_workspace/{workspace_id}/{slug}/{version}/`` (decision K3),
  so two workspaces importing the same slug never share or overwrite files.

These calls block (the S3 client is synchronous): call them from a plain-``def`` route,
which FastAPI runs in its threadpool, or through ``run_in_executor``; never directly on
the event loop.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Mapping, NamedTuple, Optional
from uuid import UUID

from config import config
from core.storage import ensure_bucket, get_s3_client
from services.skins.schema import SLUG_MAX, SLUG_PATTERN, VERSION_MAX, VERSION_PATTERN

logger = logging.getLogger(__name__)

SKINS_ROOT = "skins"
WORKSPACE_SEGMENT = "_workspace"


class StoredFile(NamedTuple):
    """One file of a skin package, with the content type its bytes were checked as."""

    data: bytes
    content_type: str


def _checked_slug_version(slug: str, version: str) -> None:
    if len(slug or "") > SLUG_MAX or not SLUG_PATTERN.fullmatch(slug or ""):
        raise ValueError(f"Not a skin slug: {slug!r}")
    if len(version or "") > VERSION_MAX or not VERSION_PATTERN.fullmatch(version or ""):
        raise ValueError(f"Not a skin version: {version!r}")


def skin_prefix(slug: str, version: str, workspace_id: Optional[UUID] = None) -> str:
    """The key prefix of a skin's files: a Marketplace skin's, or one workspace's import."""
    _checked_slug_version(slug, version)
    if workspace_id is None:
        return f"{SKINS_ROOT}/{slug}/{version}/"
    return f"{SKINS_ROOT}/{WORKSPACE_SEGMENT}/{UUID(str(workspace_id))}/{slug}/{version}/"


def _is_plain_relative(path: str) -> bool:
    """A relative POSIX path in its normal form: no '..', '.', empty segment, backslash or control character."""
    if not path or "\\" in path or any(ord(ch) < 32 or ord(ch) == 127 for ch in path):
        return False
    parts = path.split("/")
    return all(part not in ("", ".", "..") for part in parts)


def skin_key(prefix: str, path: str) -> str:
    """The object key of ``path`` (a package-relative path) under ``prefix``."""
    if not _is_plain_relative(path):
        raise ValueError(f"Unsafe skin asset path: {path!r}")
    root, _sep, rest = prefix.rstrip("/").partition("/")
    if root != SKINS_ROOT or not prefix.endswith("/") or not _is_plain_relative(rest):
        raise ValueError(f"Not a skin prefix: {prefix!r}")
    return f"{prefix}{path}"


def write_skin_files(prefix: str, files: Mapping[str, StoredFile]) -> List[str]:
    """Store each file under ``prefix`` (blocking); returns the keys written.

    Every path is checked before the first write, so an unsafe one writes nothing.
    """
    keyed = [(skin_key(prefix, path), stored) for path, stored in files.items()]
    bucket = config.MARKETPLACE_S3_BUCKET
    ensure_bucket(bucket)
    client = get_s3_client()
    for key, stored in keyed:
        client.put_object(Bucket=bucket, Key=key, Body=stored.data, ContentType=stored.content_type)
    keys = [key for key, _stored in keyed]
    logger.info("Stored %d skin files under s3://%s/%s", len(keys), bucket, prefix)
    return keys


def read_skin_file(prefix: str, path: str) -> bytes:
    """The bytes of one stored skin file (blocking)."""
    response = get_s3_client().get_object(Bucket=config.MARKETPLACE_S3_BUCKET, Key=skin_key(prefix, path))
    return response["Body"].read()


def delete_skin_files(prefix: str, paths: Iterable[str]) -> List[str]:
    """Remove the files at ``paths`` under ``prefix`` (blocking); returns the keys removed.

    Every path is checked before the first delete, like :func:`write_skin_files`.
    """
    keys = [skin_key(prefix, path) for path in paths]
    client = get_s3_client()
    for key in keys:
        client.delete_object(Bucket=config.MARKETPLACE_S3_BUCKET, Key=key)
    logger.info("Removed %d skin files under s3://%s/%s", len(keys), config.MARKETPLACE_S3_BUCKET, prefix)
    return keys
