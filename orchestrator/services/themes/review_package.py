"""PRD-261 C-03c — a short-lived link to a Marketplace version's package, for the kit's review job.

The kit's isolated review (automatos-design-kit ``.github/workflows/theme-review.yml``) downloads one
version from an ``https://`` ``package_url`` and checks it against ``expected_digest``. Play's
release asset is such a URL; for any other submission ``review_link`` gives one: it rebuilds the
version's ``.autotheme`` from its stored files, writes it to ``themes/_review/<digest>.autotheme``
in the Marketplace bucket and answers ``{url, digest, expiresAt}``, a presigned GET that lasts
``REVIEW_LINK_SECONDS``.

* A version that isn't a Marketplace submission answers 404, as a missing one does
  (``review.queued_package``).
* Every stored file must be there and match its recorded sha256, and the file list must add up to
  the row's digest, so the rebuilt package reads (``package.read_theme_package``) with the row's
  digest; anything else is 409 with a plain message.
* Storage that fails otherwise is 503, and so is a link the job would refuse: not ``https://``, or
  with the digest longer than a dispatch's inputs may be (``DISPATCH_INPUTS_MAX_CHARS``). The link
  is a bearer credential: it is never logged.

The zip is deterministic (sorted members, deflated, one fixed time and mode), so a version always
gives the same bytes under the same key. Blocking (S3): plain-``def`` routes only.
"""

from __future__ import annotations

import io
import logging
import zipfile
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Mapping, Optional
from uuid import UUID

from sqlalchemy.orm import Session

from config import config
from core.models.themes import ThemePackage
from core.storage import get_public_s3_client
from services.themes.choices import CONFLICT
from services.themes.contract import echo
from services.themes.digest import package_digest, sha256_hex
from services.themes.package import ThemeFile
from services.themes.review import ThemeReviewError, queued_package
from services.themes.serve import FileReader, ThemeStorageUnavailable, read_stored_file
from services.themes.storage import THEMES_ROOT, theme_key, write_theme_files

logger = logging.getLogger(__name__)

REVIEW_PREFIX = f"{THEMES_ROOT}/_review/"
PACKAGE_SUFFIX = ".autotheme"
PACKAGE_CONTENT_TYPE = "application/zip"
REVIEW_LINK_SECONDS = 15 * 60
LINK_SCHEME = "https://"
# GitHub caps a workflow_dispatch's inputs payload at 65,535 characters (package_url and expected_digest).
DISPATCH_INPUTS_MAX_CHARS = 65535
PRESIGNED_METHOD = "get_object"
ZIP_TIME = (1980, 1, 1, 0, 0, 0)
ZIP_FILE_MODE = 0o644 << 16
ZIP_LEVEL = 9
UNAVAILABLE = 503

FILE_LIST_BROKEN = (
    "This version's file list doesn't add up to its digest, so its package can't be rebuilt for review. Reject it."
)
FILE_MISSING = "The stored file {path} of this version is missing, so its package can't be rebuilt for review."
FILE_CHANGED = (
    "The stored file {path} of this version doesn't match its recorded hash, so its package can't be rebuilt "
    "for review."
)
STORAGE_DOWN = "Theme storage is unavailable. Try again in a moment."
LINK_UNUSABLE = "This platform's storage doesn't give a link the kit's review job accepts (https, short enough)."


def review_key(digest: str) -> str:
    """The object key of a version's rebuilt package: ``themes/_review/<digest>.autotheme``."""
    return theme_key(REVIEW_PREFIX, digest + PACKAGE_SUFFIX)


def file_hashes(package: ThemePackage) -> Dict[str, str]:
    """``{path: sha256}`` of the version's file list; 409 unless it adds up to the row's digest."""
    rows = package.file_manifest if isinstance(package.file_manifest, Mapping) else {}
    hashes = {path: row.get("sha256") for path, row in rows.items() if isinstance(row, Mapping)}
    if not hashes or len(hashes) != len(rows) or not all(isinstance(sha, str) for sha in hashes.values()):
        raise ThemeReviewError(CONFLICT, FILE_LIST_BROKEN)
    if package_digest(hashes) != package.digest:
        raise ThemeReviewError(CONFLICT, FILE_LIST_BROKEN)
    return hashes


def _stored_bytes(package: ThemePackage, path: str, sha256: str, read: FileReader) -> bytes:
    """One stored file's bytes, checked against its hash: 409 when gone or changed, 503 when storage fails."""
    try:
        data = read(package.storage_prefix, path)
    except ThemeStorageUnavailable as down:
        raise ThemeReviewError(UNAVAILABLE, STORAGE_DOWN) from down
    if data is None:
        raise ThemeReviewError(CONFLICT, FILE_MISSING.format(path=echo(path)))
    if sha256_hex(data) != sha256:
        logger.error("theme package %s: stored file %r doesn't match its recorded sha256", package.id, path)
        raise ThemeReviewError(CONFLICT, FILE_CHANGED.format(path=echo(path)))
    return data


def rebuilt_package(package: ThemePackage, read: FileReader = read_stored_file) -> bytes:
    """The version's ``.autotheme``, rebuilt from its stored files (``read``: prefix, path → bytes)."""
    hashes = file_hashes(package)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=ZIP_LEVEL) as archive:
        for path in sorted(hashes):
            info = zipfile.ZipInfo(path, date_time=ZIP_TIME)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = ZIP_FILE_MODE
            archive.writestr(info, _stored_bytes(package, path, hashes[path], read), compresslevel=ZIP_LEVEL)
    return buffer.getvalue()


def _store_copy(package: ThemePackage, data: bytes) -> str:
    """Write the rebuilt package under its review key (blocking); 503 when storage fails."""
    name = package.digest + PACKAGE_SUFFIX
    try:
        write_theme_files(REVIEW_PREFIX, {name: ThemeFile(data, PACKAGE_CONTENT_TYPE)})
    except Exception as exc:
        logger.exception("theme package %s: storing its review copy failed", package.id)
        raise ThemeReviewError(UNAVAILABLE, STORAGE_DOWN) from exc
    return review_key(package.digest)


def _presigned_get(key: str) -> str:
    """A presigned GET of ``key`` for ``REVIEW_LINK_SECONDS``, minted where the job can reach it."""
    try:
        return get_public_s3_client().generate_presigned_url(
            PRESIGNED_METHOD, Params={"Bucket": config.MARKETPLACE_S3_BUCKET, "Key": key},
            ExpiresIn=REVIEW_LINK_SECONDS,
        )
    except Exception as exc:
        logger.exception("presigning the review copy %s failed", key)
        raise ThemeReviewError(UNAVAILABLE, STORAGE_DOWN) from exc


def job_accepts(url: Any, digest: str) -> bool:
    """True when theme-review.yml takes ``url`` as its package_url: https, and short enough to dispatch."""
    return isinstance(url, str) and url.startswith(LINK_SCHEME) and len(url) + len(digest) <= DISPATCH_INPUTS_MAX_CHARS


def review_link(db: Session, package_id: UUID, now: Optional[datetime] = None) -> Dict[str, str]:
    """``{url, digest, expiresAt}`` for a Marketplace version: its rebuilt package, presigned for 15 minutes."""
    package = queued_package(db, package_id)
    key = _store_copy(package, rebuilt_package(package))
    issued = now or datetime.now(timezone.utc)
    url = _presigned_get(key)
    if not job_accepts(url, package.digest):
        logger.error("theme package %s: the presigned review link isn't one theme-review.yml accepts", package.id)
        raise ThemeReviewError(UNAVAILABLE, LINK_UNUSABLE)
    expires = issued + timedelta(seconds=REVIEW_LINK_SECONDS)
    return {"url": url, "digest": package.digest, "expiresAt": expires.isoformat()}


__all__ = [
    "DISPATCH_INPUTS_MAX_CHARS", "REVIEW_LINK_SECONDS", "REVIEW_PREFIX", "file_hashes", "job_accepts",
    "rebuilt_package", "review_key", "review_link",
]
