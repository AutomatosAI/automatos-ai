"""PRD-251 D9: where a post's rendered files live.

Rendered files go to the platform object store (S3 in SaaS, MinIO locally,
through ``core.storage``) under ``social-media/{workspace}/{post}/{file}``, in
the documents bucket. The app serves them back through
``GET /api/socials/posts/{post}/media/{file}``: that route is the Deliverable's
stable preview link, and it streams the object, never a presigned URL that
expires. Channels that fetch media by URL get presigned links at publish time
(Wave 3). A render's own inputs kept here (a voice toolkit's lines, S1.5,
``voice-<line>.<ext>``) reach media-render through short presigned links
(``presigned_get``).

A file name is lowercase letters, digits, ``.``, ``_`` and ``-`` only, so a
key can never climb out of its post's prefix.
"""
from __future__ import annotations

import logging
import mimetypes
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from config import config
from core.storage import ensure_bucket, get_s3_client, is_storage_configured

logger = logging.getLogger(__name__)

MEDIA_KEY_PREFIX = "social-media"
FILE_NAME = re.compile(r"^[a-z0-9][a-z0-9._-]{0,119}$")
STREAM_CHUNK_BYTES = 1 << 16
DEFAULT_CONTENT_TYPE = "application/octet-stream"
_MISSING_KEY_CODES = frozenset({"NoSuchKey", "404", "NotFound"})


class MediaNameError(ValueError):
    """A file name that cannot name a post's media."""


def valid_file_name(name: Any) -> bool:
    return isinstance(name, str) and bool(FILE_NAME.match(name)) and ".." not in name


def media_key(workspace_id: Any, post_id: Any, file_name: str) -> str:
    """``social-media/{workspace}/{post}/{file}`` (D9)."""
    if not valid_file_name(file_name):
        raise MediaNameError(f"{file_name!r} is not a media file name")
    return f"{MEDIA_KEY_PREFIX}/{workspace_id}/{post_id}/{file_name}"


def media_route(post_id: Any, file_name: str) -> str:
    """The app path that serves a rendered file: the Deliverable's preview link."""
    return f"/api/socials/posts/{post_id}/media/{file_name}"


def content_type_for(file_name: str) -> str:
    return mimetypes.guess_type(file_name)[0] or DEFAULT_CONTENT_TYPE


@dataclass(frozen=True)
class MediaObject:
    """A stored file opened for streaming."""

    body: Iterator[bytes]
    content_type: str
    content_length: int


def _error_code(exc: Exception) -> str:
    response = getattr(exc, "response", None) or {}
    return str(response.get("Error", {}).get("Code", ""))


def _iter_body(body: Any, chunk_size: int) -> Iterator[bytes]:
    try:
        yield from body.iter_chunks(chunk_size)
    finally:
        body.close()


class MediaStore:
    """The documents bucket, for rendered social media. Blocking: run it off the loop."""

    def __init__(self, bucket: Optional[str] = None) -> None:
        self.bucket = bucket or config.S3_DOCUMENTS_BUCKET

    @staticmethod
    def configured() -> bool:
        return is_storage_configured()

    def put_file(self, key: str, path: Path, content_type: str) -> None:
        ensure_bucket(self.bucket)
        with path.open("rb") as handle:
            get_s3_client().put_object(Bucket=self.bucket, Key=key, Body=handle, ContentType=content_type)
        logger.info("[Socials] stored s3://%s/%s (%d bytes)", self.bucket, key, path.stat().st_size)

    def presigned_get(self, key: str, ttl_seconds: int) -> str:
        """A GET link to ``key`` for media-render, which fetches a render's inputs
        from our storage (a voice toolkit's lines, S1.5). It is minted against the
        backend's own endpoint (``S3_ENDPOINT_URL``: MinIO on the compose network,
        AWS in SaaS), the one media-render's storage allowlist names; never the
        public endpoint a browser reaches."""
        return get_s3_client().generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=ttl_seconds
        )

    def open(self, key: str) -> Optional[MediaObject]:
        """The object for streaming, or ``None`` when it is not there."""
        try:
            obj = get_s3_client().get_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            if _error_code(exc) in _MISSING_KEY_CODES:
                return None
            raise
        return MediaObject(
            body=_iter_body(obj["Body"], STREAM_CHUNK_BYTES),
            content_type=obj.get("ContentType") or DEFAULT_CONTENT_TYPE,
            content_length=int(obj["ContentLength"]),
        )
