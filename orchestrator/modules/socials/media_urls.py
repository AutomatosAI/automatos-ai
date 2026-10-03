"""PRD-251 D9 (S3.4): presigned inline links to a post's media.

A post's ``media`` names Deliverables: the ids an edit sets, or the file
records a render writes (``{aspect: [id | record]}``). :func:`resolve_post_media`
finds each one through the ``deliverables`` rows of the post's own workspace,
and the object it stores. A Socials render, its footage and a generated image
are ``storage_type='s3'`` rows whose ``file_path`` is their key in the documents
bucket (``S3_DOCUMENTS_BUCKET``), under their workspace's own prefix
(``social-media/{workspace}/...``, ``generated-images/{workspace}/...``). A
Deliverable that is deleted, stored anywhere else (a workspace file, a
generated document) or keyed outside its workspace's prefix has no link, and
says why.

Every link is a presigned GET minted through the one storage factory
(``core/storage/s3.py``). It is served inline (``ResponseContentDisposition``)
with the file's own content type (``ResponseContentType``: video/mp4,
image/jpeg, image/png), and lives ``SOCIALS_MEDIA_URL_TTL_SECONDS``.

- :func:`post_media_links`: the exact media the approval view shows
  (``GET /api/socials/posts/{id}/media``). A browser opens these links, so they
  are minted against the public endpoint (``S3_PUBLIC_ENDPOINT_URL``,
  ``http://localhost:9000`` locally; AWS itself in the hosted edition).
- :func:`public_media_url`: a link a PLATFORM fetches, for a channel action that
  takes nothing but a URL (Wave 3; today only the YouTube custom thumbnail,
  because publishing is file-first). It needs public storage
  (:func:`media_public_url_available`): AWS S3 (``S3_ENDPOINT_URL`` unset, the
  hosted edition), or ``SOCIALS_PUBLIC_MEDIA_BUCKET``, into which the object is
  first copied under ``social-media/{workspace}/{post}/{file}``. With neither it
  raises :class:`NeedsPublicStorage`: "Needs public storage", which the channel
  registry shows on the step.

A SigV4 link is bound to its HTTP method: a HEAD needs a link signed for HEAD
(``method="HEAD"``). A GET link answers a Range request (206), which a video
player and a platform's fetcher both send.

The generated-images route is never used for Socials media: its 1000-key
listing and its missing Range support are why D9 exists.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any, Dict, List, Mapping, Optional, Tuple
from uuid import UUID

import sqlalchemy as sa

from config import config
from core.storage import ensure_bucket, get_public_s3_client, get_s3_client, is_storage_configured
from modules.socials.credits import MAX_MEDIA_IDS, media_entries
from modules.socials.media_store import MediaNameError, content_type_for, media_key
from modules.socials.service import SocialsError

logger = logging.getLogger(__name__)

INLINE = "inline"
GET = "GET"
# The storage type of a Deliverable whose file_path is a key in the documents bucket.
S3_STORAGE = "s3"
NEEDS_PUBLIC_STORAGE = "Needs public storage"
NOT_IN_DELIVERABLES = "This file is no longer in Deliverables."
NOT_IN_STORAGE = "This file is not in object storage, so it has no link."

_FILES = sa.text(
    """
    SELECT d.id, d.storage_type, d.file_path, d.file_name, d.file_size_bytes
      FROM deliverables d
     WHERE d.workspace_id = :workspace_id
       AND d.deleted_at IS NULL
       AND d.id IN :ids
    """
).bindparams(sa.bindparam("ids", expanding=True))


class MediaHostingError(SocialsError):
    """A post's file cannot be linked."""


class NeedsPublicStorage(MediaHostingError):
    """A platform must fetch the file, and no storage it can reach is configured."""

    def __init__(self) -> None:
        super().__init__(
            f"{NEEDS_PUBLIC_STORAGE}: this step gives the platform a link to fetch, and this "
            "instance's object storage is private. Set SOCIALS_PUBLIC_MEDIA_BUCKET to a bucket "
            "the platform can reach."
        )


class MediaUnavailable(MediaHostingError):
    """The file has no stored object to link to."""


@dataclass(frozen=True)
class MediaFile:
    """One file a post's media names, resolved in the post's workspace.

    ``key`` is its object in the documents bucket; ``None`` means it has no
    link, and ``error`` says why.
    """

    aspect: str
    deliverable_id: str
    name: Optional[str] = None
    key: Optional[str] = None
    content_type: Optional[str] = None
    bytes: Optional[int] = None
    error: Optional[str] = None

    def link(self, url: Optional[str]) -> Dict[str, Any]:
        """The approval view's entry for this file, with ``url`` its link."""
        return {
            "aspect": self.aspect,
            "deliverable_id": self.deliverable_id,
            "name": self.name,
            "url": url,
            "content_type": self.content_type,
            "bytes": self.bytes,
            "error": self.error,
        }


def media_public_url_available() -> bool:
    """Whether a platform can fetch a link to our media: AWS S3 (no custom
    endpoint, the hosted edition), or a public bucket configured for it (D9)."""
    if not is_storage_configured():
        return False
    return not config.S3_ENDPOINT_URL or bool(config.SOCIALS_PUBLIC_MEDIA_BUCKET)


def presigned_inline_url(client: Any, bucket: str, key: str, content_type: str, *, method: str = GET) -> str:
    """A presigned link to ``bucket``/``key``, served inline as ``content_type``,
    living ``SOCIALS_MEDIA_URL_TTL_SECONDS``, valid for ``method`` only."""
    return client.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": bucket,
            "Key": key,
            "ResponseContentDisposition": INLINE,
            "ResponseContentType": content_type,
        },
        ExpiresIn=config.SOCIALS_MEDIA_URL_TTL_SECONDS,
        HttpMethod=method,
    )


def _canonical(value: Any) -> str:
    return str(UUID(str(value)))


def _workspace_key(row: Any, workspace_id: Any) -> Optional[str]:
    """The row's key in the documents bucket, when it is one of the workspace's
    own objects there (``{prefix}/{workspace}/.../{file}``): a row can never
    point a link at another workspace's object."""
    key = row.file_path or ""
    parts = key.split("/")
    if row.storage_type != S3_STORAGE or len(parts) < 3 or not all(parts) or ".." in parts:
        return None
    return key if parts[1] == str(workspace_id) else None


def _media_file(aspect: str, ident: str, row: Any, workspace_id: Any) -> MediaFile:
    if row is None:
        return MediaFile(aspect=aspect, deliverable_id=ident, error=NOT_IN_DELIVERABLES)
    name = row.file_name or PurePosixPath(row.file_path or "").name or None
    size = int(row.file_size_bytes) if row.file_size_bytes is not None else None
    content_type = content_type_for(name) if name else None
    key = _workspace_key(row, workspace_id)
    return MediaFile(
        aspect=aspect,
        deliverable_id=ident,
        name=name,
        key=key,
        content_type=content_type,
        bytes=size,
        error=None if key else NOT_IN_STORAGE,
    )


def deliverable_file(db: Any, workspace_id: Any, deliverable_id: Any, aspect: str = "") -> Optional[MediaFile]:
    """One of the workspace's Deliverables as a file (its object in the documents bucket,
    ``key`` None and why when it has none), or ``None`` when the workspace has no such
    Deliverable, a deleted one or another workspace's included."""
    try:
        ident = _canonical(deliverable_id)
    except (TypeError, ValueError):
        return None
    row = db.execute(_FILES, {"workspace_id": str(workspace_id), "ids": [ident]}).first()
    return _media_file(aspect, ident, row, workspace_id) if row is not None else None


def resolve_post_media(db: Any, post: Any) -> List[MediaFile]:
    """Each file the post's media names, in media order, found through the
    ``deliverables`` rows of the post's own workspace (deleted ones excluded)."""
    entries = media_entries(post.media)[:MAX_MEDIA_IDS]
    if not entries:
        return []
    ids = list(dict.fromkeys(ident for _, ident in entries))
    rows = db.execute(_FILES, {"workspace_id": str(post.workspace_id), "ids": ids}).fetchall()
    found: Mapping[str, Any] = {_canonical(row.id): row for row in rows}
    return [_media_file(aspect, ident, found.get(ident), post.workspace_id) for aspect, ident in entries]


def post_media_links(db: Any, post: Any) -> List[Dict[str, Any]]:
    """The post's media for the approval view: each file with a presigned
    inline link a browser opens, or ``url`` None and why. Storage is touched
    only when a file has a stored object (``StorageNotConfigured`` otherwise
    propagates: the caller says storage is unavailable)."""
    files = resolve_post_media(db, post)
    if not any(f.key for f in files):
        return [f.link(None) for f in files]
    client = get_public_s3_client()
    bucket = config.S3_DOCUMENTS_BUCKET
    return [f.link(presigned_inline_url(client, bucket, f.key, f.content_type) if f.key else None) for f in files]


def _public_copy(post: Any, media: MediaFile) -> Tuple[str, str]:
    """Copy ``media`` into the public media bucket under
    ``social-media/{workspace}/{post}/{file}``: where its link points."""
    bucket = config.SOCIALS_PUBLIC_MEDIA_BUCKET
    try:
        key = media_key(post.workspace_id, post.id, PurePosixPath(media.key).name)
    except MediaNameError as exc:
        raise MediaUnavailable(f"{media.name or media.deliverable_id} cannot be published by link: {exc}") from exc
    ensure_bucket(bucket)
    get_s3_client().copy_object(
        Bucket=bucket, Key=key, CopySource={"Bucket": config.S3_DOCUMENTS_BUCKET, "Key": media.key}
    )
    logger.info("[Socials] copied s3://%s/%s to the public media bucket %s", config.S3_DOCUMENTS_BUCKET, media.key, bucket)
    return bucket, key


def public_media_url(post: Any, media: MediaFile) -> str:
    """A presigned inline link to ``media`` that a platform can fetch (D9).

    Raises :class:`MediaUnavailable` for a file with no stored object, and
    :class:`NeedsPublicStorage` when no storage a platform can reach is
    configured. With ``SOCIALS_PUBLIC_MEDIA_BUCKET`` set, the object is copied
    there first and linked from there, through the public endpoint.
    """
    if media.key is None:
        raise MediaUnavailable(media.error or NOT_IN_STORAGE)
    if not media_public_url_available():
        raise NeedsPublicStorage()
    if config.SOCIALS_PUBLIC_MEDIA_BUCKET:
        bucket, key = _public_copy(post, media)
    else:
        bucket, key = config.S3_DOCUMENTS_BUCKET, media.key
    return presigned_inline_url(get_public_s3_client(), bucket, key, media.content_type)
