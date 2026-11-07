"""PRD-273 US-273-03 — where a finished export waits for its one download: object storage.

The file goes to the documents bucket under the workspace's own prefix,
``workspaces/<id>/exports/<job id>.automatos-workspace``, so any replica of the backend can
serve the download, a restart does not lose it, and the workspace purge (which wipes that
prefix) deletes it with the workspace. It is uploaded from the temporary file in parts and
read back as a stream, never whole. ``core.storage`` gives the client, on both editions
(MinIO locally, S3 hosted).
"""
from __future__ import annotations

import logging
from typing import Any, BinaryIO, Iterator
from uuid import UUID

from botocore.exceptions import ClientError

from services.workspace_move.manifest import FILE_EXTENSION

logger = logging.getLogger(__name__)

EXPORT_KEY = "workspaces/{workspace_id}/exports/{job_id}" + FILE_EXTENSION
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
_MISSING_CODES = frozenset({"404", "NoSuchKey", "NotFound"})


def _client() -> Any:
    from core.storage import get_s3_client

    return get_s3_client()


def _bucket() -> str:
    from modules.documents.thumbnails.store import documents_bucket

    return documents_bucket()


def configured() -> bool:
    """Whether this install has object storage to hold an export."""
    from core.storage import is_storage_configured

    return is_storage_configured()


def export_key(workspace_id: UUID | str, job_id: UUID | str) -> str:
    """The object key of a job's file."""
    return EXPORT_KEY.format(workspace_id=workspace_id, job_id=job_id)


def upload(fileobj: BinaryIO, key: str) -> None:
    """Upload an open file from its start (multipart for a large one)."""
    from core.storage import ensure_bucket

    client, bucket = _client(), _bucket()
    ensure_bucket(bucket, client)
    client.upload_fileobj(fileobj, bucket, key)


def open_body(key: str) -> Any:
    """The object, opened for reading (a stream with ``read`` and ``close``)."""
    return _client().get_object(Bucket=_bucket(), Key=key)["Body"]


def chunks(body: Any) -> Iterator[bytes]:
    """An opened object's bytes, a chunk at a time; it is closed at the end."""
    try:
        yield from iter(lambda: body.read(DOWNLOAD_CHUNK_BYTES), b"")
    finally:
        body.close()


def delete(key: str) -> None:
    """Delete the object; one already gone counts as deleted."""
    try:
        _client().delete_object(Bucket=_bucket(), Key=key)
    except ClientError as error:
        if str(error.response.get("Error", {}).get("Code", "")) not in _MISSING_CODES:
            raise


__all__ = ["chunks", "configured", "delete", "export_key", "open_body", "upload"]
