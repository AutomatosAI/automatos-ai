"""F353 (issue #947): where a Deliverable's first-page picture is kept.

Beside the generated documents, in both editions, the way DocGen keeps its files
(``modules/documents/generation_service.py``):

* on disk at ``<DOCUMENT_STORAGE_DIR>/<workspace>/generated/thumb_<id>.png``;
* and, when object storage is configured (the hosted edition; MinIO locally),
  at ``workspaces/<workspace>/generated-documents/thumb_<id>.png`` in the
  documents bucket, because a container's disk does not outlive it.

The name comes from the Deliverable's id (a UUID), never from a file name, so it
cannot reach outside the workspace's folder. It is read back by the thumbnail
route only (``api/deliverable_thumbnails.py``), after the caller's workspace has
been checked against the Deliverable's.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional
from uuid import UUID

from botocore.exceptions import BotoCoreError, ClientError

from config import config
from core.storage import ensure_bucket, get_s3_client, is_storage_configured

logger = logging.getLogger(__name__)

PNG_CONTENT_TYPE = "image/png"
DEFAULT_DOCUMENTS_BUCKET = "automatos-ai"


def thumbnail_name(output_id: str | UUID) -> str:
    """``thumb_<uuid>.png``; ValueError when ``output_id`` is not a UUID."""
    return f"thumb_{UUID(str(output_id))}.png"


def generated_dir(workspace_id: str | UUID) -> Path:
    """The workspace's generated-documents folder on this container's disk."""
    return Path(config.DOCUMENT_STORAGE_DIR) / str(workspace_id) / "generated"


def generated_s3_key(workspace_id: str | UUID, filename: str) -> str:
    """The object key DocGen uses for a generated file (``_s3_key`` there)."""
    return f"workspaces/{workspace_id}/generated-documents/{filename}"


def documents_bucket() -> str:
    return config.S3_DOCUMENTS_BUCKET or DEFAULT_DOCUMENTS_BUCKET


def save_thumbnail(workspace_id: str | UUID, output_id: str | UUID, png: bytes) -> str:
    """Write the picture to disk and, when configured, to object storage. Returns its name.

    The disk copy always lands (it serves this container straight away); a failed
    upload is logged and leaves the disk copy, as DocGen does with its documents.
    """
    name = thumbnail_name(output_id)
    folder = generated_dir(workspace_id)
    os.makedirs(folder, exist_ok=True)
    (folder / name).write_bytes(png)
    if is_storage_configured():
        try:
            bucket = documents_bucket()
            ensure_bucket(bucket)
            get_s3_client().put_object(
                Bucket=bucket, Key=generated_s3_key(workspace_id, name), Body=png, ContentType=PNG_CONTENT_TYPE,
            )
        except Exception:
            logger.exception("[F353] thumbnail upload failed for %s; the disk copy stays", name)
    return name


def load_thumbnail(workspace_id: str | UUID, output_id: str | UUID) -> Optional[bytes]:
    """The picture's bytes from disk, else from object storage; None when there is none."""
    name = thumbnail_name(output_id)
    local = generated_dir(workspace_id) / name
    if local.is_file():
        return local.read_bytes()
    if not is_storage_configured():
        return None
    try:
        obj = get_s3_client().get_object(Bucket=documents_bucket(), Key=generated_s3_key(workspace_id, name))
        return obj["Body"].read()
    except (ClientError, BotoCoreError) as e:  # a missing object is the common case: no picture yet
        logger.debug("[F353] no stored thumbnail %s: %s", name, e)
        return None
