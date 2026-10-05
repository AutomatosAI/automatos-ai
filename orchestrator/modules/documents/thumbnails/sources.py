"""F353 (issue #947): a Deliverable's file, read where it lives, to draw its first page.

* ``generated`` (DocGen's PDFs, Word documents and sheets): the container's disk,
  else the persistence copy in object storage (as ``serve_generated_file`` does).
* ``workspace`` (files an agent wrote, and reports): the workspace worker, read
  synchronously — this runs on the thumbnail thread, which has no event loop.

``None`` when the file is not there (yet, or any more); a file over
``MAX_SOURCE_BYTES`` is not read at all.
"""
from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from botocore.exceptions import BotoCoreError, ClientError

from core.storage import get_s3_client, is_storage_configured
from core.workspace_client import WorkspaceClient
from modules.documents.thumbnails.store import documents_bucket, generated_dir, generated_s3_key

logger = logging.getLogger(__name__)

MAX_SOURCE_BYTES = 25 * 1024 * 1024
GENERATED_PREFIX = "generated/"


def _generated_bytes(workspace_id: str | UUID, file_path: str) -> Optional[bytes]:
    filename = file_path[len(GENERATED_PREFIX):] if file_path.startswith(GENERATED_PREFIX) else file_path
    if not filename or "/" in filename or "\\" in filename or filename in (".", ".."):
        return None
    local = generated_dir(workspace_id) / filename
    if local.is_file():
        return local.read_bytes() if local.stat().st_size <= MAX_SOURCE_BYTES else None
    if not is_storage_configured():
        return None
    try:
        obj = get_s3_client().get_object(Bucket=documents_bucket(), Key=generated_s3_key(workspace_id, filename))
    except (ClientError, BotoCoreError) as e:
        logger.info("[F353] generated file %s is not in object storage: %s", filename, e)
        return None
    if int(obj.get("ContentLength") or 0) > MAX_SOURCE_BYTES:
        return None
    return obj["Body"].read()


def read_source(workspace_id: str | UUID, storage_type: str, file_path: str) -> Optional[bytes]:
    """The Deliverable's bytes, or None when they cannot be found or are too big to draw."""
    if not file_path:
        return None
    if storage_type == "generated":
        return _generated_bytes(workspace_id, file_path)
    if storage_type == "workspace":
        return WorkspaceClient(str(workspace_id)).download_file_sync(file_path, max_bytes=MAX_SOURCE_BYTES)
    return None
