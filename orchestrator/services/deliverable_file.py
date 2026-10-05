"""F354 (5 Oct): the bytes of a Deliverable's file, wherever it is kept.

The owner's "Add to Knowledge" files the document itself (the invoice PDF, the
letter's DOCX), so it needs the file. A Deliverable keeps it in one of two places:

- ``generated``: a document ``generate_document`` rendered. It is written to
  ``DOCUMENT_STORAGE_DIR/<workspace>/generated/<name>`` and copied to S3 at
  ``workspaces/<workspace>/generated-documents/<name>``, because containers are
  ephemeral (``DocumentGenerationService._output_path`` / ``_s3_key``; the download
  route ``api/document_generation.serve_generated_file`` reads it the same way).
- ``workspace``: a file an agent or a session wrote in the workspace, read through
  the workspace worker.

Anything else (a row that points at nothing readable) has no bytes: ``None``.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

from config import config

logger = logging.getLogger(__name__)

GENERATED = "generated"
WORKSPACE = "workspace"
GENERATED_SUBDIR = "generated"
GENERATED_S3_KEY = "workspaces/{workspace_id}/generated-documents/{filename}"
DEFAULT_BUCKET = "automatos-ai"  # the writer's fallback (DocumentGenerationService._upload_to_s3)


def _generated_name(file_path: str) -> Optional[str]:
    """The file name of a ``generated/<name>`` path; None for anything that is not one
    plain name (no directory hops)."""
    name = os.path.basename(file_path or "")
    if not name or name in (".", "..") or file_path != f"{GENERATED_SUBDIR}/{name}":
        return None
    return name


def _read_local(path: str) -> Optional[bytes]:
    if not os.path.isfile(path):
        return None
    with open(path, "rb") as fh:
        return fh.read()


def _read_s3(key: str) -> Optional[bytes]:
    from core.storage import get_s3_client, is_storage_configured

    if not is_storage_configured():
        return None
    try:
        bucket = config.S3_DOCUMENTS_BUCKET or DEFAULT_BUCKET
        return get_s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
    except Exception:  # logged; the caller tells the owner the file could not be read
        logger.exception("[F354] could not read the generated document %s from storage", key)
        return None


def _read_generated(workspace_id: str, file_path: str) -> Optional[bytes]:
    name = _generated_name(file_path)
    if name is None:
        return None
    local = os.path.join(config.DOCUMENT_STORAGE_DIR, workspace_id, GENERATED_SUBDIR, name)
    found = _read_local(local)
    if found is not None:
        return found
    return _read_s3(GENERATED_S3_KEY.format(workspace_id=workspace_id, filename=name))


async def _read_workspace(workspace_id: str, file_path: str) -> Optional[bytes]:
    from core.workspace_client import WorkspaceClient

    got = await WorkspaceClient(workspace_id).download_file(file_path)
    if not got.get("success"):
        logger.warning("[F354] could not read %s from workspace %s: %s", file_path, workspace_id, got.get("error"))
        return None
    return got.get("content") or None


async def deliverable_bytes(workspace_id: Any, storage_type: Optional[str], file_path: Optional[str]) -> Optional[bytes]:
    """The file's bytes, or None when it cannot be read. The caller has checked that the
    Deliverable is in ``workspace_id``; every read here is under that workspace."""
    ws = str(workspace_id)
    if not file_path:
        return None
    if storage_type == GENERATED:
        return await asyncio.to_thread(_read_generated, ws, file_path)
    if storage_type == WORKSPACE:
        return await _read_workspace(ws, file_path)
    return None


__all__ = ["deliverable_bytes"]
