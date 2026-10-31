"""F453: a document download serves only the caller's own workspace.

``GET /api/documents/{id}/download`` finds the document in the caller's workspace
(api/documents.py) and hands back a pre-signed link to its stored object, or the local
file when no object storage holds a copy. ``GET /api/documents/download?path=`` serves a
file only from the caller's own folder under the documents root,
``<DOCUMENTS_ROOT>/<workspace_id>/``: the ``<root>/<workspace_id>`` layout the workspace
worker keeps (services/workspace-worker/worker_config.py). Both routes used to answer
without auth: any id, and any file under the root, were anyone's.

Paths are compared after resolving, so a ``..`` or a symlink that leaves the folder is
answered as not found, the same as a file that is not there.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from botocore.exceptions import BotoCoreError, ClientError
from fastapi.responses import RedirectResponse

from config import config
from core.storage import get_public_s3_client, get_s3_client, is_storage_configured

logger = logging.getLogger(__name__)

# The local documents root the path route has always served from.
DOCUMENTS_ROOT = Path("/var/automatos/documents")
PRESIGNED_URL_TTL_SECONDS = 3600
REDIRECT_MODE = "redirect"
S3_SCHEME = "s3://"
DEFAULT_MEDIA_TYPE = "application/octet-stream"
MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".json": "application/json",
}

Link = Union[Dict[str, str], RedirectResponse]


def s3_key(document: Any) -> str:
    """The object key a document's file is stored under.

    ``s3://bucket/key`` gives the key; a local path gives the key an upload stores it
    under (``workspaces/<ws>/documents/<id>_<filename>``); anything else is a bare key.
    """
    raw_path = document.file_path or ""
    if raw_path.startswith(S3_SCHEME):
        without_scheme = raw_path[len(S3_SCHEME):]
        return without_scheme.split("/", 1)[1] if "/" in without_scheme else without_scheme
    if raw_path.startswith("/"):
        return f"workspaces/{document.workspace_id}/documents/{document.id}_{document.filename}"
    return raw_path


def stored_object_link(document: Any, filename: str, mode: str) -> Optional[Link]:
    """A pre-signed link to the document's stored object: ``{"url", "filename"}``, or a
    302 to it when ``mode`` is ``redirect``. None when object storage is not set up or
    holds no copy, so the caller serves the local file instead."""
    if not is_storage_configured():
        return None
    bucket = config.S3_DOCUMENTS_BUCKET
    key = s3_key(document)
    try:
        get_s3_client().head_object(Bucket=bucket, Key=key)
        # Presigned against the browser-reachable host (PRD-151 US-006).
        url = get_public_s3_client().generate_presigned_url(
            "get_object",
            Params={"Bucket": bucket, "Key": key,
                    "ResponseContentDisposition": f'attachment; filename="{filename}"'},
            ExpiresIn=PRESIGNED_URL_TTL_SECONDS,
        )
    except (BotoCoreError, ClientError) as exc:
        logger.warning("S3 download unavailable for document %s: %s", document.id, exc)
        return None
    if mode == REDIRECT_MODE:
        return RedirectResponse(url=url, status_code=302)
    return {"url": url, "filename": filename}


def workspace_file(path: str, workspace_id: Any) -> Optional[Path]:
    """The file ``path`` names, when it sits in this workspace's own folder under the
    documents root; None otherwise, or when there is no such file. Compared after
    resolving, so neither ``..`` nor a symlink can leave the folder."""
    if not path or workspace_id is None:
        return None
    try:
        root = DOCUMENTS_ROOT.resolve()
        folder = root / str(workspace_id)
        requested = Path(path).resolve()
        # The folder itself must be a real directory right under the root, not a link
        # into another workspace's folder or out of the root.
        if folder.resolve() != folder or folder.parent != root or not requested.is_relative_to(folder):
            return None
        return requested if requested.is_file() else None
    except (OSError, RuntimeError, ValueError):
        # An unreadable, looping or malformed path is no file of this workspace.
        logger.warning("Document path %r could not be resolved", path, exc_info=True)
        return None


def media_type_for(path: Path) -> str:
    """The content type a served file is sent with, by its extension."""
    return MEDIA_TYPES.get(path.suffix.lower(), DEFAULT_MEDIA_TYPE)


__all__ = [
    "DOCUMENTS_ROOT",
    "media_type_for",
    "s3_key",
    "stored_object_link",
    "workspace_file",
]
