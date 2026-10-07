"""#848: a session's files reach the workspace when the host shares no folder with it.

Compose bind-mounts the operator's deliverables folder into the backend and the
worker, so a session's files already sit where the worker serves them and the
result only registers them (``cli_host_service._register_session_deliverables``).
On a cluster the workspace is a volume inside the cluster and the files stay on
the operator's machine. With ``CLI_SESSION_FILE_UPLOAD`` on, the claim (``upload``) asks the
host to upload what its session left in the ticket's deliverables folder
(``<root>/sessions/<ticket>``), one file at a time, before it posts the result:

* each file is written to ``sessions/<ticket>/<path>`` through the worker's
  internal file API (``WorkspaceClient.write_binary``), the place compose would
  have put it;
* the result names the files the backend took (``uploaded_files``), and they are
  registered by the same ``DeliverableService.register`` as a shared folder's,
  once this process can see them on the volume.

Limits: only the ticket's deliverables folder, the document upload limit per file
(``api.documents.MAX_UPLOAD_BYTES``), ``CLI_SESSION_UPLOAD_MAX_TOTAL_MB`` per run of
the ticket, and no path that leaves the folder. The host refuses symlinks on its side.
"""
from __future__ import annotations

import logging
from pathlib import PurePosixPath
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional

from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session

from config import config
from core.models.cli_hosts import CliHost

logger = logging.getLogger(__name__)

# The host's ``<root>/sessions/<ticket>``; the worker's ``sessions/<ticket>`` (compose maps one onto the other).
SESSIONS_FOLDER = "sessions"
# runtime_ref: the bytes this run of the ticket uploaded so far. Every claim builds a fresh ref, so it is per run.
UPLOADED_BYTES_KEY = "uploaded_bytes"
MAX_UPLOAD_PATH_CHARS = 1024
BYTES_PER_MB = 1024 * 1024


class SessionUploadRefused(Exception):
    """An upload the backend will not take; ``status_code`` is the HTTP answer."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def upload_enabled() -> bool:
    """True when hosts upload their sessions' files (a cluster: no shared folder)."""
    return bool(getattr(config, "CLI_SESSION_FILE_UPLOAD", False))


def max_file_bytes() -> int:
    """The per-file limit: the same as a document upload."""
    from api.documents import MAX_UPLOAD_BYTES

    return MAX_UPLOAD_BYTES


def max_total_bytes() -> int:
    """The limit on everything one run of a ticket uploads."""
    return int(config.CLI_SESSION_UPLOAD_MAX_TOTAL_MB) * BYTES_PER_MB


def upload_claim_fields() -> Dict[str, Any]:
    """The claim's ``upload`` (host contract 0.14.0): whether to upload, and the limits.
    An older host ignores it, and its session's files stay on the operator's machine."""
    return {"enabled": upload_enabled(), "max_file_bytes": max_file_bytes(), "max_total_bytes": max_total_bytes()}


def safe_upload_path(raw: Any) -> Optional[str]:
    """A path inside the ticket's deliverables folder, normalised, or ``None``.

    Refused: empty, absolute (``/x``, ``C:/x``), a backslash, a ``..`` or ``.``
    segment, an empty segment, a control character, or longer than the bound."""
    text = str(raw or "")
    if not text or len(text) > MAX_UPLOAD_PATH_CHARS or "\\" in text or text.startswith("/"):
        return None
    if any(ord(ch) < 32 for ch in text):
        return None
    parts = text.split("/")
    if any(part in ("", ".", "..") for part in parts) or ":" in parts[0]:
        return None
    return str(PurePosixPath(*parts))


def ticket_upload_target(task_id: Any, rel: str) -> str:
    """Where an uploaded file lands in the workspace: ``sessions/<ticket>/<rel>``."""
    return f"{SESSIONS_FOLDER}/{int(task_id)}/{rel}"


async def read_capped(chunks: AsyncIterator[bytes], limit: int) -> bytes:
    """The request body, refused with 413 the moment it passes ``limit``."""
    data = bytearray()
    async for chunk in chunks:
        data.extend(chunk)
        if len(data) > limit:
            raise SessionUploadRefused(413, f"a session file may be at most {limit // BYTES_PER_MB} MB")
    return bytes(data)


def _reserve(db: Session, task_id: int, size: int, limit: int) -> bool:
    """Add ``size`` to this run's uploaded total, ONE key in one statement (a
    whole-document write would clobber the host's concurrent event flush).
    False when it would pass ``limit``: nothing is added then."""
    row = db.execute(
        sql_text(
            """
            UPDATE board_tasks
               SET runtime_ref = jsonb_set(
                       COALESCE(runtime_ref, CAST('{}' AS jsonb)),
                       CAST(:path AS text[]),
                       to_jsonb(COALESCE((runtime_ref ->> :key)::bigint, 0) + :size),
                       true)
             WHERE id = :task_id
               AND COALESCE((runtime_ref ->> :key)::bigint, 0) + :size <= :limit
         RETURNING id
            """
        ),
        {"path": "{%s}" % UPLOADED_BYTES_KEY, "key": UPLOADED_BYTES_KEY, "size": int(size),
         "limit": int(limit), "task_id": int(task_id)},
    ).first()
    db.commit()
    return row is not None


async def _single(data: bytes) -> AsyncIterator[bytes]:
    yield data


async def store_session_file(
    db: Session, host: CliHost, task_id: int, raw_path: str, chunks: AsyncIterator[bytes],
) -> Dict[str, Any]:
    """Take one file of a running ticket's session and write it into the workspace.

    The ticket must be this host's and still running; the path must stay inside
    the ticket's deliverables folder; the file and this run's total stay within
    their limits. Returns the path the result names and where the file landed."""
    from core.workspace_client import WorkspaceClient
    from services.cli_host_service import _owned_task

    if not upload_enabled():
        raise SessionUploadRefused(404, "this instance does not take session file uploads")
    task = _owned_task(db, host, task_id)
    if task.status != "in_progress":
        raise SessionUploadRefused(409, f"task {task_id} is {task.status}, not running")
    rel = safe_upload_path(raw_path)
    if rel is None:
        raise SessionUploadRefused(400, "the path must stay inside the ticket's deliverables folder")
    data = await read_capped(chunks, max_file_bytes())
    if not _reserve(db, task.id, len(data), max_total_bytes()):
        raise SessionUploadRefused(
            413, f"this run's files would pass {config.CLI_SESSION_UPLOAD_MAX_TOTAL_MB} MB in all")
    target = ticket_upload_target(task.id, rel)
    written = await WorkspaceClient(str(task.workspace_id)).write_binary(target, _single(data))
    if not written.get("success"):
        logger.error("[cli-host] session file %s for ticket #%s not written: %s", target, task.id,
                     written.get("error"))
        raise SessionUploadRefused(502, "the workspace could not store the file")
    return {"path": rel, "workspace_path": target, "size": len(data)}


def uploaded_volume_paths(task: Any, uploaded: Iterable[Any]) -> List[str]:
    """The result's ``uploaded_files`` as paths on this process's view of the
    volume, which ``workspace_relative_path`` maps by the workspace-id segment
    (as the dispatcher's adoption does). A path that is not safe is dropped; a
    file that never landed is dropped later, when its size cannot be read."""
    if not upload_enabled():
        return []
    base = f"{config.WORKSPACE_VOLUME_PATH.rstrip('/')}/{task.workspace_id}"
    safe = (safe_upload_path(raw) for raw in uploaded or [])
    return [f"{base}/{ticket_upload_target(task.id, rel)}" for rel in safe if rel is not None]


__all__ = [
    "SESSIONS_FOLDER", "SessionUploadRefused", "UPLOADED_BYTES_KEY", "max_file_bytes", "max_total_bytes",
    "read_capped", "safe_upload_path", "store_session_file", "ticket_upload_target", "upload_claim_fields",
    "upload_enabled", "uploaded_volume_paths",
]
