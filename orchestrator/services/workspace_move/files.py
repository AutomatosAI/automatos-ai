"""PRD-273 §5.1 — ``files/<sha256>``: each document and Deliverable file, once, by content hash.

A document's file is in object storage (``documents.file_path``, ``s3://<bucket>/<key>``);
a Deliverable's is a generated document (this container's disk, else its object-storage copy)
or a file in the workspace (read through the workspace worker), as ``services/deliverable_file``
reads them. Each file is copied to a temporary file while its SHA-256 is counted, then into
the zip under that hash, so a large file is never held in memory and two rows with the same
bytes share one entry. ``files/index.jsonl`` says which row has which file.

Only objects under the workspace's own prefix (``workspaces/<id>/``) in the documents bucket
are read. A file that is missing, unreadable, outside that prefix or over
``WORKSPACE_EXPORT_MAX_FILE_MB`` is not carried; the manifest lists it under
``skipped_files`` with the reason, and its row still moves. A deleted Deliverable's row moves
with its ``deleted_at``; its file does not.
"""
from __future__ import annotations

import hashlib
import io
import logging
import shutil
import tempfile
from contextlib import closing
from dataclasses import dataclass
from typing import Any, BinaryIO, Callable, Dict, Iterator, List, Optional, Tuple
from uuid import UUID

from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy import bindparam, text
from sqlalchemy.dialects.postgresql import UUID as PGUUID
from sqlalchemy.orm import Session

from config import config
from services.workspace_move.rules import DELIVERABLES, KNOWLEDGE

logger = logging.getLogger(__name__)

FILES_PREFIX = "files/"
FILES_INDEX = "files/index.jsonl"
S3_SCHEME = "s3://"
GENERATED = "generated"
WORKSPACE = "workspace"
STORED = "s3"
# The object-key prefixes of a workspace's own files: its documents and generated files, and
# its social media (modules/socials/media_store.media_key).
OWN_PREFIXES = ("workspaces/{workspace_id}/", "{media}/{workspace_id}/")
GENERATED_PREFIX = "generated/"
CHUNK_BYTES = 1024 * 1024
MISSING = "missing"
UNREADABLE = "unreadable"
TOO_LARGE = "too large"
OUTSIDE = "not in this workspace's storage"
MEGABYTE = 1024 * 1024

_DOCUMENTS_SQL = text(
    "SELECT id, file_path, COALESCE(original_filename, filename) AS name FROM documents WHERE workspace_id = :ws"
).bindparams(bindparam("ws", type_=PGUUID(as_uuid=True)))
_DELIVERABLES_SQL = text(
    "SELECT id, storage_type, file_path, file_name AS name FROM deliverables "
    "WHERE workspace_id = :ws AND deleted_at IS NULL"
).bindparams(bindparam("ws", type_=PGUUID(as_uuid=True)))

Opener = Callable[[], Optional[BinaryIO]]


@dataclass(frozen=True)
class FileSource:
    """One row's file: where to open it (None when it points at nothing this export reads)."""

    table: str
    row_id: Any
    name: Optional[str]
    opener: Optional[Opener]
    reason: Optional[str] = None


def _workspace_key(workspace_id: str, path: Optional[str]) -> Optional[str]:
    """The object key of a ``file_path`` (``s3://<bucket>/<key>`` or a bare key), when it is in
    the documents bucket under one of the workspace's own prefixes; else None."""
    from modules.documents.thumbnails.store import documents_bucket
    from modules.socials.media_store import MEDIA_KEY_PREFIX

    if not path:
        return None
    key = path
    if path.startswith(S3_SCHEME):
        bucket, _, key = path[len(S3_SCHEME):].partition("/")
        if bucket != documents_bucket():
            return None
    prefixes = tuple(p.format(workspace_id=workspace_id, media=MEDIA_KEY_PREFIX) for p in OWN_PREFIXES)
    return key if key.startswith(prefixes) and ".." not in key.split("/") else None


def _s3_opener(key: str) -> Opener:
    def open_object() -> Optional[BinaryIO]:
        from core.storage import get_s3_client
        from modules.documents.thumbnails.store import documents_bucket

        return get_s3_client().get_object(Bucket=documents_bucket(), Key=key)["Body"]
    return open_object


def _generated_opener(workspace_id: str, file_path: str) -> Optional[Opener]:
    """The opener of a generated document: this container's copy, else its object-storage copy."""
    from modules.documents.thumbnails.store import generated_dir, generated_s3_key

    name = file_path[len(GENERATED_PREFIX):] if file_path.startswith(GENERATED_PREFIX) else ""
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        return None
    local = generated_dir(workspace_id) / name

    def open_generated() -> Optional[BinaryIO]:
        if local.is_file():
            return open(local, "rb")
        return _s3_opener(generated_s3_key(workspace_id, name))()
    return open_generated


def _worker_opener(workspace_id: str, file_path: str, max_bytes: int) -> Opener:
    def open_worker_file() -> Optional[BinaryIO]:
        from core.workspace_client import WorkspaceClient

        data = WorkspaceClient(workspace_id).download_file_sync(file_path, max_bytes=max_bytes)
        return io.BytesIO(data) if data is not None else None
    return open_worker_file


def _document_source(workspace_id: str, row: Any) -> FileSource:
    key = _workspace_key(workspace_id, row.file_path)
    if key is None:
        return FileSource("documents", row.id, row.name, None, OUTSIDE if row.file_path else MISSING)
    return FileSource("documents", row.id, row.name, _s3_opener(key))


def _deliverable_source(workspace_id: str, row: Any, max_bytes: int) -> FileSource:
    opener: Optional[Opener] = None
    if row.storage_type == GENERATED:
        opener = _generated_opener(workspace_id, row.file_path)
    elif row.storage_type == WORKSPACE:
        opener = _worker_opener(workspace_id, row.file_path, max_bytes)
    elif row.storage_type == STORED:
        key = _workspace_key(workspace_id, row.file_path)
        opener = _s3_opener(key) if key else None
    return FileSource("deliverables", row.id, row.name, opener, None if opener else OUTSIDE)


def file_sources(db: Session, workspace_id: UUID | str, categories: Tuple[str, ...],
                 max_bytes: int) -> Iterator[FileSource]:
    """Every file of the chosen categories' rows: documents (Knowledge), Deliverables."""
    ws, params = str(workspace_id), {"ws": UUID(str(workspace_id))}
    if KNOWLEDGE in categories:
        for row in db.execute(_DOCUMENTS_SQL, params).all():
            yield _document_source(ws, row)
    if DELIVERABLES in categories:
        for row in db.execute(_DELIVERABLES_SQL, params).all():
            if row.file_path:  # a Deliverable with no file has nothing to carry
                yield _deliverable_source(ws, row, max_bytes)


def _spool(stream: BinaryIO, max_bytes: int, spool_dir: Optional[str]) -> Tuple[Optional[str], int, Any]:
    """Copy ``stream`` to a temporary file while hashing it: (sha256, size, the file), or
    (None, size, None) once it passes ``max_bytes``."""
    spooled = tempfile.TemporaryFile(dir=spool_dir)
    digest, size = hashlib.sha256(), 0
    for chunk in iter(lambda: stream.read(CHUNK_BYTES), b""):
        size += len(chunk)
        if size > max_bytes:
            spooled.close()
            return None, size, None
        digest.update(chunk)
        spooled.write(chunk)
    spooled.seek(0)
    return digest.hexdigest(), size, spooled


def _open(source: FileSource) -> Tuple[Optional[BinaryIO], Optional[str]]:
    """(the open stream, None), or (None, why it could not be opened)."""
    try:
        stream = source.opener() if source.opener else None
    except ClientError as error:
        code = str(error.response.get("Error", {}).get("Code", ""))
        if code in ("404", "NoSuchKey", "NotFound"):
            return None, MISSING
        logger.exception("[workspace-move] %s %s: the file could not be read", source.table, source.row_id)
        return None, UNREADABLE
    except (BotoCoreError, OSError):
        logger.exception("[workspace-move] %s %s: the file could not be read", source.table, source.row_id)
        return None, UNREADABLE
    return stream, None if stream is not None else MISSING


def _copy(writer: Any, source: FileSource, max_bytes: int) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Put one file in the zip (once per hash): (its index line, None), or (None, why not)."""
    stream, why = _open(source) if source.reason is None else (None, source.reason)
    if stream is None:
        return None, why
    try:
        with closing(stream):
            sha, size, spooled = _spool(stream, max_bytes, config.WORKSPACE_EXPORT_TEMP_DIR or None)
    except (BotoCoreError, ClientError, OSError):
        logger.exception("[workspace-move] %s %s: the file broke off while it was read", source.table, source.row_id)
        return None, UNREADABLE
    if sha is None:
        return None, TOO_LARGE
    with spooled:
        name = f"{FILES_PREFIX}{sha}"
        if not writer.has(name):
            with writer.part(name) as sink:
                shutil.copyfileobj(spooled, _SinkWriter(sink), CHUNK_BYTES)
    return {"table": source.table, "id": source.row_id, "sha256": sha, "bytes": size, "name": source.name}, None


class _SinkWriter:
    """A file-like ``write`` over a part sink, for ``shutil.copyfileobj``."""

    def __init__(self, sink: Any) -> None:
        self._sink = sink

    def write(self, data: bytes) -> int:
        self._sink.write(data)
        return len(data)


def write_files(db: Session, writer: Any, workspace_id: UUID | str,
                categories: Tuple[str, ...]) -> Tuple[int, List[Dict[str, Any]]]:
    """Write every file and ``files/index.jsonl``; (files carried, the ones skipped and why)."""
    max_bytes = int(config.WORKSPACE_EXPORT_MAX_FILE_MB) * MEGABYTE
    index: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for source in file_sources(db, workspace_id, categories, max_bytes):
        line, why = _copy(writer, source, max_bytes)
        if line is None:
            skipped.append({"table": source.table, "id": source.row_id, "reason": why})
        else:
            index.append(line)
    writer.write_rows(FILES_INDEX, index)
    return len(index), skipped


__all__ = ["FILES_INDEX", "FILES_PREFIX", "FileSource", "file_sources", "write_files"]
