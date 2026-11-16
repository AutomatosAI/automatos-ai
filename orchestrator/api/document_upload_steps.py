"""The steps of ``POST /api/documents/upload`` (``api/documents.py`` ``handle_request``).

Moved from ``api/documents.py`` as they were (code shape: the route was one function of almost 200
lines, in a file well over the size limit). The allow-list, the size limit, ``UPLOAD_DIR`` and the
ingestion manager stay there, shared with the platform's upload tool (PRD-143 S10), and are read
from there when a step runs.

* **The checks:** the file has a name, it is at most ``max_bytes`` long, python-magic detects its
  content as a MIME type in ``allowed_mime_types``, and its extension is one that type allows.
  Anything else is a 400 saying why.
* **The tags:** the comma-separated ``tags`` field as a stripped, de-duplicated list, order kept.
  They persist to ``documents.tags`` (PostgreSQL ``text[]``). The Academy corpus sync tags each
  document ``academy,<vendor>,<track>,<domain>`` so the tutor's knowledge graph can map chunks
  back to a course.
* **Saving:** the bytes go to ``UPLOAD_DIR`` under a random name, never the uploaded one. The
  route has already refused a server key's upload that would write over a document the key
  can't read (``services.document_key_access.clashes_with_unreachable``).
* **Replacing (F087)** a document of the same name, under its id, the old source kept in its
  history; else **a new document**, ingested now. Either names its uploader through
  ``services.document_key_access.uploaded_by``: a member's sign-in id, a server key's
  ``sdk:<key id>``. A spreadsheet is also copied into the workspace (F181).
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import magic
from fastapi import HTTPException, UploadFile

from core.models import Document, DocumentUploadResponse
from services.document_key_access import uploaded_by

logger = logging.getLogger(__name__)

BYTES_PER_MB = 1024 * 1024  # the size limit as the refusal names it
PROCESSING_ERROR_CHARS = 200  # how much of a failed ingestion's error the upload's answer quotes


@dataclass(frozen=True)
class StoredUpload:
    """An upload as saved: what its document row and the ingestion pipeline are given."""

    filename: str
    content: bytes
    file_path: Path
    file_type: str
    content_hash: str


async def checked_upload(file: UploadFile, allowed_mime_types: Mapping[str, Sequence[str]],
                         max_bytes: int) -> Tuple[bytes, str]:
    """The upload's bytes and its lower-cased extension once the file passes, its pointer back at
    the start; a 400 saying why when it does not."""
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(status_code=400, detail=f"File too large (max {max_bytes // BYTES_PER_MB}MB)")

    # MIME type validation using python-magic (detect actual content type)
    detected_mime = magic.from_buffer(content, mime=True)
    file_extension = Path(file.filename).suffix.lower()

    if detected_mime not in allowed_mime_types:
        logger.warning(f"Upload rejected: {file.filename} — detected MIME '{detected_mime}', extension '{file_extension}'")
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Detected MIME type: {detected_mime}"
        )

    # Verify extension matches detected MIME type
    allowed_extensions = allowed_mime_types[detected_mime]
    if file_extension not in allowed_extensions:
        logger.warning(f"Upload rejected: {file.filename} — extension '{file_extension}' not in {allowed_extensions} for MIME '{detected_mime}'")
        raise HTTPException(
            status_code=400,
            detail=f"File extension '{file_extension}' does not match detected content type '{detected_mime}'. "
                   f"Allowed extensions for this type: {', '.join(allowed_extensions)}"
        )

    # Reset file pointer
    await file.seek(0)
    return content, file_extension


def upload_tags(tags: Optional[str]) -> List[str]:
    """The comma-separated ``tags`` form field as a stripped, de-duplicated list, order kept."""
    tag_list: List[str] = []
    seen: set[str] = set()
    for raw_tag in (tags or "").split(","):
        tag = raw_tag.strip()
        if tag and tag not in seen:
            seen.add(tag)
            tag_list.append(tag)
    return tag_list


def stored_upload(filename: str, content: bytes, file_extension: str, file_type: str,
                  content_hash: str) -> StoredUpload:
    """Save the checked bytes in ``UPLOAD_DIR`` under a random name (never the original filename)."""
    from api import documents

    documents.UPLOAD_DIR.mkdir(exist_ok=True)
    file_path = documents.UPLOAD_DIR / f"{uuid.uuid4().hex}{file_extension}"
    with open(file_path, "wb") as f:
        f.write(content)
    return StoredUpload(filename, content, file_path, file_type, content_hash)


async def replace_upload(db: Any, ctx: Any, replaced: Document, upload: StoredUpload,
                         tag_list: List[str], description: Optional[str]) -> DocumentUploadResponse:
    """F087: ingest the upload under ``replaced``'s id, the old source kept in its history."""
    from services.document_versions import replace_document, replaced_message
    # F181: a spreadsheet is also copied into the workspace, where agents
    # count it with code (services.spreadsheet_workspace). Best effort.
    from services.spreadsheet_workspace import SPREADSHEET_TYPES, copy_to_workspace

    version = await replace_document(
        db, replaced, workspace_id=ctx.workspace_id, file_path=str(upload.file_path), file_size=len(upload.content),
        content_hash=upload.content_hash, file_type=upload.file_type, replaced_by=uploaded_by(ctx),
        tags=tag_list, description=description,
    )
    if upload.file_type in SPREADSHEET_TYPES:
        await copy_to_workspace(ctx.workspace_id, upload.filename, upload.content)
    return DocumentUploadResponse(
        document_id=replaced.id,
        filename=replaced.filename,
        status=replaced.status,
        message=replaced_message(replaced.filename, version, replaced.status),
    )


async def new_upload(db: Any, ctx: Any, upload: StoredUpload, tag_list: List[str],
                     description: Optional[str], team_access_list: List[str]) -> DocumentUploadResponse:
    """A new document for the upload, ingested now; the answer says so when processing failed."""
    from services.spreadsheet_workspace import SPREADSHEET_TYPES, copy_to_workspace

    # Create document record. tags persist to documents.tags — a real PostgreSQL
    # text[] column (see migration 208275450a15: ARRAY(TEXT), default ARRAY[]::text[]).
    # The prior "SQLAlchemy array bug" was a mis-diagnosis: team_access below uses the
    # identical PG_ARRAY(String) column and writes a Python list on every upload without
    # issue. Assigning a fresh list[str] to a new row is tracked correctly (no
    # flag_modified needed — that only matters for in-place mutation of a loaded array).
    document = Document(
        workspace_id=ctx.workspace_id,
        filename=upload.filename,
        original_filename=upload.filename,
        file_type=upload.file_type,
        file_size=len(upload.content),
        file_path=str(upload.file_path),
        content_hash=upload.content_hash,
        status="uploaded",
        tags=tag_list,
        description=description,
        team_access=team_access_list,
        # PRD-168 S4: real actor. The principal lives on ctx.user —
        # RequestContext itself has no clerk_user_id; reading it there
        # raised AttributeError and 500'd every upload (2026-08-06).
        # A server key is named as sdk:<key id> (PRD-274 §5.2).
        created_by=uploaded_by(ctx),
    )
    db.add(document)
    db.commit()
    db.refresh(document)

    processing_error = await _ingest_upload(db, ctx, document, upload)
    if upload.file_type in SPREADSHEET_TYPES:
        await copy_to_workspace(ctx.workspace_id, upload.filename, upload.content)

    return DocumentUploadResponse(
        document_id=document.id,
        filename=document.filename,
        status=document.status,
        # PRD-233: never say "successfully" over a failed pipeline — every local
        # upload had ended status=failed (no embedding provider) behind this text.
        message=(
            "Document uploaded successfully"
            if document.status != "failed"
            else f"Uploaded, but processing failed: {processing_error or 'see backend logs'}"
        ),
    )


async def _ingest_upload(db: Any, ctx: Any, document: Document, upload: StoredUpload) -> Optional[str]:
    """Run the ingestion pipeline on the new document's file. None when it ran; when it failed,
    the row is ``failed`` and the error, cut short, is answered."""
    from api import documents

    try:
        # DocumentManager.upload_document creates its own DB record, so the
        # processing is triggered directly on this one.
        document.status = "processing"
        db.commit()

        # Get workspace-specific DocumentManager
        doc_manager = documents.get_document_manager(str(ctx.workspace_id))
        await doc_manager._process_document(document.id, str(upload.file_path), documents.processing_type(upload.file_type))

        # Refresh to get updated chunk_count
        db.refresh(document)
        return None
    except Exception as e:
        logger.exception(f"Error processing document {document.id}")
        document.status = "failed"
        db.commit()
        return str(e)[:PROCESSING_ERROR_CHARS]


__all__ = ["StoredUpload", "checked_upload", "new_upload", "replace_upload", "stored_upload", "upload_tags"]
