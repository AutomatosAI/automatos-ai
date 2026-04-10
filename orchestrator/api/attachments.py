"""
Attachments API — Ephemeral file uploads for chat, missions, tasks, channels (PRD-127)

POST /api/attachments      — Upload a file
GET  /api/attachments/{id} — Get metadata
DELETE /api/attachments/{id} — Delete (explicit removal before sending)

Files are stored in S3 under `workspaces/{ws}/ephemeral-attachments/` with a 7-day
lifecycle rule. NOT for long-term storage — users wanting RAG use /api/documents/upload.
"""

from __future__ import annotations

import logging
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from modules.attachments.store import (
    AttachmentNotFoundError,
    AttachmentRef,
    get_attachment_store,
)
from modules.attachments.validation import ValidationError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/attachments", tags=["attachments"])


@router.get("/health")
async def attachments_health():
    """Simple health check to verify router is loaded."""
    return {"status": "ok", "router": "attachments"}


class AttachmentResponse(BaseModel):
    """Response model for attachment uploads."""

    attachment_id: str
    filename: str
    mime: str
    media_type: str
    size_bytes: int

    @classmethod
    def from_ref(cls, ref: AttachmentRef) -> "AttachmentResponse":
        return cls(
            attachment_id=str(ref.attachment_id),
            filename=ref.filename,
            mime=ref.mime,
            media_type=ref.media_type.value,
            size_bytes=ref.size_bytes,
        )


class AttachmentMetadata(BaseModel):
    """Inline metadata for storing in JSONB columns."""

    attachment_id: str
    filename: str
    mime: str
    media_type: str


@router.post("", response_model=AttachmentResponse)
async def upload_attachment(
    file: UploadFile = File(...),
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> AttachmentResponse:
    """
    Upload an ephemeral attachment.

    Accepts images (JPEG, PNG, GIF, WebP) and documents (PDF, DOCX, XLSX, TXT, etc.).
    Files are stored with a 7-day TTL — use /api/documents/upload for persistent storage.

    Returns attachment metadata including the `attachment_id` to include in message payloads.
    """
    if not ctx.workspace_id:
        raise HTTPException(status_code=400, detail="Workspace ID required")

    # Read file content
    content = await file.read()

    # Get uploader identity
    uploaded_by = ctx.user_id or ctx.agent_id or "anonymous"

    store = get_attachment_store()

    try:
        ref = await store.put(
            workspace_id=UUID(ctx.workspace_id),
            uploaded_by=uploaded_by,
            filename=file.filename or "attachment",
            content=content,
            declared_mime=file.content_type,
        )
    except ValidationError as e:
        logger.warning("Attachment validation failed: %s", e)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error("Attachment upload failed: %s", e, exc_info=True)
        raise HTTPException(status_code=500, detail="Upload failed")

    return AttachmentResponse.from_ref(ref)


@router.get("/{attachment_id}", response_model=AttachmentResponse)
async def get_attachment(
    attachment_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> AttachmentResponse:
    """
    Get attachment metadata.

    Does not return the file content — use this to verify an attachment exists
    before including it in a message payload.
    """
    if not ctx.workspace_id:
        raise HTTPException(status_code=400, detail="Workspace ID required")

    store = get_attachment_store()

    try:
        ref = await store.get(
            attachment_id=UUID(attachment_id),
            workspace_id=UUID(ctx.workspace_id),
        )
    except AttachmentNotFoundError:
        raise HTTPException(status_code=404, detail="Attachment not found or expired")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid attachment ID")

    return AttachmentResponse.from_ref(ref)


@router.delete("/{attachment_id}")
async def delete_attachment(
    attachment_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
) -> dict:
    """
    Delete an attachment.

    Use this when the user removes an attachment before sending a message.
    The 7-day lifecycle rule handles cleanup anyway, but this provides immediate removal.
    """
    if not ctx.workspace_id:
        raise HTTPException(status_code=400, detail="Workspace ID required")

    store = get_attachment_store()

    try:
        await store.delete(
            attachment_id=UUID(attachment_id),
            workspace_id=UUID(ctx.workspace_id),
        )
    except AttachmentNotFoundError:
        raise HTTPException(status_code=404, detail="Attachment not found or expired")
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid attachment ID")

    return {"deleted": True, "attachment_id": attachment_id}
