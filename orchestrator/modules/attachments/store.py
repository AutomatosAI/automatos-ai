"""
AttachmentStore — S3 storage for ephemeral attachments (PRD-127)

Ephemeral attachments live under:
    s3://{bucket}/workspaces/{workspace_id}/ephemeral-attachments/{attachment_id}/{filename}

A 7-day S3 lifecycle rule handles garbage collection — no cron, no DB cleanup.

Storage is the platform object store through ``core.storage`` (PRD-233 S4):
AWS S3 in SaaS, MinIO locally — one code path, no filesystem fallback.

This module has NO imports from modules/rag/ or DocumentManager.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from botocore.exceptions import ClientError

from config import config
from core.storage import ensure_bucket, get_public_s3_client, get_s3_client
from modules.attachments.validation import validate_upload

logger = logging.getLogger(__name__)


class MediaType(str, Enum):
    """Attachment media type — determines how the resolver handles it."""

    IMAGE = "image"
    DOCUMENT = "document"


@dataclass(frozen=True)
class AttachmentRef:
    """Immutable reference to an uploaded attachment."""

    attachment_id: UUID
    workspace_id: UUID
    media_type: MediaType
    mime: str
    filename: str
    size_bytes: int
    s3_key: str

    def to_inline_metadata(self) -> dict:
        """Return dict for storing inline in JSONB columns."""
        return {
            "attachment_id": str(self.attachment_id),
            "filename": self.filename,
            "mime": self.mime,
            "media_type": self.media_type.value,
        }


class AttachmentNotFoundError(Exception):
    """Attachment does not exist or has expired."""

    pass


class AttachmentStore:
    """
    S3 storage for ephemeral attachments.

    All attachments live under the `ephemeral-attachments/` prefix which has
    a 7-day lifecycle rule configured in S3. No manual cleanup needed.
    """

    def __init__(self, bucket: Optional[str] = None):
        self._bucket = bucket or config.S3_DOCUMENTS_BUCKET

    @property
    def _client(self):
        """The process-wide S3 client — lazy, no network at construction."""
        return get_s3_client()

    def _s3_key(self, workspace_id: UUID, attachment_id: UUID, filename: str) -> str:
        """Build the S3 key for an attachment."""
        return f"workspaces/{workspace_id}/ephemeral-attachments/{attachment_id}/{filename}"

    async def put(
        self,
        *,
        workspace_id: UUID,
        uploaded_by: str,
        filename: str,
        content: bytes,
        declared_mime: Optional[str] = None,
    ) -> AttachmentRef:
        """
        Validate and store an attachment.

        Args:
            workspace_id: Owning workspace (enforces isolation)
            uploaded_by: User or agent ID for audit
            filename: Original filename
            content: Raw file bytes
            declared_mime: MIME type from client (validated against magic bytes)

        Returns:
            AttachmentRef with all metadata

        Raises:
            ValueError: If validation fails (bad MIME, too large, malicious)
        """
        # Validate before storing
        validated = validate_upload(content, filename, declared_mime)

        attachment_id = uuid4()
        safe_filename = validated["safe_filename"]
        mime = validated["mime"]
        media_type = MediaType.IMAGE if mime.startswith("image/") else MediaType.DOCUMENT

        key = self._s3_key(workspace_id, attachment_id, safe_filename)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: ensure_bucket(self._bucket))
        await loop.run_in_executor(
            None,
            lambda: self._client.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=content,
                ContentType=mime,
                Metadata={
                    "uploaded-by": uploaded_by,
                    "original-filename": filename[:200],
                },
            ),
        )
        logger.info(
            "Attachment stored: %s (%d bytes, %s) by %s",
            key,
            len(content),
            mime,
            uploaded_by,
        )

        return AttachmentRef(
            attachment_id=attachment_id,
            workspace_id=workspace_id,
            media_type=media_type,
            mime=mime,
            filename=safe_filename,
            size_bytes=len(content),
            s3_key=key,
        )

    async def get(self, attachment_id: UUID, workspace_id: UUID) -> AttachmentRef:
        """
        Get attachment metadata via HeadObject (no body download).

        Raises:
            AttachmentNotFoundError: If expired or doesn't exist
        """
        # List objects under the attachment_id prefix to find the filename
        prefix = f"workspaces/{workspace_id}/ephemeral-attachments/{attachment_id}/"
        loop = asyncio.get_running_loop()
        try:
            resp = await loop.run_in_executor(
                None,
                lambda: self._client.list_objects_v2(
                    Bucket=self._bucket, Prefix=prefix, MaxKeys=1
                ),
            )
            contents = resp.get("Contents", [])
            if not contents:
                raise AttachmentNotFoundError(
                    f"Attachment {attachment_id} not found or expired"
                )
            key = contents[0]["Key"]
            filename = key.split("/")[-1]

            # HeadObject for metadata
            head = await loop.run_in_executor(
                None,
                lambda: self._client.head_object(Bucket=self._bucket, Key=key),
            )
            mime = head.get("ContentType", "application/octet-stream")
            size = head.get("ContentLength", 0)

        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "404":
                raise AttachmentNotFoundError(
                    f"Attachment {attachment_id} not found or expired"
                )
            raise

        media_type = MediaType.IMAGE if mime.startswith("image/") else MediaType.DOCUMENT

        return AttachmentRef(
            attachment_id=attachment_id,
            workspace_id=workspace_id,
            media_type=media_type,
            mime=mime,
            filename=filename,
            size_bytes=size,
            s3_key=key,
        )

    async def open(self, attachment_id: UUID, workspace_id: UUID) -> bytes:
        """
        Download attachment bytes. Used by extract.py for documents.

        Raises:
            AttachmentNotFoundError: If expired or doesn't exist
        """
        ref = await self.get(attachment_id, workspace_id)

        loop = asyncio.get_running_loop()
        try:
            obj = await loop.run_in_executor(
                None,
                lambda: self._client.get_object(Bucket=self._bucket, Key=ref.s3_key),
            )
            body = await loop.run_in_executor(None, lambda: obj["Body"].read())
            return body
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "NoSuchKey":
                raise AttachmentNotFoundError(
                    f"Attachment {attachment_id} not found or expired"
                )
            raise

    async def sign_url(
        self, attachment_id: UUID, workspace_id: UUID, ttl_seconds: int = 900
    ) -> str:
        """
        Generate a presigned GET URL for image_url parts.

        The URL leaves the backend (the LLM provider fetches it), so it is
        minted against the public endpoint (PRD-151 US-006).

        Args:
            attachment_id: The attachment UUID
            workspace_id: Owning workspace (for isolation check)
            ttl_seconds: URL validity period (default 15 minutes)

        Returns:
            Presigned S3 URL

        Raises:
            AttachmentNotFoundError: If expired or doesn't exist
        """
        ref = await self.get(attachment_id, workspace_id)

        loop = asyncio.get_running_loop()
        url = await loop.run_in_executor(
            None,
            lambda: get_public_s3_client().generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": ref.s3_key},
                ExpiresIn=ttl_seconds,
            ),
        )
        return url

    async def delete(self, attachment_id: UUID, workspace_id: UUID) -> None:
        """
        Explicitly delete an attachment (e.g., user removes before sending).

        Lifecycle rule handles cleanup anyway, but this is for immediate removal.
        """
        ref = await self.get(attachment_id, workspace_id)

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self._client.delete_object(Bucket=self._bucket, Key=ref.s3_key),
        )
        logger.info("Attachment deleted: %s", ref.s3_key)


# Singleton
_store: Optional[AttachmentStore] = None


def get_attachment_store() -> AttachmentStore:
    """Get or create the AttachmentStore singleton."""
    global _store
    if _store is None:
        _store = AttachmentStore()
    return _store
