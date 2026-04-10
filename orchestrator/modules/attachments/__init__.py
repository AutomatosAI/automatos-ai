"""
Ephemeral Attachments Module (PRD-127)

Provides short-lived file uploads for chat, missions, tasks, and channels.
Files are stored in S3 with a 7-day lifecycle rule — no manual cleanup needed.

NOT for RAG/Knowledge Base. Users wanting persistent documents use /api/documents/upload.

Key components:
- AttachmentStore: S3 operations (put/get/open/sign_url/delete)
- AttachmentResolver: Converts attachment_ids → LLM content parts
- validate_upload: MIME/size/magic-byte validation
- extract_text: Text extraction for documents (PDF, DOCX, XLSX, etc.)
"""

from modules.attachments.store import (
    AttachmentRef,
    AttachmentStore,
    MediaType,
)
from modules.attachments.resolver import (
    AttachmentResolver,
    VisionNotSupportedError,
)
from modules.attachments.validation import validate_upload
from modules.attachments.extract import extract_text

__all__ = [
    "AttachmentRef",
    "AttachmentStore",
    "MediaType",
    "AttachmentResolver",
    "VisionNotSupportedError",
    "validate_upload",
    "extract_text",
]
