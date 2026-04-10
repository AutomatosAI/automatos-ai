"""
Attachment validation — MIME, size, magic bytes, filename sanitization (PRD-127)

Security checks:
- Max file size (default 50MB)
- MIME type allowlist
- Magic byte verification
- Filename traversal prevention
- NULL byte injection prevention
"""

from __future__ import annotations

import logging
import mimetypes
import os
import re
from typing import Optional

from config import config

logger = logging.getLogger(__name__)

# Maximum attachment size (default 50MB, configurable)
MAX_ATTACHMENT_SIZE = getattr(config, "MAX_ATTACHMENT_SIZE_BYTES", 50 * 1024 * 1024)

# Allowed MIME types — images and common document formats
ALLOWED_MIME_TYPES = {
    # Images
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/svg+xml",
    "image/bmp",
    "image/tiff",
    # Documents
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # .pptx
    "application/vnd.ms-excel",  # .xls
    "application/msword",  # .doc
    "application/vnd.oasis.opendocument.text",  # .odt
    "application/vnd.oasis.opendocument.spreadsheet",  # .ods
    # Text/code
    "text/plain",
    "text/markdown",
    "text/csv",
    "text/html",
    "text/css",
    "text/javascript",
    "text/typescript",
    "text/x-python",
    "text/x-java",
    "text/x-c",
    "text/x-c++",
    "text/x-go",
    "text/x-rust",
    "text/x-ruby",
    "text/x-swift",
    "text/x-kotlin",
    "text/x-scala",
    "text/x-php",
    "application/json",
    "application/xml",
    "application/x-yaml",
    "application/x-sh",
}

# Magic byte signatures for common formats
MAGIC_BYTES = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
    b"RIFF": "image/webp",  # WebP starts with RIFF
    b"%PDF": "application/pdf",
    b"PK\x03\x04": "application/zip",  # ZIP-based formats (docx, xlsx, pptx)
}

# Extension to MIME mapping for text/code files (no magic bytes)
TEXT_EXTENSIONS = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".csv": "text/csv",
    ".html": "text/html",
    ".css": "text/css",
    ".js": "text/javascript",
    ".ts": "text/typescript",
    ".py": "text/x-python",
    ".java": "text/x-java",
    ".c": "text/x-c",
    ".cpp": "text/x-c++",
    ".h": "text/x-c",
    ".hpp": "text/x-c++",
    ".go": "text/x-go",
    ".rs": "text/x-rust",
    ".rb": "text/x-ruby",
    ".swift": "text/x-swift",
    ".kt": "text/x-kotlin",
    ".scala": "text/x-scala",
    ".php": "text/x-php",
    ".json": "application/json",
    ".xml": "application/xml",
    ".yaml": "application/x-yaml",
    ".yml": "application/x-yaml",
    ".sh": "application/x-sh",
    ".bash": "application/x-sh",
    ".sql": "text/plain",
    ".graphql": "text/plain",
    ".proto": "text/plain",
}


class ValidationError(ValueError):
    """Attachment validation failed."""

    pass


def validate_upload(
    content: bytes,
    filename: str,
    declared_mime: Optional[str] = None,
) -> dict:
    """
    Validate an attachment before storage.

    Args:
        content: Raw file bytes
        filename: Original filename from client
        declared_mime: MIME type declared by client (optional)

    Returns:
        dict with validated metadata:
            - safe_filename: Sanitized filename
            - mime: Verified MIME type
            - size_bytes: Content length

    Raises:
        ValidationError: If any check fails
    """
    # Size check
    if len(content) > MAX_ATTACHMENT_SIZE:
        raise ValidationError(
            f"File too large: {len(content)} bytes (max {MAX_ATTACHMENT_SIZE})"
        )

    if len(content) == 0:
        raise ValidationError("Empty file")

    # Filename sanitization
    safe_filename = sanitize_filename(filename)
    if not safe_filename:
        raise ValidationError("Invalid filename")

    # MIME detection
    detected_mime = detect_mime(content, safe_filename)

    # If client declared a MIME, verify it's compatible
    if declared_mime:
        # Allow generic types to match specific ones
        if declared_mime != detected_mime:
            # Check if they're in the same family (e.g., application/octet-stream is generic)
            if declared_mime not in ("application/octet-stream", "binary/octet-stream"):
                logger.warning(
                    "MIME mismatch: declared=%s, detected=%s for %s",
                    declared_mime,
                    detected_mime,
                    safe_filename,
                )
            # Trust detected MIME over declared

    # Allowlist check
    if detected_mime not in ALLOWED_MIME_TYPES:
        # Check if it's a known extension even if MIME isn't in allowlist
        ext = os.path.splitext(safe_filename)[1].lower()
        if ext in TEXT_EXTENSIONS:
            detected_mime = TEXT_EXTENSIONS[ext]
        else:
            raise ValidationError(
                f"File type not allowed: {detected_mime} ({safe_filename})"
            )

    logger.debug(
        "Validated attachment: %s (%s, %d bytes)",
        safe_filename,
        detected_mime,
        len(content),
    )

    return {
        "safe_filename": safe_filename,
        "mime": detected_mime,
        "size_bytes": len(content),
    }


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal and other attacks.

    - Strips directory components
    - Removes NULL bytes
    - Limits length
    - Preserves extension
    """
    if not filename:
        return ""

    # Remove NULL bytes (injection attack)
    filename = filename.replace("\x00", "")

    # Strip directory components
    filename = os.path.basename(filename)

    # Remove any remaining path separators (paranoid)
    filename = filename.replace("/", "_").replace("\\", "_")

    # Remove control characters
    filename = re.sub(r"[\x00-\x1f\x7f]", "", filename)

    # Limit length (preserve extension)
    name, ext = os.path.splitext(filename)
    max_name_len = 200 - len(ext)
    if len(name) > max_name_len:
        name = name[:max_name_len]
    filename = name + ext

    # Final check: must have content
    if not filename or filename in (".", ".."):
        return ""

    return filename


def detect_mime(content: bytes, filename: str) -> str:
    """
    Detect MIME type from content magic bytes and filename.

    Priority:
    1. Magic bytes (for binary formats)
    2. Extension mapping (for text/code files)
    3. mimetypes module fallback
    4. application/octet-stream
    """
    # Check magic bytes
    for magic, mime in MAGIC_BYTES.items():
        if content.startswith(magic):
            # Special case: ZIP-based formats need extension check
            if mime == "application/zip":
                ext = os.path.splitext(filename)[1].lower()
                if ext == ".docx":
                    return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                elif ext == ".xlsx":
                    return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                elif ext == ".pptx":
                    return "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                elif ext == ".odt":
                    return "application/vnd.oasis.opendocument.text"
                elif ext == ".ods":
                    return "application/vnd.oasis.opendocument.spreadsheet"
                # Default to zip for unknown PK archives
                return "application/zip"
            return mime

    # Check extension for text/code files
    ext = os.path.splitext(filename)[1].lower()
    if ext in TEXT_EXTENSIONS:
        return TEXT_EXTENSIONS[ext]

    # Fallback to mimetypes module
    guessed, _ = mimetypes.guess_type(filename)
    if guessed:
        return guessed

    # Last resort
    return "application/octet-stream"
