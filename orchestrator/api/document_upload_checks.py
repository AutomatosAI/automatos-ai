"""What ``POST /api/documents/upload`` checks of a file before anything is stored, and its tags.

Moved from ``api/documents.py`` as it was (code shape: the route was one function of almost 200
lines). The allow-list and the size limit stay there, shared with the platform's upload tool
(PRD-143 S10), and are passed in.

* **The file:** it has a name, it is at most ``max_bytes`` long, python-magic detects its content
  as a MIME type in ``allowed_mime_types``, and its extension is one that type allows. Anything
  else is a 400 saying why.
* **The tags:** the comma-separated ``tags`` field as a stripped, de-duplicated list, order kept.
  They persist to ``documents.tags`` (PostgreSQL ``text[]``). The Academy corpus sync tags each
  document ``academy,<vendor>,<track>,<domain>`` so the tutor's knowledge graph can map chunks
  back to a course.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

import magic
from fastapi import HTTPException, UploadFile

logger = logging.getLogger(__name__)

BYTES_PER_MB = 1024 * 1024  # the size limit as the refusal names it


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


__all__ = ["checked_upload", "upload_tags"]
