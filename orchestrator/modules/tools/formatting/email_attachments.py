"""An email's attachments as the chat's email widget lists them.

Each attachment is ``{id, filename, mimeType, size, downloadUrl}``, from
Gmail's ``attachmentList``, Outlook's ``attachments`` or Gmail's
``payload.parts``, the first of those the email has.

``downloadUrl`` is a link the provider gave (Outlook's ``contentLocation``).
A Gmail attachment has none: its bytes come from the platform's own route, so
when the email's message id and the attachment's ``attachmentId`` are both
Gmail ids it carries ``downloadPath``, that route's API path
(``core/composio/gmail_attachments.py``). The widget fetches it with auth.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.composio.gmail_attachments import download_path

DEFAULT_FILENAME = "attachment"
OCTET_STREAM = "application/octet-stream"
# Gmail body parts, not attachments.
_BODY_MIME_TYPES = ("text/plain", "text/html")


def _dicts(value: Any) -> List[Dict[str, Any]]:
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _gmail_attachment(att: Dict[str, Any], message_id: Any) -> Dict[str, Any]:
    filename = att.get("filename") or att.get("name") or DEFAULT_FILENAME
    gmail_id = att.get("attachmentId") or att.get("attachment_id")
    return {
        "id": gmail_id or att.get("id") or att.get("fileId"),
        "filename": filename,
        "mimeType": att.get("mimeType") or att.get("contentType") or OCTET_STREAM,
        "size": att.get("size") or att.get("fileSize") or 0,
        "downloadUrl": att.get("downloadUrl") or att.get("url"),
        "downloadPath": download_path(message_id, gmail_id, filename),
    }


def _outlook_attachment(att: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": att.get("id"),
        "filename": att.get("name") or att.get("filename") or DEFAULT_FILENAME,
        "mimeType": att.get("contentType") or att.get("mimeType") or OCTET_STREAM,
        "size": att.get("size") or 0,
        "downloadUrl": att.get("contentLocation") or att.get("downloadUrl"),
    }


def _part_attachment(part: Dict[str, Any], message_id: Any) -> Optional[Dict[str, Any]]:
    """A ``payload.parts`` entry with a file name; a body part is not one."""
    mime_type = part.get("mimeType", "")
    filename = part.get("filename")
    if mime_type in _BODY_MIME_TYPES or not filename:
        return None
    body = part.get("body") if isinstance(part.get("body"), dict) else {}
    gmail_id = body.get("attachmentId")
    return {
        "id": gmail_id or part.get("partId"),
        "filename": filename,
        "mimeType": mime_type,
        "size": body.get("size") or 0,
        "downloadUrl": None,
        "downloadPath": download_path(message_id, gmail_id, filename),
    }


def extract_attachments(item: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """The email's attachments, or ``None`` when it has none."""
    message_id = item.get("messageId") or item.get("id")
    attachments = [_gmail_attachment(att, message_id) for att in _dicts(item.get("attachmentList"))]
    if not attachments:
        attachments = [_outlook_attachment(att) for att in _dicts(item.get("attachments"))]
    payload = item.get("payload")
    if not attachments and isinstance(payload, dict):
        parts = (_part_attachment(part, message_id) for part in _dicts(payload.get("parts")))
        attachments = [part for part in parts if part]
    return attachments or None
