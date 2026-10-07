"""A Gmail attachment the chat's email widget downloads: its Composio action, its API path, its id rules.

A Gmail attachment carries no URL. Its bytes come from Composio's Gmail
get-attachment action on the workspace's own connection, served by the
platform's own authenticated route (``api/widget_email.py``). The email
extractor (``modules/tools/formatting/email_attachments.py``) puts that route's
path on the attachment as ``downloadPath``; both sides read the path and the id
rules from here.

Gmail message and attachment ids are URL-safe tokens (hex, base64url). Only
``[A-Za-z0-9_-]`` of a sane length is taken, so an id can never add a path
segment, a query or another route.
"""
from __future__ import annotations

import re
from typing import Any, Optional
from urllib.parse import quote

GMAIL_APP = "GMAIL"
GET_ATTACHMENT_ACTION = "GMAIL_GET_ATTACHMENT"
DOWNLOAD_ROUTE_PREFIX = "/api/emails/attachments/gmail"

# A Gmail message id is 16 hex characters; an attachment id is a long
# base64url token (several hundred characters is normal).
MAX_MESSAGE_ID_CHARS = 128
MAX_ATTACHMENT_ID_CHARS = 2048
# The longest file name a download path carries.
MAX_FILENAME_CHARS = 255

_ID = re.compile(r"[A-Za-z0-9_-]+")


def _is_id(value: Any, max_chars: int) -> bool:
    return isinstance(value, str) and len(value) <= max_chars and _ID.fullmatch(value) is not None


def is_gmail_message_id(value: Any) -> bool:
    """A Gmail message id: ``[A-Za-z0-9_-]``, at most ``MAX_MESSAGE_ID_CHARS``."""
    return _is_id(value, MAX_MESSAGE_ID_CHARS)


def is_gmail_attachment_id(value: Any) -> bool:
    """A Gmail attachment id: ``[A-Za-z0-9_-]``, at most ``MAX_ATTACHMENT_ID_CHARS``."""
    return _is_id(value, MAX_ATTACHMENT_ID_CHARS)


def download_path(message_id: Any, attachment_id: Any, filename: Any = None) -> Optional[str]:
    """The API path that serves this attachment, or ``None`` when either id is
    not a Gmail id. The file name rides in the query, encoded."""
    if not (is_gmail_message_id(message_id) and is_gmail_attachment_id(attachment_id)):
        return None
    path = f"{DOWNLOAD_ROUTE_PREFIX}/{message_id}/{attachment_id}"
    if isinstance(filename, str) and filename.strip():
        path += f"?filename={quote(filename.strip()[:MAX_FILENAME_CHARS], safe='')}"
    return path
