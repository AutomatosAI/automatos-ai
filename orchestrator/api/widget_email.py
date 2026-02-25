"""
Widget Email API (US-012)
=========================

Backend API endpoints for the email widget.  Proxies Gmail / Outlook
operations through Composio so the frontend never talks to mail providers
directly.

All endpoints require workspace auth via ``get_request_context_hybrid``.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from core.auth.dependencies import RequestContext
from core.auth.hybrid import get_request_context_hybrid
from core.composio.client import ComposioClient, get_composio_client
from core.composio.entity_manager import EntityManager
from core.database.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/emails", tags=["Widget Emails"])


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class EmailAddress(BaseModel):
    email: str
    name: Optional[str] = None


class EmailAttachment(BaseModel):
    filename: str
    mime_type: Optional[str] = None
    size: Optional[int] = None
    url: Optional[str] = None


class EmailSummary(BaseModel):
    """Lightweight representation returned by the list endpoint."""

    id: str
    thread_id: Optional[str] = None
    subject: Optional[str] = None
    snippet: Optional[str] = None
    from_address: Optional[EmailAddress] = None
    to_addresses: List[EmailAddress] = Field(default_factory=list)
    date: Optional[str] = None
    is_read: bool = True
    has_attachments: bool = False
    labels: List[str] = Field(default_factory=list)


class EmailDetail(EmailSummary):
    """Full email body returned by the single-email endpoint."""

    body_text: Optional[str] = None
    body_html: Optional[str] = None
    cc_addresses: List[EmailAddress] = Field(default_factory=list)
    bcc_addresses: List[EmailAddress] = Field(default_factory=list)
    attachments: List[EmailAttachment] = Field(default_factory=list)
    in_reply_to: Optional[str] = None
    references: Optional[str] = None


class EmailListResponse(BaseModel):
    emails: List[EmailSummary]
    total: int
    next_page_token: Optional[str] = None
    provider: str = "composio"


class EmailDetailResponse(BaseModel):
    email: EmailDetail
    provider: str = "composio"


class SendEmailRequest(BaseModel):
    to: List[str] = Field(..., min_length=1)
    subject: str = Field(..., min_length=1, max_length=998)
    body: str = Field(..., min_length=1)
    cc: List[str] = Field(default_factory=list)
    bcc: List[str] = Field(default_factory=list)
    is_html: bool = False


class SendEmailResponse(BaseModel):
    success: bool
    message_id: Optional[str] = None
    thread_id: Optional[str] = None
    error: Optional[str] = None


class ReplyEmailRequest(BaseModel):
    body: str = Field(..., min_length=1)
    cc: List[str] = Field(default_factory=list)
    bcc: List[str] = Field(default_factory=list)
    is_html: bool = False


class ReplyEmailResponse(BaseModel):
    success: bool
    message_id: Optional[str] = None
    thread_id: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_entity_id(db: Session, workspace_id: UUID) -> str:
    """Resolve the Composio entity_id for a workspace, or raise 404."""
    manager = EntityManager(db)
    entity = manager.get_entity_by_workspace(workspace_id)
    if not entity or not entity.get("composio_entity_id"):
        raise HTTPException(
            status_code=404,
            detail="No Composio entity found for this workspace. Connect a mail provider first.",
        )
    return entity["composio_entity_id"]


def _parse_email_summary(raw: Dict[str, Any]) -> EmailSummary:
    """Best-effort parse of a Composio Gmail/Outlook list item into our model."""
    # Gmail typically returns: id, threadId, snippet, payload.headers, labelIds
    # Outlook returns: id, subject, bodyPreview, from, toRecipients, receivedDateTime

    from_addr = None
    from_raw = raw.get("from") or raw.get("sender")
    if isinstance(from_raw, dict):
        email_addr = from_raw.get("emailAddress", from_raw)
        from_addr = EmailAddress(
            email=email_addr.get("address", email_addr.get("email", "")),
            name=email_addr.get("name"),
        )
    elif isinstance(from_raw, str):
        from_addr = EmailAddress(email=from_raw)

    to_addrs: List[EmailAddress] = []
    for recip in raw.get("toRecipients", raw.get("to", [])) or []:
        if isinstance(recip, dict):
            ea = recip.get("emailAddress", recip)
            to_addrs.append(EmailAddress(
                email=ea.get("address", ea.get("email", "")),
                name=ea.get("name"),
            ))
        elif isinstance(recip, str):
            to_addrs.append(EmailAddress(email=recip))

    labels: List[str] = []
    raw_labels = raw.get("labelIds") or raw.get("categories") or []
    if isinstance(raw_labels, list):
        labels = [str(lbl) for lbl in raw_labels]

    return EmailSummary(
        id=str(raw.get("id", "")),
        thread_id=raw.get("threadId") or raw.get("conversationId"),
        subject=raw.get("subject"),
        snippet=raw.get("snippet") or raw.get("bodyPreview"),
        from_address=from_addr,
        to_addresses=to_addrs,
        date=raw.get("date") or raw.get("receivedDateTime") or raw.get("internalDate"),
        is_read=raw.get("isRead", not ("UNREAD" in labels)),
        has_attachments=bool(raw.get("hasAttachments", False)),
        labels=labels,
    )


def _parse_email_detail(raw: Dict[str, Any]) -> EmailDetail:
    """Parse a full email response into our detail model."""
    summary = _parse_email_summary(raw)

    cc_addrs: List[EmailAddress] = []
    for recip in raw.get("ccRecipients", raw.get("cc", [])) or []:
        if isinstance(recip, dict):
            ea = recip.get("emailAddress", recip)
            cc_addrs.append(EmailAddress(
                email=ea.get("address", ea.get("email", "")),
                name=ea.get("name"),
            ))
        elif isinstance(recip, str):
            cc_addrs.append(EmailAddress(email=recip))

    bcc_addrs: List[EmailAddress] = []
    for recip in raw.get("bccRecipients", raw.get("bcc", [])) or []:
        if isinstance(recip, dict):
            ea = recip.get("emailAddress", recip)
            bcc_addrs.append(EmailAddress(
                email=ea.get("address", ea.get("email", "")),
                name=ea.get("name"),
            ))
        elif isinstance(recip, str):
            bcc_addrs.append(EmailAddress(email=recip))

    attachments: List[EmailAttachment] = []
    for att in raw.get("attachments", []) or []:
        if isinstance(att, dict):
            attachments.append(EmailAttachment(
                filename=att.get("filename") or att.get("name", "unknown"),
                mime_type=att.get("mimeType") or att.get("contentType"),
                size=att.get("size"),
                url=att.get("url") or att.get("contentUrl"),
            ))

    body = raw.get("body", {})
    body_text: Optional[str] = None
    body_html: Optional[str] = None
    if isinstance(body, dict):
        content_type = body.get("contentType", "text")
        content = body.get("content", "")
        if "html" in content_type.lower():
            body_html = content
        else:
            body_text = content
    elif isinstance(body, str):
        body_text = body

    # Gmail sometimes puts body in payload.parts
    if not body_text and not body_html:
        body_text = raw.get("bodyText") or raw.get("body_text")
        body_html = raw.get("bodyHtml") or raw.get("body_html")

    return EmailDetail(
        id=summary.id,
        thread_id=summary.thread_id,
        subject=summary.subject,
        snippet=summary.snippet,
        from_address=summary.from_address,
        to_addresses=summary.to_addresses,
        date=summary.date,
        is_read=summary.is_read,
        has_attachments=summary.has_attachments,
        labels=summary.labels,
        body_text=body_text,
        body_html=body_html,
        cc_addresses=cc_addrs,
        bcc_addresses=bcc_addrs,
        attachments=attachments,
        in_reply_to=raw.get("inReplyTo") or raw.get("in_reply_to"),
        references=raw.get("references"),
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=EmailListResponse)
async def list_emails(
    limit: int = Query(20, ge=1, le=100),
    page_token: Optional[str] = Query(None, description="Pagination token from previous response"),
    query: Optional[str] = Query(None, description="Search filter (e.g. 'is:unread')"),
    label: Optional[str] = Query(None, description="Label/folder filter (e.g. 'INBOX')"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> EmailListResponse:
    """List emails from the connected Gmail or Outlook account via Composio."""
    entity_id = _get_entity_id(db, ctx.workspace_id)
    client = get_composio_client()

    # Build Composio action params.  GMAIL_FETCH_EMAILS is the standard action.
    params: Dict[str, Any] = {"max_results": limit}
    if page_token:
        params["page_token"] = page_token
    if query:
        params["query"] = query
    if label:
        params["label_ids"] = [label]

    try:
        result = client.execute_action(
            action="GMAIL_FETCH_EMAILS",
            params=params,
            entity_id=entity_id,
        )
    except Exception as exc:
        logger.error("Composio GMAIL_FETCH_EMAILS failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Email provider error: {exc}")

    if not result.get("success"):
        error_msg = result.get("error", "Unknown error fetching emails")
        logger.warning("list_emails failed: %s", error_msg)
        raise HTTPException(status_code=502, detail=error_msg)

    data = result.get("data") or {}

    # Composio may return the list under various keys
    raw_emails: List[Dict[str, Any]] = []
    if isinstance(data, list):
        raw_emails = data
    elif isinstance(data, dict):
        raw_emails = (
            data.get("messages")
            or data.get("emails")
            or data.get("data")
            or data.get("value")
            or []
        )

    emails = [_parse_email_summary(e) for e in raw_emails if isinstance(e, dict)]

    next_token: Optional[str] = None
    if isinstance(data, dict):
        next_token = data.get("nextPageToken") or data.get("next_page_token")

    return EmailListResponse(
        emails=emails,
        total=len(emails),
        next_page_token=next_token,
        provider="composio",
    )


@router.get("/{email_id}", response_model=EmailDetailResponse)
async def get_email(
    email_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> EmailDetailResponse:
    """Return a single email by its provider ID."""
    entity_id = _get_entity_id(db, ctx.workspace_id)
    client = get_composio_client()

    try:
        result = client.execute_action(
            action="GMAIL_GET_EMAIL",
            params={"message_id": email_id},
            entity_id=entity_id,
        )
    except Exception as exc:
        logger.error("Composio GMAIL_GET_EMAIL failed: %s", exc)
        raise HTTPException(status_code=502, detail=f"Email provider error: {exc}")

    if not result.get("success"):
        error_msg = result.get("error", "Unknown error fetching email")
        logger.warning("get_email(%s) failed: %s", email_id, error_msg)
        raise HTTPException(status_code=502, detail=error_msg)

    data = result.get("data") or {}
    if isinstance(data, dict) and "data" in data:
        data = data["data"]

    email = _parse_email_detail(data if isinstance(data, dict) else {})
    if not email.id:
        email.id = email_id

    return EmailDetailResponse(email=email, provider="composio")


@router.post("", response_model=SendEmailResponse)
async def send_email(
    payload: SendEmailRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> SendEmailResponse:
    """Send a new email via the connected mail provider."""
    entity_id = _get_entity_id(db, ctx.workspace_id)
    client = get_composio_client()

    params: Dict[str, Any] = {
        "recipient_email": payload.to[0] if len(payload.to) == 1 else payload.to,
        "subject": payload.subject,
        "body": payload.body,
    }
    if payload.cc:
        params["cc"] = payload.cc
    if payload.bcc:
        params["bcc"] = payload.bcc
    if payload.is_html:
        params["is_html"] = True

    try:
        result = client.execute_action(
            action="GMAIL_SEND_EMAIL",
            params=params,
            entity_id=entity_id,
        )
    except Exception as exc:
        logger.error("Composio GMAIL_SEND_EMAIL failed: %s", exc)
        return SendEmailResponse(success=False, error=str(exc))

    if not result.get("success"):
        return SendEmailResponse(
            success=False,
            error=result.get("error", "Failed to send email"),
        )

    data = result.get("data") or {}
    if isinstance(data, dict):
        return SendEmailResponse(
            success=True,
            message_id=data.get("id") or data.get("messageId") or data.get("message_id"),
            thread_id=data.get("threadId") or data.get("thread_id"),
        )

    return SendEmailResponse(success=True)


@router.post("/{email_id}/reply", response_model=ReplyEmailResponse)
async def reply_to_email(
    email_id: str,
    payload: ReplyEmailRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> ReplyEmailResponse:
    """Reply to an existing email thread."""
    entity_id = _get_entity_id(db, ctx.workspace_id)
    client = get_composio_client()

    params: Dict[str, Any] = {
        "message_id": email_id,
        "body": payload.body,
    }
    if payload.cc:
        params["cc"] = payload.cc
    if payload.bcc:
        params["bcc"] = payload.bcc
    if payload.is_html:
        params["is_html"] = True

    try:
        result = client.execute_action(
            action="GMAIL_REPLY_TO_EMAIL",
            params=params,
            entity_id=entity_id,
        )
    except Exception as exc:
        logger.error("Composio GMAIL_REPLY_TO_EMAIL failed: %s", exc)
        return ReplyEmailResponse(success=False, error=str(exc))

    if not result.get("success"):
        return ReplyEmailResponse(
            success=False,
            error=result.get("error", "Failed to reply to email"),
        )

    data = result.get("data") or {}
    if isinstance(data, dict):
        return ReplyEmailResponse(
            success=True,
            message_id=data.get("id") or data.get("messageId") or data.get("message_id"),
            thread_id=data.get("threadId") or data.get("thread_id"),
        )

    return ReplyEmailResponse(success=True)
