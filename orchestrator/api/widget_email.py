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

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from api.widget_email_attachment import (
    INVALID_LINK,
    NOT_CONNECTED,
    UNREADABLE,
    AttachmentError,
    attachment_file,
    download_headers,
    safe_content_type,
    safe_filename,
)
from core.auth.dependencies import RequestContext
from core.auth.workspace_permission import require_workspace_permission
from core.auth.hybrid import get_request_context_hybrid
from core.composio.entity_manager import EntityManager
from core.composio.gmail_attachments import (
    GET_ATTACHMENT_ACTION,
    GMAIL_APP,
    is_gmail_attachment_id,
    is_gmail_message_id,
)
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


async def _execute_email_action(
    ctx: RequestContext, action: str, params: Dict[str, Any]
) -> Dict[str, Any]:
    """Route a widget email action through the platform SPINE (PRD-192 S5).

    Previously these endpoints called the raw Composio client — no policy
    gate, no budget admission, no Art.12 audit row, no telemetry. They now
    dispatch through the same ``UnifiedToolExecutor`` chokepoint as every
    other tool, as the ``composio_execute`` meta-tool (the per-action name
    rides in ``action``), with the HUMAN-DIRECT actor marker: a user clicking
    Send in the widget IS the approval, so the route gate treats the ask as
    satisfied (locked #6) while budget admission and the audit row still
    apply. Agent-initiated calls on this integration carry no marker and
    still ask. Returns the executor's ``{success, data, error}`` envelope —
    the same shape the endpoints already parse.
    """
    from modules.tools.tool_router import execute_tool

    return await execute_tool(
        "composio_execute",
        {"action": action, "params": params},
        agent_id=0,  # no agent — a human-direct API request
        workspace_id=ctx.workspace_id,
        caller_context={
            "actor_type": "user_direct",
            "user_id": getattr(ctx.user, "id", None),
            "auth_type": ctx.auth_type,
            "surface": "widget_email",
        },
    )


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
    _get_entity_id(db, ctx.workspace_id)  # 404 early when email isn't connected

    # Build Composio action params.  GMAIL_FETCH_EMAILS is the standard action.
    params: Dict[str, Any] = {"max_results": limit}
    if page_token:
        params["page_token"] = page_token
    if query:
        params["query"] = query
    if label:
        params["label_ids"] = [label]

    try:
        result = await _execute_email_action(ctx, "GMAIL_FETCH_EMAILS", params)
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


def _gmail_connected(db: Session, workspace_id: UUID) -> bool:
    """Whether this workspace's own Composio entity has an active Gmail connection."""
    apps = EntityManager(db).get_connected_apps(workspace_id)
    return GMAIL_APP in {str(app).upper().strip() for app in apps}


@router.get("/attachments/gmail/{message_id}/{attachment_id}")
async def download_gmail_attachment(
    message_id: str,
    attachment_id: str,
    filename: Optional[str] = Query(None, max_length=1024, description="The name to save the file under"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> Response:
    """One attachment of a Gmail message in the caller's workspace, as a download.

    Gmail attachments carry no URL: the bytes come from Composio's Gmail
    get-attachment action on the workspace's own connection, through the spine
    like every widget email call. Only Gmail-shaped ids are taken; the response
    is always ``Content-Disposition: attachment`` with a sanitised name, a safe
    Content-Type and the size cap (``api/widget_email_attachment.py``).
    """
    if not (is_gmail_message_id(message_id) and is_gmail_attachment_id(attachment_id)):
        raise HTTPException(status_code=400, detail=INVALID_LINK)
    name = safe_filename(filename)
    if not await run_in_threadpool(_gmail_connected, db, ctx.workspace_id):
        raise HTTPException(status_code=404, detail=NOT_CONNECTED)
    params = {"message_id": message_id, "attachment_id": attachment_id, "file_name": name}
    try:
        result = await _execute_email_action(ctx, GET_ATTACHMENT_ACTION, params)
        data, mimetype = await attachment_file(result)
    except AttachmentError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from None
    except Exception:
        logger.exception("[EmailAttachment] %s failed for workspace %s", GET_ATTACHMENT_ACTION, ctx.workspace_id)
        raise HTTPException(status_code=502, detail=UNREADABLE) from None
    return Response(content=data, media_type=safe_content_type(mimetype, name), headers=download_headers(name))


@router.get("/{email_id}", response_model=EmailDetailResponse)
async def get_email(
    email_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> EmailDetailResponse:
    """Return a single email by its provider ID."""
    _get_entity_id(db, ctx.workspace_id)  # 404 early when email isn't connected

    try:
        result = await _execute_email_action(
            ctx, "GMAIL_GET_EMAIL", {"message_id": email_id}
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


@router.post("", response_model=SendEmailResponse, dependencies=[Depends(require_workspace_permission("agents:execute"))])
async def send_email(
    payload: SendEmailRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> SendEmailResponse:
    """Send a new email via the connected mail provider.

    PRD-192 S5: an external side-effect on the spine — the user's click IS the
    approval (human-direct rule), budget admission + the audit row still apply.
    """
    _get_entity_id(db, ctx.workspace_id)  # 404 early when email isn't connected

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
        result = await _execute_email_action(ctx, "GMAIL_SEND_EMAIL", params)
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


@router.post("/{email_id}/reply", response_model=ReplyEmailResponse, dependencies=[Depends(require_workspace_permission("agents:execute"))])
async def reply_to_email(
    email_id: str,
    payload: ReplyEmailRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
) -> ReplyEmailResponse:
    """Reply to an existing email thread.

    PRD-192 S5: an external side-effect on the spine — human-direct rule
    applies (the click is the approval); budget + audit still fire.
    """
    _get_entity_id(db, ctx.workspace_id)  # 404 early when email isn't connected

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
        result = await _execute_email_action(ctx, "GMAIL_REPLY_TO_EMAIL", params)
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
