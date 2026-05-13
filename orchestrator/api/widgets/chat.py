"""
Widget Chat API
===============

Chat endpoint for embedded SDK widgets with SSE streaming.

Reuses the existing ``ChatService`` and ``StreamingChatService`` from
``consumers.chatbot`` so that widget conversations get the same agent,
memory, and tool-loop capabilities as the main chat UI.

Auth: ``widget_auth`` + ``require_permission("chat")``.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import AsyncGenerator, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.widgets.auth import WidgetAuthContext, require_permission, widget_auth
from core.database.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Widget Chat"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class WidgetChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    agent_id: Optional[str] = None  # UUID (public_id) or legacy integer id as string
    model_id: Optional[str] = None
    # PRD-007: page context + trigger reason for proactive engagement.
    # Both backwards-compatible (default None). When trigger_reason is set
    # (e.g. "proactive_opener"), the chat service uses an opener prompt
    # variant instead of treating ``message`` as a user utterance.
    page_context: Optional[dict] = None
    trigger_reason: Optional[str] = None


# PRD-007: trigger reasons that flip the agent into opener-generation mode.
# Anything else is treated as a normal user message.
PROACTIVE_TRIGGER_REASONS: frozenset[str] = frozenset({"proactive_opener"})


def _build_proactive_opener_message(page_context: dict) -> str:
    """Synthesize the user-side message for a proactive opener request.

    The widget never sends real user text for proactive openers — instead
    we synthesize a directive describing the page context. The agent's
    ``shopify-support`` skill recognises the ``[PROACTIVE_OPENER]`` prefix
    and generates a one-sentence contextual greeting.
    """
    ctx_summary_parts: list[str] = []
    if page_context.get("pageType"):
        ctx_summary_parts.append(f"page_type={page_context['pageType']}")
    if page_context.get("productTitle"):
        ctx_summary_parts.append(f"product=\"{page_context['productTitle']}\"")
    elif page_context.get("productHandle"):
        ctx_summary_parts.append(f"product_handle={page_context['productHandle']}")
    if page_context.get("productType"):
        ctx_summary_parts.append(f"product_type={page_context['productType']}")
    if page_context.get("collectionTitle"):
        ctx_summary_parts.append(f"collection=\"{page_context['collectionTitle']}\"")
    summary = ", ".join(ctx_summary_parts) if ctx_summary_parts else "no context"
    return f"[PROACTIVE_OPENER] Generate a contextual one-sentence opener. Context: {summary}"


class WidgetMessageOut(BaseModel):
    id: str
    role: str
    content: str
    created_at: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_widget_user_id(db: Session) -> int:
    """Return a default user id for widget-initiated chats.

    Widgets authenticate via API key / JWT and are not tied to a specific
    platform user.  We use a well-known user row (id=1) as the owner of
    widget conversations so that foreign-key constraints are satisfied.
    """
    result = db.execute(text("SELECT id FROM users WHERE id = 1 LIMIT 1")).fetchone()
    if not result:
        result = db.execute(text("SELECT id FROM users LIMIT 1")).fetchone()
    if not result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="No users found in the database",
        )
    return result[0]


# ---------------------------------------------------------------------------
# POST /chat — send message, get SSE stream back
# ---------------------------------------------------------------------------

@router.post("/chat")
async def widget_chat(
    body: WidgetChatRequest,
    auth: WidgetAuthContext = Depends(require_permission("chat")),
    db: Session = Depends(get_db),
):
    """Send a message and receive a streaming SSE response.

    The endpoint creates (or continues) a conversation scoped to the
    authenticated workspace and streams back chunks using Server-Sent
    Events.

    SSE event types emitted:
    - ``message``    — text delta from the assistant
    - ``tool-start`` — a tool invocation is beginning
    - ``tool-end``   — a tool invocation finished
    - ``tool-data``  — intermediate data from a tool
    - ``done``       — stream is complete
    """

    # ------------------------------------------------------------------
    # Import chat infrastructure (may not be available in every deploy)
    # ------------------------------------------------------------------
    try:
        from consumers.chatbot import ChatService, StreamingChatService
    except ImportError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service is not available in this deployment",
        )

    workspace_id = str(auth.workspace_id)
    user_id = _get_widget_user_id(db)

    # PRD-007: rewrite ``message`` for proactive opener requests so the agent
    # sees a directive, not an empty / placeholder user utterance from the SDK.
    is_proactive = (
        body.trigger_reason in PROACTIVE_TRIGGER_REASONS
        and body.page_context is not None
    )
    if is_proactive:
        body.message = _build_proactive_opener_message(body.page_context or {})

    chat_service = ChatService(db)
    streaming_service = StreamingChatService(db, workspace_id=workspace_id, widget_mode=True)

    # ------------------------------------------------------------------
    # Resolve or create conversation
    # ------------------------------------------------------------------
    chat_id: str

    if body.conversation_id:
        ws_uuid = uuid.UUID(workspace_id) if isinstance(workspace_id, str) else workspace_id
        chat = chat_service.get_chat(body.conversation_id, workspace_id=ws_uuid)
        if not chat:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )
        chat_id = body.conversation_id
    else:
        short_id = uuid.uuid4().hex[:8]
        title = f"{body.message[:40] if body.message else 'Widget Chat'} [{short_id}]"
        ws_uuid = uuid.UUID(workspace_id) if isinstance(workspace_id, str) else workspace_id
        chat = chat_service.create_chat(
            user_id=user_id,
            title=title,
            visibility="private",
            workspace_id=ws_uuid,
        )
        chat_id = str(chat.id)

    # ------------------------------------------------------------------
    # Persist the user message
    # ------------------------------------------------------------------
    chat_service.save_message(
        chat_id=chat_id,
        role="user",
        parts=[{"type": "text", "text": body.message}],
        workspace_id=workspace_id,
    )

    # ------------------------------------------------------------------
    # Build message history for the LLM
    # ------------------------------------------------------------------
    messages = chat_service.get_messages_by_chat_id(chat_id)
    message_history = [{"role": msg.role, "parts": msg.parts} for msg in messages]

    # API key can lock the agent — ignore client-provided agent_id when set
    # Resolve public_id (UUID) or legacy int to internal id
    from core.utils.agent_resolver import resolve_agent_id as _resolve_aid
    _raw_agent_ref = auth.default_agent_id or body.agent_id
    if _raw_agent_ref:
        effective_agent_id = _resolve_aid(db, _raw_agent_ref, workspace_id)
    else:
        # Fallback to workspace Auto agent (Phase 2 will replace the `or 1`)
        from api.chat import get_default_agent_id
        effective_agent_id = get_default_agent_id(db, workspace_id)

    # PRD-124: Resolve agent team for document scoping
    agent_team: Optional[str] = None
    try:
        from core.models.core import Agent
        agent_row = db.query(Agent.team).filter(Agent.id == effective_agent_id).first()
        agent_team = agent_row.team if agent_row else None
    except Exception:
        logger.debug("Could not resolve agent team for agent_id=%s", effective_agent_id)

    # ------------------------------------------------------------------
    # Stream
    # ------------------------------------------------------------------
    async def _event_stream() -> AsyncGenerator[str, None]:
        """Wrap the existing streaming service output as SSE events.

        The streaming service yields AI-SDK formatted chunks:
        - ``0:"text"\n``        — text content
        - ``d:{...}\n``         — data events (tool-start, finish, etc.)
        - ``{"_final_response": ...}`` — internal dict (skip)

        The widget SDK expects standard SSE:
        ``event: message\ndata: {"content":"text","conversation_id":"..."}\n\n``
        """
        try:
            async for chunk in streaming_service.stream_response_with_agent(
                chat_id=chat_id,
                messages=message_history,
                agent_id=effective_agent_id,
                user_id=user_id,
                team=agent_team,
            ):
                # Skip internal dicts (e.g. _final_response)
                if isinstance(chunk, dict):
                    continue

                # Skip non-string chunks
                if not isinstance(chunk, str):
                    continue

                # Parse AI-SDK text chunks: 0:"text content"\n
                if chunk.startswith('0:'):
                    text = chunk[2:].strip()
                    # Remove surrounding quotes: "text" → text
                    if text.startswith('"') and text.endswith('"'):
                        # Unescape JSON string
                        try:
                            text = json.loads(text)
                        except (json.JSONDecodeError, ValueError):
                            text = text[1:-1]
                    if text:
                        yield f"event: message\ndata: {json.dumps({'content': text, 'conversation_id': chat_id})}\n\n"

                # Forward tool events for SDK tool-start/tool-end support
                elif chunk.startswith('d:'):
                    try:
                        data = json.loads(chunk[2:].strip())
                        event_type = data.get("type", "")
                        if event_type == "tool-start":
                            yield f"event: tool-start\ndata: {json.dumps({'tool': data.get('data', {}).get('toolName', ''), 'arguments': data.get('data', {}).get('input', {})})}\n\n"
                        elif event_type == "tool-end":
                            yield f"event: tool-end\ndata: {json.dumps({'tool': data.get('data', {}).get('toolName', ''), 'result': data.get('data', {}).get('result')})}\n\n"
                    except (json.JSONDecodeError, ValueError):
                        pass

                # Skip other AI-SDK control chunks (e:, etc.)

        except Exception as exc:
            logger.exception("Widget chat streaming error for chat_id=%s", chat_id)
            yield f"event: error\ndata: {json.dumps({'message': str(exc)})}\n\n"
        finally:
            yield f"event: done\ndata: {json.dumps({'conversation_id': chat_id})}\n\n"

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# GET /chat/{conversation_id} — conversation history
# ---------------------------------------------------------------------------

@router.get("/chat/{conversation_id}", response_model=List[WidgetMessageOut])
async def widget_chat_history(
    conversation_id: str,
    auth: WidgetAuthContext = Depends(require_permission("chat")),
    db: Session = Depends(get_db),
):
    """Return the message history for a conversation.

    Only messages belonging to the authenticated workspace are returned.
    """
    workspace_id = str(auth.workspace_id)

    rows = db.execute(
        text(
            "SELECT id, role, parts, created_at "
            "FROM messages "
            "WHERE chat_id = :chat_id AND workspace_id = :ws "
            "ORDER BY created_at ASC"
        ),
        {"chat_id": conversation_id, "ws": workspace_id},
    ).fetchall()

    if not rows:
        # Either the conversation doesn't exist or it belongs to a
        # different workspace.  Return 404 in both cases.
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found or no messages",
        )

    result: list[WidgetMessageOut] = []
    for row in rows:
        # ``parts`` is a JSONB column stored as a list of dicts.
        # Extract a plain-text representation for the widget SDK.
        parts = row.parts if isinstance(row.parts, list) else []
        content = " ".join(
            p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")
        )
        result.append(
            WidgetMessageOut(
                id=str(row.id),
                role=row.role,
                content=content,
                created_at=row.created_at.isoformat() if row.created_at else "",
            )
        )

    return result
