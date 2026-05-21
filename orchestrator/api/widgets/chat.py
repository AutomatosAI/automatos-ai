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
import time
import uuid
from typing import AsyncGenerator, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.widgets.auth import WidgetAuthContext, require_permission, widget_auth
from core.database.database import get_db
from services.sites import get_default_site

logger = logging.getLogger(__name__)

# Tag every widget-chat log line so we can grep `widget_chat` in Railway.
_LOG_TAG = "widget_chat"


def _short(s: Optional[str], n: int = 80) -> str:
    """Truncate strings for logging. Avoids dumping huge messages into logs."""
    if s is None:
        return "<none>"
    s = str(s)
    return s if len(s) <= n else s[:n] + f"…(+{len(s) - n})"

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


# PRD-008-A.2: shopper utterances that should auto-open the callback form.
# Per-Site overrides land via ``site.settings.callback.intent_phrases``;
# this set is the fallback when the Site config is missing or empty.
_DEFAULT_CALLBACK_INTENT_PHRASES: tuple[str, ...] = (
    "speak to someone",
    "call me back",
    "talk to a human",
    "phone me",
    "give me a call",
    "can someone call",
)


def _matches_callback_intent(message: Optional[str], phrases: tuple[str, ...]) -> bool:
    """Return True if any intent phrase appears in ``message`` (case-insensitive
    substring match). Empty / None messages never match.
    """
    if not message or not phrases:
        return False
    msg = message.lower()
    return any(p.lower() in msg for p in phrases if p)


# Fields from page_context that the agent gets to ground openers on.
# Order matters — first match per group wins. Numeric/boolean fields are
# coerced to strings only when present + non-default.
_OPENER_CONTEXT_FIELDS: tuple[tuple[str, str], ...] = (
    ("pageType",          "page_type"),
    ("template",          "template"),
    # Product
    ("productTitle",      "product"),
    ("productType",       "product_type"),
    ("productVendor",     "vendor"),
    ("productPrice",      "price"),
    ("productAvailable",  "in_stock"),
    ("productHandle",     "product_handle"),
    # Collection (when on a collection page)
    ("collectionTitle",   "collection"),
    ("collectionHandle",  "collection_handle"),
    # Shop / locale
    ("shopDomain",        "shop"),
    ("shopCurrency",      "currency"),
    ("shopLocale",        "locale"),
    # Customer / cart
    ("customerId",        "logged_in_customer_id"),
    ("customerTags",      "customer_tags"),
    ("cartItemCount",     "cart_item_count"),
    ("cartTotalPrice",    "cart_total"),
)


def _format_opener_context_value(key: str, value) -> Optional[str]:
    """Render a single page-context value into the directive. Returns None
    if the value is empty/zero/false and shouldn't be sent to the agent."""
    if value is None or value == "" or value == 0 or value is False:
        return None
    if isinstance(value, str):
        return f'{key}="{value}"' if " " in value or '"' in value else f"{key}={value}"
    return f"{key}={value}"


def _build_proactive_opener_message(page_context: dict) -> str:
    """Synthesize the user-side message for a proactive opener request.

    The widget never sends real user text for proactive openers — instead
    we synthesize a directive carrying the FULL page context the agent
    needs to ground a contextual one-line opener.

    PRD-007 v0.4: previously only pageType + productTitle/Type leaked into
    the directive; agents had to make up everything else (price, vendor,
    availability). Now every populated page_context field is forwarded so
    the agent can lean on real facts before reaching for a tool call.
    """
    parts: list[str] = []
    for src_key, label in _OPENER_CONTEXT_FIELDS:
        rendered = _format_opener_context_value(label, page_context.get(src_key))
        if rendered is not None:
            parts.append(rendered)
    summary = ", ".join(parts) if parts else "no context"
    return (
        "[PROACTIVE_OPENER] Generate a contextual one-sentence opener "
        "(≤140 chars). RETURN PLAIN TEXT ONLY — no tool calls, no JSON, "
        "no markdown, no greetings. Use the facts below as your source of "
        "truth — do NOT invent specs, compatibility, or pricing the context "
        "doesn't include. If a fact you'd want isn't here, ask a question "
        "instead of fabricating. "
        f"Context: {summary}"
    )


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
    request: Request,
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
    # Tag this request for log correlation
    # ------------------------------------------------------------------
    req_id = (
        request.headers.get("X-Request-ID")
        or request.headers.get("x-railway-request-id")
        or uuid.uuid4().hex[:12]
    )
    started_at = time.perf_counter()
    origin = request.headers.get("Origin") or "?"
    log_extra = (
        f"[{_LOG_TAG} req={req_id} ws={auth.workspace_id} origin={origin}]"
    )
    logger.info(
        "%s REQUEST: agent_id=%s conv_id=%s trigger_reason=%s page_type=%s msg_len=%d msg_preview=%s",
        log_extra,
        body.agent_id,
        body.conversation_id,
        body.trigger_reason,
        (body.page_context or {}).get("pageType") if body.page_context else None,
        len(body.message or ""),
        _short(body.message),
    )

    # ------------------------------------------------------------------
    # Import chat infrastructure (may not be available in every deploy)
    # ------------------------------------------------------------------
    try:
        from consumers.chatbot import ChatService, StreamingChatService
    except ImportError:
        logger.error("%s SERVICE_UNAVAILABLE: consumers.chatbot import failed", log_extra)
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
        original_len = len(body.message or "")
        body.message = _build_proactive_opener_message(body.page_context or {})
        logger.info(
            "%s PROACTIVE_REWRITE: original_msg_len=%d new_msg_len=%d new_preview=%s",
            log_extra,
            original_len,
            len(body.message),
            _short(body.message),
        )
    elif body.trigger_reason:
        logger.warning(
            "%s UNKNOWN_TRIGGER_REASON: %s (page_context=%s) — proceeding as normal chat",
            log_extra,
            body.trigger_reason,
            "present" if body.page_context else "missing",
        )

    # PRD-008-A.2: detect callback intent BEFORE the LLM call so the SDK can
    # auto-open the phone-capture form. Skip if this is a proactive opener
    # (the synthetic message isn't real user input).
    is_callback_intent = False
    callback_product_context: Optional[str] = None
    if not is_proactive and body.message:
        try:
            site = get_default_site(db, uuid.UUID(workspace_id))
        except Exception:
            site = None
        if site is not None:
            cb_config = (site.settings or {}).get("callback") or {}
            if cb_config.get("enabled"):
                phrases = cb_config.get("intent_phrases") or list(_DEFAULT_CALLBACK_INTENT_PHRASES)
                if _matches_callback_intent(body.message, tuple(phrases)):
                    is_callback_intent = True
                    # Pull product title (preferred) or handle for the form heading.
                    pc = body.page_context or {}
                    callback_product_context = (
                        pc.get("productTitle") or pc.get("productHandle") or None
                    )
                    logger.info(
                        "%s CALLBACK_INTENT_MATCHED: phrase set size=%d product_context=%s",
                        log_extra, len(phrases), callback_product_context,
                    )
                    # Augment the message so the agent acknowledges gracefully
                    # rather than suggesting email — the form is already opening.
                    original_msg = body.message
                    body.message = (
                        f"[SYSTEM_NOTE: A phone callback form has just been opened "
                        f"in the shopper's chat panel. Acknowledge briefly and ask "
                        f"them to fill in their name and number — DO NOT suggest "
                        f"email or alternative contact methods.] "
                        f"User said: \"{original_msg}\""
                    )

    chat_service = ChatService(db)
    streaming_service = StreamingChatService(db, workspace_id=workspace_id, widget_mode=True)

    # ------------------------------------------------------------------
    # Resolve or create conversation
    # ------------------------------------------------------------------
    chat_id: str

    t_conv = time.perf_counter()
    if body.conversation_id:
        ws_uuid = uuid.UUID(workspace_id) if isinstance(workspace_id, str) else workspace_id
        chat = chat_service.get_chat(body.conversation_id, workspace_id=ws_uuid)
        if not chat:
            logger.warning(
                "%s CONV_NOT_FOUND: %s",
                log_extra,
                body.conversation_id,
            )
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Conversation not found",
            )
        chat_id = body.conversation_id
        logger.info("%s CONV_RESUMED: chat_id=%s", log_extra, chat_id)
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
        logger.info(
            "%s CONV_CREATED: chat_id=%s title=%s (took %.0fms)",
            log_extra,
            chat_id,
            _short(title, 60),
            (time.perf_counter() - t_conv) * 1000,
        )

    # ------------------------------------------------------------------
    # Persist the user message
    # ------------------------------------------------------------------
    t_save = time.perf_counter()
    chat_service.save_message(
        chat_id=chat_id,
        role="user",
        parts=[{"type": "text", "text": body.message}],
        workspace_id=workspace_id,
    )
    logger.debug(
        "%s USER_MSG_SAVED: chat_id=%s (took %.0fms)",
        log_extra,
        chat_id,
        (time.perf_counter() - t_save) * 1000,
    )

    # ------------------------------------------------------------------
    # Build message history for the LLM
    # ------------------------------------------------------------------
    messages = chat_service.get_messages_by_chat_id(chat_id)
    message_history = [{"role": msg.role, "parts": msg.parts} for msg in messages]
    logger.debug(
        "%s HISTORY_LOADED: %d messages", log_extra, len(message_history)
    )

    # API key can lock the agent — ignore client-provided agent_id when set
    # Resolve public_id (UUID) or legacy int to internal id
    from core.utils.agent_resolver import resolve_agent_id as _resolve_aid
    _raw_agent_ref = auth.default_agent_id or body.agent_id
    if _raw_agent_ref:
        effective_agent_id = _resolve_aid(db, _raw_agent_ref, workspace_id)
        logger.info(
            "%s AGENT_RESOLVED: input=%s -> id=%s (source=%s)",
            log_extra,
            _raw_agent_ref,
            effective_agent_id,
            "key_lock" if auth.default_agent_id else "body",
        )
    else:
        # Fallback to workspace Auto agent (Phase 2 will replace the `or 1`)
        from api.chat import get_default_agent_id
        effective_agent_id = get_default_agent_id(db, workspace_id)
        logger.info(
            "%s AGENT_DEFAULT: id=%s (workspace auto)",
            log_extra,
            effective_agent_id,
        )

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
        stream_start = time.perf_counter()
        first_chunk_at: Optional[float] = None
        first_yield_at: Optional[float] = None
        counts = {
            "total": 0, "dict": 0, "non_str": 0,
            "text": 0, "data_event": 0, "other": 0,
            "yielded_message": 0, "yielded_tool_start": 0, "yielded_tool_end": 0,
            "text_chars": 0,
        }
        logger.info(
            "%s STREAM_START: chat_id=%s agent_id=%s team=%s proactive=%s callback_intent=%s",
            log_extra, chat_id, effective_agent_id, agent_team, is_proactive, is_callback_intent,
        )
        # PRD-008-A.2: emit the open-callback-form signal first so the SDK
        # opens the phone-capture form even before the agent's text arrives.
        # Idempotent on the SDK side — listener guards against double-open.
        if is_callback_intent:
            payload = {
                "conversation_id": chat_id,
                "product_context": callback_product_context,
            }
            yield f"event: open-callback-form\ndata: {json.dumps(payload)}\n\n"
            logger.info(
                "%s OPEN_CALLBACK_FORM emitted (product_context=%s)",
                log_extra, callback_product_context,
            )
        try:
            async for chunk in streaming_service.stream_response_with_agent(
                chat_id=chat_id,
                messages=message_history,
                agent_id=effective_agent_id,
                user_id=user_id,
                team=agent_team,
                # PRD-007 fix: proactive openers must NOT call Composio tools.
                # The skill prompt forbids them (rule 5), but the LLM has been
                # ignoring the directive and calling tools anyway — yielding
                # data_event chunks but zero text (STREAM_NO_TEXT). Skipping
                # Composio injection at the source forces text-only output and
                # keeps openers within their <2s latency budget.
                skip_composio=is_proactive,
            ):
                counts["total"] += 1
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter()
                    logger.info(
                        "%s STREAM_FIRST_CHUNK: after %.0fms (type=%s, preview=%s)",
                        log_extra,
                        (first_chunk_at - stream_start) * 1000,
                        type(chunk).__name__,
                        _short(repr(chunk), 100),
                    )

                # Skip internal dicts (e.g. _final_response)
                if isinstance(chunk, dict):
                    counts["dict"] += 1
                    continue

                # Skip non-string chunks
                if not isinstance(chunk, str):
                    counts["non_str"] += 1
                    continue

                # Parse AI-SDK text chunks: 0:"text content"\n
                if chunk.startswith('0:'):
                    counts["text"] += 1
                    text = chunk[2:].strip()
                    # Remove surrounding quotes: "text" → text
                    if text.startswith('"') and text.endswith('"'):
                        # Unescape JSON string
                        try:
                            text = json.loads(text)
                        except (json.JSONDecodeError, ValueError):
                            text = text[1:-1]
                    if text:
                        counts["text_chars"] += len(text)
                        if first_yield_at is None:
                            first_yield_at = time.perf_counter()
                            logger.info(
                                "%s STREAM_FIRST_TEXT_YIELD: after %.0fms",
                                log_extra,
                                (first_yield_at - stream_start) * 1000,
                            )
                        counts["yielded_message"] += 1
                        yield f"event: message\ndata: {json.dumps({'content': text, 'conversation_id': chat_id})}\n\n"

                # Forward tool events for SDK tool-start/tool-end support
                elif chunk.startswith('d:'):
                    counts["data_event"] += 1
                    try:
                        data = json.loads(chunk[2:].strip())
                        event_type = data.get("type", "")
                        if event_type == "tool-start":
                            counts["yielded_tool_start"] += 1
                            yield f"event: tool-start\ndata: {json.dumps({'tool': data.get('data', {}).get('toolName', ''), 'arguments': data.get('data', {}).get('input', {})})}\n\n"
                        elif event_type == "tool-end":
                            counts["yielded_tool_end"] += 1
                            yield f"event: tool-end\ndata: {json.dumps({'tool': data.get('data', {}).get('toolName', ''), 'result': data.get('data', {}).get('result')})}\n\n"
                    except (json.JSONDecodeError, ValueError):
                        pass

                # Skip other AI-SDK control chunks (e:, etc.)
                else:
                    counts["other"] += 1

        except Exception as exc:
            logger.exception(
                "%s STREAM_ERROR: chat_id=%s after %.0fms",
                log_extra, chat_id, (time.perf_counter() - stream_start) * 1000,
            )
            yield f"event: error\ndata: {json.dumps({'message': str(exc)})}\n\n"
        finally:
            elapsed_ms = (time.perf_counter() - stream_start) * 1000
            total_ms = (time.perf_counter() - started_at) * 1000
            log_level = logger.info if counts["yielded_message"] > 0 or counts["yielded_tool_start"] > 0 else logger.warning
            log_level(
                "%s STREAM_DONE: chat_id=%s stream=%.0fms total=%.0fms counts=%s",
                log_extra, chat_id, elapsed_ms, total_ms, counts,
            )
            if counts["total"] == 0:
                logger.warning(
                    "%s STREAM_EMPTY: no chunks from streaming service. Agent=%s may be misconfigured or LLM call returned nothing.",
                    log_extra, effective_agent_id,
                )
            elif counts["yielded_message"] == 0:
                logger.warning(
                    "%s STREAM_NO_TEXT: agent produced %d chunks but no text (text_count=%d, data_event=%d, dict=%d). Skill prompt may be missing or agent is tool-only.",
                    log_extra, counts["total"], counts["text"], counts["data_event"], counts["dict"],
                )
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
