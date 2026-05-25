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
from modules.tools.widget_callback import (
    WIDGET_OPEN_CALLBACK_FORM_NAME,
    WIDGET_SIGNAL_KEY,
    WIDGET_SIGNAL_OPEN_CALLBACK_FORM,
)

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


def _build_proactive_opener_message(
    page_context: dict,
    related_products: Optional[list] = None,
) -> str:
    """Synthesize the user-side message for a proactive opener request.

    The widget never sends real user text for proactive openers — instead
    we synthesize a directive carrying the FULL page context the agent
    needs to ground a contextual one-line opener.

    PRD-007 v0.4: previously only pageType + productTitle/Type leaked into
    the directive; agents had to make up everything else (price, vendor,
    availability). Now every populated page_context field is forwarded so
    the agent can lean on real facts before reaching for a tool call.

    PRD-007 addendum (post PRD-009 Layer 2): when ``related_products`` is
    supplied — top FBT pair, top in-collection sibling, top vendor sibling
    — they're appended as facts the agent can weave into the opener. The
    agent is instructed to lead with FBT signal when present (real
    customer co-purchase pattern, highest leverage) and fall back to the
    catalog siblings otherwise. Always real data; never invented.
    """
    parts: list[str] = []
    for src_key, label in _OPENER_CONTEXT_FIELDS:
        rendered = _format_opener_context_value(label, page_context.get(src_key))
        if rendered is not None:
            parts.append(rendered)
    summary = ", ".join(parts) if parts else "no context"

    related_block = ""
    if related_products:
        # Render each related product as a one-line fact with provenance,
        # so the agent can cite naturally (e.g. "often bought with X — 12 of
        # 57 orders"). Order matters: FBT first (strongest signal), then
        # collection / vendor siblings as fall-backs.
        rel_order = {
            "frequently_bought_with": 0,
            "in_collection": 1,
            "by_vendor": 2,
        }
        sorted_rel = sorted(
            related_products,
            key=lambda p: rel_order.get(p.get("relation", ""), 99),
        )
        rendered_rel = []
        for p in sorted_rel:
            label = p.get("label", "?")
            rel = p.get("relation", "")
            if rel == "frequently_bought_with" and p.get("co_count"):
                rendered_rel.append(
                    f'"{label}" (bought together in {p["co_count"]} '
                    f'of {p.get("total_orders", "?")} orders)'
                )
            elif rel == "in_collection":
                rendered_rel.append(f'"{label}" (same collection)')
            elif rel == "by_vendor":
                rendered_rel.append(f'"{label}" (same vendor)')
            else:
                rendered_rel.append(f'"{label}"')
        related_block = (
            " Related from order/catalog graph (use these naturally — "
            "prefer the order-pair signal when present, else mention the "
            "collection/vendor sibling as a starter for conversation): "
            + "; ".join(rendered_rel)
        )

    return (
        "[PROACTIVE_OPENER] Generate a contextual one-sentence opener "
        "(≤140 chars). RETURN PLAIN TEXT ONLY — no tool calls, no JSON, "
        "no markdown, no greetings. Use the facts below as your source of "
        "truth — do NOT invent specs, compatibility, or pricing the context "
        "doesn't include. If a fact you'd want isn't here, ask a question "
        "instead of fabricating. "
        f"Context: {summary}.{related_block}"
    )


async def _resolve_graph_related_products(
    workspace_id: str,
    page_context: dict,
    *,
    max_per_relation: int = 1,
) -> list[dict]:
    """Pull top related products from the workspace knowledge graph.

    Looks up the seed product node by Shopify handle in page_context, then
    walks 1-hop edges by relation type:
      - frequently_bought_with (highest co_count wins — real customer signal)
      - in_collection (most-connected sibling — likely a category anchor)
      - by_vendor (any same-vendor sibling)

    Returns at most ``max_per_relation`` entries per relation type. Empty
    list when no seed product, no graph, or any failure — caller treats
    that as "fall back to plain Layer-1 opener".
    """
    handle = (page_context or {}).get("productHandle")
    title = (page_context or {}).get("productTitle")
    if not handle and not title:
        return []  # No seed (cart/checkout/collection page) — nothing to traverse

    try:
        from modules.knowledge.graph_service import GraphifyService

        gs = GraphifyService()
        graph = await gs.load_graph(workspace_id)
        if graph is None:
            return []

        # Find the seed node by handle (preferred) or label match.
        seed_id = None
        for node_id, attrs in graph.nodes(data=True):
            node_attrs = attrs.get("attrs") or {}
            if handle and node_attrs.get("handle") == handle:
                seed_id = node_id
                break
        if seed_id is None and title:
            for node_id, attrs in graph.nodes(data=True):
                if (attrs.get("file_type") == "shopify_product"
                        and attrs.get("label") == title):
                    seed_id = node_id
                    break
        if seed_id is None:
            return []

        # Walk 1-hop, group by relation type, sort each group by signal strength.
        by_relation: dict[str, list[dict]] = {}
        for u, v, edata in graph.edges(seed_id, data=True):
            rel = (edata.get("relation") or "").lower()
            if rel not in ("frequently_bought_with", "in_collection", "by_vendor"):
                continue
            other = v if u == seed_id else u
            other_attrs = graph.nodes[other]
            # Only return product nodes (skip variants/vendors as targets —
            # those aren't recommendable on their own to a shopper)
            if other_attrs.get("file_type") not in ("shopify_product", "shopify_collection"):
                # in_collection targets ARE collections; that's fine for catalog framing.
                if rel != "in_collection":
                    continue
            edge_attrs = edata.get("attrs") or {}
            by_relation.setdefault(rel, []).append({
                "relation": rel,
                "label": other_attrs.get("label") or other,
                "type": other_attrs.get("file_type", ""),
                "confidence": edata.get("confidence_score", 0),
                "co_count": edge_attrs.get("co_count"),
                "total_orders": edge_attrs.get("total_orders"),
                "weight": edata.get("weight", 0),
            })

        # FBT: sort by co_count (raw signal strength).
        by_relation.get("frequently_bought_with", []).sort(
            key=lambda p: -(p.get("co_count") or 0)
        )
        # Collection / vendor: arbitrary — take first.

        out: list[dict] = []
        for rel in ("frequently_bought_with", "in_collection", "by_vendor"):
            out.extend(by_relation.get(rel, [])[:max_per_relation])
        return out

    except Exception as e:  # noqa: BLE001 — opener falls back gracefully
        logger.warning("_resolve_graph_related_products failed: %s", e)
        return []


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
        # PRD-007 addendum (post PRD-009 Layer 2): for product pages, pull
        # the top 1 FBT pair + collection sibling + vendor sibling from the
        # workspace graph and inject them as opener facts. Returns empty
        # list for non-product pages or when the graph has no signal for
        # this product — opener silently falls back to Layer-1-only.
        related_products = await _resolve_graph_related_products(
            workspace_id, body.page_context or {},
        )
        body.message = _build_proactive_opener_message(
            body.page_context or {},
            related_products=related_products,
        )
        logger.info(
            "%s PROACTIVE_REWRITE: original_msg_len=%d new_msg_len=%d related=%d new_preview=%s",
            log_extra,
            original_len,
            len(body.message),
            len(related_products),
            _short(body.message),
        )
    elif body.trigger_reason:
        logger.warning(
            "%s UNKNOWN_TRIGGER_REASON: %s (page_context=%s) — proceeding as normal chat",
            log_extra,
            body.trigger_reason,
            "present" if body.page_context else "missing",
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
            "%s STREAM_START: chat_id=%s agent_id=%s team=%s proactive=%s",
            log_extra, chat_id, effective_agent_id, agent_team, is_proactive,
        )
        # PRD-008-A.2: track tool calls so the widget_open_callback_form
        # tool-end (or its tool-data signal) can be converted into the
        # SSE `event: open-callback-form` event for the SDK. We capture
        # product_context from tool-start args so it survives even if the
        # LLM omits it from the result payload.
        callback_form_args: dict = {}
        callback_form_emitted = False

        def _emit_open_callback_form(product_context: Optional[str]) -> str:
            payload = {
                "conversation_id": chat_id,
                "product_context": product_context,
            }
            logger.info(
                "%s OPEN_CALLBACK_FORM emitted (product_context=%s)",
                log_extra, product_context,
            )
            return f"event: open-callback-form\ndata: {json.dumps(payload)}\n\n"

        try:
            async for chunk in streaming_service.stream_response_with_agent(
                chat_id=chat_id,
                messages=message_history,
                agent_id=effective_agent_id,
                user_id=user_id,
                team=agent_team,
                # PRD-007 v0.5 — proactive openers must produce plain text only.
                # `skip_composio` alone wasn't enough: the agent still had ~45
                # platform tools loaded from its skill and would call one (40s
                # with text_count=0). force_text_only=True clears use_tools
                # entirely so the LLM can only emit text. Keeps openers within
                # the <2s latency budget.
                skip_composio=is_proactive,
                force_text_only=is_proactive,
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
                        inner = data.get("data") or {}
                        tool_name_in_event = inner.get("toolName") or ""
                        if event_type == "tool-start":
                            counts["yielded_tool_start"] += 1
                            if tool_name_in_event == WIDGET_OPEN_CALLBACK_FORM_NAME:
                                # Stash args so the tool-data / tool-end bridge can read
                                # product_context even when the LLM omits it from the result.
                                callback_form_args = inner.get("input") or {}
                            yield f"event: tool-start\ndata: {json.dumps({'tool': tool_name_in_event, 'arguments': inner.get('input', {})})}\n\n"
                        elif event_type == "tool-end":
                            counts["yielded_tool_end"] += 1
                            yield f"event: tool-end\ndata: {json.dumps({'tool': tool_name_in_event, 'result': inner.get('result')})}\n\n"
                        elif event_type == "tool-data":
                            # PRD-008-A.2: bridge widget signals to SSE events.
                            # The widget_open_callback_form tool emits a tool-data
                            # chunk carrying `_widget_signal: "open-callback-form"`
                            # which we translate into the dedicated SSE event the
                            # SDK already subscribes to.
                            if (
                                not callback_form_emitted
                                and inner.get(WIDGET_SIGNAL_KEY) == WIDGET_SIGNAL_OPEN_CALLBACK_FORM
                            ):
                                product_context = (
                                    inner.get("product_context")
                                    or callback_form_args.get("product_context")
                                    or (body.page_context or {}).get("productTitle")
                                    or (body.page_context or {}).get("productHandle")
                                )
                                yield _emit_open_callback_form(product_context)
                                callback_form_emitted = True
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
