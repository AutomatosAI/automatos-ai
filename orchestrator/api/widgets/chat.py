"""
Widget Chat API
===============

Chat endpoint for embedded SDK widgets with SSE streaming.

Reuses the existing ``ChatService`` and ``StreamingChatService`` from
``consumers.chatbot`` so that widget conversations get the same agent,
memory, and tool-loop capabilities as the main chat UI.

Auth: ``widget_auth`` + ``require_permission("chat")``.

The SSE stream itself lives in ``api/widgets/chat_stream.py``. PRD-264
EUAI-02a: its first event is the AI notice, and every assistant message
event carries ``"ai_generated": true``.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from api.widgets.auth import WidgetAuthContext, require_permission, widget_auth
from api.widgets.chat_stream import WidgetStreamContext, widget_event_stream
from api.widgets.config import resolve_ai_disclosure
from core.security.log_safe import log_safe
from core.database.database import get_db
from integrations import PLUGIN_REGISTRY

logger = logging.getLogger(__name__)

# Tag every widget-chat log line so we can grep `widget_chat` in Railway.
_LOG_TAG = "widget_chat"


# Truncate for logging, with control characters made spaces (F155).
_short = log_safe

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


# PRD-007 / PRD-008-B: trigger reasons that flip the agent into opener-generation
# mode. Anything else is treated as a normal user message.
PROACTIVE_TRIGGER_REASONS: frozenset[str] = frozenset({
    "proactive_opener",  # product-page contextual opener (PRD-007)
    "cart_idle",         # cart-page idle nudge w/ FBT recs (PRD-008-B Feature C2)
})


class WidgetMessageOut(BaseModel):
    id: str
    role: str
    content: str
    created_at: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _retrieval_team(db: Session, auth: WidgetAuthContext, agent_id: Optional[int]) -> Optional[str]:
    """PRD-124: the team that scopes a widget chat's documents. F155: the key's
    team lock wins, as /search and /docs already honour it; otherwise the
    answering agent's team."""
    from core.team_access import effective_team

    agent_team: Optional[str] = None
    try:
        from core.models.core import Agent

        agent_row = db.query(Agent.team).filter(Agent.id == agent_id).first()
        agent_team = agent_row.team if agent_row else None
    except Exception:
        logger.debug("Could not resolve agent team for agent_id=%s", agent_id)
    return effective_team(auth.team, agent_team)


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


def _key_started(db: Session, conversation_id: str, auth: WidgetAuthContext) -> bool:
    """F155: a key reads and resumes only the conversations it started. Any
    other id — another key's, the dashboard's, not a chat — is not found."""
    try:
        chat_id = str(uuid.UUID(str(conversation_id)))
    except ValueError:
        return False
    row = db.execute(
        text("SELECT 1 FROM chats WHERE id = CAST(:id AS uuid) AND workspace_id = CAST(:ws AS uuid) "
             "AND widget_key_id = CAST(:key AS uuid)"),
        {"id": chat_id, "ws": str(auth.workspace_id), "key": str(auth.api_key_id)},
    ).first()
    return row is not None


def _resolve_workspace_vertical(db: Session, workspace_id: str) -> str:
    """PRD-141: read ``workspace.settings.vertical`` for plugin dispatch.

    Returns ``"generic"`` when the workspace has no row, no settings,
    or no ``vertical`` field — that's the default pass-through plugin.
    Same raw-SQL pattern as ``channels/sender.py`` so we don't pull the
    full ORM object just to read one JSONB key.
    """
    row = db.execute(
        text("SELECT settings FROM workspaces WHERE id = :ws"),
        {"ws": workspace_id},
    ).fetchone()
    settings = row.settings if (row and isinstance(row.settings, dict)) else {}
    return settings.get("vertical") or "generic"


# ---------------------------------------------------------------------------
# POST /chat — the steps of one widget turn
# ---------------------------------------------------------------------------

_SSE_HEADERS: dict[str, str] = {
    "Cache-Control": "no-cache, no-store, must-revalidate",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@dataclass(frozen=True)
class _WidgetTurn:
    """The visitor's message after the vertical plugin has seen it."""

    message: str
    system_preamble: Optional[str]
    vertical: str
    is_proactive: bool


def _log_tag(request: Request, auth: WidgetAuthContext) -> str:
    """The ``[widget_chat req=… ws=… origin=…]`` prefix of every line of a turn."""
    req_id = _short(
        request.headers.get("X-Request-ID")
        or request.headers.get("x-railway-request-id")
        or uuid.uuid4().hex[:12],
        64,
    )
    origin = _short(request.headers.get("Origin") or "?", 120)
    return f"[{_LOG_TAG} req={req_id} ws={auth.workspace_id} origin={origin}]"


def _log_request(log_extra: str, body: WidgetChatRequest) -> None:
    logger.info(
        "%s REQUEST: agent_id=%s conv_id=%s trigger_reason=%s page_type=%s msg_len=%d msg_preview=%s",
        log_extra,
        _short(body.agent_id, 40),
        _short(body.conversation_id, 64),
        _short(body.trigger_reason, 40),
        _short((body.page_context or {}).get("pageType") if body.page_context else None, 40),
        len(body.message or ""),
        _short(body.message),
    )


def _chat_services(log_extra: str) -> Tuple[Any, Any]:
    """``ChatService`` and ``StreamingChatService``, imported per call: the chat
    infrastructure may not be available in every deploy (503 when it isn't)."""
    try:
        from consumers.chatbot import ChatService, StreamingChatService
    except ImportError:
        logger.error("%s SERVICE_UNAVAILABLE: consumers.chatbot import failed", log_extra)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Chat service is not available in this deployment",
        ) from None
    return ChatService, StreamingChatService


def _log_plugin_outcome(body: WidgetChatRequest, rewritten: str, telemetry: Any,
                        vertical: str, is_proactive: bool, log_extra: str) -> None:
    if is_proactive:
        logger.info(
            "%s PROACTIVE_REWRITE: vertical=%s trigger=%s original_msg_len=%d new_msg_len=%d telemetry=%s new_preview=%s",
            log_extra, vertical, _short(body.trigger_reason, 40), len(body.message or ""),
            len(rewritten), telemetry, _short(rewritten),
        )
    elif body.trigger_reason:
        logger.warning(
            "%s UNKNOWN_TRIGGER_REASON: %s vertical=%s (page_context=%s) — proceeding as normal chat",
            log_extra, _short(body.trigger_reason, 40), vertical,
            "present" if body.page_context else "missing",
        )


async def _run_vertical_plugin(body: WidgetChatRequest, auth: WidgetAuthContext,
                               db: Session, log_extra: str) -> _WidgetTurn:
    """PRD-141: dispatch the message through the workspace's vertical plugin.

    Generic surfaces (this file) hold zero vertical-specific keys — all
    rewrite logic lives under ``integrations/<vertical>/``. ``is_proactive``
    (a known proactive trigger + populated context) controls the LLM-call
    shape (force_text_only, skip_composio); the plugin's decision to rewrite
    or pass through is independent of it.
    """
    vertical = _resolve_workspace_vertical(db, str(auth.workspace_id))
    plugin = PLUGIN_REGISTRY.get(vertical)
    if plugin is None:
        logger.warning("%s UNKNOWN_VERTICAL: %s — falling back to generic plugin", log_extra, vertical)
        vertical = "generic"
        plugin = PLUGIN_REGISTRY["generic"]
    is_proactive = body.trigger_reason in PROACTIVE_TRIGGER_REASONS and body.page_context is not None
    result = await plugin.handle_widget_message(
        message=body.message,
        page_context=body.page_context,
        trigger_reason=body.trigger_reason,
        workspace_id=auth.workspace_id,
        db=db,
    )
    _log_plugin_outcome(body, result.message, result.telemetry, vertical, is_proactive, log_extra)
    return _WidgetTurn(message=result.message, system_preamble=result.system_preamble,
                       vertical=vertical, is_proactive=is_proactive)


def _open_conversation(db: Session, chat_service: Any, body: WidgetChatRequest, turn: _WidgetTurn,
                       auth: WidgetAuthContext, user_id: int, log_extra: str) -> str:
    """Resume a conversation this key started (404 otherwise), or start one."""
    if body.conversation_id:
        if not _key_started(db, body.conversation_id, auth):
            logger.warning("%s CONV_NOT_FOUND: %s", log_extra, _short(body.conversation_id, 64))
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")
        logger.info("%s CONV_RESUMED: chat_id=%s", log_extra, body.conversation_id)
        return body.conversation_id

    t_conv = time.perf_counter()
    title = f"{turn.message[:40] if turn.message else 'Widget Chat'} [{uuid.uuid4().hex[:8]}]"
    chat = chat_service.create_chat(
        user_id=user_id,
        title=title,
        visibility="private",
        workspace_id=uuid.UUID(str(auth.workspace_id)),
        widget_key_id=auth.api_key_id,
    )
    logger.info(
        "%s CONV_CREATED: chat_id=%s title=%s (took %.0fms)",
        log_extra, chat.id, _short(title, 60), (time.perf_counter() - t_conv) * 1000,
    )
    return str(chat.id)


def _ground_with_preamble(history: List[dict], turn: _WidgetTurn, log_extra: str) -> List[dict]:
    """PRD-141: prepend the plugin's per-turn grounding preamble (e.g. the page
    the visitor is viewing) to the latest user turn, in memory only, so the
    persisted transcript and title stay the verbatim user text. Vertical-agnostic:
    the preamble's shape is never inspected."""
    if not (turn.system_preamble and history and history[-1].get("role") == "user"):
        return history
    last_parts = history[-1].get("parts") or [{"type": "text", "text": ""}]
    original = next(
        (p.get("text", "") for p in last_parts if isinstance(p, dict) and p.get("type") == "text"),
        "",
    )
    grounded = {"role": "user", "parts": [{"type": "text", "text": f"{turn.system_preamble}\n\n{original}"}]}
    logger.info(
        "%s PAGE_CONTEXT_GROUNDED: vertical=%s preamble_len=%d",
        log_extra, turn.vertical, len(turn.system_preamble),
    )
    return history[:-1] + [grounded]


def _history_for_turn(chat_service: Any, chat_id: str, turn: _WidgetTurn,
                      workspace_id: str, log_extra: str) -> List[dict]:
    """Persist the visitor's message, then build the LLM's message history (a failed
    turn's note reaches the model as a marker, consumers.chatbot.model_history)."""
    from consumers.chatbot import history_for_model

    t_save = time.perf_counter()
    chat_service.save_message(
        chat_id=chat_id,
        role="user",
        parts=[{"type": "text", "text": turn.message}],
        workspace_id=workspace_id,
    )
    logger.debug(
        "%s USER_MSG_SAVED: chat_id=%s (took %.0fms)",
        log_extra, chat_id, (time.perf_counter() - t_save) * 1000,
    )
    messages = chat_service.get_messages_by_chat_id(chat_id)
    history = history_for_model(messages)
    logger.debug("%s HISTORY_LOADED: %d messages", log_extra, len(history))
    return _ground_with_preamble(history, turn, log_extra)


def _effective_agent_id(db: Session, auth: WidgetAuthContext, body: WidgetChatRequest,
                        workspace_id: str, log_extra: str) -> int:
    """The agent that answers: the key's lock, else the visitor's choice
    (public_id UUID or legacy int), else the workspace's Auto."""
    from core.utils.agent_resolver import resolve_agent_id as _resolve_aid

    raw_agent_ref = auth.default_agent_id or body.agent_id
    if not raw_agent_ref:
        from api.chat import get_default_agent_id

        agent_id = get_default_agent_id(db, workspace_id)
        logger.info("%s AGENT_DEFAULT: id=%s (workspace auto)", log_extra, agent_id)
        return agent_id

    agent_id = _resolve_aid(db, raw_agent_ref, workspace_id)
    logger.info(
        "%s AGENT_RESOLVED: input=%s -> id=%s (source=%s)",
        log_extra, _short(str(raw_agent_ref), 40), agent_id,
        "key_lock" if auth.default_agent_id else "body",
    )
    if not auth.default_agent_id:
        # F155 (a), log-only: which agents each key's visitors name, before
        # a per-key agent allow-list is decided.
        logger.info(
            "%s AGENT_CENSUS: key=%s named=%s resolved=%s",
            log_extra, auth.api_key_id, _short(str(body.agent_id), 40), agent_id,
        )
    return agent_id


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
    - ``ai_disclosure`` — always first, before any token: the AI notice
      (PRD-264 EUAI-02a), ``{"type": "ai_disclosure", "badge", "line", "locale"}``
    - ``message``    — text delta from the assistant, ``"ai_generated": true``
    - ``tool-start`` — a tool invocation is beginning
    - ``tool-end``   — a tool invocation finished
    - ``open-callback-form`` — the agent asked the widget to open its callback form
    - ``error``      — the stream failed
    - ``done``       — stream is complete
    """
    started_at = time.perf_counter()
    log_extra = _log_tag(request, auth)
    _log_request(log_extra, body)
    chat_service_cls, streaming_service_cls = _chat_services(log_extra)

    workspace_id = str(auth.workspace_id)
    user_id = _get_widget_user_id(db)
    turn = await _run_vertical_plugin(body, auth, db, log_extra)
    chat_service = chat_service_cls(db)
    chat_id = _open_conversation(db, chat_service, body, turn, auth, user_id, log_extra)
    message_history = _history_for_turn(chat_service, chat_id, turn, workspace_id, log_extra)
    effective_agent_id = _effective_agent_id(db, auth, body, workspace_id, log_extra)

    # PRD-124: the team that scopes this chat's documents.
    agent_team = _retrieval_team(db, auth, effective_agent_id)
    # F155: the key's scopes, team lock and agent lock mark every call of the turn.
    streaming_service = streaming_service_cls(
        db, workspace_id=workspace_id, widget_mode=True, widget_scopes=auth.permissions, widget_team=auth.team,
        widget_agent_lock=effective_agent_id if auth.default_agent_id else None)
    stream_ctx = WidgetStreamContext(
        chat_id=chat_id, agent_id=effective_agent_id, team=agent_team, is_proactive=turn.is_proactive,
        log_extra=log_extra, started_at=started_at, disclosure=resolve_ai_disclosure(db, auth.workspace_id),
    )

    def _open_chunks() -> AsyncIterator[Any]:
        # PRD-007 v0.5 — proactive openers must produce plain text only.
        # `skip_composio` alone wasn't enough: the agent still had ~45
        # platform tools loaded from its skill and would call one (40s
        # with text_count=0). force_text_only=True clears use_tools
        # entirely so the LLM can only emit text. Keeps openers within
        # the <2s latency budget.
        return streaming_service.stream_response_with_agent(
            chat_id=chat_id, messages=message_history, agent_id=effective_agent_id, user_id=user_id,
            team=agent_team, skip_composio=turn.is_proactive, force_text_only=turn.is_proactive,
        )

    return StreamingResponse(
        widget_event_stream(stream_ctx, _open_chunks),
        media_type="text/event-stream",
        headers=_SSE_HEADERS,
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
    """Return the message history for a conversation this key started."""
    if not _key_started(db, conversation_id, auth):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found or no messages",
        )
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
