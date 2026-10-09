"""
Chat API
========
PRD-27: Chat endpoints for streaming conversations, history, and voting.

Secured with hybrid auth (Clerk JWT + API key).
"""

import logging
import uuid as _uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import text, func, or_
from pydantic import BaseModel

from core.database.database import SessionLocal, get_db
from consumers.chatbot import ChatService, StreamingChatService
from api.chat_dispatch import TurnLane, auto_lane, chosen_agent_lane, response_headers, said_before
from core.auth.hybrid import get_request_context_hybrid
from core.auth.workspace_permission import require_workspace_permission
from core.auth.dependencies import RequestContext
from core.auth.principal import resolve_user_pk
from core.models.core import User
from core.session_queue import get_session_queue
from core.utils.timestamps import utc_iso
from services.board_events import notify_chat_event
from services.chat_agent_switch import record_agent_switch
from services.chat_turns import get_turn_registry, run_detached_turn
from services.page_context import inject_page_preamble, sanitize_page_context

logger = logging.getLogger(__name__)


router = APIRouter(prefix="/api/chat", tags=["💬 Chat"])


# Request/Response Models
class MessagePart(BaseModel):
    type: str
    text: Optional[str] = None
    filename: Optional[str] = None
    mediaType: Optional[str] = None
    url: Optional[str] = None
    # PRD-127: ephemeral attachment reference (sent by multimodal-input.tsx)
    attachment_id: Optional[str] = None


class ChatMessageRequest(BaseModel):
    role: str = "user"
    parts: Optional[List[MessagePart]] = None
    # Compatibility with older/alternate clients
    content: Optional[str] = None
    # PRD-127: top-level list of ephemeral attachment ids for the current message
    attachment_ids: Optional[List[str]] = None


class ChatRequest(BaseModel):
    id: Optional[str] = None
    message: ChatMessageRequest
    # Compatibility with AI SDK "messages" payloads
    messages: Optional[List[ChatMessageRequest]] = None
    # PRD-180 S3 (F035): the per-message model selector was a placebo — nothing
    # ever read the chosen model. Field removed; the model resolves via the Auto
    # tier / the selected agent's own config, never a client-picked override.
    selectedVisibilityType: Optional[str] = "private"
    context: Optional[dict] = None
    # PRD: Unified Agent-Chat System
    agentId: Optional[int] = None  # Selected agent ID (default: system agent id=1)
    # PRD-82A: Mission mode — conversational mission planning
    missionMode: Optional[bool] = False
    # Plan mode — research and strategy output, no execution
    planMode: Optional[bool] = False


class UpdateTitleRequest(BaseModel):
    title: str


class VoteRequest(BaseModel):
    chatId: str
    messageId: str
    isUpvoted: bool


# PRD-237 S6: the hosted edition's open-conversation tabs (per user, per
# workspace). Mirrors frontend/lib/chat/chat-session.ts — ids and timestamps
# only, never message content.
MAX_OPEN_CHATS = 8
MAX_TRACKED_THREADS = 50
_EMPTY_SESSION_DOC: Dict[str, Any] = {
    "activeChatId": None,
    "draftOpen": False,
    "openChatIds": [],
    "lastReadAt": {},
    "updatedAt": None,
}


class ChatSessionRequest(BaseModel):
    activeChatId: Optional[str] = None
    draftOpen: bool = False
    openChatIds: List[str] = []
    lastReadAt: Dict[str, Any] = {}


# Helper function to get user ID from database
def get_user_id(db: Session, ctx=None) -> int:
    """Resolve the authenticated principal to the integer ``users.id`` PK (PRD-185 S6).

    Prefer the real principal from the request context; fall back to a default
    user only for genuinely principal-less (system/anonymous) paths — never
    hardcode id=1 for a logged-in caller. The old id=1 default mis-attributed
    every chat, message save, vote-ownership check, and mid-chat mission approval
    (_driving_clerk derives from this) to user 1.

    IMPORTANT: ``UserContext.id`` carries the *Clerk subject string* (or email)
    for SaaS auth — NOT the integer ``users.id`` (see
    ``core.auth.principal.resolve_user_pk``, the one resolver every integer-FK
    site must use; PRD-242 moved the lookup there so the document renderers
    stopped 500-ing on the same bug).
    """
    resolved = resolve_user_pk(db, ctx)
    if resolved is not None:
        return resolved
    # Principal-less (system/anonymous) or unresolvable — the chat default user.
    result = db.execute(text("SELECT id FROM users WHERE id = 1 LIMIT 1")).fetchone()
    if not result:
        result = db.execute(text("SELECT id FROM users LIMIT 1")).fetchone()
    if not result:
        raise HTTPException(status_code=500, detail="No users found")
    return result[0]

def _explicitly_chosen_agent(db: Session, workspace_id, agent_id: int) -> int:
    """F149: the agent a user picks is one of this workspace's, or a platform
    system agent (as the agent switch allows) — never another workspace's."""
    from core.security.workspace_scope import agent_in_workspace

    if not agent_in_workspace(db, agent_id, workspace_id):
        raise HTTPException(status_code=404, detail="Agent not found")
    return agent_id


def get_default_agent_id(db: Session, workspace_id) -> int:
    """Return the workspace's Auto agent (per-workspace system agent).

    Looks up the agent with slug='auto-{workspace_id}'. If missing (workspace
    created before the migration), lazy-seeds one with deployment defaults.

    Never returns agent id=1 or any agent from another workspace.
    """
    from core.models.core import Agent

    slug = f"auto-{workspace_id}"
    row = db.query(Agent.id).filter(
        Agent.slug == slug,
        Agent.is_system_agent.is_(True),
        Agent.workspace_id == workspace_id,
    ).scalar()
    if row:
        return int(row)

    # Lazy-seed for workspaces created before the migration
    try:
        from core.seeds.seed_auto_agent import seed_auto_agent
        auto = seed_auto_agent(db, workspace_id)
        db.commit()
        logger.info("Lazy-seeded Auto agent for workspace %s → agent.id=%s", workspace_id, auto.id)
        return auto.id
    except Exception:
        logger.exception("Failed to seed Auto agent for workspace %s", workspace_id)

    # Last resort: first agent in THIS workspace (never cross-tenant)
    first = db.query(Agent.id).filter(
        Agent.workspace_id == workspace_id,
    ).order_by(Agent.id.asc()).scalar()
    if first:
        return int(first)

    raise HTTPException(
        status_code=500,
        detail="No agent available for workspace. Configure Auto at Settings > Orchestrator.",
    )


def _parts_text(parts) -> str:
    """Flatten a message's ``parts`` JSONB into plain text."""
    if not isinstance(parts, list):
        return ""
    return " ".join(
        p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")
    ).strip()


_PREVIEW_MAX_CHARS = 80


def _last_message_previews(db: Session, chat_ids: List[str]) -> dict:
    """Latest message text per chat, truncated — one query for the whole page (PRD-220 S2)."""
    if not chat_ids:
        return {}
    rows = db.execute(
        text("""
            SELECT DISTINCT ON (m.chat_id) m.chat_id, m.parts
            FROM messages m
            WHERE m.chat_id = ANY(CAST(:ids AS uuid[]))
            ORDER BY m.chat_id, m.created_at DESC
        """),
        {"ids": chat_ids},
    ).fetchall()
    previews = {}
    for r in rows:
        content = _parts_text(r.parts)
        if len(content) > _PREVIEW_MAX_CHARS:
            content = content[: _PREVIEW_MAX_CHARS - 1] + "…"
        previews[str(r.chat_id)] = content
    return previews


_UNTITLED_CHAT = "New Chat"
_TITLE_MAX_CHARS = 50


def _message_parts(msg: ChatMessageRequest) -> List[MessagePart]:
    """A message's parts; a ``content``-only message is one text part."""
    if msg.parts:
        return msg.parts
    if msg.content:
        return [MessagePart(type="text", text=msg.content)]
    return []


def _current_message(request: ChatRequest) -> ChatMessageRequest:
    """The message this turn answers: ``message``, or the last of ``messages[]``."""
    current = request.message or (request.messages[-1] if request.messages else None)
    if not current:
        raise HTTPException(status_code=400, detail="No message provided")
    return current


def _message_text(parts: List[MessagePart]) -> str:
    """The text AutoBrain classifies: the first part, when it is text."""
    return (parts[0].text or "") if parts and parts[0].type == "text" else ""


def _unique_title(db: Session, user_id: int, parts: List[MessagePart]) -> str:
    """A new chat's title: its first words, numbered when this user already has one."""
    first = parts[0] if parts else None
    base = first.text[:_TITLE_MAX_CHARS] if first and first.text else _UNTITLED_CHAT
    title, counter = base, 1
    while db.execute(
        text("SELECT 1 FROM chats WHERE user_id = :user_id AND title = :title LIMIT 1"),
        {"user_id": user_id, "title": title},
    ).fetchone():
        counter += 1
        title = f"{base} ({counter})"
    return title


def _open_chat(db: Session, chat_service: ChatService, request: ChatRequest, ctx: RequestContext,
               user_id: int, parts: List[MessagePart]) -> str:
    """The turn's chat id: the request's own chat, or a new one. A stale or unknown id
    gets a new chat, never an error; another user's chat is refused."""
    if request.id:
        chat = chat_service.get_chat(request.id, workspace_id=ctx.workspace_id)
        if chat:
            if chat.user_id != user_id:
                raise HTTPException(status_code=403, detail="Access denied")
            return request.id
    chat = chat_service.create_chat(
        user_id=user_id,
        title=_unique_title(db, user_id, parts),
        visibility=request.selectedVisibilityType,
        workspace_id=ctx.workspace_id,
    )
    return str(chat.id)


def _with_attachments(history: List[Dict[str, Any]], current_msg: ChatMessageRequest) -> List[Dict[str, Any]]:
    """PRD-127: the request's ephemeral attachment ids ride on the latest user message.
    Attachments are request-scoped (7-day S3 TTL) and never persisted in chat history;
    AttachmentResolver resolves them inline. The frontend sends ids both at the top
    level and inside file parts."""
    ids = list(current_msg.attachment_ids or [])
    for part in current_msg.parts or []:
        if part.attachment_id and part.attachment_id not in ids:
            ids.append(part.attachment_id)
    logger.info(
        "[PRD-127] chat request attachments: top_level_ids=%s parts=%s collected=%s",
        current_msg.attachment_ids, [p.dict(exclude_none=True) for p in current_msg.parts or []], ids,
    )
    if not ids:
        return history
    for index in range(len(history) - 1, -1, -1):
        if history[index].get("role") == "user":
            logger.info("[PRD-127] injected %d attachment_ids into message_history[%d]", len(ids), index)
            return history[:index] + [{**history[index], "attachment_ids": ids}] + history[index + 1:]
    return history


def _turn_history(chat_service: ChatService, chat_id: str, current_msg: ChatMessageRequest,
                  page_ctx: Any) -> List[Dict[str, Any]]:
    """The conversation the turn reads: the chat's messages, the attachments, and the page.

    PRD-221 S2 (extends PRD-220): the page context is the structured reference set
    {page, route, tab, selected, filters, visible_ids}, or the legacy {"page": <label>};
    one renderer serves both. It is sanitized against the allow-list (authz-looking
    fields never survive) and injected prompt-side only: the user message is already
    saved clean, so chat titles and reloaded history never show the hint."""
    history = [{"role": m.role, "parts": m.parts} for m in chat_service.get_messages_by_chat_id(chat_id)]
    return inject_page_preamble(_with_attachments(history, current_msg), page_ctx)


def _is_super_admin(ctx: RequestContext) -> bool:
    """PRD-143: the su surface is derived from system_role ONLY, never from workspace
    role, is_admin or autonomy level (fail-closed boundary)."""
    role = getattr(ctx.user, "system_role", "user") if ctx.user else "user"
    logger.info("[PRD-67] user_role=%r, user_id=%s", role, getattr(ctx.user, "id", "?"))
    return role == "super_admin"


async def _turn_lane(db: Session, ctx: RequestContext, request: ChatRequest, message_text: str,
                     history_length: int, before: str = "") -> TurnLane:
    """PRD-256 US-010 (D2): the agent the owner chose answers; otherwise Auto does.

    Every workspace has its own Auto agent: its model, persona and tools come from that
    agent's config (Settings > Orchestrator), never a hardcoded agent id."""
    if request.agentId:
        return chosen_agent_lane(db, _explicitly_chosen_agent(db, ctx.workspace_id, request.agentId))
    return await auto_lane(
        db, ctx.workspace_id, auto_agent_id=get_default_agent_id(db, ctx.workspace_id),
        message_text=message_text, history_length=history_length, before=before,
    )


@dataclass(frozen=True)
class _Turn:
    """What the detached turn needs: plain values, never the request's DB session."""

    workspace_id: Any
    chat_id: str
    user_id: int
    messages: List[Dict[str, Any]]
    lane: TurnLane
    message_text: str
    mission_mode: bool
    plan_mode: bool
    is_super_admin: bool
    page_context: Any


async def _turn_chunks(service: StreamingChatService, task_db: Session, turn: _Turn):
    """The turn's frames, from the agent its lane names."""
    lane = turn.lane
    if lane.session_agent:
        # PRD-239 S7 v2: a session agent lives in the Runtime Canvas; the operator talks
        # to it in the terminal, never through this lane. Say so (S4 renders `e:` frames).
        from services.cli_ticket_lane import session_agent_terminal_message

        yield service.streaming_handler.format_aisdk_error(
            session_agent_terminal_message(task_db, lane.agent_id), code="session_agent_terminal",
        )
        return
    if lane.suggest_mission:
        # PRD-125: the mission suggestion card (the frontend renders it); Auto still answers.
        yield service.streaming_handler.format_aisdk_data("mission-suggestion", {
            "goal": turn.message_text, "complexity": lane.assessment.complexity.value, "agent_id": lane.agent_id,
        })
    async for chunk in service.stream_response_with_agent(
        chat_id=turn.chat_id, messages=turn.messages, agent_id=lane.agent_id, user_id=turn.user_id,
        skip_composio=lane.skip_composio, complexity_assessment=lane.assessment,
        mission_mode=turn.mission_mode, plan_mode=turn.plan_mode, suggest_mission=lane.suggest_mission,
        is_super_admin=turn.is_super_admin,
        # PRD-221 S3/S4: the sanitized reference set rides into the turn's
        # context_trace and the page-prior action exposure.
        page_context=turn.page_context,
    ):
        yield chunk


async def _produce_turn(turn: _Turn):
    """PRD-237 S7: the turn is a producer task with its own DB session; the HTTP response
    only consumes it (services.chat_turns). A reload or navigation that drops the
    connection no longer kills the reply; Stop is explicit via POST /{chat_id}/cancel.
    Concurrent requests for the same chat are serialized by the session queue."""
    task_db = SessionLocal()
    try:
        service = StreamingChatService(task_db, workspace_id=turn.workspace_id)
        async with get_session_queue().acquire(f"{turn.workspace_id}:{turn.chat_id}"):
            async for chunk in _turn_chunks(service, task_db, turn):
                yield chunk
    finally:
        task_db.close()


def _notify_a_missed_reply(workspace_id: Any, chat_id: str, user_id: int):
    """The turn's ``on_complete``: a client that missed the end of the turn (reload,
    navigation) is told the reply landed, and the PRD-205 S7 lane merges it live. A
    connected client already has the reply; notifying it would duplicate."""
    async def _on_complete(*, completed: bool, cancelled: bool, client_gone: bool) -> None:
        if not (completed and client_gone):
            return
        notify_db = SessionLocal()
        try:
            notify_chat_event(notify_db, workspace_id=workspace_id, chat_id=chat_id, user_id=user_id)
            notify_db.commit()
        finally:
            notify_db.close()
    return _on_complete


# Endpoints
@router.post("", dependencies=[Depends(require_workspace_permission("agents:execute"))])
async def stream_chat(
    request: ChatRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Stream chat messages using AI SDK Data Stream format (text/plain)"""
    logger.info("[chat] RequestContext workspace_id=%s agentId=%s", ctx.workspace_id, request.agentId)
    chat_service = ChatService(db)
    # F145: an admin_only tool is an admin's when the call is made for an active
    # owner/admin of the workspace (the chat threads the driving user) or for a
    # super admin — core.security.driving_user; no "workspace has an admin" fallback.
    user_id = get_user_id(db, ctx)
    current_msg = _current_message(request)
    parts = _message_parts(current_msg)
    chat_id = _open_chat(db, chat_service, request, ctx, user_id, parts)
    chat_service.save_message(
        chat_id=chat_id, role="user", parts=[part.dict() for part in parts], workspace_id=ctx.workspace_id,
    )
    page_ctx = sanitize_page_context(request.context)
    history = _turn_history(chat_service, chat_id, current_msg, page_ctx)
    message_text = _message_text(parts)
    lane = await _turn_lane(db, ctx, request, message_text, len(history), said_before(history))
    logger.info("[chat] agent_id=%s answers the turn", lane.agent_id)
    # The request-scoped ``db`` is never handed to the turn: FastAPI closes it when the
    # response ends, which may be before the turn does.
    turn = _Turn(
        workspace_id=ctx.workspace_id, chat_id=chat_id, user_id=user_id, messages=history, lane=lane,
        message_text=message_text, mission_mode=bool(request.missionMode), plan_mode=bool(request.planMode),
        is_super_admin=_is_super_admin(ctx), page_context=page_ctx,
    )
    return StreamingResponse(
        run_detached_turn(
            chat_id=chat_id,
            produce=lambda: _produce_turn(turn),
            on_complete=_notify_a_missed_reply(ctx.workspace_id, chat_id, user_id),
        ),
        media_type="text/plain; charset=utf-8",
        headers=response_headers(lane.assessment),
    )


@router.get("/history")
async def get_chat_history(
    limit: int = 20,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Get chat history for the current user within their workspace"""
    chat_service = ChatService(db)
    user_id = get_user_id(db, ctx)

    chats = chat_service.get_chat_history(user_id=user_id, limit=limit, workspace_id=ctx.workspace_id)
    previews = _last_message_previews(db, [str(chat.id) for chat in chats])

    return [
        {
            "id": str(chat.id),
            "userId": chat.user_id,
            "title": chat.title,
            "createdAt": utc_iso(chat.created_at),
            "updatedAt": utc_iso(chat.updated_at),
            "visibility": chat.visibility,
            "lastContext": chat.last_context,
            "lastMessagePreview": previews.get(str(chat.id)),
            # PRD-205 S7: 'auto' marks the thread where Auto speaks unprompted.
            "kind": chat.kind,
        }
        for chat in chats
    ]


# ---------------------------------------------------------------------------
# PRD-237 S6: open-conversation tabs, server-side for the hosted edition so they
# follow the user across devices (owner decision D1, 2026-09-07). The local
# edition keeps its session in the browser and never calls these. MUST stay
# above ``GET /{chat_id}`` — the PRD-220 /search failure mode.
# ---------------------------------------------------------------------------

def _validate_chat_ids(ids: List[Any], *, cap: int, field: str) -> List[str]:
    """Canonical, de-duplicated chat ids — or a 422 that names the field."""
    out: List[str] = []
    for raw in ids:
        try:
            cid = str(_uuid.UUID(str(raw)))
        except (ValueError, TypeError, AttributeError):
            raise HTTPException(status_code=422, detail=f"{field}: {raw!r} is not a chat id")
        if cid not in out:
            out.append(cid)
    if len(out) > cap:
        raise HTTPException(status_code=422, detail=f"{field}: at most {cap} entries")
    return out


def _read_stamps(raw: Dict[str, Any]) -> Dict[str, float]:
    stamps: Dict[str, float] = {}
    for raw_id, raw_ts in raw.items():
        if isinstance(raw_ts, bool) or not isinstance(raw_ts, (int, float)):
            raise HTTPException(status_code=422, detail="lastReadAt values must be timestamps (ms)")
        stamps[_validate_chat_ids([raw_id], cap=1, field="lastReadAt")[0]] = float(raw_ts)
    if len(stamps) > MAX_TRACKED_THREADS:
        keep = sorted(stamps, key=stamps.__getitem__, reverse=True)[:MAX_TRACKED_THREADS]
        stamps = {k: stamps[k] for k in keep}
    return stamps


def _session_doc_from_request(body: ChatSessionRequest) -> Dict[str, Any]:
    open_ids = _validate_chat_ids(body.openChatIds, cap=MAX_OPEN_CHATS, field="openChatIds")
    active = None
    if body.activeChatId is not None:
        active = _validate_chat_ids([body.activeChatId], cap=1, field="activeChatId")[0]
        if active not in open_ids:
            raise HTTPException(status_code=422, detail="activeChatId must be one of openChatIds")
    return {
        "activeChatId": active,
        "draftOpen": bool(body.draftOpen),
        "openChatIds": open_ids,
        "lastReadAt": _read_stamps(body.lastReadAt),
    }


@router.get("/session")
async def get_chat_session(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """The caller's open-conversation tabs in this workspace (empty doc when none)."""
    user_id = get_user_id(db, ctx)
    row = db.query(User).filter(User.id == user_id).first()
    sessions = (getattr(row, "chat_sessions", None) or {}) if row is not None else {}
    doc = sessions.get(str(ctx.workspace_id))
    if not isinstance(doc, dict):
        return dict(_EMPTY_SESSION_DOC)
    return {**_EMPTY_SESSION_DOC, **doc}


@router.put("/session", dependencies=[Depends(require_workspace_permission("agents:execute"))])
async def put_chat_session(
    body: ChatSessionRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Replace the caller's tabs for this workspace (validated: ids, cap, active ∈ open)."""
    doc = _session_doc_from_request(body)
    user_id = get_user_id(db, ctx)
    row = db.query(User).filter(User.id == user_id).first()
    if row is None:
        raise HTTPException(status_code=404, detail="User not found")
    stored = {**doc, "updatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    # Rebuild, never mutate: SQLAlchemy only sees a JSONB change on reassignment.
    row.chat_sessions = {**(row.chat_sessions or {}), str(ctx.workspace_id): stored}
    db.commit()
    return stored


@router.get("/search")
async def search_chat_history(
    q: str,
    limit: int = 20,
    days: int = 30,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Search across chat messages by keyword (workspace-scoped).

    MUST be registered before ``GET /{chat_id}`` — routes match in declaration
    order, so with this below the param route ``/api/chat/search`` resolved as
    ``chat_id="search"`` and 404'd (PRD-220 drive-by fix).
    """
    from datetime import datetime, timedelta

    user_id = get_user_id(db, ctx)
    since = datetime.utcnow() - timedelta(days=min(days, 365))
    search_term = f"%{q}%"

    rows = db.execute(
        text("""
            SELECT m.id, m.chat_id, m.role, m.parts, m.created_at,
                   c.title AS chat_title
            FROM messages m
            JOIN chats c ON c.id = m.chat_id
            WHERE c.user_id = :user_id
              AND c.workspace_id = :workspace_id
              AND m.created_at >= :since
              AND EXISTS (
                  SELECT 1 FROM jsonb_array_elements(m.parts) AS p
                  WHERE p->>'text' ILIKE :search
              )
            ORDER BY m.created_at DESC
            LIMIT :lim
        """),
        {"user_id": user_id, "workspace_id": str(ctx.workspace_id), "since": since, "search": search_term, "lim": min(limit, 100)},
    ).fetchall()

    results = []
    for r in rows:
        # Extract text content from parts
        parts = r.parts if isinstance(r.parts, list) else []
        text_content = " ".join(
            p.get("text", "") for p in parts if isinstance(p, dict) and p.get("text")
        )
        results.append({
            "message_id": str(r.id),
            "chat_id": str(r.chat_id),
            "chat_title": r.chat_title,
            "role": r.role,
            "content": text_content[:500],
            "created_at": utc_iso(r.created_at),
        })

    return {"query": q, "total": len(results), "results": results}


# PRD-205 S8: /vote and /agents MUST be declared before the /{chat_id}
# routes below — declared after, FastAPI matched chat_id="vote"/"agents"
# and both endpoints were dead (the exact PRD-220 /search failure mode).
# Locked by route-order regression tests.
@router.patch("/vote", dependencies=[Depends(require_workspace_permission("agents:execute"))])
async def vote_message(
    request: VoteRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Vote on a message (workspace-scoped)"""
    chat_service = ChatService(db)
    user_id = get_user_id(db, ctx)

    chat = chat_service.get_chat(request.chatId, workspace_id=ctx.workspace_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    
    if chat.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    success = chat_service.vote_message(
        request.chatId,
        request.messageId,
        request.isUpvoted
    )

    # PRD-185 S7: feed the RAG feedback loop. Write a rag_feedback row from the
    # voted assistant message's retrieved doc ids so the PRD-179 live ranker can
    # learn from thumbs. Best-effort — never fail the vote — but log loudly if it
    # breaks (this wave exists because of silent swallows).
    if success:
        try:
            from modules.rag.feedback_writer import feedback_from_retrieval_context
            message = chat_service.get_message(request.chatId, request.messageId)
            if message is not None:
                feedback_from_retrieval_context(
                    db,
                    retrieval_context=message.retrieval_context,
                    is_upvoted=request.isUpvoted,
                    workspace_id=ctx.workspace_id,
                    user_id=user_id,
                )
        except Exception:
            # The vote itself is already committed; roll back only the failed
            # feedback partial so the per-request session isn't left poisoned.
            db.rollback()
            logger.warning(
                "[PRD-185 S7] rag_feedback write from chat vote failed "
                "(chat=%s message=%s)",
                request.chatId, request.messageId, exc_info=True,
            )

    return {"success": success}


# PRD: Unified Agent-Chat System - Agent Endpoints
@router.get("/agents")
async def get_available_agents(
    status: str = "active",
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Get list of available agents for chat selection."""
    from core.models import Agent
    
    query = db.query(Agent).filter(Agent.status == status, Agent.workspace_id == ctx.workspace_id)
    agents = query.all()
    
    return {
        "agents": [
            {
                "id": agent.id,
                "name": agent.name,
                "agent_type": agent.agent_type,
                "description": agent.description,
                "status": agent.status,
                "skills": agent.configuration.get("skills", []) if agent.configuration else [],
                "model_config": agent.model_config or {},
                "is_default": agent.id == 1,
                "tags": agent.tags or []
            }
            for agent in agents
        ]
    }


class SwitchAgentRequest(BaseModel):
    newAgentId: int
    reason: Optional[str] = None


@router.get("/{chat_id}")
async def get_chat(
    chat_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Get a specific chat (workspace-scoped)"""
    chat_service = ChatService(db)
    user_id = get_user_id(db, ctx)

    chat = chat_service.get_chat(chat_id, workspace_id=ctx.workspace_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    if chat.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    return {
        "id": str(chat.id),
        "userId": chat.user_id,
        "title": chat.title,
        "createdAt": utc_iso(chat.created_at),
        "updatedAt": utc_iso(chat.updated_at),
        "visibility": chat.visibility,
        "lastContext": chat.last_context,
        # PRD-205 S7: 'auto' marks the thread where Auto speaks unprompted.
        "kind": chat.kind,
        # PRD-237 S7: a turn is still being produced for this chat (the page
        # reloaded mid-reply) — the client shows the typing state until the
        # reply merges in via chat_changed.
        "turnInFlight": await get_turn_registry().is_in_flight(chat_id),
    }


@router.get("/{chat_id}/messages")
async def get_chat_messages(
    chat_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Get all messages for a chat (workspace-scoped)"""
    chat_service = ChatService(db)
    user_id = get_user_id(db, ctx)

    # Verify chat access within workspace
    chat = chat_service.get_chat(chat_id, workspace_id=ctx.workspace_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    if chat.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    messages = chat_service.get_messages_by_chat_id(chat_id)
    
    return [
        {
            "id": str(msg.id),
            "role": msg.role,
            "parts": msg.parts,
            "attachments": msg.attachments,
            # PRD-205 S3: background-author provenance — the frontend maps it
            # into the message badge slot ("Auto · background"). null for
            # every in-turn message, incl. all rows predating the column.
            "source": msg.source,
            "createdAt": utc_iso(msg.created_at)
        }
        for msg in messages
    ]


@router.delete("/{chat_id}", dependencies=[Depends(require_workspace_permission("agents:execute"))])
async def delete_chat(
    chat_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Delete a chat (workspace-scoped)"""
    chat_service = ChatService(db)
    user_id = get_user_id(db, ctx)

    chat = chat_service.get_chat(chat_id, workspace_id=ctx.workspace_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    if chat.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    success = chat_service.delete_chat(chat_id)
    return {"success": success}


@router.patch("/{chat_id}", dependencies=[Depends(require_workspace_permission("agents:execute"))])
async def update_chat(
    chat_id: str,
    request: UpdateTitleRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Update chat title (workspace-scoped)"""
    chat_service = ChatService(db)
    user_id = get_user_id(db, ctx)

    chat = chat_service.get_chat(chat_id, workspace_id=ctx.workspace_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    if chat.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    success = chat_service.update_chat_title(chat_id, request.title)
    return {"success": success}


@router.post("/{chat_id}/switch-agent", dependencies=[Depends(require_workspace_permission("agents:execute"))])
async def switch_agent(
    chat_id: str,
    request: SwitchAgentRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Switch to a different agent mid-conversation."""
    from core.models import Agent

    chat_service = ChatService(db)
    user_id = get_user_id(db, ctx)
    
    chat = chat_service.get_chat(chat_id, workspace_id=ctx.workspace_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    if chat.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    # PRD-67: Allow switching to system agents (CTO) if user has the required role
    new_agent = db.query(Agent).filter(
        Agent.id == request.newAgentId,
        or_(Agent.workspace_id == ctx.workspace_id, Agent.is_system_agent.is_(True)),
    ).first()
    if not new_agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    # Verify role access for system agents
    if new_agent.is_system_agent and new_agent.required_role:
        _switch_user_role = getattr(ctx.user, "system_role", "user") if ctx.user else "user"
        _switch_hierarchy = {"super_admin": {"super_admin", "admin"}, "admin": {"admin"}}
        if new_agent.required_role not in _switch_hierarchy.get(_switch_user_role, set()):
            raise HTTPException(status_code=403, detail="Insufficient role for this agent")
    
    old_agent_id = getattr(chat, 'current_agent_id', None) or 1
    
    db.execute(
        text("UPDATE chats SET current_agent_id = :new_agent_id WHERE id = :chat_id"),
        {"new_agent_id": request.newAgentId, "chat_id": chat.id}
    )
    
    record_agent_switch(db, chat, old_agent_id, request.newAgentId, request.reason)
    db.commit()
    
    return {
        "success": True,
        "agent": {
            "id": new_agent.id,
            "name": new_agent.name,
            "type": new_agent.agent_type,
            "message": f"Switched to {new_agent.name}. How can I help?"
        }
    }


@router.post("/{chat_id}/cancel", dependencies=[Depends(require_workspace_permission("agents:execute"))])
async def cancel_chat_turn(
    chat_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """PRD-237 S7: Stop. The reply no longer dies with the connection, so the
    frontend's Stop must say so — this reaches whichever worker holds the turn
    (process-local cancel, plus the Redis marker the producer polls)."""
    chat_service = ChatService(db)
    user_id = get_user_id(db, ctx)

    chat = chat_service.get_chat(chat_id, workspace_id=ctx.workspace_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")
    if chat.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    cancelled = await get_turn_registry().request_cancel(chat_id)
    return {"cancelled": cancelled}
