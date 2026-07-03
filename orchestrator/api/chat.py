"""
Chat API
========
PRD-27: Chat endpoints for streaming conversations, history, and voting.

Secured with hybrid auth (Clerk JWT + API key).
"""

import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import text, func, or_
from pydantic import BaseModel

from core.database.database import get_db
from consumers.chatbot import ChatService, StreamingChatService
from consumers.chatbot.auto import AutoBrain, Action
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext
from core.routing.cache import get_routing_cache
from core.routing.engine import UniversalRouter
from core.routing.ingestors.chatbot import ChatbotIngestor
from core.session_queue import get_session_queue

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


# Helper function to get user ID from database
def get_user_id(db: Session) -> int:
    """Get default user ID (id=1) for MVP"""
    result = db.execute(text("SELECT id FROM users WHERE id = 1 LIMIT 1")).fetchone()
    if not result:
        result = db.execute(text("SELECT id FROM users LIMIT 1")).fetchone()
    if not result:
        raise HTTPException(status_code=500, detail="No users found")
    return result[0]

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


# Endpoints
@router.post("")
async def stream_chat(
    request: ChatRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Stream chat messages using AI SDK Data Stream format (text/plain)"""
    logger.info(f"[chat] RequestContext workspace_id={ctx.workspace_id}")
    chat_service = ChatService(db)
    # PRD-122: Admin gate is workspace-scoped, not user-scoped.
    # caller_context=None lets PlatformActionExecutor fall through to
    # _workspace_has_admin_owner() which checks if the workspace has an
    # admin/owner member.  Admin workspace → all tools; user workspace → restricted.
    streaming_service = StreamingChatService(db, workspace_id=ctx.workspace_id)
    user_id = get_user_id(db)

    def get_parts(msg: ChatMessageRequest) -> List[MessagePart]:
        if msg.parts:
            return msg.parts
        if msg.content:
            return [MessagePart(type="text", text=msg.content)]
        return []

    # Support both {message} and {messages[]} payloads
    current_msg: Optional[ChatMessageRequest] = request.message
    if (not current_msg) and request.messages:
        current_msg = request.messages[-1]

    if not current_msg:
        raise HTTPException(status_code=400, detail="No message provided")
    
    # Get or create chat
    chat_id = request.id
    if not chat_id:
        parts = get_parts(current_msg)
        first_part = parts[0] if parts else None
        base_title = first_part.text[:50] if first_part and first_part.text else "New Chat"
        
        # Make title unique by checking existing titles first
        title = base_title
        counter = 1
        while True:
            # Check if title already exists for this user
            existing = db.execute(
                text("SELECT 1 FROM chats WHERE user_id = :user_id AND title = :title LIMIT 1"),
                {"user_id": user_id, "title": title}
            ).fetchone()
            
            if not existing:
                break
            
            counter += 1
            title = f"{base_title} ({counter})"
        
        chat = chat_service.create_chat(
            user_id=user_id,
            title=title,
            visibility=request.selectedVisibilityType,
            workspace_id=ctx.workspace_id,
        )
        chat_id = str(chat.id)
    else:
        chat = chat_service.get_chat(chat_id, workspace_id=ctx.workspace_id)
        if not chat:
            # Be forgiving: if client sends stale/invalid chat id, create a new chat
            parts = get_parts(current_msg)
            first_part = parts[0] if parts else None
            base_title = first_part.text[:50] if first_part and first_part.text else "New Chat"
            title = base_title
            counter = 1
            while True:
                existing = db.execute(
                    text("SELECT 1 FROM chats WHERE user_id = :user_id AND title = :title LIMIT 1"),
                    {"user_id": user_id, "title": title},
                ).fetchone()
                if not existing:
                    break
                counter += 1
                title = f"{base_title} ({counter})"

            chat = chat_service.create_chat(
                user_id=user_id,
                title=title,
                visibility=request.selectedVisibilityType,
                workspace_id=ctx.workspace_id,
            )
            chat_id = str(chat.id)
        
        if chat.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
    
    # Save user message
    parts = get_parts(current_msg)
    chat_service.save_message(
        chat_id=chat_id,
        role="user",
        parts=[part.dict() for part in parts],
        workspace_id=ctx.workspace_id
    )
    
    
    # Get chat history
    messages = chat_service.get_messages_by_chat_id(chat_id)
    message_history = [{'role': msg.role, 'parts': msg.parts} for msg in messages]

    # PRD-127: Attach ephemeral attachment_ids from the incoming request to the
    # latest user message. Attachments are request-scoped (7-day S3 TTL) and not
    # persisted in chat history — resolved inline by AttachmentResolver.
    _incoming_attachment_ids: List[str] = []
    if current_msg.attachment_ids:
        _incoming_attachment_ids.extend(current_msg.attachment_ids)
    # Also collect attachment_ids embedded inside file parts (frontend sends both)
    for _p in (current_msg.parts or []):
        if _p.attachment_id and _p.attachment_id not in _incoming_attachment_ids:
            _incoming_attachment_ids.append(_p.attachment_id)
    # PRD-127 diagnostics: log what the client actually sent
    try:
        _parts_debug = [
            {k: v for k, v in (_p.dict() if hasattr(_p, "dict") else {}).items() if v is not None}
            for _p in (current_msg.parts or [])
        ]
        logger.info(
            f"[PRD-127] chat request attachments: top_level_ids={current_msg.attachment_ids} "
            f"parts={_parts_debug} collected={_incoming_attachment_ids}"
        )
    except Exception:
        pass
    if _incoming_attachment_ids and message_history:
        for _i in range(len(message_history) - 1, -1, -1):
            if message_history[_i].get("role") == "user":
                message_history[_i]["attachment_ids"] = _incoming_attachment_ids
                logger.info(
                    f"[PRD-127] injected {len(_incoming_attachment_ids)} attachment_ids "
                    f"into message_history[{_i}]"
                )
                break
    
    # DEBUG: Log incoming request
    logger.info(f"Chat request - agentId: {request.agentId}")

    # --- PRD-50: Universal Router Integration ---
    # Extract message text for the ingestor
    message_text = ""
    if parts:
        message_text = (parts[0].text or "") if parts[0].type == "text" else ""

    routing_decision = None
    routing_request_id = None
    # PRD-137 Fix #2: when True, agent_factory uses orchestrator-tier defaults
    # (system_settings.orchestrator_llm.*) instead of the agent's model_config.
    use_orchestrator_llm = False
    complexity_assessment = None
    _suggest_mission = False     # PRD-125: True when ORGAN/ORGANISM → suggest mission

    # Every workspace has its own Auto agent — the model, persona, and tools
    # come from that agent's config (set via Settings > Orchestrator).
    # No hardcoded agent IDs. Admins get elevated tool access on the Auto agent.
    _user_role = getattr(ctx.user, "system_role", "user") if ctx.user else "user"
    _is_admin = _user_role in ("admin", "super_admin")
    # PRD-143: the su surface is derived from system_role ONLY — never from
    # workspace role, is_admin, or autonomy level (fail-closed boundary).
    _is_super_admin = _user_role == "super_admin"
    logger.info(f"[PRD-67] user_role={_user_role!r}, is_admin={_is_admin}, user_id={getattr(ctx.user, 'id', '?')}")

    _fallback_agent_id = get_default_agent_id(db, ctx.workspace_id)

    if request.agentId:
        # User explicitly selected an agent — skip Auto, use directly
        effective_agent_id = request.agentId
        logger.info(f"[chat] Direct mode: agent_id={effective_agent_id}")
    else:
        # --- Auto mode: the brain decides (admins included, PRD-67 CTO is fallback) ---
        auto_brain = AutoBrain(db, str(ctx.workspace_id))
        complexity_assessment = await auto_brain.assess(message_text, len(message_history))
        logger.info(
            f"[Auto] Complexity={complexity_assessment.complexity.value} "
            f"action={complexity_assessment.action.value} "
            f"tools={complexity_assessment.matched_tools} "
            f"reasoning={complexity_assessment.reasoning}"
        )

        # Platform management is Auto's core job — never delegate it.
        # When tool_hints include "platform", Auto handles directly with
        # all its platform tools (create agents, read workspace, plan, etc.)
        _platform_hints = "platform" in (complexity_assessment.tool_hints or [])

        if complexity_assessment.action == Action.RESPOND or _platform_hints:
            # Auto handles directly — no routing, no delegation.
            # Auto uses its own agent model (not orchestrator LLM).
            effective_agent_id = _fallback_agent_id
            if _platform_hints:
                # Override action so we don't fall into the DELEGATE branch below
                complexity_assessment.action = Action.RESPOND
                logger.info(
                    f"[Auto] Platform hint detected — Auto handles directly "
                    f"(complexity={complexity_assessment.complexity.value}, "
                    f"hints={complexity_assessment.tool_hints}): "
                    f"agent_id={effective_agent_id}"
                )
            else:
                logger.info(
                    f"[Auto] Direct response (complexity={complexity_assessment.complexity.value}): "
                    f"agent_id={effective_agent_id}"
                )
        elif complexity_assessment.action == Action.MISSION:
            # PRD-125: Complex task detected — suggest mission to user,
            # but still delegate to a single agent for an immediate response.
            logger.info(
                f"[Auto] MISSION suggested (complexity={complexity_assessment.complexity.value})"
            )
            _suggest_mission = True

        if complexity_assessment.action in (Action.DELEGATE, Action.MISSION):
            # Universal Router picks the right specialized agent.
            try:
                ingestor = ChatbotIngestor()
                envelope = ingestor.ingest(
                    message=message_text,
                    agent_id=None,  # no override — let router decide
                    session_id=chat_id,
                    request_context=ctx,
                )
                routing_request_id = str(envelope.id)
                universal_router = UniversalRouter(db, cache=get_routing_cache())
                routing_decision = await universal_router.route(envelope)
            except Exception:
                logger.exception("[chat] Router failed — falling back to fallback agent")
                routing_decision = None

            if routing_decision is not None and routing_decision.route_type == "agent" and routing_decision.agent_id is not None:
                effective_agent_id = routing_decision.agent_id
                logger.info(
                    f"[Auto] Router → agent_id={effective_agent_id} "
                    f"(confidence={routing_decision.confidence:.2f}, reasoning={routing_decision.reasoning})"
                )
            elif routing_decision is not None and routing_decision.route_type == "orchestrate":
                # LLM explicitly chose Auto / orchestrate — Auto uses its own model
                effective_agent_id = _fallback_agent_id
                logger.info(
                    f"[Auto] Router → orchestrate "
                    f"(confidence={routing_decision.confidence:.2f}, "
                    f"reasoning={routing_decision.reasoning}): "
                    f"agent_id={effective_agent_id}"
                )
            else:
                # Router couldn't decide — fall back to Auto with its own model
                effective_agent_id = _fallback_agent_id
                logger.info(f"[Auto] Router returned no match — fallback agent_id={effective_agent_id}")

    # Build response headers (include routing metadata when available)
    response_headers = {
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "x-vercel-ai-data-stream": "v1",
    }
    if routing_decision is not None:
        response_headers["x-routing-agent-id"] = str(routing_decision.agent_id or "")
        response_headers["x-routing-confidence"] = f"{routing_decision.confidence:.2f}"
        response_headers["x-routing-type"] = routing_decision.route_type
        response_headers["x-routing-reasoning"] = routing_decision.reasoning[:200].encode("ascii", "replace").decode("ascii")
        if routing_request_id:
            response_headers["x-routing-request-id"] = routing_request_id
    if complexity_assessment is not None:
        response_headers["x-auto-complexity"] = complexity_assessment.complexity.value
        response_headers["x-auto-action"] = complexity_assessment.action.value
        response_headers["x-auto-confidence"] = f"{complexity_assessment.confidence:.2f}"
        response_headers["x-auto-needs-memory"] = str(complexity_assessment.needs_memory).lower()
        if complexity_assessment.tool_hints:
            response_headers["x-auto-tool-hints"] = ",".join(complexity_assessment.tool_hints)

    # PRD: Unified Agent-Chat System
    # Use agent-based streaming for all resolved agents
    logger.info(f"Using agent-based streaming with agent_id={effective_agent_id}")

    # Session-scoped queue: serialize concurrent requests for the same chat
    session_key = f"{ctx.workspace_id}:{chat_id}"
    session_queue = get_session_queue()

    # Skip Composio tool loading for simple conversational messages (RESPOND)
    _skip_composio = (
        complexity_assessment is not None
        and complexity_assessment.action == Action.RESPOND
    )

    async def _guarded_stream():
        import json as _json

        async with session_queue.acquire(session_key):
            # PRD-125: Emit mission suggestion data event (for frontend to render card)
            if _suggest_mission:
                suggestion_event = streaming_service.streaming_handler.format_aisdk_data(
                    "mission-suggestion",
                    {
                        "goal": message_text,
                        "complexity": complexity_assessment.complexity.value if complexity_assessment else "organ",
                        "agent_id": effective_agent_id,
                    },
                )
                yield suggestion_event

            # Normal agent streaming (RESPOND, DELEGATE, or MISSION fallback)
            async for chunk in streaming_service.stream_response_with_agent(
                chat_id=chat_id,
                messages=message_history,
                agent_id=effective_agent_id,
                user_id=user_id,
                use_orchestrator_llm=use_orchestrator_llm,
                skip_composio=_skip_composio,
                complexity_assessment=complexity_assessment,
                mission_mode=bool(request.missionMode),
                plan_mode=bool(request.planMode),
                suggest_mission=_suggest_mission,
                is_super_admin=_is_super_admin,
            ):
                yield chunk

    return StreamingResponse(
        _guarded_stream(),
        media_type="text/plain; charset=utf-8",
        headers=response_headers,
    )


@router.get("/history")
async def get_chat_history(
    limit: int = 20,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Get chat history for the current user within their workspace"""
    chat_service = ChatService(db)
    user_id = get_user_id(db)

    chats = chat_service.get_chat_history(user_id=user_id, limit=limit, workspace_id=ctx.workspace_id)
    
    return [
        {
            "id": str(chat.id),
            "userId": chat.user_id,
            "title": chat.title,
            "createdAt": chat.created_at.isoformat(),
            "updatedAt": chat.updated_at.isoformat(),
            "visibility": chat.visibility,
            "lastContext": chat.last_context
        }
        for chat in chats
    ]


@router.get("/{chat_id}")
async def get_chat(
    chat_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Get a specific chat (workspace-scoped)"""
    chat_service = ChatService(db)
    user_id = get_user_id(db)

    chat = chat_service.get_chat(chat_id, workspace_id=ctx.workspace_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    if chat.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    return {
        "id": str(chat.id),
        "userId": chat.user_id,
        "title": chat.title,
        "createdAt": chat.created_at.isoformat(),
        "updatedAt": chat.updated_at.isoformat(),
        "visibility": chat.visibility,
        "lastContext": chat.last_context
    }


@router.get("/{chat_id}/messages")
async def get_chat_messages(
    chat_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Get all messages for a chat (workspace-scoped)"""
    chat_service = ChatService(db)
    user_id = get_user_id(db)

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
            "createdAt": msg.created_at.isoformat()
        }
        for msg in messages
    ]


@router.get("/search")
async def search_chat_history(
    q: str,
    limit: int = 20,
    days: int = 30,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Search across chat messages by keyword (workspace-scoped)."""
    from datetime import datetime, timedelta

    user_id = get_user_id(db)
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
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })

    return {"query": q, "total": len(results), "results": results}


@router.delete("/{chat_id}")
async def delete_chat(
    chat_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Delete a chat (workspace-scoped)"""
    chat_service = ChatService(db)
    user_id = get_user_id(db)

    chat = chat_service.get_chat(chat_id, workspace_id=ctx.workspace_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    if chat.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")

    success = chat_service.delete_chat(chat_id)
    return {"success": success}


@router.patch("/{chat_id}")
async def update_chat(
    chat_id: str,
    request: UpdateTitleRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Update chat title (workspace-scoped)"""
    chat_service = ChatService(db)
    user_id = get_user_id(db)

    chat = chat_service.get_chat(chat_id, workspace_id=ctx.workspace_id)
    if not chat:
        raise HTTPException(status_code=404, detail="Chat not found")

    if chat.user_id != user_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    success = chat_service.update_chat_title(chat_id, request.title)
    return {"success": success}


@router.patch("/vote")
async def vote_message(
    request: VoteRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Vote on a message (workspace-scoped)"""
    chat_service = ChatService(db)
    user_id = get_user_id(db)

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


@router.post("/{chat_id}/switch-agent")
async def switch_agent(
    chat_id: str,
    request: SwitchAgentRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Switch to a different agent mid-conversation."""
    from core.models import Chat, Agent
    from datetime import datetime
    import json
    
    chat_service = ChatService(db)
    user_id = get_user_id(db)
    
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
    
    switch_record = {
        "timestamp": datetime.utcnow().isoformat(),
        "from_agent_id": old_agent_id,
        "to_agent_id": request.newAgentId,
        "reason": request.reason or "User requested switch"
    }
    
    existing_switches = getattr(chat, 'agent_switches', None) or []
    if isinstance(existing_switches, str):
        existing_switches = json.loads(existing_switches)
    
    existing_switches.append(switch_record)
    
    db.execute(
        text("UPDATE chats SET agent_switches = :switches WHERE id = :chat_id"),
        {"switches": json.dumps(existing_switches), "chat_id": chat.id}
    )
    
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
