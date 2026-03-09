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


# ---------------------------------------------------------------------------
# PRD-67: CTO Agent lookup (cached in-process)
# ---------------------------------------------------------------------------
_CTO_CACHE_SENTINEL = object()
_cto_agent_id_cache: dict = {}  # {"id": int|None|SENTINEL, "ts": float}
_CTO_CACHE_TTL = 300  # 5 minutes


def _get_cto_agent_id(db: Session) -> Optional[int]:
    """Get the CTO Agent's ID (cached). Returns None if not seeded."""
    import time

    cached = _cto_agent_id_cache.get("id", _CTO_CACHE_SENTINEL)
    ts = _cto_agent_id_cache.get("ts", 0)
    if cached is not _CTO_CACHE_SENTINEL and (time.time() - ts) < _CTO_CACHE_TTL:
        return cached  # may be None — means CTO not seeded yet, respect TTL

    try:
        from core.models.core import Agent
        row = db.query(Agent.id).filter(
            Agent.slug == "auto-cto",
            Agent.is_system_agent.is_(True),
            Agent.status == "active",
        ).first()
        agent_id = row[0] if row else None
        _cto_agent_id_cache["id"] = agent_id
        _cto_agent_id_cache["ts"] = time.time()
        return agent_id
    except Exception:
        logger.debug("CTO Agent lookup failed", exc_info=True)
        return None


router = APIRouter(prefix="/api/chat", tags=["💬 Chat"])


# ---------------------------------------------------------------------------
# PRD-68 Phase 2: Workflow Bridge — ORGAN/ORGANISM chat → workflow engine
# ---------------------------------------------------------------------------

async def _stream_workflow_bridge(
    db: Session,
    chat_id: str,
    message_text: str,
    workspace_id,
    agent_id: int,
    user_id: int,
    streaming_service,
    complexity_assessment=None,
):
    """
    Bridge chat messages to the PRD-59 workflow engine for ORGAN/ORGANISM tasks.

    Creates a transient workflow from the user's message, executes it through
    the full pipeline (PLAN → PREPARE → EXECUTE → EVALUATE → LEARN), and
    streams stage events back as AI SDK format into the chat response.

    The workflow is tagged 'chat_generated' so users can find/re-run it.
    """
    import asyncio
    import json

    # 1. Send chat_id to frontend (same as normal chat flow)
    yield streaming_service.streaming_handler.format_aisdk_chat_id(chat_id)
    await asyncio.sleep(0)

    try:
        from core.models.core import Workflow, WorkflowExecution
        from consumers.workflows.streaming import stream_workflow_as_aisdk, get_stream_manager

        # 2. Create transient workflow from the user's message
        workflow = Workflow(
            name=f"Chat workflow: {message_text[:60]}{'...' if len(message_text) > 60 else ''}",
            description=message_text,
            goal=message_text,
            context=f"Generated from chat {chat_id} by AutoBrain (complexity={complexity_assessment.complexity.value})" if complexity_assessment else "",
            workflow_definition={
                "steps": [{
                    "name": "Execute task",
                    "description": message_text,
                    "agent_id": agent_id,
                }],
                "source": "chat_generated",
            },
            status="active",
            workspace_id=workspace_id,
            tags=["chat_generated", "auto"],
        )
        db.add(workflow)
        db.commit()
        db.refresh(workflow)

        logger.info(f"[PRD-68] Created transient workflow id={workflow.id} from chat")

        # 3. Create execution record
        execution = WorkflowExecution(
            workflow_id=workflow.id,
            agent_id=agent_id,
            workspace_id=workspace_id,
            input_data={
                "message": message_text,
                "chat_id": chat_id,
                "user_id": user_id,
            },
            status="pending",
        )
        db.add(execution)
        db.commit()
        db.refresh(execution)

        logger.info(f"[PRD-68] Created workflow execution id={execution.id}")

        # 4. Send workflow-started event to frontend
        yield streaming_service.streaming_handler.format_aisdk_data({
            "type": "workflow-update",
            "status": "started",
            "workflow_id": workflow.id,
            "execution_id": execution.id,
            "complexity": complexity_assessment.complexity.value if complexity_assessment else "organ",
        })
        await asyncio.sleep(0)

        # 5. Kick off execution in background
        from api.workflows import execute_workflow_with_progress

        async def _run_workflow():
            try:
                await execute_workflow_with_progress(execution.id, {})
            except Exception as e:
                logger.error(f"[PRD-68] Workflow execution failed: {e}", exc_info=True)

        asyncio.create_task(_run_workflow())

        # 6. Stream workflow events as AI SDK format back to chat
        # Small delay to let the workflow register with the stream manager
        await asyncio.sleep(0.3)

        async for chunk in stream_workflow_as_aisdk(execution.id):
            yield chunk

        # 7. Save the workflow result as an assistant message in the chat
        db.refresh(execution)
        if execution.output_data:
            final_text = execution.output_data.get("final_response", "")
            if final_text:
                streaming_service.chat_service.save_message(
                    chat_id=chat_id,
                    role="assistant",
                    parts=[{"type": "text", "text": final_text}],
                    workspace_id=workspace_id,
                )

        # 8. Emit finish event
        yield streaming_service.streaming_handler.format_aisdk_data({
            "type": "workflow-update",
            "status": "completed",
            "workflow_id": workflow.id,
            "execution_id": execution.id,
        })
        yield 'd:{"finishReason":"stop","usage":{"promptTokens":0,"completionTokens":0}}\n'

    except Exception as e:
        logger.error(f"[PRD-68] Workflow bridge failed: {e}", exc_info=True)
        # Fall back to normal chat response
        error_text = f"Workflow execution failed. Falling back to direct response.\n\nError: {e}"
        yield f'0:{json.dumps(error_text)}\n'
        yield f'e:{json.dumps({"message": str(e)})}\n'


# Request/Response Models
class MessagePart(BaseModel):
    type: str
    text: Optional[str] = None
    filename: Optional[str] = None
    mediaType: Optional[str] = None
    url: Optional[str] = None


class ChatMessageRequest(BaseModel):
    role: str = "user"
    parts: Optional[List[MessagePart]] = None
    # Compatibility with older/alternate clients
    content: Optional[str] = None


class ChatRequest(BaseModel):
    id: Optional[str] = None
    message: ChatMessageRequest
    # Compatibility with AI SDK "messages" payloads
    messages: Optional[List[ChatMessageRequest]] = None
    selectedChatModel: Optional[str] = "gpt-4"
    selectedVisibilityType: Optional[str] = "private"
    context: Optional[dict] = None
    # PRD: Unified Agent-Chat System
    agentId: Optional[int] = None  # Selected agent ID (default: system agent id=1)


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
    """
    Pick a sensible default agent for chat when the client does not send agentId.

    Preference:
    - Any agent in this workspace with active EXTERNAL app assignments (Composio),
      ordered by number of assignments (desc).
    - Fallback to agent id=1.
    """
    try:
        from core.models import Agent
        from core.models.composio_cache import AgentAppAssignment
        from core.composio.entity_manager import EntityManager

        # Connected apps for this workspace (must be connected, otherwise Composio gating will deny)
        connected_apps: set[str] = set()
        try:
            manager = EntityManager(db)
            connected_apps = {a.upper().strip() for a in manager.get_connected_apps(workspace_id)}
        except Exception:
            connected_apps = set()

        # Candidate agents with EXTERNAL app assignments (descending by assignment count)
        candidates = (
            db.query(AgentAppAssignment.agent_id)
            .join(Agent, Agent.id == AgentAppAssignment.agent_id)
            .filter(
                Agent.workspace_id == workspace_id,
                AgentAppAssignment.is_active == True,  # noqa: E712
                AgentAppAssignment.app_type == "EXTERNAL",
            )
            .group_by(AgentAppAssignment.agent_id)
            .order_by(func.count(AgentAppAssignment.id).desc(), AgentAppAssignment.agent_id.asc())
            .limit(10)
            .all()
        )

        # Pick the first candidate that has at least one assigned app connected in this workspace.
        for (agent_id,) in candidates:
            if not agent_id:
                continue
            if connected_apps:
                assigned_apps = {
                    (r[0] or "").upper().strip()
                    for r in db.query(AgentAppAssignment.app_name)
                    .filter(
                        AgentAppAssignment.agent_id == agent_id,
                        AgentAppAssignment.is_active == True,  # noqa: E712
                        AgentAppAssignment.app_type == "EXTERNAL",
                    )
                    .all()
                }
                assigned_apps.discard("")
                if not assigned_apps.intersection(connected_apps):
                    continue
            return int(agent_id)
    except Exception:
        pass
    return 1


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
            visibility=request.selectedVisibilityType
        )
        chat_id = str(chat.id)
    else:
        chat = chat_service.get_chat(chat_id)
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
    
    # DEBUG: Log incoming request
    logger.info(f"Chat request - agentId: {request.agentId}, model: {request.selectedChatModel}")

    # --- PRD-50: Universal Router Integration ---
    # Extract message text for the ingestor
    message_text = ""
    if parts:
        message_text = (parts[0].text or "") if parts[0].type == "text" else ""

    routing_decision = None
    routing_request_id = None
    use_system_llm = False  # True = use orchestrator LLM settings, not agent's model
    complexity_assessment = None
    _use_workflow_bridge = False  # PRD-68: True when ORGAN/ORGANISM triggers workflow

    # PRD-67: Admin persona is CTO Agent (Auto with elevated access).
    # AutoBrain + Router run for ALL users — the CTO agent is the fallback
    # persona for admins (RESPOND, orchestrate, router-miss), not a routing bypass.
    _user_role = getattr(ctx.user, "system_role", "user") if ctx.user else "user"
    _is_admin = _user_role in ("admin", "super_admin")
    logger.info(f"[PRD-67] user_role={_user_role!r}, is_admin={_is_admin}, user_id={getattr(ctx.user, 'id', '?')}")

    _cto_id = _get_cto_agent_id(db) if _is_admin else None
    _fallback_agent_id = _cto_id or get_default_agent_id(db, ctx.workspace_id)

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
            effective_agent_id = _fallback_agent_id
            use_system_llm = True
            if _platform_hints:
                # Override action so we don't fall into the DELEGATE branch below
                complexity_assessment.action = Action.RESPOND
                logger.info(
                    f"[Auto] Platform hint detected — Auto handles directly "
                    f"(complexity={complexity_assessment.complexity.value}, "
                    f"hints={complexity_assessment.tool_hints}): "
                    f"agent_id={effective_agent_id} with orchestrator LLM"
                )
            else:
                logger.info(
                    f"[Auto] Direct response (complexity={complexity_assessment.complexity.value}): "
                    f"agent_id={effective_agent_id} with orchestrator LLM"
                )
        elif complexity_assessment.action == Action.WORKFLOW:
            # PRD-68 Phase 2: ORGAN/ORGANISM → workflow execution via chat.
            # Create transient workflow, execute through PRD-59 pipeline,
            # stream stage events as AI SDK format back to chat.
            logger.info(
                f"[Auto] WORKFLOW detected (complexity={complexity_assessment.complexity.value})"
            )
            _use_workflow_bridge = True

        if complexity_assessment.action in (Action.DELEGATE, Action.WORKFLOW):
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
                # LLM explicitly chose Auto / orchestrate — use fallback agent with system LLM
                effective_agent_id = _fallback_agent_id
                use_system_llm = True
                logger.info(
                    f"[Auto] Router → orchestrate "
                    f"(confidence={routing_decision.confidence:.2f}, "
                    f"reasoning={routing_decision.reasoning}): "
                    f"agent_id={effective_agent_id} with orchestrator LLM"
                )
            else:
                # Router couldn't decide — fall back
                effective_agent_id = _fallback_agent_id
                use_system_llm = True
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
        async with session_queue.acquire(session_key):
            if _use_workflow_bridge:
                # PRD-68 Phase 2: Stream workflow execution as AI SDK events
                async for chunk in _stream_workflow_bridge(
                    db=db,
                    chat_id=chat_id,
                    message_text=message_text,
                    workspace_id=ctx.workspace_id,
                    agent_id=effective_agent_id,
                    user_id=user_id,
                    streaming_service=streaming_service,
                    complexity_assessment=complexity_assessment,
                ):
                    yield chunk
            else:
                async for chunk in streaming_service.stream_response_with_agent(
                    chat_id=chat_id,
                    messages=message_history,
                    agent_id=effective_agent_id,
                    user_id=user_id,
                    use_system_llm=use_system_llm,
                    skip_composio=_skip_composio,
                    complexity_assessment=complexity_assessment,
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
    """Get chat history for the current user"""
    chat_service = ChatService(db)
    user_id = get_user_id(db)
    
    chats = chat_service.get_chat_history(user_id=user_id, limit=limit)
    
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
    """Get a specific chat"""
    chat_service = ChatService(db)
    user_id = get_user_id(db)
    
    chat = chat_service.get_chat(chat_id)
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
    """Get all messages for a chat"""
    chat_service = ChatService(db)
    user_id = get_user_id(db)
    
    # Verify chat access
    chat = chat_service.get_chat(chat_id)
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


@router.delete("/{chat_id}")
async def delete_chat(
    chat_id: str,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """Delete a chat"""
    chat_service = ChatService(db)
    user_id = get_user_id(db)
    
    # Verify chat access
    chat = chat_service.get_chat(chat_id)
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
    """Update chat title"""
    chat_service = ChatService(db)
    user_id = get_user_id(db)
    
    # Verify chat access
    chat = chat_service.get_chat(chat_id)
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
    """Vote on a message"""
    chat_service = ChatService(db)
    user_id = get_user_id(db)
    
    # Verify chat access
    chat = chat_service.get_chat(request.chatId)
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
    
    chat = chat_service.get_chat(chat_id)
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
