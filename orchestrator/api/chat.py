"""
Chat API
========
PRD-27: Chat endpoints for streaming conversations, history, and voting.

Follows standard API pattern with require_api_key dependency.
"""

import os
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Header
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel

from core.database.database import get_db
from consumers.chatbot import ChatService, StreamingChatService, get_chat_tools
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext

logger = logging.getLogger(__name__)

# Standard API key auth (matches all other APIs)
# For chat endpoint, API key is optional if not set in env (user-facing feature)
def require_api_key(x_api_key: str = Header(None)):
    required = os.getenv("API_KEY")
    # If API_KEY is not set in env, allow requests (for user-facing chat)
    if not required:
        return True
    # If API_KEY is set, require it
    if x_api_key != required:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True

router = APIRouter(prefix="/api/chat", tags=["💬 Chat"])


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


# Endpoints
@router.post("")
async def stream_chat(
    request: ChatRequest,
    _x_api_key: bool = Depends(require_api_key),
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
    
    # PRD: Unified Agent-Chat System
    # Use agent-based streaming if agentId is provided
    if request.agentId:
        logger.info(f"Using agent-based streaming with agent_id={request.agentId}")
        return StreamingResponse(
            streaming_service.stream_response_with_agent(
                chat_id=chat_id,
                messages=message_history,
                agent_id=request.agentId,
                user_id=user_id
            ),
            media_type="text/plain; charset=utf-8",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "x-vercel-ai-data-stream": "v1"
            }
        )
    
    # Legacy: Stream response using AI SDK Data Stream format with model selection
    return StreamingResponse(
        streaming_service.stream_response_aisdk(
            chat_id=chat_id,
            messages=message_history,
            tools=get_chat_tools(agent_id=request.agentId or 1),
            selected_model=request.selectedChatModel,
            agent_id=request.agentId or 1
        ),
        media_type="text/plain; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "x-vercel-ai-data-stream": "v1"
        }
    )


@router.get("/history")
async def get_chat_history(
    limit: int = 20,
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
    db: Session = Depends(get_db)
):
    """Get list of available agents for chat selection."""
    from core.models import Agent
    
    query = db.query(Agent).filter(Agent.status == status)
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
    
    new_agent = db.query(Agent).filter(Agent.id == request.newAgentId).first()
    if not new_agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
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
