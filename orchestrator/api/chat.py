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

# Get tools from consumers.chatbot (uses modules.tools)
CHAT_TOOLS = get_chat_tools()

logger = logging.getLogger(__name__)

# Standard API key auth (matches all other APIs)
def require_api_key(x_api_key: str = Header(None)):
    required = os.getenv("API_KEY")
    if required and x_api_key != required:
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
    db: Session = Depends(get_db)
):
    """Stream chat messages using AI SDK Data Stream format (text/plain)"""
    chat_service = ChatService(db)
    streaming_service = StreamingChatService(db)
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
        parts=[part.dict() for part in parts]
    )
    
    # Get chat history
    messages = chat_service.get_messages_by_chat_id(chat_id)
    message_history = [{'role': msg.role, 'parts': msg.parts} for msg in messages]
    
    # Stream response using AI SDK Data Stream format
    # Pass selected model from frontend request
    return StreamingResponse(
        streaming_service.stream_response_aisdk(
            chat_id=chat_id,
            messages=message_history,
            tools=CHAT_TOOLS,
            selected_model=request.selectedChatModel
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

