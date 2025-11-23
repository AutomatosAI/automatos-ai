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

from database.database import get_db
from services.chat_service import ChatService, StreamingChatService

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
    parts: List[MessagePart]


class ChatRequest(BaseModel):
    id: Optional[str] = None
    message: ChatMessageRequest
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
    db: Session = Depends(get_db)
):
    """Stream chat messages using Server-Sent Events"""
    chat_service = ChatService(db)
    streaming_service = StreamingChatService(db)
    user_id = get_user_id(db)
    
    # Get or create chat
    chat_id = request.id
    if not chat_id:
        first_part = request.message.parts[0] if request.message.parts else None
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
            raise HTTPException(status_code=404, detail="Chat not found")
        
        if chat.user_id != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
    
    # Save user message
    chat_service.save_message(
        chat_id=chat_id,
        role="user",
        parts=[part.dict() for part in request.message.parts]
    )
    
    # Get chat history
    messages = chat_service.get_messages_by_chat_id(chat_id)
    message_history = [{'role': msg.role, 'parts': msg.parts} for msg in messages]
    
    # Stream response
    return StreamingResponse(
        streaming_service.stream_response(
            chat_id=chat_id,
            messages=message_history
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
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

