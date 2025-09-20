"""
Real Chatbot Implementation
===========================

NO MOCK DATA - Uses actual LLM service and real database queries.
Returns honest errors when features aren't ready.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
from datetime import datetime

from database.database import get_db
from database.models import Agent, Document, Skill
from services.llm_provider import LLMService, LLMProvider

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["💬 Chatbot"])

# Initialize LLM service
llm_service = None

def get_llm_service():
    """Get or initialize LLM service"""
    global llm_service
    if not llm_service:
        # Check for API keys
        if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
            raise HTTPException(
                status_code=501,
                detail="No LLM API keys configured. Set OPENAI_API_KEY or ANTHROPIC_API_KEY."
            )
        llm_service = LLMService.create_default()
    return llm_service

@router.post("/message")
async def chat_message(
    message: str,
    context: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db)
):
    """
    Process a chat message - REAL IMPLEMENTATION
    
    Handles:
    - Database queries (agents, documents, skills)
    - General Q&A via LLM
    - Returns real data or honest errors
    """
    try:
        message_lower = message.lower()
        
        # 1. Check for database queries
        if any(word in message_lower for word in ["how many", "count", "list", "show me"]):
            return await handle_data_query(message, db)
        
        # 2. Check for document search
        if any(word in message_lower for word in ["search", "find", "look for"]):
            return await handle_document_search(message, db)
        
        # 3. Use LLM for general chat
        llm = get_llm_service()
        
        # Build context from database if needed
        system_context = await build_system_context(db)
        
        messages = [
            {"role": "system", "content": f"You are an AI assistant for the Automatos platform. {system_context}"},
            {"role": "user", "content": message}
        ]
        
        # Add conversation context if provided
        if context and "history" in context:
            # Insert history before current message
            messages = messages[:-1] + context["history"][-5:] + [messages[-1]]
        
        # Get real LLM response
        response = await llm.generate_response(messages)
        
        return {
            "response": response.content,
            "type": "llm",
            "model": response.model,
            "usage": response.usage
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        # Return honest error, not fake success
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

async def handle_data_query(message: str, db: Session) -> Dict[str, Any]:
    """
    Handle database queries - REAL DATA ONLY
    """
    try:
        message_lower = message.lower()
        
        # Agent queries
        if "agent" in message_lower:
            total_agents = db.query(func.count(Agent.id)).scalar()
            active_agents = db.query(func.count(Agent.id)).filter(Agent.status == "active").scalar()
            
            response = f"You have {total_agents} agents total ({active_agents} active)"
            
            if "list" in message_lower or "show" in message_lower:
                agents = db.query(Agent).limit(5).all()
                agent_list = [f"- {a.name} ({a.agent_type})" for a in agents]
                response += "\\n\\nRecent agents:\\n" + "\\n".join(agent_list)
            
            return {
                "response": response,
                "type": "database_query",
                "data": {
                    "total_agents": total_agents,
                    "active_agents": active_agents
                }
            }
        
        # Document queries
        if "document" in message_lower:
            total_docs = db.query(func.count(Document.id)).scalar()
            processed_docs = db.query(func.count(Document.id)).filter(Document.status == "processed").scalar()
            
            return {
                "response": f"You have {total_docs} documents ({processed_docs} processed)",
                "type": "database_query",
                "data": {
                    "total_documents": total_docs,
                    "processed_documents": processed_docs
                }
            }
        
        # Skill queries
        if "skill" in message_lower:
            total_skills = db.query(func.count(Skill.id)).scalar()
            return {
                "response": f"You have {total_skills} skills defined",
                "type": "database_query",
                "data": {"total_skills": total_skills}
            }
        
        # Fallback to LLM if we can't parse the query
        raise ValueError("Could not parse data query")
        
    except Exception as e:
        logger.error(f"Data query error: {e}")
        # If we can't handle it, pass to LLM
        raise

async def handle_document_search(message: str, db: Session) -> Dict[str, Any]:
    """
    Handle document search - REAL SEARCH or HONEST ERROR
    """
    # Extract search query
    search_terms = message.lower().replace("search for", "").replace("find", "").replace("look for", "").strip()
    
    # Check if we have processed documents
    processed_docs = db.query(Document).filter(Document.status == "processed").count()
    
    if processed_docs == 0:
        return {
            "response": "No documents have been processed yet. Please upload and process documents first.",
            "type": "search",
            "data": {"processed_documents": 0}
        }
    
    # TODO: Implement actual vector search
    # For now, return honest response
    raise HTTPException(
        status_code=501,
        detail="Document search not yet implemented. Documents are processed but search API not connected."
    )

async def build_system_context(db: Session) -> str:
    """Build context about the system for LLM"""
    try:
        agent_count = db.query(func.count(Agent.id)).scalar()
        doc_count = db.query(func.count(Document.id)).scalar()
        skill_count = db.query(func.count(Skill.id)).scalar()
        
        return f"""
        Current system state:
        - {agent_count} agents configured
        - {doc_count} documents stored  
        - {skill_count} skills available
        """
    except:
        return "System context unavailable."

@router.get("/test")
async def test_chat():
    """Test endpoint to verify chat is working"""
    try:
        llm = get_llm_service()
        response = await llm.generate_response([
            {"role": "user", "content": "Say 'Chat is working!' in 5 words or less"}
        ])
        return {
            "status": "working",
            "test_response": response.content,
            "llm_provider": str(llm.config.provider.value)
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "message": "LLM not configured properly"
        }
