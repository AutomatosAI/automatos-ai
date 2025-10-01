"""
Chatbot Suggestions API
=======================

Provides contextual suggestions for dashboard elements based on user role and current page.
"""

from fastapi import APIRouter, Query, HTTPException
from typing import List, Dict, Any, Optional
import logging
from pydantic import BaseModel
import json

# Setup router and logger
router = APIRouter(prefix="/api/chatbot", tags=["chatbot"])
logger = logging.getLogger(__name__)

# Pydantic model for suggestions
class ChatbotSuggestion(BaseModel):
    text: str
    action: Optional[str] = None
    category: Optional[str] = None
    priority: Optional[int] = 0

# Simple suggestion sets with minimal context
# Keeps dashboard working without complex data structures
def get_page_suggestions(page):
    """Get basic suggestions based on current page"""
    if page == "dashboard":
        return [
            {"text": "How is system performance?", "action": "system_health"},
            {"text": "Show me agent metrics", "action": "agent_metrics"}
        ]
    elif page == "agents":
        return [
            {"text": "List all agents", "action": "list_agents"},
            {"text": "Show agent status", "action": "agent_status"}
        ]
    elif page == "documents":
        return [
            {"text": "List all documents", "action": "list_documents"},
            {"text": "Show document statistics", "action": "document_stats"}
        ]
    elif page == "workflows":
        return [
            {"text": "Show active workflows", "action": "active_workflows"},
            {"text": "List all workflows", "action": "list_workflows"}
        ]
    else:
        # Default suggestions for any page
        return [
            {"text": "Help me with this page", "action": "help"},
            {"text": "Show system status", "action": "system_status"}
        ]

@router.get("/suggestions")
async def get_chatbot_suggestions(
    page: str = Query(..., description="Current page: dashboard, agents, documents, workflows"),
    user_role: str = Query("user", description="User role: admin, developer, user")
):
    """
    Get contextual chatbot suggestions based on current page and user role
    """
    try:
        # Get base suggestions for the page
        page = page.lower()
        suggestions = get_page_suggestions(page)
        
        # Convert to Pydantic models
        return [ChatbotSuggestion(**suggestion) for suggestion in suggestions]
        
    except Exception as e:
        logger.error(f"Error generating chatbot suggestions: {e}")
        # Return a minimal set rather than failing
        return [
            ChatbotSuggestion(text="How can I help you today?"),
            ChatbotSuggestion(text="Tell me about the system status")
        ]

# Health check endpoint
@router.get("/health")
async def health_check():
    """Check if suggestions service is healthy"""
    from datetime import datetime
    return {
        "status": "healthy",
        "service": "chatbot_suggestions",
        "timestamp": datetime.now().isoformat(),
    }
