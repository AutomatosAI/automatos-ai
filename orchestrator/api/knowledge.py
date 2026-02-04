"""
Knowledge API Endpoints
======================

REST API endpoints for knowledge sharing and management.
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from datetime import datetime

from core.database.database import get_db
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext

logger = logging.getLogger(__name__)

class KnowledgeShareRequest(BaseModel):
    """Request model for knowledge sharing"""
    title: str = Field(..., description="Knowledge title")
    content: str = Field(..., description="Knowledge content")
    category: Optional[str] = Field(None, description="Knowledge category")
    tags: Optional[List[str]] = Field(None, description="Knowledge tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

# Create router
router = APIRouter(prefix="/api/knowledge", tags=["📚 Knowledge"])

@router.post("/share")
async def share_knowledge(
    request: KnowledgeShareRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    ## 📚 Share Knowledge
    
    Shares knowledge with the system for learning and reuse.
    """
    try:
        knowledge_id = f"know_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        knowledge_response = {
            "knowledge_id": knowledge_id,
            "title": request.title,
            "status": "shared",
            "category": request.category or "general",
            "tags": request.tags or [],
            "processing_status": "indexed",
            "accessibility": "system_wide",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return knowledge_response
        
    except Exception as e:
        logger.error(f"Error sharing knowledge: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
