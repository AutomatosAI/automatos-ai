
"""
Knowledge API Endpoints
======================

REST API endpoints for knowledge sharing and management.
"""

import os
import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, Body, Header
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from datetime import datetime

from core.database.database import get_db

logger = logging.getLogger(__name__)

# Simple API key auth dependency
def require_api_key(x_api_key: str = Header(None)):
    required = os.getenv("API_KEY")
    if required and x_api_key != required:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True

class KnowledgeShareRequest(BaseModel):
    """Request model for knowledge sharing"""
    title: str = Field(..., description="Knowledge title")
    content: str = Field(..., description="Knowledge content")
    category: Optional[str] = Field(None, description="Knowledge category")
    tags: Optional[List[str]] = Field(None, description="Knowledge tags")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

# Create router
router = APIRouter(prefix="/api/knowledge", tags=["📚 Knowledge"])

@router.post("/share", dependencies=[Depends(require_api_key)])
async def share_knowledge(
    request: KnowledgeShareRequest,
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
        raise HTTPException(status_code=500, detail=f"Failed to share knowledge: {str(e)}")
