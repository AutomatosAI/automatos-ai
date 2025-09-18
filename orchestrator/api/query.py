"""
Query API Endpoints
==================

REST API endpoints for platform knowledge queries and help system.
"""

import os
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Body, Header
from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from datetime import datetime

from database.database import get_db

logger = logging.getLogger(__name__)

# Simple API key auth dependency
def require_api_key(x_api_key: str = Header(None)):
    required = os.getenv("API_KEY")
    if required and x_api_key != required:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True

class PlatformHelpRequest(BaseModel):
    """Request model for platform help queries"""
    query: str = Field(..., description="Help query or question")
    context: Optional[str] = Field(None, description="Additional context")
    category: Optional[str] = Field(None, description="Help category")

# Create router
router = APIRouter(prefix="/api/query", tags=["🔍 Query"])

@router.post("/platform-help", dependencies=[Depends(require_api_key)])
async def platform_help(
    request: PlatformHelpRequest,
    db: Session = Depends(get_db)
):
    """
    ## 🆘 Platform Help Query
    
    Provides intelligent help and guidance for platform usage.
    """
    try:
        # Mock help response
        help_response = {
            "query": request.query,
            "category": request.category or "general",
            "response": f"Here's help for your query: {request.query}",
            "suggestions": [
                "Check the documentation",
                "Try the getting started guide",
                "Contact support if needed"
            ],
            "related_topics": ["API usage", "Authentication", "Best practices"],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return help_response
        
    except Exception as e:
        logger.error(f"Error processing platform help query: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process help query: {str(e)}")

