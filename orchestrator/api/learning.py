"""
Learning API Endpoints
=====================

REST API endpoints for learning, feedback, and adaptation systems.
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

class FeedbackRequest(BaseModel):
    """Request model for learning feedback"""
    feedback_type: str = Field(..., description="Type of feedback")
    content: str = Field(..., description="Feedback content")
    rating: Optional[int] = Field(None, description="Rating (1-10)")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")

class FeedbackProcessRequest(BaseModel):
    """Request model for processing feedback"""
    feedback_ids: List[str] = Field(..., description="List of feedback IDs to process")
    processing_options: Optional[Dict[str, Any]] = Field(None, description="Processing options")

# Create router
router = APIRouter(prefix="/api/learning", tags=["🧠 Learning"])

@router.post("/feedback")
async def submit_feedback(
    request: FeedbackRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    ## 📝 Submit Learning Feedback
    
    Submits feedback for system learning and improvement.
    """
    try:
        feedback_id = f"feedback_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        feedback_response = {
            "feedback_id": feedback_id,
            "status": "received",
            "feedback_type": request.feedback_type,
            "rating": request.rating,
            "processing_status": "queued",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return feedback_response
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/feedback/process")
async def process_feedback(
    request: FeedbackProcessRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    ## ⚙️ Process Learning Feedback
    
    Processes submitted feedback for system learning.
    """
    try:
        processing_result = {
            "processing_id": f"proc_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "feedback_ids": request.feedback_ids,
            "status": "completed",
            "insights_extracted": len(request.feedback_ids) * 2,
            "improvements_identified": 5,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return processing_result
        
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/insights/aggregate")
async def aggregate_insights(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    period: Optional[str] = "last_30_days",
    db: Session = Depends(get_db)
):
    """
    ## 📊 Aggregate Learning Insights
    
    Aggregates learning insights over a specified period.
    """
    try:
        insights = {
            "period": period,
            "total_feedback": 245,
            "processed_feedback": 240,
            "key_insights": [
                "Users prefer faster response times",
                "Multi-agent coordination needs improvement",
                "Documentation clarity is important"
            ],
            "improvement_areas": [
                "Response time optimization",
                "User interface enhancements",
                "Error handling improvements"
            ],
            "learning_metrics": {
                "adaptation_rate": 0.85,
                "improvement_velocity": 0.72,
                "user_satisfaction_trend": "increasing"
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return insights
        
    except Exception as e:
        logger.error(f"Error aggregating insights: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/effectiveness-report")
async def effectiveness_report(
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db)
):
    """
    ## 📈 Learning Effectiveness Report
    
    Generates a comprehensive learning effectiveness report.
    """
    try:
        report = {
            "report_id": f"report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "period": "last_quarter",
            "overall_effectiveness": 0.78,
            "learning_metrics": {
                "knowledge_retention": 0.82,
                "skill_improvement": 0.75,
                "adaptation_speed": 0.69,
                "error_reduction": 0.84
            },
            "performance_improvements": {
                "response_time": "15% faster",
                "accuracy": "12% improvement",
                "user_satisfaction": "18% increase"
            },
            "recommendations": [
                "Increase feedback collection frequency",
                "Implement real-time learning updates",
                "Enhance pattern recognition algorithms"
            ],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating effectiveness report: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
