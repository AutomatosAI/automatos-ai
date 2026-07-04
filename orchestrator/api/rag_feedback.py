"""
RAG Feedback API Endpoints
===========================

Collect user feedback on RAG-generated responses and provide quality analytics.
"""

import logging
from typing import List, Optional
from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException, Depends, Query
from sqlalchemy.orm import Session
from sqlalchemy import text
from pydantic import BaseModel, Field

from core.database.database import get_db
from core.auth.hybrid import get_request_context_hybrid
from core.auth.dependencies import RequestContext
from modules.rag.feedback_writer import write_rag_feedback

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/rag/feedback", tags=["rag-feedback"])


# --- Request / Response models ---

class RAGFeedbackRequest(BaseModel):
    query: str = Field(..., min_length=1)
    response_text: Optional[str] = None
    chunk_ids: Optional[List[int]] = None
    document_ids: Optional[List[int]] = None
    rating: Optional[int] = Field(None, ge=1, le=5)
    feedback_type: str = Field("thumbs_up", pattern="^(thumbs_up|thumbs_down|star_rating|correction)$")
    correction_text: Optional[str] = None
    rag_config_id: Optional[int] = None
    execution_time_ms: Optional[float] = None


class FeedbackStatsResponse(BaseModel):
    total_feedback: int = 0
    avg_rating: Optional[float] = None
    thumbs_up_pct: float = 0.0
    thumbs_down_pct: float = 0.0
    correction_count: int = 0
    time_range: str = "7d"


# --- Endpoints ---

@router.post("")
async def submit_feedback(
    feedback: RAGFeedbackRequest,
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Submit feedback on a RAG-generated response."""
    try:
        # PRD-185 S7: single shared writer — the chat vote path lands rows here too.
        feedback_id = write_rag_feedback(
            db,
            query=feedback.query,
            workspace_id=ctx.workspace_id,
            user_id=getattr(ctx, "user_id", None),
            document_ids=feedback.document_ids,
            chunk_ids=feedback.chunk_ids,
            rating=feedback.rating,
            feedback_type=feedback.feedback_type,
            correction_text=feedback.correction_text,
            rag_config_id=feedback.rag_config_id,
            execution_time_ms=feedback.execution_time_ms,
            response_text=feedback.response_text,
        )
        return {"success": True, "feedback_id": feedback_id}

    except Exception as e:
        db.rollback()
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/stats")
async def get_feedback_stats(
    time_range: str = Query("7d", pattern="^(7d|30d|90d|all)$"),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Get aggregated feedback statistics for the workspace."""
    try:
        days_map = {"7d": 7, "30d": 30, "90d": 90, "all": 36500}
        days = days_map.get(time_range, 7)
        cutoff = datetime.utcnow() - timedelta(days=days)

        stats_query = text("""
            SELECT
                COUNT(*) as total,
                AVG(rating) as avg_rating,
                COUNT(*) FILTER (WHERE feedback_type = 'thumbs_up') as thumbs_up,
                COUNT(*) FILTER (WHERE feedback_type = 'thumbs_down') as thumbs_down,
                COUNT(*) FILTER (WHERE feedback_type = 'correction') as corrections
            FROM rag_feedback
            WHERE workspace_id = :workspace_id
              AND created_at >= :cutoff
        """)

        row = db.execute(stats_query, {
            "workspace_id": str(ctx.workspace_id),
            "cutoff": cutoff,
        }).fetchone()

        total = row.total if row else 0
        thumbs_up = row.thumbs_up if row else 0
        thumbs_down = row.thumbs_down if row else 0

        return {
            "total_feedback": total,
            "avg_rating": round(float(row.avg_rating), 2) if row and row.avg_rating else None,
            "thumbs_up_pct": round(thumbs_up / total * 100, 1) if total > 0 else 0.0,
            "thumbs_down_pct": round(thumbs_down / total * 100, 1) if total > 0 else 0.0,
            "correction_count": row.corrections if row else 0,
            "time_range": time_range,
        }

    except Exception as e:
        logger.error(f"Error getting feedback stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/recent")
async def get_recent_feedback(
    limit: int = Query(20, ge=1, le=100),
    ctx: RequestContext = Depends(get_request_context_hybrid),
    db: Session = Depends(get_db),
):
    """Get recent feedback entries."""
    try:
        query = text("""
            SELECT id, query, feedback_type, rating, correction_text, created_at
            FROM rag_feedback
            WHERE workspace_id = :workspace_id
            ORDER BY created_at DESC
            LIMIT :limit
        """)

        rows = db.execute(query, {
            "workspace_id": str(ctx.workspace_id),
            "limit": limit,
        }).fetchall()

        return {
            "feedback": [
                {
                    "id": r.id,
                    "query": r.query,
                    "feedback_type": r.feedback_type,
                    "rating": r.rating,
                    "correction_text": r.correction_text,
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in rows
            ]
        }

    except Exception as e:
        logger.error(f"Error getting recent feedback: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
