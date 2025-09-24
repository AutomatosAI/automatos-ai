"""
PRD-06: Analytics API Endpoints
REST API endpoints for dashboard data
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from services.analytics_engine import AnalyticsEngine
from services.dashboard_realtime import get_websocket_manager
from database.database import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

# Global analytics engine instance
analytics_engine = None

def get_analytics_engine() -> AnalyticsEngine:
    """Get the global analytics engine instance"""
    global analytics_engine
    if analytics_engine is None:
        analytics_engine = AnalyticsEngine()
    return analytics_engine

@router.get("/dashboard/overview")
async def get_dashboard_overview():
    """
    Get comprehensive dashboard overview data
    """
    try:
        engine = get_analytics_engine()
        data = await engine.get_dashboard_overview()
        return JSONResponse(content=data)
    except Exception as e:
        logger.error(f"Error getting dashboard overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents/{agent_id}")
async def get_agent_analytics(
    agent_id: int,
    period: str = Query("7d", description="Time period: 1h, 24h, 7d, 30d")
):
    """
    Get detailed analytics for a specific agent
    """
    try:
        engine = get_analytics_engine()
        data = await engine.get_agent_analytics(agent_id, period)
        return JSONResponse(content=data)
    except Exception as e:
        logger.error(f"Error getting agent analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents")
async def get_all_agents_analytics():
    """
    Get analytics for all agents
    """
    try:
        engine = get_analytics_engine()
        # Get basic agent metrics
        data = await engine._get_agent_metrics()
        return JSONResponse(content=data)
    except Exception as e:
        logger.error(f"Error getting all agents analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/context")
async def get_context_analytics():
    """
    Get context optimization analytics
    """
    try:
        engine = get_analytics_engine()
        data = await engine._get_context_metrics()
        return JSONResponse(content=data)
    except Exception as e:
        logger.error(f"Error getting context analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/learning")
async def get_learning_analytics():
    """
    Get learning and memory analytics
    """
    try:
        engine = get_analytics_engine()
        data = await engine._get_learning_metrics()
        return JSONResponse(content=data)
    except Exception as e:
        logger.error(f"Error getting learning analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system/health")
async def get_system_health():
    """
    Get real-time system health metrics
    """
    try:
        engine = get_analytics_engine()
        data = await engine._get_system_health()
        return JSONResponse(content=data)
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/realtime")
async def get_realtime_metrics():
    """
    Get real-time metrics for WebSocket updates
    """
    try:
        engine = get_analytics_engine()
        data = await engine.get_real_time_metrics()
        return JSONResponse(content=data)
    except Exception as e:
        logger.error(f"Error getting real-time metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/track/agent-execution")
async def track_agent_execution(
    agent_id: int,
    task_id: int,
    execution_time: float,
    tokens_used: int = 0,
    success: bool = True,
    error_message: Optional[str] = None,
    context_optimization_applied: bool = False,
    memory_items_created: int = 0,
    collaboration_sessions: int = 0
):
    """
    Track agent execution for analytics
    """
    try:
        engine = get_analytics_engine()
        await engine.track_agent_execution(
            agent_id=agent_id,
            task_id=task_id,
            execution_time=execution_time,
            tokens_used=tokens_used,
            success=success,
            error_message=error_message,
            context_optimization_applied=context_optimization_applied,
            memory_items_created=memory_items_created,
            collaboration_sessions=collaboration_sessions
        )
        return JSONResponse(content={"status": "success"})
    except Exception as e:
        logger.error(f"Error tracking agent execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/track/context-optimization")
async def track_context_optimization(
    original_tokens: int,
    optimized_tokens: int,
    optimization_type: str,
    pattern_used: Optional[str] = None,
    execution_time: Optional[float] = None
):
    """
    Track context optimization for analytics
    """
    try:
        engine = get_analytics_engine()
        await engine.track_context_optimization(
            original_tokens=original_tokens,
            optimized_tokens=optimized_tokens,
            optimization_type=optimization_type,
            pattern_used=pattern_used,
            execution_time=execution_time
        )
        return JSONResponse(content={"status": "success"})
    except Exception as e:
        logger.error(f"Error tracking context optimization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/track/learning-progress")
async def track_learning_progress(
    agent_id: int,
    knowledge_items: int = 0,
    memory_consolidations: int = 0,
    performance_improvement: float = 0.0,
    knowledge_transfers: int = 0
):
    """
    Track learning progress for analytics
    """
    try:
        engine = get_analytics_engine()
        await engine.track_learning_progress(
            agent_id=agent_id,
            knowledge_items=knowledge_items,
            memory_consolidations=memory_consolidations,
            performance_improvement=performance_improvement,
            knowledge_transfers=knowledge_transfers
        )
        return JSONResponse(content={"status": "success"})
    except Exception as e:
        logger.error(f"Error tracking learning progress: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/websocket/status")
async def get_websocket_status():
    """
    Get WebSocket connection status
    """
    try:
        manager = await get_websocket_manager()
        connection_count = await manager.get_connection_count()
        return JSONResponse(content={
            "active_connections": connection_count,
            "status": "running" if manager.is_running else "stopped"
        })
    except Exception as e:
        logger.error(f"Error getting WebSocket status: {e}")
        raise HTTPException(status_code=500, detail=str(e))
