"""
PRD-06: Dashboard Integration Helpers
Integration functions to connect analytics with existing systems

NOTE: WebSocket removed - using AI SDK SSE streaming instead
"""

import asyncio
import logging
from typing import Optional
from fastapi import FastAPI

from core.services.analytics_engine import AnalyticsEngine

logger = logging.getLogger(__name__)

# Global instances
analytics_engine = None
redis_client = None

async def startup_dashboard(app: FastAPI):
    """Initialize dashboard services on startup"""
    global analytics_engine, redis_client
    
    try:
        # Initialize analytics engine
        analytics_engine = AnalyticsEngine()
        
        # Use centralized Redis client for real-time updates
        from core.redis.client import get_redis_client
        redis_client = get_redis_client()
        if redis_client:
            analytics_engine.redis_client = redis_client
            logger.info("Redis connection established for real-time updates")
        else:
            logger.warning("Redis client not initialized for real-time updates")
        
        logger.info("Dashboard services initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing dashboard services: {e}")

async def shutdown_dashboard(app: FastAPI):
    """Cleanup dashboard services on shutdown"""
    try:
        # Close Redis connection
        if redis_client:
            try:
                redis_client.close()
            except AttributeError:
                pass  # Redis client may not have close method
        
        logger.info("Dashboard services shutdown complete")
        
    except Exception as e:
        logger.error(f"Error shutting down dashboard services: {e}")

def register_dashboard_routes(app: FastAPI):
    """Register dashboard API routes"""
    from api.analytics_api import router as analytics_router
    
    app.include_router(analytics_router)
    # WebSocket router removed - using AI SDK SSE streaming
    
    logger.info("Dashboard API routes registered")

async def track_agent_execution(
    agent_id: int,
    task_id: int,
    execution_time: float,
    tokens_used: int = 0,
    success: bool = True,
    error_message: Optional[str] = None,
    context_optimization_applied: bool = False,
    memory_items_created: int = 0,
    tools_used: int = 0
):
    """Track agent execution for analytics"""
    if analytics_engine:
        try:
            await analytics_engine.track_agent_execution(
                agent_id=agent_id,
                task_id=task_id,
                execution_time=execution_time,
                tokens_used=tokens_used,
                success=success,
                error_message=error_message,
                context_optimization_applied=context_optimization_applied,
                memory_items_created=memory_items_created,
                tools_used=tools_used
            )
        except Exception as e:
            logger.error(f"Error tracking agent execution: {e}")

async def track_workflow_execution(
    workflow_id: int,
    execution_id: int,
    execution_time: float,
    stages_completed: int = 0,
    total_tokens: int = 0,
    success: bool = True,
    error_message: Optional[str] = None
):
    """Track workflow execution for analytics"""
    if analytics_engine:
        try:
            await analytics_engine.track_workflow_execution(
                workflow_id=workflow_id,
                execution_id=execution_id,
                execution_time=execution_time,
                stages_completed=stages_completed,
                total_tokens=total_tokens,
                success=success,
                error_message=error_message
            )
        except Exception as e:
            logger.error(f"Error tracking workflow execution: {e}")

async def broadcast_dashboard_update(update_type: str, data: dict):
    """Broadcast update to dashboard - now via Redis pub/sub only"""
    if redis_client:
        try:
            import json
            message = json.dumps({"type": update_type, "data": data})
            await redis_client.publish("dashboard_updates", message)
        except Exception as e:
            logger.error(f"Error broadcasting dashboard update: {e}")

def get_analytics_engine() -> Optional[AnalyticsEngine]:
    """Get the global analytics engine instance"""
    return analytics_engine
