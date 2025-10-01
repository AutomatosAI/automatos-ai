"""
PRD-06: Dashboard Integration Helpers
Integration functions to connect analytics with existing systems
"""

import asyncio
import logging
from typing import Optional
from fastapi import FastAPI

from services.analytics_engine import AnalyticsEngine
from services.dashboard_realtime import websocket_manager

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
        
        # Initialize Redis client for real-time updates
        try:
            import redis
            redis_client = redis.Redis(
                host="127.0.0.1",
                port=6379,
                password="redis_password_123",
                decode_responses=True
            )
            # Test connection
            redis_client.ping()
            analytics_engine.redis_client = redis_client
            websocket_manager.redis_client = redis_client
            logger.info("Redis connection established for real-time updates")
        except Exception as e:
            logger.warning(f"Redis not available for real-time updates: {e}")
        
        # Start WebSocket manager background tasks
        asyncio.create_task(websocket_manager.start_redis_listener())
        asyncio.create_task(websocket_manager.send_periodic_updates())
        
        logger.info("Dashboard services initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing dashboard services: {e}")

async def shutdown_dashboard(app: FastAPI):
    """Cleanup dashboard services on shutdown"""
    try:
        # Stop WebSocket manager
        await websocket_manager.stop_redis_listener()
        
        # Close Redis connection
        if redis_client:
            redis_client.close()
        
        logger.info("Dashboard services shutdown complete")
        
    except Exception as e:
        logger.error(f"Error shutting down dashboard services: {e}")

def register_dashboard_routes(app: FastAPI):
    """Register dashboard API routes"""
    from api.analytics_api import router as analytics_router
    from api.websocket_api import router as websocket_router
    
    app.include_router(analytics_router)
    app.include_router(websocket_router)
    
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
    collaboration_sessions: int = 0
):
    """
    Track agent execution for analytics
    Call this from your agent factory when agents execute tasks
    """
    try:
        if analytics_engine:
            await analytics_engine.track_agent_execution(
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
    except Exception as e:
        logger.error(f"Error tracking agent execution: {e}")

async def track_context_optimization(
    original_tokens: int,
    optimized_tokens: int,
    optimization_type: str,
    pattern_used: Optional[str] = None,
    execution_time: Optional[float] = None
):
    """
    Track context optimization for analytics
    Call this from your context optimizer when optimizations are applied
    """
    try:
        if analytics_engine:
            await analytics_engine.track_context_optimization(
                original_tokens=original_tokens,
                optimized_tokens=optimized_tokens,
                optimization_type=optimization_type,
                pattern_used=pattern_used,
                execution_time=execution_time
            )
    except Exception as e:
        logger.error(f"Error tracking context optimization: {e}")

async def track_learning_progress(
    agent_id: int,
    knowledge_items: int = 0,
    memory_consolidations: int = 0,
    performance_improvement: float = 0.0,
    knowledge_transfers: int = 0
):
    """
    Track learning progress for analytics
    Call this from your memory system when learning occurs
    """
    try:
        if analytics_engine:
            await analytics_engine.track_learning_progress(
                agent_id=agent_id,
                knowledge_items=knowledge_items,
                memory_consolidations=memory_consolidations,
                performance_improvement=performance_improvement,
                knowledge_transfers=knowledge_transfers
            )
    except Exception as e:
        logger.error(f"Error tracking learning progress: {e}")

def get_analytics_engine() -> Optional[AnalyticsEngine]:
    """Get the global analytics engine instance"""
    return analytics_engine