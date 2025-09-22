"""
Dashboard Integration Module
============================

Integrates the real-time dashboard with the FastAPI application.
Initializes WebSocket connections, Redis subscriptions, and analytics engine.
"""

from fastapi import FastAPI
from sqlalchemy.orm import Session
import redis.asyncio as redis
import asyncio
import logging

from api.analytics_real import router as analytics_router
from services.websocket_manager import ConnectionManager
from services.dashboard_realtime import initialize_realtime_service, shutdown_realtime_service
from database.database import SessionLocal
from config import Config

logger = logging.getLogger(__name__)

# Global instances
redis_client = None
connection_manager = ConnectionManager()
db_session = None

async def startup_dashboard(app: FastAPI):
    """
    Initialize dashboard services on application startup
    """
    global redis_client, db_session
    
    try:
        # Initialize configuration
        config = Config()
        
        # Connect to Redis
        logger.info("Connecting to Redis for dashboard...")
        redis_client = await redis.from_url(config.REDIS_URL)
        await redis_client.ping()
        logger.info("Redis connection established")
        
        # Create database session
        db_session = SessionLocal()
        
        # Initialize real-time service
        logger.info("Starting dashboard real-time service...")
        await initialize_realtime_service(
            connection_manager,
            redis_client,
            db_session
        )
        logger.info("Dashboard real-time service started")
        
        # Store instances in app state for access in endpoints
        app.state.redis_client = redis_client
        app.state.connection_manager = connection_manager
        app.state.dashboard_db = db_session
        
        logger.info("Dashboard services initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize dashboard services: {e}")
        raise

async def shutdown_dashboard(app: FastAPI):
    """
    Cleanup dashboard services on application shutdown
    """
    global redis_client, db_session
    
    try:
        logger.info("Shutting down dashboard services...")
        
        # Stop real-time service
        await shutdown_realtime_service()
        
        # Close Redis connection
        if redis_client:
            await redis_client.close()
            redis_client = None
        
        # Close database session
        if db_session:
            db_session.close()
            db_session = None
        
        logger.info("Dashboard services shut down successfully")
        
    except Exception as e:
        logger.error(f"Error during dashboard shutdown: {e}")

def register_dashboard_routes(app: FastAPI):
    """
    Register dashboard API routes
    """
    # Include analytics router
    app.include_router(analytics_router)
    
    # Add dashboard-specific middleware if needed
    @app.middleware("http")
    async def dashboard_middleware(request, call_next):
        # Add any dashboard-specific request processing
        response = await call_next(request)
        return response
    
    logger.info("Dashboard routes registered")

async def publish_agent_event(event_type: str, agent_id: str, data: dict):
    """
    Publish agent event to Redis for dashboard updates
    """
    if not redis_client:
        return
    
    try:
        import json
        from datetime import datetime
        
        message = json.dumps({
            "event_type": event_type,
            "agent_id": agent_id,
            "data": data,
            "timestamp": datetime.now().isoformat()
        })
        
        await redis_client.publish(f"agent:{event_type}", message)
        
    except Exception as e:
        logger.error(f"Error publishing agent event: {e}")

async def publish_task_event(event_type: str, task_id: str, data: dict):
    """
    Publish task event to Redis for dashboard updates
    """
    if not redis_client:
        return
    
    try:
        import json
        from datetime import datetime
        
        message = json.dumps({
            "event_type": event_type,
            "task_id": task_id,
            "data": data,
            "timestamp": datetime.now().isoformat()
        })
        
        await redis_client.publish(f"task:{event_type}", message)
        
    except Exception as e:
        logger.error(f"Error publishing task event: {e}")

async def publish_memory_event(event_type: str, memory_id: str, data: dict):
    """
    Publish memory event to Redis for dashboard updates
    """
    if not redis_client:
        return
    
    try:
        import json
        from datetime import datetime
        
        message = json.dumps({
            "event_type": event_type,
            "memory_id": memory_id,
            "data": data,
            "timestamp": datetime.now().isoformat()
        })
        
        await redis_client.publish(f"memory:{event_type}", message)
        
    except Exception as e:
        logger.error(f"Error publishing memory event: {e}")

async def publish_collaboration_event(event_type: str, session_id: str, data: dict):
    """
    Publish collaboration event to Redis for dashboard updates
    """
    if not redis_client:
        return
    
    try:
        import json
        from datetime import datetime
        
        message = json.dumps({
            "event_type": event_type,
            "session_id": session_id,
            "data": data,
            "timestamp": datetime.now().isoformat()
        })
        
        await redis_client.publish(f"collaboration:{event_type}", message)
        
    except Exception as e:
        logger.error(f"Error publishing collaboration event: {e}")

# Helper functions for tracking metrics

async def track_agent_execution(
    agent_id: str,
    task_id: str,
    tokens_used: int,
    execution_time: float,
    success: bool
):
    """
    Track agent execution metrics in database and publish event
    """
    if not db_session:
        return
    
    try:
        from sqlalchemy import text
        from datetime import datetime
        
        # Insert performance record
        db_session.execute(text("""
            INSERT INTO agent_performance 
            (agent_id, task_id, tokens_used, execution_time, success, recorded_at)
            VALUES (:agent_id, :task_id, :tokens_used, :execution_time, :success, :recorded_at)
        """), {
            "agent_id": agent_id,
            "task_id": task_id,
            "tokens_used": tokens_used,
            "execution_time": execution_time,
            "success": success,
            "recorded_at": datetime.now()
        })
        
        # Update agent totals
        db_session.execute(text("""
            UPDATE agents
            SET execution_count = execution_count + 1,
                total_tokens_used = total_tokens_used + :tokens,
                success_rate = (
                    SELECT AVG(CASE WHEN success THEN 100 ELSE 0 END)
                    FROM agent_performance
                    WHERE agent_id = :agent_id
                ),
                avg_execution_time = (
                    SELECT AVG(execution_time)
                    FROM agent_performance
                    WHERE agent_id = :agent_id
                )
            WHERE id = :agent_id
        """), {
            "agent_id": agent_id,
            "tokens": tokens_used
        })
        
        db_session.commit()
        
        # Publish event
        await publish_agent_event("execution_completed", agent_id, {
            "task_id": task_id,
            "tokens_used": tokens_used,
            "execution_time": execution_time,
            "success": success
        })
        
    except Exception as e:
        logger.error(f"Error tracking agent execution: {e}")
        db_session.rollback()

async def track_context_optimization(
    agent_id: str,
    original_tokens: int,
    optimized_tokens: int,
    information_density: float,
    pattern_type: str = None
):
    """
    Track context optimization metrics
    """
    if not db_session:
        return
    
    try:
        from sqlalchemy import text
        from datetime import datetime
        
        tokens_saved = original_tokens - optimized_tokens
        
        # Insert optimization record
        db_session.execute(text("""
            INSERT INTO context_optimizations
            (agent_id, original_tokens, optimized_tokens, tokens_saved, 
             information_density, pattern_type, created_at)
            VALUES (:agent_id, :original, :optimized, :saved, 
                    :density, :pattern, :created_at)
        """), {
            "agent_id": agent_id,
            "original": original_tokens,
            "optimized": optimized_tokens,
            "saved": tokens_saved,
            "density": information_density,
            "pattern": pattern_type,
            "created_at": datetime.now()
        })
        
        db_session.commit()
        
        # Publish to dashboard
        await connection_manager.broadcast({
            "type": "context_optimization",
            "data": {
                "agent_id": agent_id,
                "tokens_saved": tokens_saved,
                "compression_ratio": original_tokens / optimized_tokens if optimized_tokens > 0 else 1,
                "information_density": information_density
            }
        })
        
    except Exception as e:
        logger.error(f"Error tracking context optimization: {e}")
        db_session.rollback()

async def track_memory_consolidation(
    memory_id: str,
    from_level: str,
    to_level: str,
    importance: float
):
    """
    Track memory consolidation events
    """
    if not db_session:
        return
    
    try:
        from sqlalchemy import text
        from datetime import datetime
        
        # Update memory level
        db_session.execute(text("""
            UPDATE memory_items
            SET memory_level = :to_level,
                consolidation_count = consolidation_count + 1,
                last_accessed = :now
            WHERE id = :memory_id
        """), {
            "memory_id": memory_id,
            "to_level": to_level,
            "now": datetime.now()
        })
        
        db_session.commit()
        
        # Publish event
        event_type = "promoted" if to_level == "long_term" else "consolidated"
        await publish_memory_event(event_type, memory_id, {
            "from_level": from_level,
            "to_level": to_level,
            "importance": importance
        })
        
    except Exception as e:
        logger.error(f"Error tracking memory consolidation: {e}")
        db_session.rollback()

