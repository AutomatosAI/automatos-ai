"""
Real Analytics API - PRD-06 Implementation
==========================================

REAL metrics from database - NO MOCK DATA.
Integrates with AnalyticsEngine for aggregating actual system metrics.
"""

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import asyncio
import json
import logging
import redis.asyncio as redis

from database.database import get_db
from services.analytics_engine import AnalyticsEngine
from services.websocket_manager import ConnectionManager
from config import Config

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/analytics", tags=["analytics"])

# Initialize services
config = Config()
redis_client = None
connection_manager = ConnectionManager()

# Initialize Redis connection
async def get_redis():
    global redis_client
    if not redis_client:
        redis_client = await redis.from_url(config.REDIS_URL)
    return redis_client

# ==================== REAL DASHBOARD ENDPOINTS ====================

@router.get("/dashboard/overview")
async def get_dashboard_overview(
    db: Session = Depends(get_db),
    time_range: Optional[str] = "24h"
) -> Dict[str, Any]:
    """
    Get complete dashboard overview with REAL metrics.
    Aggregates data from all system components.
    """
    try:
        # Parse time range
        time_delta = None
        if time_range:
            if time_range.endswith('h'):
                hours = int(time_range[:-1])
                time_delta = timedelta(hours=hours)
            elif time_range.endswith('d'):
                days = int(time_range[:-1])
                time_delta = timedelta(days=days)
        
        # Initialize analytics engine
        redis_conn = await get_redis()
        analytics = AnalyticsEngine(db, redis_conn)
        
        # Get complete overview
        overview = await analytics.get_dashboard_overview()
        
        # Publish update to WebSocket clients
        await connection_manager.broadcast({
            "type": "dashboard_overview",
            "data": overview
        })
        
        return overview
        
    except Exception as e:
        logger.error(f"Error getting dashboard overview: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/agents")
async def get_agent_analytics(
    agent_id: Optional[str] = None,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get detailed agent analytics from REAL execution data"""
    try:
        redis_conn = await get_redis()
        analytics = AnalyticsEngine(db, redis_conn)
        
        # Get agent metrics
        agent_metrics = await analytics.get_agent_metrics()
        
        if agent_id:
            # Filter for specific agent
            agent_data = [a for a in agent_metrics if a.agent_id == agent_id]
            if not agent_data:
                raise HTTPException(status_code=404, detail="Agent not found")
            
            # Get additional details for specific agent
            from sqlalchemy import text
            
            # Get recent executions
            recent_executions = db.execute(text("""
                SELECT 
                    task_id,
                    execution_time,
                    tokens_used,
                    success,
                    recorded_at
                FROM agent_performance
                WHERE agent_id = :agent_id
                ORDER BY recorded_at DESC
                LIMIT 20
            """), {"agent_id": agent_id}).fetchall()
            
            # Get collaboration history
            collaborations = db.execute(text("""
                SELECT 
                    cs.id,
                    cs.session_type,
                    cs.status,
                    cs.created_at,
                    json_array_length(cs.participants) as participant_count
                FROM collaboration_sessions cs
                WHERE cs.initiator_agent_id = :agent_id
                   OR :agent_id::text = ANY(string_to_array(cs.participants::text, ','))
                ORDER BY cs.created_at DESC
                LIMIT 10
            """), {"agent_id": agent_id}).fetchall()
            
            return {
                "agent": agent_data[0].__dict__,
                "recent_executions": [
                    {
                        "task_id": e.task_id,
                        "execution_time": e.execution_time,
                        "tokens_used": e.tokens_used,
                        "success": e.success,
                        "recorded_at": e.recorded_at.isoformat() if e.recorded_at else None
                    }
                    for e in recent_executions
                ],
                "collaborations": [
                    {
                        "session_id": c.id,
                        "type": c.session_type,
                        "status": c.status,
                        "participants": c.participant_count,
                        "created_at": c.created_at.isoformat() if c.created_at else None
                    }
                    for c in collaborations
                ]
            }
        else:
            # Return all agents with summary
            return {
                "agents": [a.__dict__ for a in agent_metrics],
                "summary": {
                    "total_agents": len(agent_metrics),
                    "active_agents": len([a for a in agent_metrics if a.status == 'active']),
                    "total_executions": sum(a.execution_count for a in agent_metrics),
                    "total_tokens": sum(a.total_tokens_used for a in agent_metrics),
                    "avg_success_rate": sum(a.success_rate for a in agent_metrics) / len(agent_metrics) if agent_metrics else 0
                }
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/context")
async def get_context_analytics(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get REAL context optimization metrics"""
    try:
        redis_conn = await get_redis()
        analytics = AnalyticsEngine(db, redis_conn)
        
        # Get context metrics
        context_metrics = await analytics.get_context_metrics()
        
        # Get recent optimizations for timeline
        from sqlalchemy import text
        recent_optimizations = db.execute(text("""
            SELECT 
                DATE(created_at) as date,
                COUNT(*) as optimization_count,
                SUM(tokens_saved) as tokens_saved,
                AVG(information_density) as avg_density
            FROM context_optimizations
            WHERE created_at >= NOW() - INTERVAL '7 days'
            GROUP BY DATE(created_at)
            ORDER BY date DESC
        """)).fetchall()
        
        timeline = [
            {
                "date": r.date.isoformat(),
                "optimizations": r.optimization_count,
                "tokens_saved": r.tokens_saved,
                "avg_density": round(r.avg_density, 3) if r.avg_density else 0
            }
            for r in recent_optimizations
        ]
        
        return {
            "metrics": context_metrics.__dict__,
            "timeline": timeline,
            "savings_summary": {
                "total_tokens_saved": context_metrics.total_tokens_saved,
                "avg_compression": f"{context_metrics.compression_ratio:.1f}x",
                "success_rate": f"{context_metrics.optimization_success_rate:.1f}%",
                "daily_avg_savings": context_metrics.total_tokens_saved / 7 if timeline else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting context analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/learning")
async def get_learning_analytics(
    days: int = 30,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get REAL learning and memory consolidation analytics"""
    try:
        redis_conn = await get_redis()
        analytics = AnalyticsEngine(db, redis_conn)
        
        # Get memory metrics
        memory_metrics = await analytics.get_memory_metrics()
        
        # Get learning progress timeline
        learning_timeline = await analytics.get_learning_progress_timeline(days)
        
        return {
            "memory_metrics": memory_metrics.__dict__,
            "learning_progress": learning_timeline,
            "knowledge_graph": {
                "nodes": memory_metrics.knowledge_nodes,
                "edges": memory_metrics.knowledge_edges,
                "density": round(memory_metrics.knowledge_edges / max(memory_metrics.knowledge_nodes, 1), 2)
            },
            "consolidation": {
                "rate": f"{memory_metrics.consolidation_rate:.1f}%",
                "working_to_short": memory_metrics.short_term_count / max(memory_metrics.working_memory_count, 1),
                "short_to_long": memory_metrics.long_term_count / max(memory_metrics.short_term_count, 1)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting learning analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/collaboration")
async def get_collaboration_analytics(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get REAL collaboration and communication analytics"""
    try:
        redis_conn = await get_redis()
        analytics = AnalyticsEngine(db, redis_conn)
        
        # Get collaboration metrics
        collab_metrics = await analytics.get_collaboration_metrics()
        
        # Get collaboration network data
        from sqlalchemy import text
        network_data = db.execute(text("""
            WITH agent_pairs AS (
                SELECT 
                    cs.initiator_agent_id as agent1,
                    unnest(string_to_array(cs.participants::text, ','))::integer as agent2,
                    COUNT(*) as collaboration_count
                FROM collaboration_sessions cs
                WHERE cs.status = 'completed'
                GROUP BY cs.initiator_agent_id, agent2
            )
            SELECT 
                a1.name as agent1_name,
                a2.name as agent2_name,
                ap.collaboration_count
            FROM agent_pairs ap
            JOIN agents a1 ON a1.id = ap.agent1
            JOIN agents a2 ON a2.id = ap.agent2
            ORDER BY ap.collaboration_count DESC
            LIMIT 50
        """)).fetchall()
        
        # Build network graph data
        nodes = set()
        edges = []
        for row in network_data:
            nodes.add(row.agent1_name)
            nodes.add(row.agent2_name)
            edges.append({
                "source": row.agent1_name,
                "target": row.agent2_name,
                "weight": row.collaboration_count
            })
        
        return {
            "metrics": collab_metrics.__dict__,
            "network": {
                "nodes": list(nodes),
                "edges": edges,
                "total_connections": len(edges)
            },
            "communication": {
                "total_messages": collab_metrics.message_exchange_count,
                "avg_messages_per_session": round(collab_metrics.message_exchange_count / max(collab_metrics.total_sessions, 1), 1),
                "active_channels": collab_metrics.active_sessions
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting collaboration analytics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/system-health")
async def get_system_health(
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get REAL system health metrics"""
    try:
        redis_conn = await get_redis()
        analytics = AnalyticsEngine(db, redis_conn)
        
        # Get system health
        health = await analytics.get_system_health()
        
        # Add queue metrics
        from sqlalchemy import text
        queue_stats = db.execute(text("""
            SELECT 
                status,
                COUNT(*) as count,
                AVG(EXTRACT(EPOCH FROM (NOW() - created_at))) as avg_wait_time
            FROM workflow_executions
            WHERE status IN ('pending', 'queued', 'running')
            GROUP BY status
        """)).fetchall()
        
        health["queue"] = {
            "pending": 0,
            "queued": 0,
            "running": 0,
            "avg_wait_time_seconds": 0
        }
        
        for stat in queue_stats:
            health["queue"][stat.status] = stat.count
            if stat.avg_wait_time:
                health["queue"]["avg_wait_time_seconds"] = round(stat.avg_wait_time, 1)
        
        return health
        
    except Exception as e:
        logger.error(f"Error getting system health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/activity-heatmap")
async def get_activity_heatmap(
    days: int = 7,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get agent activity heatmap for visualization"""
    try:
        redis_conn = await get_redis()
        analytics = AnalyticsEngine(db, redis_conn)
        
        # Get heatmap data
        heatmap = await analytics.get_agent_activity_heatmap(days)
        
        # Add day labels
        day_labels = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        heatmap["day_labels"] = day_labels
        heatmap["hour_labels"] = [f"{h:02d}:00" for h in range(24)]
        
        return heatmap
        
    except Exception as e:
        logger.error(f"Error getting activity heatmap: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trends")
async def get_trend_analysis(
    metric: str,
    period: str = "7d",
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Get trend analysis for specific metrics"""
    try:
        # Parse period
        if period.endswith('d'):
            days = int(period[:-1])
        elif period.endswith('h'):
            hours = int(period[:-1])
            days = max(1, hours // 24)
        else:
            days = 7
        
        from sqlalchemy import text
        
        if metric == "token_usage":
            # Token usage trend
            trend_data = db.execute(text("""
                SELECT 
                    DATE(recorded_at) as date,
                    SUM(tokens_used) as total_tokens,
                    AVG(tokens_used) as avg_tokens,
                    COUNT(*) as execution_count
                FROM agent_performance
                WHERE recorded_at >= NOW() - INTERVAL :days DAY
                GROUP BY DATE(recorded_at)
                ORDER BY date
            """), {"days": days}).fetchall()
            
            return {
                "metric": "token_usage",
                "data": [
                    {
                        "date": row.date.isoformat(),
                        "total": row.total_tokens,
                        "average": round(row.avg_tokens, 0),
                        "executions": row.execution_count
                    }
                    for row in trend_data
                ]
            }
            
        elif metric == "success_rate":
            # Success rate trend
            trend_data = db.execute(text("""
                SELECT 
                    DATE(recorded_at) as date,
                    COUNT(CASE WHEN success = true THEN 1 END) as successful,
                    COUNT(*) as total,
                    AVG(CASE WHEN success = true THEN 100 ELSE 0 END) as success_rate
                FROM agent_performance
                WHERE recorded_at >= NOW() - INTERVAL :days DAY
                GROUP BY DATE(recorded_at)
                ORDER BY date
            """), {"days": days}).fetchall()
            
            return {
                "metric": "success_rate",
                "data": [
                    {
                        "date": row.date.isoformat(),
                        "successful": row.successful,
                        "total": row.total,
                        "rate": round(row.success_rate, 1)
                    }
                    for row in trend_data
                ]
            }
            
        elif metric == "memory_growth":
            # Memory growth trend
            trend_data = db.execute(text("""
                SELECT 
                    DATE(created_at) as date,
                    COUNT(*) as memories_created,
                    COUNT(CASE WHEN memory_level = 'working' THEN 1 END) as working,
                    COUNT(CASE WHEN memory_level = 'short_term' THEN 1 END) as short_term,
                    COUNT(CASE WHEN memory_level = 'long_term' THEN 1 END) as long_term
                FROM memory_items
                WHERE created_at >= NOW() - INTERVAL :days DAY
                GROUP BY DATE(created_at)
                ORDER BY date
            """), {"days": days}).fetchall()
            
            cumulative_total = 0
            data = []
            for row in trend_data:
                cumulative_total += row.memories_created
                data.append({
                    "date": row.date.isoformat(),
                    "created": row.memories_created,
                    "cumulative": cumulative_total,
                    "working": row.working,
                    "short_term": row.short_term,
                    "long_term": row.long_term
                })
            
            return {
                "metric": "memory_growth",
                "data": data
            }
            
        else:
            raise HTTPException(status_code=400, detail=f"Unknown metric: {metric}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting trend analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== WEBSOCKET ENDPOINT ====================

@router.websocket("/ws/dashboard")
async def dashboard_websocket(
    websocket: WebSocket,
    db: Session = Depends(get_db)
):
    """
    WebSocket endpoint for real-time dashboard updates.
    Subscribes to Redis pub/sub for agent events.
    """
    await connection_manager.connect(websocket, client_id=f"dashboard_{datetime.now().timestamp()}")
    
    try:
        # Send initial dashboard state
        redis_conn = await get_redis()
        analytics = AnalyticsEngine(db, redis_conn)
        initial_data = await analytics.get_dashboard_overview()
        
        await connection_manager.send_personal_message({
            "type": "initial_state",
            "data": initial_data
        }, websocket)
        
        # Create Redis subscription for real-time updates
        pubsub = redis_conn.pubsub()
        await pubsub.subscribe("dashboard_updates", "agent_events", "task_events", "memory_events")
        
        # Background task to listen for Redis messages
        async def redis_listener():
            async for message in pubsub.listen():
                if message["type"] == "message":
                    try:
                        data = json.loads(message["data"])
                        await connection_manager.send_personal_message(data, websocket)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON in Redis message: {message['data']}")
                    except Exception as e:
                        logger.error(f"Error processing Redis message: {e}")
        
        # Start Redis listener
        redis_task = asyncio.create_task(redis_listener())
        
        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for client message
                message = await websocket.receive_text()
                data = json.loads(message)
                
                # Handle different message types
                if data.get("type") == "ping":
                    await connection_manager.send_personal_message({"type": "pong"}, websocket)
                    
                elif data.get("type") == "refresh":
                    # Send updated overview
                    overview = await analytics.get_dashboard_overview()
                    await connection_manager.send_personal_message({
                        "type": "dashboard_update",
                        "data": overview
                    }, websocket)
                    
                elif data.get("type") == "subscribe":
                    # Subscribe to specific metric updates
                    metric = data.get("metric")
                    logger.info(f"Dashboard subscribed to metric: {metric}")
                    
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await connection_manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid JSON"
                }, websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
    
    finally:
        # Clean up
        redis_task.cancel()
        await pubsub.unsubscribe()
        await pubsub.close()
        connection_manager.disconnect(websocket)
        logger.info("Dashboard WebSocket disconnected")

# ==================== ALERT ENDPOINTS ====================

@router.post("/alerts/configure")
async def configure_alert(
    metric_type: str,
    threshold: float,
    comparison: str,
    channel: str = "dashboard",
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Configure a metric alert"""
    try:
        from sqlalchemy import text
        
        # Insert alert configuration
        result = db.execute(text("""
            INSERT INTO alert_configs (metric_type, threshold_value, comparison_operator, alert_channel)
            VALUES (:metric, :threshold, :comparison, :channel)
            RETURNING id
        """), {
            "metric": metric_type,
            "threshold": threshold,
            "comparison": comparison,
            "channel": channel
        })
        
        db.commit()
        alert_id = result.scalar()
        
        return {
            "alert_id": alert_id,
            "status": "configured",
            "metric_type": metric_type,
            "threshold": threshold,
            "comparison": comparison
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error configuring alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts/active")
async def get_active_alerts(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """Get all active alerts"""
    try:
        from sqlalchemy import text
        
        alerts = db.execute(text("""
            SELECT 
                id,
                metric_type,
                threshold_value,
                comparison_operator,
                alert_channel,
                created_at
            FROM alert_configs
            WHERE is_active = true
            ORDER BY created_at DESC
        """)).fetchall()
        
        return [
            {
                "id": a.id,
                "metric_type": a.metric_type,
                "threshold": a.threshold_value,
                "comparison": a.comparison_operator,
                "channel": a.alert_channel,
                "created_at": a.created_at.isoformat() if a.created_at else None
            }
            for a in alerts
        ]
        
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

