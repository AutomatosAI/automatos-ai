"""
Dashboard Real-time Update Service
===================================

Bridges Redis pub/sub events with WebSocket connections for live dashboard updates.
Listens to agent, task, memory, and collaboration events.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Set
import redis.asyncio as redis
from sqlalchemy.orm import Session

from services.websocket_manager import ConnectionManager
from services.analytics_engine import AnalyticsEngine
from config import Config

logger = logging.getLogger(__name__)

class DashboardRealtimeService:
    """
    Service that listens to Redis events and broadcasts dashboard updates
    """
    
    def __init__(
        self, 
        connection_manager: ConnectionManager,
        redis_client: redis.Redis,
        db_session: Session
    ):
        self.connection_manager = connection_manager
        self.redis = redis_client
        self.db = db_session
        self.analytics_engine = AnalyticsEngine(db_session, redis_client)
        self.active = False
        self.listeners: Set[asyncio.Task] = set()
        self.update_queue = asyncio.Queue()
        self.batch_interval = 1.0  # Batch updates every second
    
    async def start(self):
        """Start the real-time update service"""
        if self.active:
            return
        
        self.active = True
        logger.info("Starting Dashboard Real-time Service")
        
        # Start Redis listeners
        self.listeners.add(asyncio.create_task(self._listen_agent_events()))
        self.listeners.add(asyncio.create_task(self._listen_task_events()))
        self.listeners.add(asyncio.create_task(self._listen_memory_events()))
        self.listeners.add(asyncio.create_task(self._listen_collaboration_events()))
        
        # Start update broadcaster
        self.listeners.add(asyncio.create_task(self._broadcast_updates()))
        
        logger.info("Dashboard Real-time Service started")
    
    async def stop(self):
        """Stop the real-time update service"""
        if not self.active:
            return
        
        self.active = False
        logger.info("Stopping Dashboard Real-time Service")
        
        # Cancel all listeners
        for task in self.listeners:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.listeners, return_exceptions=True)
        self.listeners.clear()
        
        logger.info("Dashboard Real-time Service stopped")
    
    async def _listen_agent_events(self):
        """Listen for agent events from Redis"""
        try:
            pubsub = self.redis.pubsub()
            await pubsub.subscribe("agent:*")
            
            async for message in pubsub.listen():
                if not self.active:
                    break
                
                if message["type"] == "pmessage" or message["type"] == "message":
                    try:
                        # Parse agent event
                        channel = message["channel"].decode() if isinstance(message["channel"], bytes) else message["channel"]
                        data = json.loads(message["data"]) if isinstance(message["data"], (str, bytes)) else message["data"]
                        
                        if "agent:" in channel:
                            event_type = channel.split(":")[-1]
                            await self._handle_agent_event(event_type, data)
                    
                    except Exception as e:
                        logger.error(f"Error processing agent event: {e}")
            
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Agent event listener error: {e}")
        finally:
            await pubsub.unsubscribe()
            await pubsub.close()
    
    async def _listen_task_events(self):
        """Listen for task events from Redis"""
        try:
            pubsub = self.redis.pubsub()
            await pubsub.subscribe("task:created", "task:assigned", "task:completed", "task:failed")
            
            async for message in pubsub.listen():
                if not self.active:
                    break
                
                if message["type"] == "message":
                    try:
                        channel = message["channel"].decode() if isinstance(message["channel"], bytes) else message["channel"]
                        data = json.loads(message["data"]) if isinstance(message["data"], (str, bytes)) else message["data"]
                        
                        event_type = channel.split(":")[-1]
                        await self._handle_task_event(event_type, data)
                    
                    except Exception as e:
                        logger.error(f"Error processing task event: {e}")
            
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Task event listener error: {e}")
        finally:
            await pubsub.unsubscribe()
            await pubsub.close()
    
    async def _listen_memory_events(self):
        """Listen for memory consolidation events from Redis"""
        try:
            pubsub = self.redis.pubsub()
            await pubsub.subscribe("memory:created", "memory:consolidated", "memory:promoted")
            
            async for message in pubsub.listen():
                if not self.active:
                    break
                
                if message["type"] == "message":
                    try:
                        channel = message["channel"].decode() if isinstance(message["channel"], bytes) else message["channel"]
                        data = json.loads(message["data"]) if isinstance(message["data"], (str, bytes)) else message["data"]
                        
                        event_type = channel.split(":")[-1]
                        await self._handle_memory_event(event_type, data)
                    
                    except Exception as e:
                        logger.error(f"Error processing memory event: {e}")
            
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Memory event listener error: {e}")
        finally:
            await pubsub.unsubscribe()
            await pubsub.close()
    
    async def _listen_collaboration_events(self):
        """Listen for collaboration events from Redis"""
        try:
            pubsub = self.redis.pubsub()
            await pubsub.subscribe("collaboration:started", "collaboration:message", "collaboration:completed")
            
            async for message in pubsub.listen():
                if not self.active:
                    break
                
                if message["type"] == "message":
                    try:
                        channel = message["channel"].decode() if isinstance(message["channel"], bytes) else message["channel"]
                        data = json.loads(message["data"]) if isinstance(message["data"], (str, bytes)) else message["data"]
                        
                        event_type = channel.split(":")[-1]
                        await self._handle_collaboration_event(event_type, data)
                    
                    except Exception as e:
                        logger.error(f"Error processing collaboration event: {e}")
            
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Collaboration event listener error: {e}")
        finally:
            await pubsub.unsubscribe()
            await pubsub.close()
    
    async def _handle_agent_event(self, event_type: str, data: Dict[str, Any]):
        """Process agent events and queue updates"""
        update = {
            "type": "agent_update",
            "event": event_type,
            "data": {
                "agent_id": data.get("agent_id"),
                "agent_name": data.get("agent_name"),
                "status": data.get("status"),
                "tokens_used": data.get("tokens_used", 0),
                "execution_time": data.get("execution_time", 0),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        await self.update_queue.put(update)
        
        # If critical event, get fresh metrics
        if event_type in ["execution_completed", "status_changed"]:
            await self._queue_metric_refresh("agents")
    
    async def _handle_task_event(self, event_type: str, data: Dict[str, Any]):
        """Process task events and queue updates"""
        update = {
            "type": "task_update",
            "event": event_type,
            "data": {
                "task_id": data.get("task_id"),
                "task_name": data.get("task_name"),
                "status": event_type,
                "agent_id": data.get("agent_id"),
                "subtask_count": data.get("subtask_count", 0),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        await self.update_queue.put(update)
        
        # Queue metric refresh for task metrics
        await self._queue_metric_refresh("tasks")
    
    async def _handle_memory_event(self, event_type: str, data: Dict[str, Any]):
        """Process memory events and queue updates"""
        update = {
            "type": "memory_update",
            "event": event_type,
            "data": {
                "memory_id": data.get("memory_id"),
                "memory_level": data.get("memory_level"),
                "importance": data.get("importance", 0),
                "agent_id": data.get("agent_id"),
                "timestamp": datetime.now().isoformat()
            }
        }
        
        await self.update_queue.put(update)
        
        # Queue metric refresh for memory metrics
        if event_type in ["consolidated", "promoted"]:
            await self._queue_metric_refresh("memory")
    
    async def _handle_collaboration_event(self, event_type: str, data: Dict[str, Any]):
        """Process collaboration events and queue updates"""
        update = {
            "type": "collaboration_update",
            "event": event_type,
            "data": {
                "session_id": data.get("session_id"),
                "participants": data.get("participants", []),
                "session_type": data.get("session_type"),
                "status": event_type,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        await self.update_queue.put(update)
        
        # Queue metric refresh for collaboration metrics
        await self._queue_metric_refresh("collaboration")
    
    async def _queue_metric_refresh(self, metric_type: str):
        """Queue a metric refresh update"""
        update = {
            "type": "metric_refresh",
            "metric": metric_type,
            "timestamp": datetime.now().isoformat()
        }
        await self.update_queue.put(update)
    
    async def _broadcast_updates(self):
        """Batch and broadcast updates to connected clients"""
        batch = []
        last_broadcast = datetime.now()
        
        try:
            while self.active:
                try:
                    # Collect updates for batching
                    timeout = self.batch_interval - (datetime.now() - last_broadcast).total_seconds()
                    if timeout <= 0:
                        timeout = 0.1
                    
                    update = await asyncio.wait_for(
                        self.update_queue.get(),
                        timeout=timeout
                    )
                    batch.append(update)
                    
                except asyncio.TimeoutError:
                    pass
                
                # Broadcast if we have updates and interval has passed
                if batch and (datetime.now() - last_broadcast).total_seconds() >= self.batch_interval:
                    # Process metric refreshes
                    metrics_to_refresh = set()
                    events = []
                    
                    for update in batch:
                        if update["type"] == "metric_refresh":
                            metrics_to_refresh.add(update["metric"])
                        else:
                            events.append(update)
                    
                    # Get fresh metrics if needed
                    fresh_metrics = {}
                    if metrics_to_refresh:
                        fresh_metrics = await self._get_fresh_metrics(metrics_to_refresh)
                    
                    # Broadcast combined update
                    broadcast_data = {
                        "type": "dashboard_batch_update",
                        "events": events,
                        "metrics": fresh_metrics,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    await self.connection_manager.broadcast(broadcast_data)
                    
                    # Clear batch and reset timer
                    batch.clear()
                    last_broadcast = datetime.now()
                    
                    logger.debug(f"Broadcasted {len(events)} events and {len(fresh_metrics)} metric updates")
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Broadcast error: {e}")
    
    async def _get_fresh_metrics(self, metric_types: Set[str]) -> Dict[str, Any]:
        """Get fresh metrics for specified types"""
        metrics = {}
        
        try:
            if "agents" in metric_types:
                agent_metrics = await self.analytics_engine.get_agent_metrics()
                metrics["agents"] = {
                    "total": len(agent_metrics),
                    "active": len([a for a in agent_metrics if a.status == "active"]),
                    "total_executions": sum(a.execution_count for a in agent_metrics),
                    "total_tokens": sum(a.total_tokens_used for a in agent_metrics)
                }
            
            if "tasks" in metric_types:
                task_metrics = await self.analytics_engine.get_task_metrics()
                metrics["tasks"] = {
                    "total": task_metrics.total_tasks,
                    "completed": task_metrics.completed_tasks,
                    "pending": task_metrics.pending_tasks,
                    "avg_completion_time": task_metrics.avg_completion_time
                }
            
            if "memory" in metric_types:
                memory_metrics = await self.analytics_engine.get_memory_metrics()
                metrics["memory"] = {
                    "total": memory_metrics.total_memories,
                    "consolidation_rate": memory_metrics.consolidation_rate,
                    "knowledge_nodes": memory_metrics.knowledge_nodes,
                    "growth_rate": memory_metrics.memory_growth_rate
                }
            
            if "collaboration" in metric_types:
                collab_metrics = await self.analytics_engine.get_collaboration_metrics()
                metrics["collaboration"] = {
                    "active_sessions": collab_metrics.active_sessions,
                    "total_sessions": collab_metrics.total_sessions,
                    "success_rate": collab_metrics.collaboration_success_rate
                }
            
        except Exception as e:
            logger.error(f"Error getting fresh metrics: {e}")
        
        return metrics


# Global instance
dashboard_realtime_service: Optional[DashboardRealtimeService] = None

async def initialize_realtime_service(
    connection_manager: ConnectionManager,
    redis_client: redis.Redis,
    db_session: Session
) -> DashboardRealtimeService:
    """Initialize the dashboard real-time service"""
    global dashboard_realtime_service
    
    if not dashboard_realtime_service:
        dashboard_realtime_service = DashboardRealtimeService(
            connection_manager,
            redis_client,
            db_session
        )
        await dashboard_realtime_service.start()
    
    return dashboard_realtime_service

async def shutdown_realtime_service():
    """Shutdown the dashboard real-time service"""
    global dashboard_realtime_service
    
    if dashboard_realtime_service:
        await dashboard_realtime_service.stop()
        dashboard_realtime_service = None

