"""
PRD-06: Real-time Dashboard Updates
WebSocket handler for live dashboard updates
"""

import asyncio
import json
import logging
from typing import Set, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
import redis.asyncio as redis

logger = logging.getLogger(__name__)

class DashboardWebSocketManager:
    """
    Manages WebSocket connections for real-time dashboard updates
    """
    
    def __init__(self, redis_client: redis.Redis = None):
        self.active_connections: Set[WebSocket] = set()
        self.redis_client = redis_client
        self.pubsub = None
        self.is_running = False
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.add(websocket)
        logger.info(f"Dashboard WebSocket connected. Total connections: {len(self.active_connections)}")
        
        # Send initial dashboard state
        try:
            from services.analytics_engine import AnalyticsEngine
            analytics = AnalyticsEngine()
            initial_data = await analytics.get_dashboard_overview()
            
            await websocket.send_json({
                "type": "initial_state",
                "data": initial_data
            })
        except Exception as e:
            logger.error(f"Error sending initial state: {e}")
            await websocket.send_json({
                "type": "error",
                "message": "Failed to load initial dashboard data"
            })
    
    async def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        self.active_connections.discard(websocket)
        logger.info(f"Dashboard WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast_update(self, update: Dict[str, Any]):
        """Broadcast an update to all connected clients"""
        if not self.active_connections:
            return
        
        message = json.dumps(update)
        disconnected = set()
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send update to WebSocket: {e}")
                disconnected.add(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.active_connections.discard(connection)
    
    async def start_redis_listener(self):
        """Start listening to Redis pub/sub for updates"""
        if not self.redis_client:
            logger.warning("No Redis client available for real-time updates")
            return
        
        try:
            self.pubsub = self.redis_client.pubsub()
            await self.pubsub.subscribe("dashboard_updates")
            self.is_running = True
            
            logger.info("Started Redis listener for dashboard updates")
            
            async for message in self.pubsub.listen():
                if not self.is_running:
                    break
                
                if message["type"] == "message":
                    try:
                        update_data = json.loads(message["data"])
                        await self.broadcast_update(update_data)
                    except Exception as e:
                        logger.error(f"Error processing Redis message: {e}")
        
        except Exception as e:
            logger.error(f"Error in Redis listener: {e}")
        finally:
            if self.pubsub:
                await self.pubsub.unsubscribe("dashboard_updates")
                await self.pubsub.close()
    
    async def stop_redis_listener(self):
        """Stop the Redis listener"""
        self.is_running = False
        if self.pubsub:
            await self.pubsub.close()
        logger.info("Stopped Redis listener")
    
    async def send_periodic_updates(self):
        """Send periodic updates to all connected clients"""
        while self.is_running and self.active_connections:
            try:
                from services.analytics_engine import AnalyticsEngine
                analytics = AnalyticsEngine()
                metrics = await analytics.get_real_time_metrics()
                
                await self.broadcast_update({
                    "type": "periodic_update",
                    "data": metrics
                })
                
                # Wait 30 seconds before next update
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in periodic updates: {e}")
                await asyncio.sleep(30)
    
    async def get_connection_count(self) -> int:
        """Get the number of active connections"""
        return len(self.active_connections)

# Global WebSocket manager instance
websocket_manager = DashboardWebSocketManager()

async def get_websocket_manager() -> DashboardWebSocketManager:
    """Get the global WebSocket manager instance"""
    return websocket_manager