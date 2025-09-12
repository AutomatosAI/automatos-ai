
"""
WebSocket Manager for Real-time Updates
======================================

WebSocket connection management and real-time event broadcasting.
"""

import json
import logging
from typing import List, Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
from datetime import datetime

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_info: Dict[WebSocket, Dict[str, Any]] = {}

    async def connect(self, websocket: WebSocket, client_id: str = None):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        self.connection_info[websocket] = {
            "client_id": client_id,
            "connected_at": datetime.now(),
            "last_ping": datetime.now()
        }
        logger.info(f"WebSocket connection established. Client ID: {client_id}")

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            client_info = self.connection_info.pop(websocket, {})
            client_id = client_info.get("client_id", "unknown")
            logger.info(f"WebSocket connection closed. Client ID: {client_id}")

    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send a message to a specific WebSocket connection"""
        try:
            message_with_timestamp = {
                **message,
                "timestamp": datetime.now().isoformat()
            }
            await websocket.send_text(json.dumps(message_with_timestamp))
        except Exception as e:
            logger.error(f"Error sending personal message: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected clients"""
        if not self.active_connections:
            return

        message_with_timestamp = {
            **message,
            "timestamp": datetime.now().isoformat()
        }
        
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(json.dumps(message_with_timestamp))
            except Exception as e:
                logger.error(f"Error broadcasting to connection: {e}")
                disconnected.append(connection)

        # Clean up disconnected connections
        for connection in disconnected:
            self.disconnect(connection)

    async def send_to_client(self, client_id: str, message: Dict[str, Any]):
        """Send a message to a specific client by ID"""
        target_connection = None
        for connection, info in self.connection_info.items():
            if info.get("client_id") == client_id:
                target_connection = connection
                break

        if target_connection:
            await self.send_personal_message(message, target_connection)
        else:
            logger.warning(f"Client {client_id} not found for targeted message")

    def get_connection_count(self) -> int:
        """Get the number of active connections"""
        return len(self.active_connections)

    def get_connection_info(self) -> List[Dict[str, Any]]:
        """Get information about all active connections"""
        return [
            {
                "client_id": info.get("client_id"),
                "connected_at": info.get("connected_at").isoformat() if info.get("connected_at") else None,
                "last_ping": info.get("last_ping").isoformat() if info.get("last_ping") else None
            }
            for info in self.connection_info.values()
        ]

    async def ping_all(self):
        """Send ping to all connections to keep them alive"""
        ping_message = {
            "type": "ping",
            "data": {"message": "ping"}
        }
        await self.broadcast(ping_message)

    async def update_last_ping(self, websocket: WebSocket):
        """Update last ping time for a connection"""
        if websocket in self.connection_info:
            self.connection_info[websocket]["last_ping"] = datetime.now()

# Global connection manager instance
manager = ConnectionManager()

# WebSocket event types
class WebSocketEventType:
    # Agent events
    AGENT_CREATED = "agent_created"
    AGENT_UPDATED = "agent_updated"
    AGENT_DELETED = "agent_deleted"
    AGENT_STATUS_CHANGED = "agent_status_changed"
    
    # Workflow events
    WORKFLOW_CREATED = "workflow_created"
    WORKFLOW_UPDATED = "workflow_updated"
    WORKFLOW_DELETED = "workflow_deleted"
    EXECUTION_STARTED = "execution_started"
    EXECUTION_STATUS = "execution_status"
    EXECUTION_COMPLETED = "execution_completed"
    EXECUTION_FAILED = "execution_failed"
    
    # Document events
    DOCUMENT_UPLOADED = "document_uploaded"
    DOCUMENT_PROCESSED = "document_processed"
    DOCUMENT_FAILED = "document_failed"
    DOCUMENT_DELETED = "document_deleted"
    
    # System events
    SYSTEM_CONFIG_UPDATED = "system_config_updated"
    RAG_CONFIG_UPDATED = "rag_config_updated"
    SYSTEM_HEALTH_ALERT = "system_health_alert"
    
    # General events
    PING = "ping"
    PONG = "pong"
    ERROR = "error"
    NOTIFICATION = "notification"

async def broadcast_agent_event(event_type: str, agent_data: Dict[str, Any]):
    """Broadcast agent-related events"""
    await manager.broadcast({
        "type": event_type,
        "category": "agent",
        "data": agent_data
    })

async def broadcast_workflow_event(event_type: str, workflow_data: Dict[str, Any]):
    """Broadcast workflow-related events"""
    await manager.broadcast({
        "type": event_type,
        "category": "workflow",
        "data": workflow_data
    })

async def broadcast_document_event(event_type: str, document_data: Dict[str, Any]):
    """Broadcast document-related events"""
    await manager.broadcast({
        "type": event_type,
        "category": "document",
        "data": document_data
    })

async def broadcast_system_event(event_type: str, system_data: Dict[str, Any]):
    """Broadcast system-related events"""
    await manager.broadcast({
        "type": event_type,
        "category": "system",
        "data": system_data
    })

async def send_notification(message: str, level: str = "info", client_id: str = None):
    """Send a notification message"""
    notification = {
        "type": WebSocketEventType.NOTIFICATION,
        "data": {
            "message": message,
            "level": level,  # info, warning, error, success
            "timestamp": datetime.now().isoformat()
        }
    }
    
    if client_id:
        await manager.send_to_client(client_id, notification)
    else:
        await manager.broadcast(notification)
