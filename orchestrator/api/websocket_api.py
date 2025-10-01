"""
PRD-06: WebSocket API for Real-time Dashboard Updates
"""

import logging
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from services.dashboard_realtime import get_websocket_manager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ws", tags=["websocket"])

@router.websocket("/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    """
    WebSocket endpoint for real-time dashboard updates
    """
    manager = await get_websocket_manager()
    
    try:
        await manager.connect(websocket)
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for messages from client (ping/pong)
                data = await websocket.receive_text()
                
                # Echo back for ping/pong
                if data == "ping":
                    await websocket.send_text("pong")
                elif data == "get_status":
                    status = {
                        "type": "status",
                        "active_connections": await manager.get_connection_count(),
                        "is_running": manager.is_running
                    }
                    await websocket.send_json(status)
                
            except WebSocketDisconnect:
                logger.info("Dashboard WebSocket disconnected")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket: {e}")
                break
                
    except Exception as e:
        logger.error(f"Error setting up WebSocket: {e}")
    finally:
        await manager.disconnect(websocket)
