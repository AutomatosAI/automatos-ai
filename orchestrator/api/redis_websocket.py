"""
Redis-backed WebSocket endpoint for real-time workflow updates
This decouples messaging from the execution loop, eliminating buffering issues.
"""
import json
import logging
import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from typing import Dict, Any

logger = logging.getLogger(__name__)

router = APIRouter()


@router.websocket("/ws/{workflow_id}/{execution_id}")
async def websocket_endpoint(websocket: WebSocket, workflow_id: int, execution_id: int):
    """
    WebSocket endpoint that subscribes to Redis channel and forwards messages to client
    
    This endpoint:
    1. Accepts WebSocket connection
    2. Subscribes to Redis channel for this specific workflow execution
    3. Forwards all Redis messages to the WebSocket client in real-time
    4. Handles disconnections and cleanup gracefully
    """
    await websocket.accept()
    
    channel = f"workflow:{workflow_id}:execution:{execution_id}"
    logger.info(f"🔌 WebSocket connected for {channel}")
    
    redis_async = None
    pubsub = None
    
    try:
        from core.redis_client import get_redis_client
        
        redis_client = get_redis_client()
        if not redis_client:
            logger.error("Redis client not initialized")
            await websocket.send_json({
                "type": "error",
                "data": {"message": "Redis not available"}
            })
            await websocket.close()
            return
        
        # Get async Redis pubsub for non-blocking real-time streaming
        redis_async, pubsub = await redis_client.get_async_pubsub(channel)
        logger.info(f"📡 Async pubsub subscribed to Redis channel: {channel}")
        
        # Send connection confirmation
        await websocket.send_json({
            "type": "connected",
            "data": {
                "channel": channel,
                "workflow_id": workflow_id,
                "execution_id": execution_id
            }
        })
        
        # Listen for messages from Redis and forward to WebSocket (ASYNC, NON-BLOCKING)
        while True:
            try:
                # Get next message from Redis (async, waits for new messages)
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=30.0)
                
                if message and message['type'] == 'message':
                    # Parse and forward the message IMMEDIATELY
                    try:
                        data = json.loads(message['data'])
                        logger.info(f"📨 LIVE: Forwarding Redis message to WebSocket: {data.get('type', 'unknown')}")
                        await websocket.send_json(data)
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse Redis message: {e}")
                
                # No sleep needed - async pubsub waits for messages naturally
                    
            except WebSocketDisconnect:
                logger.info(f"🔌 Client disconnected from {channel}")
                break
            except Exception as e:
                logger.error(f"Error in WebSocket message loop: {e}", exc_info=True)
                break
            
    except Exception as e:
        logger.error(f"❌ WebSocket error for {channel}: {e}", exc_info=True)
    finally:
        # Clean up async Redis connection
        if pubsub:
            try:
                await pubsub.unsubscribe(channel)
                await pubsub.close()
                logger.info(f"📡 Unsubscribed from Redis channel: {channel}")
            except Exception as e:
                logger.error(f"Error closing pubsub: {e}")
        
        if redis_async:
            try:
                await redis_async.close()
            except Exception as e:
                logger.error(f"Error closing Redis connection: {e}")
        
        # Close WebSocket
        try:
            await websocket.close()
            logger.info(f"🔌 WebSocket closed for {channel}")
        except:
            pass


@router.get("/ws/health")
async def websocket_health():
    """Health check for WebSocket/Redis integration"""
    from core.redis_client import get_redis_client
    
    redis_client = get_redis_client()
    if not redis_client:
        return {"status": "error", "message": "Redis client not initialized"}
    
    if redis_client.test_connection():
        return {"status": "healthy", "redis": "connected"}
    else:
        return {"status": "unhealthy", "redis": "disconnected"}

