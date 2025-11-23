"""
Workflow Execution SSE Streaming Service
=========================================

Provides real-time workflow execution updates via Server-Sent Events.
Replaces Redis pubsub + WebSocket architecture from PRD-28.

Based on successful PRD-27 chat SSE implementation.
"""

from typing import AsyncGenerator, Dict, Any, Optional, List
from datetime import datetime
import asyncio
import json
import logging

logger = logging.getLogger(__name__)


class WorkflowStreamManager:
    """
    Manages SSE streams for workflow executions.
    
    Features:
    - Multiple concurrent streams per workflow execution
    - In-memory event queues (no Redis dependency)
    - Automatic cleanup on client disconnect
    - Heartbeat to keep connections alive
    - Zero polling required
    """
    
    def __init__(self):
        # {execution_id: [queue1, queue2, ...]}
        self._streams: Dict[int, List[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()
    
    async def create_stream(self, execution_id: int) -> asyncio.Queue:
        """Create a new SSE stream for an execution"""
        async with self._lock:
            queue = asyncio.Queue(maxsize=1000)  # Large queue for busy workflows
            
            if execution_id not in self._streams:
                self._streams[execution_id] = []
            
            self._streams[execution_id].append(queue)
            logger.info(f"📡 SSE stream created for execution {execution_id} (total streams: {len(self._streams[execution_id])})")
            return queue
    
    async def remove_stream(self, execution_id: int, queue: asyncio.Queue):
        """Remove a stream when client disconnects"""
        async with self._lock:
            if execution_id in self._streams:
                try:
                    self._streams[execution_id].remove(queue)
                    remaining = len(self._streams[execution_id])
                    
                    if remaining == 0:
                        del self._streams[execution_id]
                        logger.info(f"📡 Last SSE stream removed for execution {execution_id}")
                    else:
                        logger.info(f"📡 SSE stream removed for execution {execution_id} ({remaining} remaining)")
                except ValueError:
                    pass
    
    async def broadcast_event(
        self,
        execution_id: int,
        event_type: str,
        data: Dict[str, Any]
    ):
        """
        Broadcast event to all streams for this execution.
        Non-blocking - drops events if queue is full.
        """
        async with self._lock:
            if execution_id not in self._streams:
                # No active streams, skip broadcast
                return
            
            event = {
                "type": event_type,
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
            
            # Send to all active streams
            dead_queues = []
            success_count = 0
            
            for queue in self._streams[execution_id]:
                try:
                    # Non-blocking put - if queue full, skip this client
                    queue.put_nowait(event)
                    success_count += 1
                except asyncio.QueueFull:
                    logger.warning(f"⚠️ Queue full for execution {execution_id}, dropping event")
                    dead_queues.append(queue)
                except Exception as e:
                    logger.error(f"❌ Error broadcasting to queue: {e}")
                    dead_queues.append(queue)
            
            # Clean up dead queues
            for queue in dead_queues:
                try:
                    self._streams[execution_id].remove(queue)
                except ValueError:
                    pass
            
            if success_count > 0:
                logger.debug(f"📤 Broadcast {event_type} to {success_count} stream(s) for execution {execution_id}")
    
    def has_active_streams(self, execution_id: int) -> bool:
        """Check if execution has any active streams"""
        return execution_id in self._streams and len(self._streams[execution_id]) > 0
    
    def get_active_stream_count(self, execution_id: int) -> int:
        """Get number of active streams for execution"""
        return len(self._streams.get(execution_id, []))
    
    async def broadcast_log(
        self,
        execution_id: int,
        level: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Broadcast a structured log event.
        
        Args:
            execution_id: Workflow execution ID
            level: Log level (INFO, DEBUG, WARNING, ERROR)
            message: Short summary message
            details: Full details dictionary (not truncated)
        """
        await self.broadcast_event(
            execution_id=execution_id,
            event_type="execution_log",
            data={
                "level": level,
                "message": message,
                "details": details or {},
                "truncated": False  # Never truncate!
            }
        )


# Global singleton instance
_stream_manager: Optional[WorkflowStreamManager] = None


def get_stream_manager() -> WorkflowStreamManager:
    """Get or create global stream manager singleton"""
    global _stream_manager
    if _stream_manager is None:
        _stream_manager = WorkflowStreamManager()
    return _stream_manager


async def stream_workflow_execution(
    execution_id: int
) -> AsyncGenerator[str, None]:
    """
    SSE generator for workflow execution updates.
    
    Usage in FastAPI:
        return StreamingResponse(
            stream_workflow_execution(execution_id),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    Args:
        execution_id: Workflow execution ID to stream
        
    Yields:
        SSE formatted events: "data: {...}\n\n"
    """
    manager = get_stream_manager()
    queue = await manager.create_stream(execution_id)
    
    try:
        logger.info(f"🚀 Starting SSE stream for execution {execution_id}")
        
        # Send initial connection event
        initial_event = {
            "type": "connected",
            "execution_id": execution_id,
            "timestamp": datetime.now().isoformat()
        }
        yield f"data: {json.dumps(initial_event)}\n\n"
        
        # Stream events until client disconnects
        heartbeat_counter = 0
        while True:
            try:
                # Wait for next event with timeout for heartbeat
                event = await asyncio.wait_for(queue.get(), timeout=15.0)
                
                # Send event as SSE
                yield f"data: {json.dumps(event)}\n\n"
                
            except asyncio.TimeoutError:
                # Send heartbeat to keep connection alive
                heartbeat_counter += 1
                heartbeat = f": heartbeat {heartbeat_counter}\n\n"
                yield heartbeat
                continue
                
    except asyncio.CancelledError:
        logger.info(f"🔌 SSE stream cancelled for execution {execution_id}")
    except GeneratorExit:
        logger.info(f"🔌 SSE stream closed by client for execution {execution_id}")
    except Exception as e:
        logger.error(f"❌ SSE stream error for execution {execution_id}: {e}", exc_info=True)
        # Send error event before closing
        error_event = {
            "type": "error",
            "message": str(e),
            "timestamp": datetime.now().isoformat()
        }
        try:
            yield f"data: {json.dumps(error_event)}\n\n"
        except:
            pass
    finally:
        await manager.remove_stream(execution_id, queue)
        logger.info(f"✅ SSE stream cleanup complete for execution {execution_id}")


def format_execution_log(
    level: str,
    message: str,
    details: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Format execution log with full details for SSE streaming.
    
    Frontend can show summary in UI and expand for full details.
    NO TRUNCATION - everything is preserved.
    
    Args:
        level: Log level (INFO, DEBUG, WARNING, ERROR)
        message: Short summary message for UI
        details: Full details dictionary
        
    Returns:
        Formatted log event
    """
    return {
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "message": message,
        "details": details or {},
        "truncated": False
    }

