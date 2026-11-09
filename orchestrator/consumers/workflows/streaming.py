"""
Workflow Execution SSE Streaming Service
=========================================

Provides real-time workflow execution updates via Server-Sent Events.
Replaces Redis pubsub + WebSocket architecture from PRD-28.

Based on successful PRD-27 chat SSE implementation.
"""

from typing import AsyncGenerator, Dict, Any, Optional, List, Tuple
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
    - EVENT REPLAY: Stores recent events for late-connecting clients
    """
    
    def __init__(self):
        # {execution_id: [queue1, queue2, ...]}
        self._streams: Dict[int, List[asyncio.Queue]] = {}
        # {execution_id: [event1, event2, ...]} - Recent events for replay
        self._event_history: Dict[int, List[Dict[str, Any]]] = {}
        self._lock = asyncio.Lock()
        self._max_history = 100  # Keep last 100 events per execution
    
    async def create_stream(self, execution_id: int) -> asyncio.Queue:
        """Create a new SSE stream for an execution with event replay"""
        async with self._lock:
            queue = asyncio.Queue(maxsize=1000)  # Large queue for busy workflows
            
            if execution_id not in self._streams:
                self._streams[execution_id] = []
            
            self._streams[execution_id].append(queue)
            
            # REPLAY: Send any missed events to the new subscriber
            if execution_id in self._event_history:
                for event in self._event_history[execution_id]:
                    try:
                        queue.put_nowait(event)
                    except asyncio.QueueFull:
                        logger.warning(f"⚠️ Queue full during replay for execution {execution_id}")
                        break
                logger.info(f"📡 SSE stream created for execution {execution_id} - replayed {len(self._event_history[execution_id])} events")
            else:
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
    
    async def cleanup_execution(self, execution_id: int):
        """Clean up all data for a completed execution"""
        async with self._lock:
            if execution_id in self._event_history:
                del self._event_history[execution_id]
                logger.debug(f"🧹 Cleaned up event history for execution {execution_id}")
            if execution_id in self._streams:
                del self._streams[execution_id]
                logger.debug(f"🧹 Cleaned up streams for execution {execution_id}")
    
    async def broadcast_event(
        self,
        execution_id: int,
        event_type: str,
        data: Dict[str, Any]
    ):
        """
        Broadcast event to all streams for this execution.
        Non-blocking - drops events if queue is full.
        ALSO stores events for replay to late-connecting clients.
        """
        async with self._lock:
            event = {
                "type": event_type,
                "data": data,
                "timestamp": datetime.now().isoformat()
            }
            
            # ALWAYS store event for replay, even if no streams yet
            if execution_id not in self._event_history:
                self._event_history[execution_id] = []
            self._event_history[execution_id].append(event)
            
            # Trim history if too long
            if len(self._event_history[execution_id]) > self._max_history:
                self._event_history[execution_id] = self._event_history[execution_id][-self._max_history:]
            
            if execution_id not in self._streams:
                # No active streams yet, but event is stored for replay
                logger.debug(f"📦 Stored {event_type} event for execution {execution_id} (no active streams yet)")
                return
            
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

    async def broadcast_workflow_update(
        self,
        execution_id: int,
        workflow_id: str,
        status: str,
        current_step: Optional[str] = None,
        progress: Optional[float] = None,
    ):
        """
        Broadcast a workflow-update event (US-015).

        Args:
            execution_id: Workflow execution ID (for stream routing)
            workflow_id: Workflow definition or execution ID string
            status: Current status (running, completed, failed, paused)
            current_step: Name of the currently executing step
            progress: Completion ratio 0.0 - 1.0
        """
        payload: Dict[str, Any] = {
            "workflowId": workflow_id,
            "status": status,
        }
        if current_step is not None:
            payload["currentStep"] = current_step
        if progress is not None:
            payload["progress"] = progress
        await self.broadcast_event(
            execution_id=execution_id,
            event_type="workflow-update",
            data=payload,
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
        # CodeQL py/stack-trace-exposure: the exception's text (e) stays in the
        # server log; the client's SSE event carries a fixed message.
        logger.error(f"❌ SSE stream error for execution {execution_id}: {e}", exc_info=True)
        # Send error event before closing
        error_event = {
            "type": "error",
            "message": "A streaming error occurred",
            "timestamp": datetime.now().isoformat()
        }
        try:
            yield f"data: {json.dumps(error_event)}\n\n"
        except Exception:
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


def _format_stage_event_chunks(event_type: str, data: Dict[str, Any], event: Dict[str, Any]) -> List[str]:
    """Format a stage_start / stage_complete event into its text + data chunks."""
    stage_num = data.get("stage", "?")
    stage_name = data.get("stage_name", "Unknown")
    prefix = "⚡ Stage" if event_type == "stage_start" else "✓ Completed Stage"
    text = f"{prefix} {stage_num}: {stage_name}\\n"
    return [f'0:{json.dumps(text)}\n', f'd:{json.dumps(event)}\n']


def _format_execution_log_chunks(data: Dict[str, Any], event: Dict[str, Any]) -> List[str]:
    """Format an execution_log event's message (with level emoji) + data chunks."""
    level = data.get("level", "INFO")
    message = data.get("message", "")
    emoji = {"INFO": "ℹ️", "DEBUG": "🔍", "WARNING": "⚠️", "ERROR": "❌"}.get(level, "•")
    text = f"{emoji} {message}\\n"
    return [f'0:{json.dumps(text)}\n', f'd:{json.dumps(event)}\n']


def _format_subtask_update_chunks(data: Dict[str, Any], event: Dict[str, Any]) -> List[str]:
    """Format a subtask_update event's status text + data chunks."""
    description = data.get("description", "Task")
    status = data.get("status", "running")
    if status == "completed":
        text = f"✓ {description}\\n"
    elif status == "failed":
        text = f"✗ {description}\\n"
    else:
        text = f"⋯ {description}\\n"
    return [f'0:{json.dumps(text)}\n', f'd:{json.dumps(event)}\n']


def _format_error_event_chunks(event: Dict[str, Any]) -> List[str]:
    """Format an error event's text chunk + error-protocol chunk."""
    error_msg = event.get("message", "Unknown error")
    text = f"❌ Error: {error_msg}\\n"
    error_data = json.dumps({"message": error_msg, "details": event.get("data", {})})
    return [f'0:{json.dumps(text)}\n', f'e:{error_data}\n']


def _format_stream_event_chunks(event: Dict[str, Any]) -> Tuple[List[str], bool]:
    """Format one workflow stream event into AI SDK data-stream chunks.

    Returns the ordered chunks to yield and whether the caller should stop
    streaming after them (true for workflow_complete and error events).
    """
    event_type = event.get("type", "")
    data = event.get("data", {})

    if event_type in ("stage_start", "stage_complete"):
        return _format_stage_event_chunks(event_type, data, event), False
    if event_type == "execution_log":
        return _format_execution_log_chunks(data, event), False
    if event_type == "subtask_update":
        return _format_subtask_update_chunks(data, event), False
    if event_type == "workflow_complete":
        text = "✅ Workflow execution completed!\\n"
        return [f'0:{json.dumps(text)}\n', f'd:{json.dumps(event)}\n'], True
    if event_type == "token":
        content = event.get("content", "")
        return [f'0:{json.dumps(content)}\n'], False
    if event_type == "error":
        return _format_error_event_chunks(event), True
    return [f'd:{json.dumps(event)}\n'], False


async def stream_workflow_as_aisdk(
    execution_id: int
) -> AsyncGenerator[str, None]:
    """
    Stream workflow execution using Vercel AI SDK Data Stream Protocol.
    
    Protocol Format:
    0:"text chunk"  -> Text delta (for stage updates, progress messages)
    d:{"key":"val"} -> Data payload (for structured logs, metrics, stage info)
    e:{"err":"msg"} -> Error information
    
    This provides smooth progressive updates in the UI:
    - Text deltas show in the message stream (stage transitions, progress)
    - Data payloads provide structured information for metrics/logs
    
    CRITICAL: Each yield must be followed by flush to prevent buffering!
    """
    manager = get_stream_manager()
    queue = await manager.create_stream(execution_id)
    
    try:
        logger.info(f"🚀 Starting AI SDK stream for execution {execution_id}")
        
        # Initial connection - send IMMEDIATELY with flush
        yield '0:"🚀 Workflow execution started...\\n"\n'
        await asyncio.sleep(0)  # Yield control to flush
        yield f'd:{{"type":"connected","execution_id":{execution_id},"status":"ok"}}\n'
        await asyncio.sleep(0)  # Yield control to flush
        
        # Small delay to allow workflow to catch this stream
        await asyncio.sleep(0.1)
        
        while True:
            try:
                # Wait for next event with SHORT timeout for responsiveness
                event = await asyncio.wait_for(queue.get(), timeout=5.0)

                # Format by event type, then stream the resulting chunks —
                # each yield flushes immediately via the sleep(0) after it.
                chunks, should_stop = _format_stream_event_chunks(event)
                for chunk in chunks:
                    yield chunk
                    await asyncio.sleep(0)
                if should_stop:
                    break

            except asyncio.TimeoutError:
                # Keep-alive heartbeat - more frequent for better responsiveness
                yield 'd:{"type":"heartbeat"}\n'
                await asyncio.sleep(0)
                continue
                
    except asyncio.CancelledError:
        logger.info(f"🔌 AI SDK stream cancelled for execution {execution_id}")
    except GeneratorExit:
        logger.info(f"🔌 AI SDK stream closed by client for execution {execution_id}")
    except Exception as e:
        # CodeQL py/stack-trace-exposure: the exception's text (e) stays in the
        # server log; the client's data-stream events carry a fixed message.
        logger.error(f"❌ AI SDK stream error for execution {execution_id}: {e}", exc_info=True)
        text = "❌ Stream error occurred\\n"
        try:
            yield f'0:{json.dumps(text)}\n'
            err_json = json.dumps({"message": "A streaming error occurred"})
            yield f'e:{err_json}\n'
        except Exception:
            pass
    finally:
        await manager.remove_stream(execution_id, queue)
        logger.info(f"✅ AI SDK stream cleanup complete for execution {execution_id}")


