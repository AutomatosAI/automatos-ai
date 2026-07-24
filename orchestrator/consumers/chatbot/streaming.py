"""
Streaming Handler - SSE Response Formatting
============================================

Handles:
- Formatting SSE chunks for legacy format
- Formatting AI SDK Data Stream format
- Word-boundary-aware chunking for smooth streaming
- Widget-related SSE events (memory, workflow)
"""

import json
import logging
import uuid
from typing import Dict, Any, AsyncGenerator, List, Optional
import asyncio

logger = logging.getLogger(__name__)


class StreamingHandler:
    """
    Handles SSE formatting for chat responses.
    Supports both legacy SSE format and AI SDK Data Stream format.
    """

    # ==========================================================================
    # LEGACY SSE FORMAT (data: {json}\n\n)
    # ==========================================================================

    def format_sse_chunk(self, chunk: Dict[str, Any]) -> str:
        """Format chunk as SSE data (legacy format)."""
        if chunk.get('type') == 'text':
            data = {
                'type': 'text-delta',
                'id': str(uuid.uuid4()),
                'delta': chunk.get('text', '')
            }
        elif chunk.get('type') == 'tool_call':
            data = {
                'type': 'tool-result',
                'toolName': chunk.get('tool_name'),
                'result': chunk.get('result')
            }
        elif chunk.get('type') == 'usage':
            data = {
                'type': 'data-usage',
                'data': {
                    'promptTokens': chunk.get('prompt_tokens', 0),
                    'completionTokens': chunk.get('completion_tokens', 0),
                    'totalTokens': chunk.get('total_tokens', 0),
                    'cost': chunk.get('cost')
                }
            }
        else:
            data = chunk

        return f"data: {json.dumps(data)}\n\n"

    def format_sse_tool_data(self, tool_data: Dict[str, Any]) -> str:
        """Format tool data for SSE."""
        return f"data: {json.dumps({'type': 'tool-data', 'data': tool_data})}\n\n"

    def format_sse_text_start(self, message_id: str) -> str:
        """Format text-start event."""
        return f"data: {json.dumps({'type': 'text-start', 'id': message_id})}\n\n"

    def format_sse_text_delta(self, message_id: str, delta: str) -> str:
        """Format text-delta event."""
        return f"data: {json.dumps({'type': 'text-delta', 'id': message_id, 'delta': delta})}\n\n"

    def format_sse_text_end(self, message_id: str) -> str:
        """Format text-end event."""
        return f"data: {json.dumps({'type': 'text-end', 'id': message_id})}\n\n"

    def format_sse_done(self) -> str:
        """Format done event."""
        return f"data: {json.dumps({'type': 'done'})}\n\n"

    def format_sse_error(self, error: str) -> str:
        """Format error event."""
        return f"data: {json.dumps({'type': 'error', 'error': error})}\n\n"

    async def stream_text_legacy(
        self,
        text: str,
        message_id: str = None
    ) -> AsyncGenerator[str, None]:
        """Stream text word-by-word in legacy SSE format."""
        message_id = message_id or str(uuid.uuid4())

        yield self.format_sse_text_start(message_id)

        words = text.split(' ')
        for i, word in enumerate(words):
            chunk_text = word + (' ' if i < len(words) - 1 else '')
            yield self.format_sse_text_delta(message_id, chunk_text)

        yield self.format_sse_text_end(message_id)

    # ==========================================================================
    # AI SDK DATA STREAM FORMAT (0:"text"\n, d:{json}\n, e:{json}\n)
    # ==========================================================================

    def format_aisdk_text(self, text: str) -> str:
        """Format text chunk for AI SDK Data Stream."""
        escaped = json.dumps(text)
        return f'0:{escaped}\n'

    def format_aisdk_data(self, event_type: str, data: Dict[str, Any] = None) -> str:
        """Format data event for AI SDK Data Stream."""
        payload = {"type": event_type}
        if data:
            payload["data"] = data
        return f'd:{json.dumps(payload)}\n'

    def format_aisdk_limit_reached(self, limit: str, value: int, message: str) -> str:
        """Format a limit_reached event so the user is told an agent stopped
        because it hit a cap (instead of silently bailing). Carries limit/value
        under the AI SDK data envelope like every other data event."""
        return self.format_aisdk_data(
            "limit_reached",
            {"limit": limit, "value": value, "message": message},
        )

    def format_aisdk_chat_id(self, chat_id: str) -> str:
        """Format chat-id data event."""
        return f'd:{{"type":"chat-id","chatId":"{chat_id}"}}\n'

    def format_aisdk_tool_data(self, tool_data: Dict[str, Any]) -> str:
        """Format tool-data event for AI SDK."""
        return f'd:{{"type":"tool-data","data":{json.dumps(tool_data)}}}\n'

    def format_aisdk_tool_start(
        self,
        tool_call_id: str,
        tool_name: str,
        tool_input: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format tool-start event for AI SDK (tool lifecycle UI)."""
        return self.format_aisdk_data(
            "tool-start",
            {
                "toolCallId": tool_call_id,
                "toolName": tool_name,
                "input": tool_input or {},
            },
        )

    def format_aisdk_tool_end(
        self,
        tool_call_id: str,
        tool_name: str,
        success: bool,
        error: Optional[str] = None,
        duration_ms: Optional[int] = None
    ) -> str:
        """Format tool-end event for AI SDK (tool lifecycle UI)."""
        payload: Dict[str, Any] = {
            "toolCallId": tool_call_id,
            "toolName": tool_name,
            "success": bool(success),
        }
        if error:
            payload["error"] = error
        if duration_ms is not None:
            payload["durationMs"] = int(duration_ms)
        return self.format_aisdk_data("tool-end", payload)

    def format_aisdk_usage(self, prompt_tokens: int, completion_tokens: int, total_tokens: int) -> str:
        """Format usage data event."""
        usage_data = {
            "promptTokens": prompt_tokens,
            "completionTokens": completion_tokens,
            "totalTokens": total_tokens
        }
        return f'd:{{"type":"usage","data":{json.dumps(usage_data)}}}\n'

    def format_aisdk_finish(self, reason: str = "stop") -> str:
        """Format finish event."""
        return f'd:{{"type":"finish","finishReason":"{reason}"}}\n'

    def format_aisdk_error(self, error: str) -> str:
        """Format error event for AI SDK."""
        return f'e:{json.dumps({"message": error})}\n'

    # ==========================================================================
    # WIDGET SSE EVENTS (US-015)
    # ==========================================================================

    def format_aisdk_memory_injected(
        self,
        memories: List[Dict[str, Any]],
        total_matched: int,
    ) -> str:
        """Format memory-injected event for AI SDK Data Stream."""
        return self.format_aisdk_data(
            "memory-injected",
            {
                "memories": memories,
                "totalMatched": total_matched,
            },
        )

    def format_aisdk_memory_stored(
        self,
        memory: Dict[str, Any],
        reason: str,
    ) -> str:
        """Format memory-stored event for AI SDK Data Stream."""
        return self.format_aisdk_data(
            "memory-stored",
            {
                "memory": memory,
                "reason": reason,
            },
        )

    def format_aisdk_workflow_update(
        self,
        workflow_id: str,
        status: str,
        current_step: Optional[str] = None,
        progress: Optional[float] = None,
    ) -> str:
        """Format workflow-update event for AI SDK Data Stream."""
        payload: Dict[str, Any] = {
            "workflowId": workflow_id,
            "status": status,
        }
        if current_step is not None:
            payload["currentStep"] = current_step
        if progress is not None:
            payload["progress"] = progress
        return self.format_aisdk_data("workflow-update", payload)

    # ==========================================================================
    # PRD-123 Pattern #11: Typed StreamEvent emission
    # ==========================================================================

    def format_stream_event(self, event: "StreamEvent") -> str:
        """Format a typed StreamEvent as AI SDK Data Stream line."""
        return event.to_sse()

    def format_agent_assigned(
        self,
        agent_id: int,
        agent_name: str,
        route_method: str = "semantic",
    ) -> str:
        """Format agent-assigned event."""
        return self.format_aisdk_data(
            "agent-assigned",
            {
                "agentId": agent_id,
                "agentName": agent_name,
                "routeMethod": route_method,
            },
        )

    def format_tool_permission_denied(
        self,
        tool_name: str,
        agent_id: int,
        reason: str,
    ) -> str:
        """Format tool-permission-denied event."""
        return self.format_aisdk_data(
            "tool-permission-denied",
            {
                "toolName": tool_name,
                "agentId": agent_id,
                "reason": reason,
            },
        )

    def format_budget_warning(
        self,
        tokens_used: int,
        token_budget: int,
        percent_used: float,
    ) -> str:
        """Format budget-warning event."""
        return self.format_aisdk_data(
            "budget-warning",
            {
                "tokensUsed": tokens_used,
                "tokenBudget": token_budget,
                "percentUsed": round(percent_used, 1),
            },
        )

    def format_done_with_metadata(
        self,
        stop_reason: Optional[str] = None,
        tokens_used: Optional[int] = None,
        cost: Optional[float] = None,
    ) -> str:
        """Format done event with metadata (PRD-123 enriched finish)."""
        payload: Dict[str, Any] = {"finishReason": "stop"}
        if stop_reason:
            payload["stopReason"] = stop_reason
        if tokens_used is not None:
            payload["tokensUsed"] = tokens_used
        if cost is not None:
            payload["cost"] = cost
        return f'd:{json.dumps({**{"type": "finish"}, **payload})}\n'

    async def stream_text_aisdk(
        self,
        text: str,
        chunk_size: int = 10
    ) -> AsyncGenerator[str, None]:
        """
        Stream text in AI SDK format with smooth word-boundary-aware chunking.

        Produces natural typewriter output by grouping 3-8 words per chunk
        with variable pacing: shorter delays for common words, tiny pauses
        at sentence boundaries for natural reading rhythm.

        Args:
            text: Full text to stream
            chunk_size: Ignored (kept for backward compat). Word-based chunking is used instead.
        """
        if not text:
            return

        # Split into words preserving whitespace boundaries
        words = text.split(' ')
        if not words:
            return

        # Sentence-ending punctuation for rhythm pauses
        _SENTENCE_END = {'.', '!', '?'}

        # Target 3-8 words per chunk for natural feel
        _MIN_WORDS = 3
        _MAX_WORDS = 8

        idx = 0
        total = len(words)

        while idx < total:
            # Determine chunk size: 3-8 words
            # Use shorter chunks near sentence boundaries for rhythm
            chunk_words = []
            words_in_chunk = 0

            while idx < total and words_in_chunk < _MAX_WORDS:
                word = words[idx]
                chunk_words.append(word)
                words_in_chunk += 1
                idx += 1

                # Break at sentence boundaries once we have minimum words
                if words_in_chunk >= _MIN_WORDS and word and word[-1] in _SENTENCE_END:
                    break

            # Reconstruct chunk text with spaces
            chunk_text = ' '.join(chunk_words)
            # Add trailing space unless this is the last chunk
            if idx < total:
                chunk_text += ' '

            yield self.format_aisdk_text(chunk_text)

            # Variable delay for natural pacing
            if chunk_words and chunk_words[-1] and chunk_words[-1][-1] in _SENTENCE_END:
                # Sentence boundary: slightly longer pause
                await asyncio.sleep(0.025)
            elif words_in_chunk <= 3:
                # Short chunk: minimal delay
                await asyncio.sleep(0.008)
            else:
                # Normal chunk
                await asyncio.sleep(0.012)


# Module-level instance
_streaming_handler = None

def get_streaming_handler() -> StreamingHandler:
    """Get or create the global StreamingHandler instance."""
    global _streaming_handler
    if _streaming_handler is None:
        _streaming_handler = StreamingHandler()
    return _streaming_handler
