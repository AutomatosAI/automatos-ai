"""
Streaming Handler - SSE Response Formatting
============================================

Handles:
- Formatting SSE chunks for legacy format
- Formatting AI SDK Data Stream format
- Chunking text for smooth streaming
"""

import json
import logging
import uuid
from typing import Dict, Any, AsyncGenerator
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
    
    def format_aisdk_chat_id(self, chat_id: str) -> str:
        """Format chat-id data event."""
        return f'd:{{"type":"chat-id","chatId":"{chat_id}"}}\n'
    
    def format_aisdk_tool_data(self, tool_data: Dict[str, Any]) -> str:
        """Format tool-data event for AI SDK."""
        return f'd:{{"type":"tool-data","data":{json.dumps(tool_data)}}}\n'
    
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
    
    async def stream_text_aisdk(
        self,
        text: str,
        chunk_size: int = 10
    ) -> AsyncGenerator[str, None]:
        """
        Stream text in AI SDK format with smooth chunking.
        
        Args:
            text: Full text to stream
            chunk_size: Characters per chunk
        """
        for i in range(0, len(text), chunk_size):
            chunk = text[i:i + chunk_size]
            yield self.format_aisdk_text(chunk)
            await asyncio.sleep(0.01)  # Small delay for smooth streaming


# Module-level instance
_streaming_handler = None

def get_streaming_handler() -> StreamingHandler:
    """Get or create the global StreamingHandler instance."""
    global _streaming_handler
    if _streaming_handler is None:
        _streaming_handler = StreamingHandler()
    return _streaming_handler

