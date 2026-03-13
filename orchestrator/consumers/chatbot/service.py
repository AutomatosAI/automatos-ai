"""
Chat Service - Consumer for Chat Functionality (Re-export Layer)
================================================================

This module re-exports the public API from responsibility-based modules:
- tool_loop.py: ToolExecutionTracker + loop prevention helpers
- chat_crud.py: ChatService (DB operations for chats/messages)
- stream_orchestrator.py: StreamingChatService (SSE streaming orchestrator)

Internal modules (not re-exported):
- message_prep.py: Message preparation pipeline
- tool_integration.py: Tool schema retrieval, Composio injection, CTO override
- tool_loop_handler.py: Tool loop execution logic

Usage:
    from consumers.chatbot.service import ChatService, StreamingChatService
"""

from consumers.chatbot.tool_loop import ToolExecutionTracker
from consumers.chatbot.chat_crud import ChatService
from consumers.chatbot.stream_orchestrator import StreamingChatService

__all__ = [
    "ToolExecutionTracker",
    "ChatService",
    "StreamingChatService",
]
