"""
Chat Service - Backward Compatibility Re-export
================================================

This file re-exports from the modular chat service architecture.
All functionality has been moved to services/chat/ for maintainability.

Modules:
- services/chat/chat_service.py: Main ChatService and StreamingChatService
- services/chat/prompt_analyzer.py: Message conversion and tool intent
- services/chat/memory_injector.py: Memory retrieval and storage
- services/chat/tool_router.py: Tool execution and formatting
- services/chat/streaming_handler.py: SSE formatting
"""

# Re-export everything from the new modular location
from services.chat.chat_service import (
    ChatService,
    StreamingChatService,
    CHAT_TOOLS,
    get_chat_tools,
)

# Also export commonly used functions for backward compatibility
from services.chat.tool_router import (
    execute_tool as execute_tool_unified,
    get_chatbot_tools as get_chatbot_tools_from_registry,
)

from services.chat.memory_injector import (
    get_memory_system,
    get_context_retrieval_engine,
)

from services.chat.prompt_analyzer import (
    get_prompt_analyzer,
)

__all__ = [
    'ChatService',
    'StreamingChatService',
    'CHAT_TOOLS',
    'get_chat_tools',
    'execute_tool_unified',
    'get_chatbot_tools_from_registry',
    'get_memory_system',
    'get_context_retrieval_engine',
    'get_prompt_analyzer',
]
