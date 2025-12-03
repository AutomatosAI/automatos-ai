"""
Chat Service Module
====================

Modular chat service architecture.

Components:
- ChatService: Database operations for chats/messages
- StreamingChatService: SSE streaming orchestrator
- PromptAnalyzer: Message conversion and tool intent detection
- MemoryInjector: Memory retrieval and storage
- ToolRouter: Unified tool execution
- StreamingHandler: SSE formatting
"""

from services.chat.chat_service import ChatService, StreamingChatService, CHAT_TOOLS, get_chat_tools

__all__ = [
    'ChatService',
    'StreamingChatService', 
    'CHAT_TOOLS',
    'get_chat_tools',
]

