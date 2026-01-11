"""
Chatbot Consumer - Chat Interface
=================================

Consumer that combines modules for chat functionality:
- shared.llm: LLM providers
- modules.memory: Memory injection/storage  
- modules.tools: Tool execution

Components:
- ChatService: Database operations for chats/messages
- StreamingChatService: SSE streaming orchestrator
- PromptAnalyzer: Message conversion, intent detection
- StreamingHandler: SSE/AI SDK formatting
- ToolRouter: Routes to modules.tools

Usage:
    from consumers.chatbot import ChatService, StreamingChatService
    
    # In API endpoint:
    chat_service = ChatService(db)
    streaming_service = StreamingChatService(db)
"""

from consumers.chatbot.service import ChatService, StreamingChatService
from consumers.chatbot.prompt_analyzer import PromptAnalyzer, get_prompt_analyzer
from consumers.chatbot.streaming import StreamingHandler, get_streaming_handler
from consumers.chatbot.tool_router import (
    ToolRouter,
    get_tool_router,
    get_chat_tools,
    get_chatbot_tools,
    execute_tool,
)

# Convenience export
CHAT_TOOLS = None

def get_tools():
    """Get chatbot tools (lazy loaded)."""
    global CHAT_TOOLS
    # Retain compatibility: defaults to agent_id=None (all allowed)
    CHAT_TOOLS = get_chat_tools()
    return CHAT_TOOLS

__all__ = [
    # Main services
    'ChatService',
    'StreamingChatService',
    # Components
    'PromptAnalyzer',
    'get_prompt_analyzer',
    'StreamingHandler', 
    'get_streaming_handler',
    'ToolRouter',
    'get_tool_router',
    # Tools
    'get_chat_tools',
    'get_chatbot_tools',
    'get_tools',
    'execute_tool',
    'CHAT_TOOLS',
]

