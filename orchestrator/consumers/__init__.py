"""
Consumers - Platform Integration Layer
======================================

Consumers combine modules for specific use cases.
Business logic lives here, APIs are thin wrappers.

Components:
- chatbot/    - Chat interface (uses shared.llm, modules.memory, modules.tools)
- workflows/  - Workflow execution (uses modules.agents, modules.tools)
- external/   - Third-party API (exposes modules)

Usage:
    from consumers.chatbot import ChatService, StreamingChatService
    from consumers.workflows import WorkflowIntegrator
"""

# Export chatbot consumer
from consumers.chatbot import (
    ChatService,
    StreamingChatService,
    get_chat_tools,
)

__all__ = [
    'ChatService',
    'StreamingChatService',
    'get_chat_tools',
]
