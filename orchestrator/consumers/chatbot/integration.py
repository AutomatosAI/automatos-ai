"""
Smart Orchestrator Integration
==============================

This module provides integration helpers to use the new Smart Orchestrator
with the existing StreamingChatService.

Usage in service.py:

    from consumers.chatbot.integration import SmartChatIntegration

    # In StreamingChatService.__init__:
    self.smart_chat = SmartChatIntegration(workspace_id, agent_id, agent_name)

    # In stream_response_with_agent:
    orchestrated = await self.smart_chat.prepare(messages, tools, chat_id, complexity_assessment)
    # Use orchestrated.system_prompt, orchestrated.tools, orchestrated.tool_choice

    # After response:
    await self.smart_chat.store(user_message, assistant_response, chat_id)
"""

import logging
from typing import Dict, List, Optional, Any

from .smart_orchestrator import SmartChatOrchestrator, OrchestratedRequest
from .smart_memory import get_smart_memory_manager
from .intent_classifier import get_intent_classifier

logger = logging.getLogger(__name__)


class SmartChatIntegration:
    """
    Integration layer for the Smart Chat Orchestrator.

    Drop-in replacement for scattered memory/tool logic in StreamingChatService.
    """

    def __init__(
        self,
        workspace_id: str,
        agent_id: Optional[int] = None,
        agent_name: Optional[str] = None,
        widget_mode: bool = False,
        db_session: Any = None,
    ):
        """
        Initialize the smart chat integration.

        Args:
            workspace_id: Workspace ID for memory scoping
            agent_id: Agent ID for memory scoping
            agent_name: Agent name for personalization
            widget_mode: When True, restrict memory to agent-only (no global workspace memories)
            db_session: Optional DB session for ContextService (PRD-80)
        """
        self.orchestrator = SmartChatOrchestrator(
            workspace_id=workspace_id,
            agent_id=agent_id,
            agent_name=agent_name,
            widget_mode=widget_mode,
            db_session=db_session,
        )
        self.workspace_id = workspace_id
        self.agent_id = agent_id
        logger.info(f"[SmartChat] Integration initialized for ws={workspace_id}, agent={agent_id}")

    async def prepare(
        self,
        messages: List[Dict[str, Any]],
        available_tools: List[Dict[str, Any]],
        chat_id: Optional[str] = None,
        complexity_assessment: Optional[Any] = None,
        attachment_ids: Optional[List[str]] = None,
        model_id: Optional[str] = None,
    ) -> OrchestratedRequest:
        """
        Prepare a chat request using smart orchestration.

        This replaces:
        - Memory injection logic
        - Tool filtering logic
        - System prompt building

        Args:
            messages: Conversation messages
            available_tools: All tools available to agent
            chat_id: Optional chat session ID
            complexity_assessment: Optional PRD-68 AutoBrain assessment
            attachment_ids: PRD-127 ephemeral attachments to resolve
            model_id: PRD-127 model identifier for vision capability check

        Returns:
            OrchestratedRequest ready for LLM
        """
        return await self.orchestrator.prepare_request(
            messages=messages,
            available_tools=available_tools,
            chat_id=chat_id,
            complexity_assessment=complexity_assessment,
            attachment_ids=attachment_ids,
            model_id=model_id,
        )

    async def store(
        self,
        user_message: str,
        assistant_response: str,
        chat_id: Optional[str] = None
    ) -> bool:
        """
        Store a conversation exchange in memory.

        Call this after the LLM response is complete.

        Args:
            user_message: The user's message
            assistant_response: The assistant's response
            chat_id: Optional chat session ID

        Returns:
            Success status
        """
        return await self.orchestrator.store_exchange(
            user_message=user_message,
            assistant_response=assistant_response,
            chat_id=chat_id
        )

    def get_user_name(self) -> Optional[str]:
        """Get the user's name if known from memory."""
        return self.orchestrator.get_user_name()

    def should_skip_tools(self, query: str) -> bool:
        """
        Quick check: should we skip tools for this message?

        Use this before loading tools to save time.
        """
        classifier = get_intent_classifier()
        result = classifier.classify(query)
        return not result.requires_tools


def apply_orchestration_to_messages(
    orchestrated: OrchestratedRequest
) -> List[Dict[str, str]]:
    """
    Apply orchestration results to build final message list.

    Returns messages with system prompt at the start.
    """
    final_messages = [
        {"role": "system", "content": orchestrated.system_prompt}
    ]
    final_messages.extend(orchestrated.messages)
    return final_messages


def get_tool_choice_for_llm(orchestrated: OrchestratedRequest) -> Optional[str]:
    """
    Get the tool_choice parameter for the LLM call.

    Returns None if no tools should be used.
    """
    if orchestrated.tool_choice == "none" or not orchestrated.tools:
        return None
    return orchestrated.tool_choice


