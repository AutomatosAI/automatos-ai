"""
Smart Chat Orchestrator
=======================

The central coordinator for the intelligent Automatos chat system.

This module ties together:
- Intent Classification (what does the user want?)
- Memory Management (what do we know about them?)
- Tool Routing (what tools do they need?)
- Personality (how do we respond?)

Usage:
    orchestrator = SmartChatOrchestrator(workspace_id, agent_id)
    result = await orchestrator.prepare_request(messages, available_tools)
    # result contains: system_prompt, filtered_tools, tool_choice, memory_context
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .intent_classifier import Intent, IntentResult, get_intent_classifier
from .smart_memory import MemoryResult, get_smart_memory_manager
from .smart_tool_router import ToolRoutingResult, get_smart_tool_router
from .personality import get_happy_system_prompt, AutomatosPersonality, load_orchestrator_settings

logger = logging.getLogger(__name__)


@dataclass
class OrchestratedRequest:
    """
    Result of orchestrating a chat request.

    Contains everything needed to make the LLM call.
    """
    # Messages to send to LLM
    system_prompt: str
    messages: List[Dict[str, str]]

    # Tool configuration
    tools: List[Dict[str, Any]]
    tool_choice: str  # "auto", "required", or "none"

    # Context that was injected
    memory_context: Optional[str]
    user_name: Optional[str]

    # Classification results (for logging/debugging)
    intent: Intent
    intent_confidence: float
    requires_tools: bool
    requires_memory: bool

    # Timing
    preparation_time_ms: float


@dataclass
class ConversationState:
    """
    Tracks state across a conversation.

    Used to avoid redundant work and maintain context.
    """
    last_intent: Optional[Intent] = None
    last_tool_calls: List[str] = field(default_factory=list)
    messages_since_memory_fetch: int = 0
    user_name: Optional[str] = None
    memory_fetched_at: Optional[float] = None


class SmartChatOrchestrator:
    """
    Central orchestrator for intelligent chat processing.

    This replaces the scattered logic in the old chat service with
    a clean, unified approach.
    """

    # How often to refresh memory (in messages)
    MEMORY_REFRESH_INTERVAL = 5

    def __init__(
        self,
        workspace_id: str,
        agent_id: Optional[int] = None,
        agent_name: Optional[str] = None,
        widget_mode: bool = False
    ):
        """
        Initialize the orchestrator.

        Args:
            workspace_id: Workspace ID for memory scoping
            agent_id: Agent ID for memory scoping
            agent_name: Agent name for personalization
            widget_mode: When True, restrict memory to agent-only (no global workspace memories)
        """
        self.workspace_id = workspace_id
        self.agent_id = agent_id
        self.agent_name = agent_name or "Automatos"
        self.widget_mode = widget_mode

        # Components
        self.classifier = get_intent_classifier()
        self.memory_manager = get_smart_memory_manager()
        self.tool_router = get_smart_tool_router()

        # Conversation state
        self.state = ConversationState()

        logger.info(f"[Orchestrator] Initialized for workspace={workspace_id}, agent={agent_id}")

    async def prepare_request(
        self,
        messages: List[Dict[str, Any]],
        available_tools: List[Dict[str, Any]],
        chat_id: Optional[str] = None,
        complexity_assessment: Optional[Any] = None,
    ) -> OrchestratedRequest:
        """
        Prepare a chat request for the LLM.

        This is the main entry point. It:
        1. Classifies the user's intent
        2. Retrieves relevant memories (if needed)
        3. Routes to appropriate tools (if needed)
        4. Builds the system prompt with personality
        5. Returns everything needed for the LLM call

        Args:
            messages: Conversation messages
            available_tools: All tools available to this agent
            chat_id: Optional chat session ID

        Returns:
            OrchestratedRequest ready for LLM
        """
        import time
        start_time = time.time()

        # Extract latest user message
        latest_query = self._extract_latest_user_message(messages)
        logger.debug(f"[Orchestrator] Processing: {latest_query[:50]}...")

        # 1. Classify Intent
        intent_result = self.classifier.classify(latest_query, messages)
        self.state.last_intent = intent_result.primary_intent

        logger.info(f"[Orchestrator] Intent: {intent_result.primary_intent.value} "
                   f"(tools: {intent_result.requires_tools}, memory: {intent_result.requires_memory})")

        # 2. Retrieve Memory (if needed or stale)
        # PRD-68: ComplexityAssessment can override memory decision
        memory_result = None
        _wants_memory = self._should_fetch_memory(intent_result)
        if complexity_assessment and not complexity_assessment.needs_memory:
            _wants_memory = False
            logger.info(f"[Orchestrator] Memory SKIPPED by ComplexityAssessment ({complexity_assessment.complexity.value})")
        if _wants_memory:
            memory_result = await self.memory_manager.retrieve_memories(
                workspace_id=self.workspace_id,
                agent_id=self.agent_id,
                query=latest_query,
                widget_mode=self.widget_mode
            )
            self.state.messages_since_memory_fetch = 0
            self.state.memory_fetched_at = time.time()

            # Update user name if found
            if memory_result and memory_result.user_context.name:
                self.state.user_name = memory_result.user_context.name
        else:
            self.state.messages_since_memory_fetch += 1

        # US-015: Store last memory result so callers can emit SSE events
        self._last_memory_result = memory_result

        # 3. Route Tools (if needed)
        # PRD-68: ComplexityAssessment tool_hints override intent-based routing
        tool_result = None
        _wants_tools = intent_result.requires_tools
        _tool_hints = None
        if complexity_assessment and complexity_assessment.tool_hints:
            _wants_tools = True
            _tool_hints = complexity_assessment.tool_hints
            logger.info(f"[Orchestrator] Tools ENABLED by tool_hints={_tool_hints}")
        if _wants_tools and available_tools:
            tool_result = await self.tool_router.route(
                query=latest_query,
                available_tools=available_tools,
                conversation_context=messages,
                tool_hints=_tool_hints,
            )
        else:
            # Even when intent says "no tools", always include platform_* tools
            # so Auto can answer platform self-awareness queries (PRD-64)
            platform_tools = [
                t for t in (available_tools or [])
                if t.get("function", {}).get("name", "").startswith("platform_")
            ]
            if platform_tools:
                tool_result = ToolRoutingResult(
                    should_include_tools=True,
                    filtered_tools=platform_tools,
                    priority_tools=[],
                    tool_choice="auto",
                    reasoning="Platform tools always available for self-awareness"
                )
            else:
                tool_result = ToolRoutingResult(
                    should_include_tools=False,
                    filtered_tools=[],
                    priority_tools=[],
                    tool_choice="none",
                    reasoning="No tools needed for this intent"
                )

        # 4. Build System Prompt
        memory_strings = []
        if memory_result and memory_result.memories:
            memory_strings = [m.get("memory", "") for m in memory_result.memories if m.get("memory")]

        daily_logs = ""
        try:
            from config import config
            if getattr(config, "INJECT_DAILY_LOGS", True):
                daily_logs = await self.memory_manager.get_daily_logs(
                    workspace_id=self.workspace_id,
                    max_chars=2000,
                )
        except Exception as exc:
            logger.debug("[Orchestrator] Daily logs skipped: %s", exc)

        tool_names = []
        if tool_result.filtered_tools:
            for t in tool_result.filtered_tools:
                fn = t.get("function", {})
                if fn.get("name"):
                    tool_names.append(fn["name"])

        # Load workspace personality settings (cached, ~0ms on hit)
        orch_settings = load_orchestrator_settings(self.workspace_id)

        system_prompt = get_happy_system_prompt(
            user_name=self.state.user_name,
            agent_name=self.agent_name,
            msg_count=len(messages),
            memories=memory_strings,
            tool_names=tool_names,
            orchestrator_settings=orch_settings,
        )

        if daily_logs:
            system_prompt = f"{system_prompt}\n\n## Recent Activity\n\n{daily_logs}"

        # 5. Convert messages to LLM format
        llm_messages = self._convert_messages(messages)

        # Add current datetime context
        now = datetime.utcnow()
        date_msg = {
            "role": "system",
            "content": f"Current date/time (UTC): {now.strftime('%Y-%m-%d %H:%M')}. "
                      f"Use this for any time-relative queries."
        }
        llm_messages.insert(1, date_msg)

        preparation_time = (time.time() - start_time) * 1000

        return OrchestratedRequest(
            system_prompt=system_prompt,
            messages=llm_messages,
            tools=tool_result.filtered_tools,
            tool_choice=tool_result.tool_choice,
            memory_context=memory_result.formatted_context if memory_result else None,
            user_name=self.state.user_name,
            intent=intent_result.primary_intent,
            intent_confidence=intent_result.confidence,
            requires_tools=tool_result.should_include_tools or intent_result.requires_tools,
            requires_memory=intent_result.requires_memory,
            preparation_time_ms=preparation_time
        )

    def _should_fetch_memory(self, intent_result: IntentResult) -> bool:
        """Determine if we should fetch memories for this request."""
        # Always fetch if intent requires it
        if intent_result.requires_memory:
            return True

        # Fetch if it's been a while
        if self.state.messages_since_memory_fetch >= self.MEMORY_REFRESH_INTERVAL:
            return True

        # Fetch if we don't have user name yet
        if not self.state.user_name:
            return True

        return False

    def _extract_latest_user_message(self, messages: List[Dict]) -> str:
        """Extract the text of the latest user message."""
        for msg in reversed(messages):
            if msg.get("role") != "user":
                continue

            # Handle parts format
            if msg.get("parts"):
                for part in msg["parts"]:
                    if part.get("type") == "text" and part.get("text"):
                        return part["text"]

            # Handle content format
            if msg.get("content"):
                return msg["content"]

        return ""

    def _convert_messages(self, messages: List[Dict]) -> List[Dict[str, str]]:
        """Convert messages to simple role/content format for LLM.

        Filters out system-role messages since the orchestrator builds
        its own system prompt via personality.py.
        """
        converted = []

        for msg in messages:
            role = msg.get("role", "user")

            # Skip system messages — we build our own system prompt
            if role == "system":
                continue

            content = ""

            # Handle parts format
            if msg.get("parts"):
                text_parts = []
                for part in msg["parts"]:
                    if part.get("type") == "text" and part.get("text"):
                        text_parts.append(part["text"])
                    elif part.get("type") == "file":
                        # File parts should already be resolved by _resolve_file_parts
                        # but handle gracefully if they slip through
                        filename = part.get("filename", "file")
                        text_parts.append(f"[Attached file: {filename} — content not available]")
                content = "\n".join(text_parts)
            else:
                content = msg.get("content", "")

            if content:
                converted.append({"role": role, "content": content})

        return converted

    async def store_exchange(
        self,
        user_message: str,
        assistant_response: str,
        chat_id: Optional[str] = None
    ) -> bool:
        """
        Store a conversation exchange in memory.

        Call this after the LLM responds to save the exchange.

        Args:
            user_message: The user's message
            assistant_response: The assistant's response
            chat_id: Optional chat session ID

        Returns:
            Success status
        """
        stored = await self.memory_manager.store_conversation(
            workspace_id=self.workspace_id,
            agent_id=self.agent_id,
            user_message=user_message,
            assistant_response=assistant_response,
            chat_id=chat_id,
            widget_mode=self.widget_mode
        )

        try:
            await self.memory_manager.store_daily_summary(
                workspace_id=self.workspace_id,
                user_message=user_message,
                assistant_response=assistant_response,
                agent_id=self.agent_id,
            )
        except Exception as exc:
            logger.debug("[Orchestrator] Daily summary storage skipped: %s", exc)

        return stored

    def get_user_name(self) -> Optional[str]:
        """Get the user's name if known."""
        return self.state.user_name


# Convenience function for quick orchestration
async def orchestrate_chat_request(
    workspace_id: str,
    agent_id: Optional[int],
    messages: List[Dict],
    available_tools: List[Dict],
    agent_name: Optional[str] = None,
    chat_id: Optional[str] = None
) -> OrchestratedRequest:
    """
    Quick helper to orchestrate a single chat request.

    For stateless usage when you don't need to maintain ConversationState.
    """
    orchestrator = SmartChatOrchestrator(
        workspace_id=workspace_id,
        agent_id=agent_id,
        agent_name=agent_name
    )
    return await orchestrator.prepare_request(messages, available_tools, chat_id)
