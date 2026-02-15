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
        agent_name: Optional[str] = None
    ):
        """
        Initialize the orchestrator.

        Args:
            workspace_id: Workspace ID for memory scoping
            agent_id: Agent ID for memory scoping
            agent_name: Agent name for personalization
        """
        self.workspace_id = workspace_id
        self.agent_id = agent_id
        self.agent_name = agent_name or "Automatos"

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
        chat_id: Optional[str] = None
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
        memory_result = None
        if self._should_fetch_memory(intent_result):
            memory_result = await self.memory_manager.retrieve_memories(
                workspace_id=self.workspace_id,
                agent_id=self.agent_id,
                query=latest_query
            )
            self.state.messages_since_memory_fetch = 0
            self.state.memory_fetched_at = time.time()

            # Update user name if found
            if memory_result and memory_result.user_context.name:
                self.state.user_name = memory_result.user_context.name
        else:
            self.state.messages_since_memory_fetch += 1

        # 3. Route Tools (if needed)
        tool_result = None
        if intent_result.requires_tools and available_tools:
            tool_result = self.tool_router.route(
                query=latest_query,
                available_tools=available_tools,
                conversation_context=messages
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
            orchestrator_settings=orch_settings
        )

        # PRD-55: Inject available skills as XML metadata
        skills_xml = self._build_skills_xml()
        if skills_xml:
            system_prompt += f"\n\n{skills_xml}"

        # 5. Convert messages to LLM format
        llm_messages = self._convert_messages(messages)

        # Add memory context as a separate system message if we have specific context
        if memory_result and memory_result.formatted_context:
            memory_msg = {
                "role": "system",
                "content": f"## Current User Context\n\n{memory_result.formatted_context}"
            }
            # Insert after main system prompt
            llm_messages.insert(1, memory_msg)

        # Inject daily logs for temporal awareness (PRD-55)
        try:
            daily_logs = await self.memory_manager.get_daily_logs(self.workspace_id)
            if daily_logs:
                daily_msg = {
                    "role": "system",
                    "content": f"## Recent Activity\n\n{daily_logs}"
                }
                llm_messages.insert(1, daily_msg)
        except Exception as e:
            logger.debug(f"[Orchestrator] Could not load daily logs: {e}")

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
            requires_tools=intent_result.requires_tools,
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

    def _build_skills_xml(self) -> str:
        """Build XML block of available skills for LLM context (PRD-55)."""
        try:
            from core.database.database import SessionLocal
            from core.models import Skill

            db = SessionLocal()
            try:
                skills = (
                    db.query(Skill)
                    .filter(Skill.is_active == True)
                    .filter(
                        (Skill.workspace_id == self.workspace_id)
                        | (Skill.workspace_id == None)
                    )
                    .limit(20)
                    .all()
                )
                if not skills:
                    return ""

                lines = ["<available_skills>"]
                for s in skills:
                    name = s.name or "unknown"
                    desc = (s.description or "")[:100]
                    cat = s.category or ""
                    lines.append(
                        f'  <skill name="{name}" category="{cat}">{desc}</skill>'
                    )
                lines.append("</available_skills>")
                return "\n".join(lines)
            finally:
                db.close()
        except Exception as e:
            logger.debug(f"[Orchestrator] Could not load skills XML: {e}")
            return ""

    def _convert_messages(self, messages: List[Dict]) -> List[Dict[str, str]]:
        """Convert messages to simple role/content format for LLM."""
        converted = []

        for msg in messages:
            role = msg.get("role", "user")
            content = ""

            # Handle parts format
            if msg.get("parts"):
                text_parts = []
                for part in msg["parts"]:
                    if part.get("type") == "text" and part.get("text"):
                        text_parts.append(part["text"])
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
        result = await self.memory_manager.store_conversation(
            workspace_id=self.workspace_id,
            agent_id=self.agent_id,
            user_message=user_message,
            assistant_response=assistant_response,
            chat_id=chat_id
        )

        # Store daily log entry (PRD-55: Temporal Memory)
        try:
            await self.memory_manager.store_daily_summary(
                workspace_id=self.workspace_id,
                user_message=user_message,
                assistant_response=assistant_response,
            )
        except Exception as e:
            logger.debug(f"[Orchestrator] Daily log storage failed: {e}")

        return result

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
