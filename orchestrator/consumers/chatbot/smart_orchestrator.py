"""
Smart Chat Orchestrator
=======================

The central coordinator for the intelligent Automatos chat system.

This module ties together:
- Intent Classification (what does the user want?)
- ContextService (unified prompt building, memory, tools, personality)
- Conversation State (tracking across messages)

Usage:
    orchestrator = SmartChatOrchestrator(workspace_id, agent_id)
    result = await orchestrator.prepare_request(messages, available_tools)
    # result contains: system_prompt, filtered_tools, tool_choice, memory_context
"""

import asyncio
import logging
import time
import types
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from .intent_classifier import Intent, IntentResult, get_intent_classifier

logger = logging.getLogger(__name__)

# Strong references to in-flight fire-and-forget tasks. Without this the event
# loop only holds a weak reference and a background write can be garbage
# collected mid-flight ("Task was destroyed but it is pending").
_BACKGROUND_TASKS: set = set()


def _spawn_background(coro, *, label: str) -> None:
    """Schedule a coroutine fire-and-forget, retaining a strong reference.

    Memory writes (Mem0 fact extraction, daily summary, L1/L2 persistence) are
    network-bound and must never block the streaming response. They run on the
    module-level UnifiedMemoryService, so they are safe to outlive the request's
    DB session.
    """
    try:
        task = asyncio.create_task(coro)
    except RuntimeError:
        # No running event loop (shouldn't happen on the async request path).
        logger.debug("[Orchestrator] No event loop for background task '%s' — skipping", label)
        return
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


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

    # PRD-201 S1: the per-turn context-assembly trace (mode, per-section
    # token/trim detail, model, budget ceiling). Persisted on messages.context_trace
    # for the assistant turn so "what did Auto know?" is answerable. None when
    # the build predates this or produced no trace.
    context_trace: Optional[Dict[str, Any]] = None


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

    Uses ContextService (PRD-80) for unified prompt building, memory
    retrieval, and tool loading.  Intent classification remains separate
    for tool_choice and response routing decisions.
    """

    # How often to refresh memory (in messages)
    MEMORY_REFRESH_INTERVAL = 5

    def __init__(
        self,
        workspace_id: str,
        agent_id: Optional[int] = None,
        agent_name: Optional[str] = None,
        widget_mode: bool = False,
        db_session: Any = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            workspace_id: Workspace ID for memory scoping
            agent_id: Agent ID for memory scoping
            agent_name: Agent name for personalization
            widget_mode: When True, restrict memory to agent-only
            db_session: Optional DB session for ContextService
        """
        self.workspace_id = workspace_id
        self.agent_id = agent_id
        self.agent_name = agent_name or "Automatos"
        self.widget_mode = widget_mode
        self._db_session = db_session

        # Components
        self.classifier = get_intent_classifier()

        # SmartMemoryManager owns the L3/L2 write fan-out for a chat turn
        # (distilled facts + verbatim transcript). UnifiedMemoryService is the
        # shared singleton; we hold a reference for L1 session updates.
        from .smart_memory import get_smart_memory_manager
        self.memory_manager = get_smart_memory_manager()

        from modules.memory.unified_memory_service import get_unified_memory_service
        self._unified_memory = get_unified_memory_service()

        # Conversation state
        self.state = ConversationState()

        logger.info(f"[Orchestrator] Initialized for workspace={workspace_id}, agent={agent_id}")

    async def prepare_request(
        self,
        messages: List[Dict[str, Any]],
        available_tools: List[Dict[str, Any]],
        chat_id: Optional[str] = None,
        complexity_assessment: Optional[Any] = None,
        attachment_ids: Optional[List[str]] = None,
        model_id: Optional[str] = None,
    ) -> OrchestratedRequest:
        """
        Prepare a chat request for the LLM.

        Uses ContextService (PRD-80) for unified prompt building, memory
        retrieval, and tool loading.  Intent classification stays separate
        for tool_choice and response routing decisions.

        Args:
            messages: Conversation messages
            available_tools: All tools available to this agent (kept for
                backward compatibility — ContextService loads tools internally)
            chat_id: Optional chat session ID
            complexity_assessment: Optional PRD-68 AutoBrain assessment
            attachment_ids: PRD-127 ephemeral attachments to resolve
            model_id: PRD-127 model identifier for vision capability check

        Returns:
            OrchestratedRequest ready for LLM
        """
        start_time = time.time()

        # Extract latest user message
        latest_query = self._extract_latest_user_message(messages)
        logger.debug(f"[Orchestrator] Processing: {latest_query[:50]}...")

        # ─── 1. Classify Intent (KEPT — needed for response routing) ───
        intent_result = self.classifier.classify(latest_query, messages)
        self.state.last_intent = intent_result.primary_intent

        logger.info(
            f"[Orchestrator] Intent: {intent_result.primary_intent.value} "
            f"(tools: {intent_result.requires_tools}, memory: {intent_result.requires_memory})"
        )

        # ─── 2. Memory decision (KEPT — chatbot-specific optimisation) ───
        _wants_memory = self._should_fetch_memory(intent_result)
        if complexity_assessment and not getattr(complexity_assessment, "needs_memory", True):
            # Don't let complexity override when intent explicitly requires memory
            if intent_result.requires_memory:
                logger.info(
                    "[Orchestrator] Memory KEPT — intent requires_memory=True overrides ComplexityAssessment (%s)",
                    getattr(complexity_assessment, "complexity", "?"),
                )
            else:
                _wants_memory = False
                logger.info(
                    "[Orchestrator] Memory SKIPPED by ComplexityAssessment (%s)",
                    getattr(complexity_assessment, "complexity", "?"),
                )

        # Extract tool hints from complexity assessment
        _tool_hints = None
        if complexity_assessment and getattr(complexity_assessment, "tool_hints", None):
            _tool_hints = complexity_assessment.tool_hints
            logger.info(f"[Orchestrator] Tools ENABLED by tool_hints={_tool_hints}")

        # ─── 3. Build context via ContextService (PRD-80) ───
        # Replaces: memory retrieval, tool routing, prompt building,
        # daily log injection, platform summary injection, datetime injection
        from modules.context import ContextService, ContextMode

        agent = self._load_agent()

        context = await ContextService(self._db_session).build_context(
            mode=ContextMode.CHATBOT,
            agent=agent,
            workspace_id=self.workspace_id,
            messages=messages,
            widget_mode=self.widget_mode,
            complexity_assessment=complexity_assessment,
            tool_hints=_tool_hints,
            attachment_ids=attachment_ids,  # PRD-127
            model_id=model_id,  # PRD-127
            # Kwargs passed through to sections:
            intent_result=intent_result,
            skip_memory=not _wants_memory,
            chat_id=chat_id,
            query=latest_query,
            agent_name=self.agent_name,
            user_name=self.state.user_name,
        )

        # ─── 4. Update conversation state ───
        if _wants_memory:
            self.state.messages_since_memory_fetch = 0
            self.state.memory_fetched_at = time.time()
            if context.user_name:
                self.state.user_name = context.user_name
        else:
            self.state.messages_since_memory_fetch += 1

        # ─── 5. Build compat MemoryResult for SSE events / CTO override ───
        self._last_memory_result = self._build_compat_memory_result(context)

        preparation_time = (time.time() - start_time) * 1000

        return OrchestratedRequest(
            system_prompt=context.system_prompt,
            messages=context.messages,
            tools=context.tools,
            tool_choice=context.tool_choice,
            memory_context=context.memory_context,
            user_name=context.user_name or self.state.user_name,
            intent=intent_result.primary_intent,
            intent_confidence=intent_result.confidence,
            context_trace=context.to_assembly_trace(),  # PRD-201 S1
            requires_tools=bool(context.tools) or intent_result.requires_tools,
            requires_memory=intent_result.requires_memory,
            preparation_time_ms=preparation_time,
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

    def _load_agent(self) -> Any:
        """Load the full agent record from DB for ContextService.

        Falls back to a SimpleNamespace with basic fields if DB is
        unavailable or the agent is not found.
        """
        if self._db_session and self.agent_id:
            try:
                from modules.agents.queries import get_agent_with_context
                agent = get_agent_with_context(self._db_session, self.agent_id)
                if agent:
                    return agent
            except Exception:
                logger.warning(
                    "[Orchestrator] Failed to load agent %s from DB — using fallback",
                    self.agent_id,
                    exc_info=True,
                )

        # Fallback: minimal pseudo-agent
        return types.SimpleNamespace(
            id=self.agent_id,
            name=self.agent_name,
            agent_type="chatbot",
            description=None,
            skills=[],
        )

    def _build_compat_memory_result(self, context: Any) -> Any:
        """Build a MemoryResult-compatible object for SSE events and CTO override.

        service.py accesses ``self._last_memory_result`` to:
        1. Emit 'memory_retrieved' SSE events (reads .formatted_context)
        2. Build CTO override prompts (reads .memories list)
        """
        from .smart_memory import MemoryResult, UserContext

        # Use raw memories stashed by MemorySection if available
        raw_memories = []
        formatted_context = ""

        if hasattr(context, "memory_context") and context.memory_context:
            formatted_context = context.memory_context

        # ContextService stashes raw memory dicts in kwargs via MemorySection
        # We access them through the formatted_context as fallback
        if formatted_context and not raw_memories:
            # Parse bullet points from formatted memory text
            for line in formatted_context.split("\n"):
                line = line.strip()
                if line.startswith("- "):
                    raw_memories.append({"memory": line[2:]})

        return MemoryResult(
            memories=raw_memories,
            user_context=UserContext(
                name=context.user_name or self.state.user_name
            ),
            formatted_context=formatted_context,
            retrieval_time_ms=0.0,
        )

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

    async def store_exchange(
        self,
        user_message: str,
        assistant_response: str,
        chat_id: Optional[str] = None,
        subject_id: Optional[str] = None,
    ) -> bool:
        """
        Store a conversation exchange in memory.

        Call this after the LLM responds to save the exchange. All writes are
        scheduled fire-and-forget so the response is never held open on memory
        persistence.

        Args:
            user_message: The user's message
            assistant_response: The assistant's response
            chat_id: Optional chat session ID

        Returns:
            True once the writes have been scheduled (not a persistence ack).
        """
        # All memory writes are fire-and-forget. Mem0 fact extraction runs a
        # synchronous server-side LLM call (seconds), and awaiting it here blocks
        # the streaming generator — and therefore the HTTP connection — long
        # after the user has seen the full response. None of these writes feed
        # back into the current turn, so schedule them and return immediately.

        # L3 distilled facts (two-tier global/agent Mem0) + L2 verbatim
        # transcript — both fan out from store_conversation. Per W3-S7 / G12
        # (write-once-per-layer) the L2 transcript IS the L2 write for a chat
        # turn; the older direct ``_unified_memory.store_exchange`` spawn was
        # a duplicate L2 row (content_type='exchange') that this collapse
        # retires.
        _spawn_background(
            self.memory_manager.store_conversation(
                workspace_id=self.workspace_id,
                agent_id=self.agent_id,
                user_message=user_message,
                assistant_response=assistant_response,
                chat_id=chat_id,
                widget_mode=self.widget_mode,
                subject_id=subject_id,
            ),
            label="store_conversation",
        )

        # Daily activity log entry
        _spawn_background(
            self.memory_manager.store_daily_summary(
                workspace_id=self.workspace_id,
                user_message=user_message,
                assistant_response=assistant_response,
                agent_id=self.agent_id,
            ),
            label="store_daily_summary",
        )

        # L1: session in Redis
        if chat_id:
            _spawn_background(
                self._unified_memory.update_session(
                    workspace_id=self.workspace_id,
                    conversation_id=chat_id,
                    user_msg=user_message,
                    assistant_msg=assistant_response,
                ),
                label="l1_update_session",
            )

        # Writes are scheduled; the turn does not wait on their outcome.
        return True

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
