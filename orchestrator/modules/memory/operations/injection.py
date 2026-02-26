"""
Memory Injector - Mem0 Memory Retrieval and Storage for Chat
=============================================================

PURE MEM0 IMPLEMENTATION - No fallbacks to old memory system.

Handles:
- Retrieving relevant memories from Mem0
- Storing conversation memories to Mem0
- Smart memory injection into LLM context
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MemoryInjector:
    """
    Handles memory retrieval and storage for the chat service.

    PURE MEM0 - Uses SmartMemoryManager which connects only to Mem0.
    No fallbacks to old HierarchicalMemorySystem or database queries.
    """

    def __init__(self):
        self._smart_memory = None
        self._intent_classifier = None

    @property
    def smart_memory(self):
        """Lazy load SmartMemoryManager."""
        if self._smart_memory is None:
            from consumers.chatbot.smart_memory import get_smart_memory_manager
            self._smart_memory = get_smart_memory_manager()
        return self._smart_memory

    @property
    def intent_classifier(self):
        """Lazy load intent classifier."""
        if self._intent_classifier is None:
            from consumers.chatbot.intent_classifier import get_intent_classifier
            self._intent_classifier = get_intent_classifier()
        return self._intent_classifier

    async def should_retrieve_memories(self, query: str, chat_id: str, agent_id: Optional[int] = None) -> bool:
        """
        Intelligent decision on whether to retrieve memories for a given query.
        Uses the new SmartIntentClassifier.
        """
        if not query or len(query) < 5:
            return False

        try:
            result = self.intent_classifier.classify(query)
            logger.debug(f"[Memory] Intent: {result.primary_intent.value} (requires_memory: {result.requires_memory})")
            return result.requires_memory
        except Exception as e:
            logger.warning(f"[Memory] Classification failed, using heuristics: {e}")

        # Simple heuristics fallback
        query_lower = query.lower().strip()

        # Skip greetings
        greetings = {"hi", "hello", "hey", "thanks", "thank you", "bye", "ok", "okay"}
        if query_lower in greetings:
            return False

        # Check for memory references
        memory_indicators = {"remember", "my name", "earlier", "before", "last time", "you said", "we discussed"}
        if any(ind in query_lower for ind in memory_indicators):
            return True

        # Default: retrieve for longer queries
        return len(query) > 20

    async def retrieve_relevant_memories(
        self,
        chat_id: str,
        query: str,
        workspace_id: Optional[str] = None,
        agent_id: Optional[int] = None,
        widget_mode: bool = False
    ) -> Optional[str]:
        """
        Retrieve relevant memories from Mem0 for injection into LLM context.

        PURE MEM0 - No fallbacks.

        Args:
            chat_id: Current chat session ID
            query: The user's query
            workspace_id: Workspace ID for scoping
            agent_id: Agent ID for scoping

        Returns:
            Formatted memory context string, or None
        """
        if not query or len(query) < 3:
            return None

        # Check if we should retrieve
        should_retrieve = await self.should_retrieve_memories(query, chat_id, agent_id)
        if not should_retrieve:
            logger.debug(f"[Memory] Skipping retrieval for query: '{query[:50]}...'")
            return None

        try:
            # Use SmartMemoryManager (Mem0 only)
            result = await self.smart_memory.retrieve_memories(
                workspace_id=str(workspace_id) if workspace_id else "default",
                agent_id=agent_id,
                query=query,
                limit=10,
                widget_mode=widget_mode
            )

            if not result or not result.memories:
                logger.debug("[Memory] Mem0 returned no memories")
                return "No previous conversation memories found."

            logger.info(f"[Memory] Retrieved {len(result.memories)} memories from Mem0")

            # Use the pre-formatted context from SmartMemoryManager
            if result.formatted_context:
                return result.formatted_context

            # Manual formatting as fallback
            memory_lines = []

            # Add user name if found
            if result.user_context and result.user_context.name:
                memory_lines.append(f"User's name: {result.user_context.name}")

            # Add memories
            for mem in result.memories[:8]:
                memory_text = mem.get("memory", "")
                if memory_text:
                    memory_lines.append(f"- {memory_text[:200]}")

            return "\n".join(memory_lines) if memory_lines else "No previous conversation memories found."

        except Exception as e:
            logger.error(f"[Memory] Mem0 retrieval failed: {e}")
            return None

    async def store_conversation_memory(
        self,
        chat_id: str,
        user_message: str,
        assistant_response: str,
        workspace_id: Optional[str] = None,
        agent_id: Optional[int] = None,
        widget_mode: bool = False
    ):
        """
        Store conversation as memory in Mem0.

        PURE MEM0 - No fallbacks.

        Args:
            chat_id: Chat session ID
            user_message: The user's message
            assistant_response: The assistant's response
            workspace_id: Workspace to scope memory to
            agent_id: Agent to scope memory to
        """
        try:
            logger.debug(f"[Memory] Storing to Mem0: {user_message[:50]}...")

            success = await self.smart_memory.store_conversation(
                workspace_id=str(workspace_id) if workspace_id else "default",
                agent_id=agent_id,
                user_message=user_message,
                assistant_response=assistant_response,
                chat_id=chat_id,
                widget_mode=widget_mode
            )

            if success:
                logger.info("[Memory] Stored conversation in Mem0")
            else:
                logger.warning("[Memory] Mem0 storage returned False")

        except Exception as e:
            logger.error(f"[Memory] Mem0 storage FAILED: {e}")

    def build_memory_injection_message(self, memory_context: str) -> Dict[str, str]:
        """
        Build the system message for memory injection.

        Args:
            memory_context: Formatted memory context string

        Returns:
            System message dict for LLM
        """
        return {
            "role": "system",
            "content": f"""## What I Remember About You

{memory_context}

**How to use these memories:**
- Use this context naturally - if you know their name, use it warmly!
- Reference past conversations when relevant ("As we discussed before...")
- Don't pretend you don't have memory - you clearly do!
- If they ask "do you remember me?" - yes, check above!
- Don't search databases for user info - trust what's here
- If memory is empty, that's okay - just mention we haven't chatted before"""
        }


# Module-level instances
_memory_injector = None
_memory_system = None


def get_memory_injector() -> MemoryInjector:
    """Get or create the global MemoryInjector instance.

    .. deprecated::
        Use ``consumers.chatbot.integration.SmartChatIntegration`` instead.
        This legacy injector will be removed in a future release.
    """
    import warnings
    warnings.warn(
        "get_memory_injector() is deprecated. Use SmartChatIntegration from "
        "consumers.chatbot.integration instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    global _memory_injector
    if _memory_injector is None:
        _memory_injector = MemoryInjector()
    return _memory_injector


def get_memory_system():
    """
    Get the memory system instance (Mem0 only).

    This function exists for backward compatibility.
    Returns Mem0MemorySystem - no fallbacks.
    """
    global _memory_system
    if _memory_system is None:
        try:
            from modules.memory.storage.mem0_system import Mem0MemorySystem
            _memory_system = Mem0MemorySystem()
            logger.info("[Memory] Using Mem0MemorySystem (pure Mem0)")
        except Exception as e:
            logger.error(f"[Memory] Failed to initialize Mem0: {e}")
            return None
    return _memory_system
