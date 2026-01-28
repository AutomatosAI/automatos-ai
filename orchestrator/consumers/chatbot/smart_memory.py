"""
Smart Memory Manager
====================

Intelligent memory management for the Automatos assistant.

Features:
- Proper Mem0 integration with correct user scoping
- Smart retrieval based on intent classification
- Extracts user facts (name, preferences) from memories
- Caches recent memories for fast access
- Background storage to not block responses
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class UserContext:
    """Extracted user context from memories."""
    name: Optional[str] = None
    preferences: List[str] = None
    facts: List[str] = None
    recent_topics: List[str] = None
    last_interaction: Optional[datetime] = None

    def __post_init__(self):
        self.preferences = self.preferences or []
        self.facts = self.facts or []
        self.recent_topics = self.recent_topics or []


@dataclass
class MemoryResult:
    """Result of memory retrieval."""
    memories: List[Dict[str, Any]]
    user_context: UserContext
    formatted_context: str
    retrieval_time_ms: float


class SmartMemoryManager:
    """
    Intelligent memory management for chat.

    Key improvements over previous system:
    1. Consistent user_id scoping (workspace + agent)
    2. Smart retrieval based on intent
    3. User fact extraction (name, preferences)
    4. Memory caching for performance
    5. Background storage to not block responses
    """

    def __init__(self):
        self._mem0_client = None
        self._cache: Dict[str, Tuple[float, MemoryResult]] = {}
        self._cache_ttl = 120  # 2 minutes
        self._storage_queue: List[Tuple] = []
        self._storage_task = None

    @property
    def mem0_client(self):
        """Lazy initialization of Mem0 client."""
        if self._mem0_client is None:
            try:
                from modules.memory.integrations.mem0_client import Mem0Client
                self._mem0_client = Mem0Client()
                logger.info("[SmartMemory] Mem0 client initialized")
            except Exception as e:
                logger.warning(f"[SmartMemory] Could not initialize Mem0: {e}")
        return self._mem0_client

    def _get_user_id(self, workspace_id: str, agent_id: Optional[int] = None) -> str:
        """
        Create consistent user ID for Mem0.

        Format: ws_{workspace_id}_agent_{agent_id}

        IMPORTANT: This must be consistent between storage and retrieval!
        """
        ws = str(workspace_id) if workspace_id else "default"
        ag = str(agent_id) if agent_id else "global"
        user_id = f"ws_{ws}_agent_{ag}"
        logger.debug(f"[SmartMemory] Using user_id: {user_id}")
        return user_id

    def _get_cache_key(self, workspace_id: str, agent_id: Optional[int], query: str) -> str:
        """Create cache key for memory lookups."""
        return f"{workspace_id}:{agent_id}:{query[:50]}"

    async def retrieve_memories(
        self,
        workspace_id: str,
        agent_id: Optional[int],
        query: str,
        limit: int = 8
    ) -> MemoryResult:
        """
        Retrieve relevant memories for a query.

        Args:
            workspace_id: Workspace ID for scoping
            agent_id: Agent ID for scoping
            query: The user's query to match against
            limit: Maximum memories to retrieve

        Returns:
            MemoryResult with memories and extracted context
        """
        start_time = time.time()

        # Check cache first
        cache_key = self._get_cache_key(workspace_id, agent_id, query)
        cached = self._cache.get(cache_key)
        if cached and (time.time() - cached[0]) < self._cache_ttl:
            logger.debug("[SmartMemory] Using cached memory result")
            return cached[1]

        memories = []
        user_context = UserContext()

        try:
            if self.mem0_client:
                user_id = self._get_user_id(workspace_id, agent_id)
                logger.debug(f"[SmartMemory] Searching memories for {user_id}: {query[:50]}...")

                # Run search (Mem0 client is sync, so run in executor)
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None,
                    lambda: self.mem0_client.search(query=query, user_id=user_id, limit=limit)
                )

                if results:
                    memories = results
                    logger.info(f"[SmartMemory] Found {len(memories)} memories")

                    # Extract user context from memories
                    user_context = self._extract_user_context(memories)
                else:
                    logger.debug("[SmartMemory] No memories found")

        except Exception as e:
            logger.warning(f"[SmartMemory] Retrieval failed: {e}")

        # Format memories for LLM context
        formatted = self._format_memories_for_llm(memories, user_context)

        retrieval_time = (time.time() - start_time) * 1000

        result = MemoryResult(
            memories=memories,
            user_context=user_context,
            formatted_context=formatted,
            retrieval_time_ms=retrieval_time
        )

        # Cache result
        self._cache[cache_key] = (time.time(), result)

        return result

    def _extract_user_context(self, memories: List[Dict]) -> UserContext:
        """
        Extract user facts from memories.

        Looks for:
        - Name mentions ("Name is X", "I'm X", "My name is X")
        - Preferences ("prefer", "like", "favorite")
        - Facts about the user
        """
        context = UserContext()
        name_patterns = ["name is", "i'm", "my name is", "call me"]

        for mem in memories:
            content = mem.get("memory", "").lower() if isinstance(mem.get("memory"), str) else ""

            # Look for name
            if not context.name:
                for pattern in name_patterns:
                    if pattern in content:
                        # Extract name after pattern
                        idx = content.find(pattern) + len(pattern)
                        words = content[idx:].strip().split()
                        if words:
                            # Capitalize the name
                            potential_name = words[0].strip(".,!?")
                            if len(potential_name) > 1:
                                context.name = potential_name.capitalize()
                                break

            # Look for preferences
            if "prefer" in content or "like" in content or "favorite" in content:
                context.preferences.append(mem.get("memory", ""))

            # Add as general fact
            if mem.get("memory"):
                context.facts.append(mem.get("memory"))

        return context

    def _format_memories_for_llm(
        self,
        memories: List[Dict],
        user_context: UserContext
    ) -> str:
        """
        Format memories into a string for LLM context injection.
        """
        if not memories and not user_context.name:
            return ""

        lines = []

        # User identity first (most important)
        if user_context.name:
            lines.append(f"User's name: {user_context.name}")

        # User facts/preferences
        if user_context.preferences:
            lines.append(f"User preferences: {'; '.join(user_context.preferences[:3])}")

        # Recent conversation memories
        if memories:
            lines.append("\nRecent interactions:")
            for mem in memories[:6]:
                memory_text = mem.get("memory", "")
                if memory_text and len(memory_text) > 5:
                    # Truncate long memories
                    if len(memory_text) > 200:
                        memory_text = memory_text[:200] + "..."
                    lines.append(f"  - {memory_text}")

        return "\n".join(lines)

    async def store_conversation(
        self,
        workspace_id: str,
        agent_id: Optional[int],
        user_message: str,
        assistant_response: str,
        chat_id: Optional[str] = None
    ) -> bool:
        """
        Store a conversation exchange in memory.

        This runs asynchronously to not block the response.

        Args:
            workspace_id: Workspace ID
            agent_id: Agent ID
            user_message: The user's message
            assistant_response: The assistant's response
            chat_id: Optional chat session ID

        Returns:
            Success status
        """
        if not user_message or not assistant_response:
            return False

        # Skip storing very short exchanges (greetings, etc.)
        if len(user_message) < 10 and len(assistant_response) < 50:
            logger.debug("[SmartMemory] Skipping storage for short exchange")
            return False

        try:
            if not self.mem0_client:
                logger.warning("[SmartMemory] No Mem0 client available for storage")
                return False

            user_id = self._get_user_id(workspace_id, agent_id)
            logger.debug(f"[SmartMemory] Storing conversation for {user_id}")

            messages = [
                {"role": "user", "content": user_message[:500]},
                {"role": "assistant", "content": assistant_response[:500]}
            ]

            metadata = {
                "chat_id": chat_id,
                "timestamp": datetime.utcnow().isoformat(),
                "workspace_id": workspace_id,
                "agent_id": agent_id
            }

            # Run storage in background
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self.mem0_client.add(
                    messages=messages,
                    user_id=user_id,
                    metadata=metadata
                )
            )

            if result and not result.get("error"):
                logger.info(f"[SmartMemory] Stored conversation successfully")
                # Invalidate cache for this user
                self._invalidate_cache(workspace_id, agent_id)
                return True
            else:
                logger.warning(f"[SmartMemory] Storage returned error: {result}")
                return False

        except Exception as e:
            logger.error(f"[SmartMemory] Storage failed: {e}")
            return False

    def _invalidate_cache(self, workspace_id: str, agent_id: Optional[int]):
        """Invalidate cached memories for a user."""
        prefix = f"{workspace_id}:{agent_id}:"
        keys_to_remove = [k for k in self._cache.keys() if k.startswith(prefix)]
        for key in keys_to_remove:
            del self._cache[key]

    def get_user_name(self, memories_result: MemoryResult) -> Optional[str]:
        """Quick helper to get user's name from memory result."""
        return memories_result.user_context.name if memories_result else None

    def clear_cache(self):
        """Clear all cached memories."""
        self._cache.clear()


# Module-level singleton
_memory_manager = None

def get_smart_memory_manager() -> SmartMemoryManager:
    """Get the global smart memory manager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = SmartMemoryManager()
    return _memory_manager
