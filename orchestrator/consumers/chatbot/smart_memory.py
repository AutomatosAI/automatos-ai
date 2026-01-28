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

    def _get_global_user_id(self, workspace_id: str) -> str:
        """
        Get user ID for GLOBAL memories (shared across all agents).

        Format: ws_{workspace_id}

        Global memories include:
        - User's name, location, job
        - General preferences
        - Personal facts about the user
        """
        ws = str(workspace_id) if workspace_id else "default"
        return f"ws_{ws}"

    def _get_agent_user_id(self, workspace_id: str, agent_id: Optional[int]) -> str:
        """
        Get user ID for AGENT-SPECIFIC memories.

        Format: ws_{workspace_id}_agent_{agent_id}

        Agent-specific memories include:
        - Tool preferences (Slack channels, email contacts)
        - Workflow patterns for this agent's domain
        - Agent-specific preferences
        """
        ws = str(workspace_id) if workspace_id else "default"
        ag = str(agent_id) if agent_id else "default"
        return f"ws_{ws}_agent_{ag}"

    def _classify_memory_tier(self, user_message: str, assistant_response: str) -> str:
        """
        Classify where a memory should be stored.

        Returns: "global", "agent", or "both"

        Classification rules:
        - Personal facts (name, location, job, general info) → global
        - Tool/workflow-related (Slack, email patterns, contacts for tools) → agent
        - Preferences → both (useful everywhere)
        """
        combined = (user_message + " " + assistant_response).lower()

        # Tool/workflow-specific keywords → agent-specific memory
        # These are things specific to how tools are used
        tool_keywords = [
            # Communication tools
            "slack", "channel", "#", "dm", "message to",
            "gmail", "email to", "send email", "cc", "bcc", "forward",
            # Dev tools
            "github", "repository", "repo", "branch", "pr", "pull request",
            "jira", "ticket", "issue",
            # Data tools
            "database", "table", "query", "sql", "api", "endpoint",
            # File tools
            "spreadsheet", "document", "file", "folder", "drive", "upload",
            # Workflow patterns
            "when i ask", "for this agent", "in this context"
        ]

        # Personal/global keywords → global memory (who the user IS)
        personal_keywords = [
            "my name", "i am", "i'm", "call me",
            "i work at", "i work for", "my job", "my role",
            "i live", "from ireland", "from portugal", "i'm from",
            "born", "age", "founder", "ceo", "coo", "manager",
            "my company", "my team", "my organization"
        ]

        # Preference keywords → both tiers (useful everywhere)
        preference_keywords = [
            "prefer", "favorite", "like to", "don't like",
            "usually", "my style", "i want", "i need"
        ]

        # Strong agent indicators (override others)
        strong_agent_keywords = [
            "always cc", "default channel", "send to", "post to",
            "my slack", "my email", "contact", "@"
        ]

        has_tool = any(kw in combined for kw in tool_keywords)
        has_personal = any(kw in combined for kw in personal_keywords)
        has_preference = any(kw in combined for kw in preference_keywords)
        has_strong_agent = any(kw in combined for kw in strong_agent_keywords)

        # Strong agent indicators take precedence
        if has_strong_agent:
            return "agent"
        elif has_preference and has_tool:
            return "agent"  # Tool preference → agent specific
        elif has_preference:
            return "both"  # General preference → both
        elif has_tool and not has_personal:
            return "agent"  # Tool-specific only
        else:
            return "global"  # Default to global (personal facts)

    def _get_cache_key(self, workspace_id: str, agent_id: Optional[int], query: str) -> str:
        """Create cache key for memory lookups (includes agent for agent-specific cache)."""
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
                # TWO-TIER MEMORY RETRIEVAL
                # 1. Global memories (user facts shared across all agents)
                # 2. Agent-specific memories (tool preferences, workflow patterns)

                global_user_id = self._get_global_user_id(workspace_id)
                agent_user_id = self._get_agent_user_id(workspace_id, agent_id)

                logger.info(f"[SmartMemory] Two-tier search: global={global_user_id}, agent={agent_user_id}")

                loop = asyncio.get_event_loop()

                # Fetch both tiers in parallel
                global_task = loop.run_in_executor(
                    None,
                    lambda: self.mem0_client.search(query=query, user_id=global_user_id, limit=limit)
                )
                agent_task = loop.run_in_executor(
                    None,
                    lambda: self.mem0_client.search(query=query, user_id=agent_user_id, limit=limit)
                )

                global_memories, agent_memories = await asyncio.gather(global_task, agent_task)

                # Merge: global first (who the user is), then agent-specific (how they use this agent)
                global_memories = global_memories or []
                agent_memories = agent_memories or []

                logger.info(f"[SmartMemory] ✅ Found {len(global_memories)} global + {len(agent_memories)} agent-specific memories")

                # Combine with global first, agent-specific second
                # Mark agent-specific memories so we can format them differently
                for mem in agent_memories:
                    mem["_tier"] = "agent"

                memories = global_memories + agent_memories

                if memories:
                    for i, mem in enumerate(memories[:3]):
                        tier = mem.get("_tier", "global")
                        logger.info(f"[SmartMemory]   Memory {i+1} [{tier}]: {mem.get('memory', '')[:80]}...")

                    # Extract user context from memories
                    user_context = self._extract_user_context(memories)
                else:
                    logger.info(f"[SmartMemory] ❌ No memories found for user")

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

        Two-tier formatting:
        1. Global memories (who the user is)
        2. Agent-specific memories (how they use this agent's tools)
        """
        if not memories and not user_context.name:
            return ""

        lines = []

        # User identity first (most important)
        if user_context.name:
            lines.append(f"User's name: {user_context.name}")

        # Separate global and agent-specific memories
        global_memories = [m for m in memories if m.get("_tier") != "agent"]
        agent_memories = [m for m in memories if m.get("_tier") == "agent"]

        # Global facts about the user
        if global_memories:
            lines.append("\nAbout this user:")
            for mem in global_memories[:5]:
                memory_text = mem.get("memory", "")
                if memory_text and len(memory_text) > 5:
                    if len(memory_text) > 150:
                        memory_text = memory_text[:150] + "..."
                    lines.append(f"  - {memory_text}")

        # Agent-specific context
        if agent_memories:
            lines.append("\nWith this agent specifically:")
            for mem in agent_memories[:4]:
                memory_text = mem.get("memory", "")
                if memory_text and len(memory_text) > 5:
                    if len(memory_text) > 150:
                        memory_text = memory_text[:150] + "..."
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

            # TWO-TIER MEMORY STORAGE
            # Classify where this memory should be stored
            tier = self._classify_memory_tier(user_message, assistant_response)
            logger.info(f"[SmartMemory] Memory classified as: {tier}")

            messages = [
                {"role": "user", "content": user_message[:500]},
                {"role": "assistant", "content": assistant_response[:500]}
            ]

            base_metadata = {
                "chat_id": chat_id,
                "timestamp": datetime.utcnow().isoformat(),
                "workspace_id": workspace_id,
                "agent_id": agent_id
            }

            loop = asyncio.get_event_loop()
            results = []

            # Store in global tier (personal facts, shared across all agents)
            if tier in ("global", "both"):
                global_user_id = self._get_global_user_id(workspace_id)
                global_metadata = {**base_metadata, "tier": "global"}

                global_result = await loop.run_in_executor(
                    None,
                    lambda: self.mem0_client.add(
                        messages=messages,
                        user_id=global_user_id,
                        metadata=global_metadata
                    )
                )
                results.append(("global", global_result))
                logger.info(f"[SmartMemory] Global storage result: {global_result}")

            # Store in agent-specific tier (tool preferences, workflow patterns)
            if tier in ("agent", "both"):
                agent_user_id = self._get_agent_user_id(workspace_id, agent_id)
                agent_metadata = {**base_metadata, "tier": "agent"}

                agent_result = await loop.run_in_executor(
                    None,
                    lambda: self.mem0_client.add(
                        messages=messages,
                        user_id=agent_user_id,
                        metadata=agent_metadata
                    )
                )
                results.append(("agent", agent_result))
                logger.info(f"[SmartMemory] Agent-specific storage result: {agent_result}")

            # Check if any storage succeeded
            success = any(r[1] and not r[1].get("error") for r in results)

            if success:
                tiers_stored = [r[0] for r in results if r[1] and not r[1].get("error")]
                logger.info(f"[SmartMemory] ✅ Stored in tiers: {tiers_stored}")
                # Invalidate cache
                self._invalidate_cache(workspace_id, agent_id)
                return True
            else:
                errors = [f"{r[0]}: {r[1].get('error') if r[1] else 'None'}" for r in results]
                logger.warning(f"[SmartMemory] ❌ Storage failed: {errors}")
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
