"""
Unified Memory Service
======================

Single entry point for all memory operations across all consumers.
Replaces 12 scattered Mem0Client instances with ONE shared service.

5-Layer Memory Stack:
  L0: Focus (context window — no code needed)
  L1: Working Memory (Redis session cache)
  L2: Short-term Memory (Postgres + time-based decay)
  L3: Long-term Memory (Mem0 with fact extraction)
  L4: Organizational Knowledge (RAG/NL2SQL — tools, not pre-fetched)

Usage:
    from modules.memory.unified_memory_service import get_unified_memory_service

    service = get_unified_memory_service()
    await service.store_long_term(workspace_id, content)
    results = await service.search_long_term(workspace_id, query)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MemoryNamespace — builds standardised user_id strings for Mem0
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemoryNamespace:
    """
    Builds standardised, scoped user_id strings for Mem0 and Redis keys.

    All memory consumers MUST use this helper instead of raw string
    concatenation. This prevents the 5+ user_id format inconsistencies
    documented in PRD-79 § 1.2.
    """

    workspace_id: str

    # --- L3 Long-term (Mem0) namespaces ---

    def workspace(self) -> str:
        """Workspace-wide facts (L3 global)."""
        return f"mem:{self.workspace_id}"

    def agent(self, agent_id: int) -> str:
        """Agent-specific memories (L3 per-agent)."""
        return f"mem:{self.workspace_id}:agent:{agent_id}"

    def recipe(self, recipe_id: int) -> str:
        """Recipe learnings (L3 per-recipe)."""
        return f"mem:{self.workspace_id}:recipe:{recipe_id}"

    def recipe_agent(self, recipe_id: int, agent_id: int) -> str:
        """Per-agent step memories within a recipe (L3)."""
        return f"mem:{self.workspace_id}:recipe:{recipe_id}:agent:{agent_id}"

    def daily(self) -> str:
        """Daily activity logs (L2)."""
        return f"mem:{self.workspace_id}:daily"

    # --- L1 Session (Redis) namespaces ---

    def session(self, conversation_id: str) -> str:
        """Session cache key (L1 Redis)."""
        return f"mem:session:{self.workspace_id}:{conversation_id}"

    # --- L3 Cache (Redis) namespaces ---

    def cache_key(self, agent_id: Optional[int], query_hash: str) -> str:
        """Cache key for L3 Mem0 search results cached in Redis."""
        scope = str(agent_id) if agent_id is not None else "global"
        return f"mem:cache:{self.workspace_id}:{scope}:{query_hash}"

    def cache_pattern(self) -> str:
        """Pattern to match all cache keys for this workspace (for invalidation)."""
        return f"mem:cache:{self.workspace_id}:*"

    # --- Awareness cache ---

    def awareness(self) -> str:
        """Knowledge awareness text cache key (Redis)."""
        return f"mem:awareness:{self.workspace_id}"

    # --- User profile cache ---

    def profile(self) -> str:
        """User profile cache key (Redis)."""
        return f"mem:profile:{self.workspace_id}"

    # --- Resolve user_id for Mem0 calls ---

    def resolve(self, agent_id: Optional[int] = None) -> str:
        """
        Resolve the correct Mem0 user_id for a given scope.

        If agent_id is provided, returns the agent-scoped namespace.
        Otherwise, returns the workspace-wide namespace.
        """
        if agent_id is not None:
            return self.agent(agent_id)
        return self.workspace()


# ---------------------------------------------------------------------------
# UnifiedMemoryService — singleton
# ---------------------------------------------------------------------------

class UnifiedMemoryService:
    """
    Single entry point for all memory operations across all consumers.

    Holds ONE shared Mem0Client, ONE Redis client.
    DB sessions are acquired per-request from the session pool — never stored
    on the singleton to prevent cross-tenant data leaks.
    """

    _instance: Optional["UnifiedMemoryService"] = None

    @classmethod
    def get_instance(cls) -> "UnifiedMemoryService":
        """Return the singleton instance, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing only)."""
        cls._instance = None

    def __init__(self) -> None:
        # Shared Mem0Client (L3 long-term)
        from modules.memory.integrations.mem0_client import Mem0Client

        self._mem0 = Mem0Client()

        # Shared Redis client (L1 session + caching)
        from core.redis.client import get_redis_client

        self._redis_client_getter = get_redis_client
        logger.info("[UnifiedMemoryService] Initialised with shared Mem0Client and Redis")

    @property
    def is_mem0_configured(self) -> bool:
        """Check if Mem0 backend is configured (has a valid API URL)."""
        return bool(getattr(self._mem0, "api_url", None))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_redis(self):
        """
        Get the Redis client instance.

        Returns None if Redis is not configured — callers MUST handle this
        gracefully (Redis failures must never break chat).
        """
        return self._redis_client_getter()

    @staticmethod
    def namespace(workspace_id: str) -> MemoryNamespace:
        """Create a MemoryNamespace for the given workspace."""
        return MemoryNamespace(workspace_id=str(workspace_id))

    # ------------------------------------------------------------------
    # L3: Long-term Memory (Mem0)
    # ------------------------------------------------------------------

    async def store_long_term(
        self,
        workspace_id: str,
        content: str,
        agent_id: Optional[int] = None,
        category: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Store content in L3 long-term memory via Mem0 with fact extraction.

        Args:
            workspace_id: Workspace scope.
            content: Text to store (Mem0 extracts facts automatically).
            agent_id: Optional agent scope.
            category: Optional category tag.
            metadata: Optional additional metadata.

        Returns:
            Mem0 response dict, or error dict on failure.
        """
        ns = self.namespace(workspace_id)
        user_id = ns.resolve(agent_id)

        meta = dict(metadata) if metadata else {}
        if category:
            meta["category"] = category

        messages = [{"role": "user", "content": content}]

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._mem0.add(messages=messages, user_id=user_id, metadata=meta or None),
            )
            logger.info(
                "[UnifiedMemoryService] store_long_term user_id=%s len=%d",
                user_id,
                len(content),
            )
            return result
        except Exception:
            logger.error(
                "[UnifiedMemoryService] store_long_term failed for user_id=%s",
                user_id,
                exc_info=True,
            )
            return {"success": False, "error": "store_long_term failed"}

    async def search_long_term(
        self,
        workspace_id: str,
        query: str,
        agent_id: Optional[int] = None,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search L3 long-term memory via Mem0 semantic search.

        Args:
            workspace_id: Workspace scope.
            query: Natural-language search query.
            agent_id: Optional agent scope.
            limit: Maximum results to return.

        Returns:
            List of memory item dicts (may be empty on failure).
        """
        ns = self.namespace(workspace_id)
        user_id = ns.resolve(agent_id)

        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._mem0.search(query=query, user_id=user_id, limit=limit),
            )
            logger.debug(
                "[UnifiedMemoryService] search_long_term user_id=%s query=%r → %d results",
                user_id,
                query[:60],
                len(results),
            )
            return results
        except Exception:
            logger.error(
                "[UnifiedMemoryService] search_long_term failed for user_id=%s",
                user_id,
                exc_info=True,
            )
            return []

    async def get_all_memories(
        self,
        workspace_id: str,
        agent_id: Optional[int] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all L3 long-term memories for a workspace/agent scope.

        Args:
            workspace_id: Workspace scope.
            agent_id: Optional agent scope.
            limit: Maximum items.

        Returns:
            List of memory item dicts.
        """
        ns = self.namespace(workspace_id)
        user_id = ns.resolve(agent_id)

        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._mem0.get_all(user_id=user_id, limit=limit),
            )
            logger.debug(
                "[UnifiedMemoryService] get_all_memories user_id=%s → %d items",
                user_id,
                len(results),
            )
            return results
        except Exception:
            logger.error(
                "[UnifiedMemoryService] get_all_memories failed for user_id=%s",
                user_id,
                exc_info=True,
            )
            return []

    async def delete_memory(self, memory_id: str) -> bool:
        """
        Delete a specific memory by ID from L3 (Mem0).

        Args:
            memory_id: The Mem0 memory ID to delete.

        Returns:
            True if deleted, False on failure.
        """
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._mem0.delete(memory_id=memory_id),
            )
            logger.info("[UnifiedMemoryService] delete_memory id=%s success=%s", memory_id, result)
            return result
        except Exception:
            logger.error(
                "[UnifiedMemoryService] delete_memory failed for id=%s",
                memory_id,
                exc_info=True,
            )
            return False

    # ------------------------------------------------------------------
    # L3: Daily Logs (Mem0 with daily namespace)
    # ------------------------------------------------------------------

    async def store_daily_log(
        self,
        workspace_id: str,
        content: str,
        agent_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Store a daily log entry in Mem0 under the daily namespace.

        Args:
            workspace_id: Workspace scope.
            content: Log entry text.
            agent_id: Optional agent that generated the log.
            metadata: Additional metadata (must include 'date' and 'type').

        Returns:
            Mem0 response dict, or error dict on failure.
        """
        ns = self.namespace(workspace_id)
        user_id = ns.daily()
        messages = [{"role": "system", "content": content}]

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._mem0.add(messages=messages, user_id=user_id, metadata=metadata),
            )
            logger.info(
                "[UnifiedMemoryService] store_daily_log user_id=%s len=%d",
                user_id,
                len(content),
            )
            return result
        except Exception:
            logger.error(
                "[UnifiedMemoryService] store_daily_log failed for user_id=%s",
                user_id,
                exc_info=True,
            )
            return {"success": False, "error": "store_daily_log failed"}

    async def get_all_daily_logs(
        self,
        workspace_id: str,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all daily log entries from Mem0 for a workspace.

        Args:
            workspace_id: Workspace scope.
            limit: Maximum items.

        Returns:
            List of memory item dicts.
        """
        ns = self.namespace(workspace_id)
        user_id = ns.daily()

        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._mem0.get_all(user_id=user_id, limit=limit),
            )
            logger.debug(
                "[UnifiedMemoryService] get_all_daily_logs user_id=%s → %d items",
                user_id,
                len(results),
            )
            return results
        except Exception:
            logger.error(
                "[UnifiedMemoryService] get_all_daily_logs failed for user_id=%s",
                user_id,
                exc_info=True,
            )
            return []

    # ------------------------------------------------------------------
    # L3: Two-tier store (global + agent-specific)
    # ------------------------------------------------------------------

    async def store_two_tier(
        self,
        workspace_id: str,
        messages: List[Dict[str, str]],
        agent_id: Optional[int],
        tier: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[tuple]:
        """
        Store content in global and/or agent-specific tiers.

        Args:
            workspace_id: Workspace scope.
            messages: Mem0-format messages list.
            agent_id: Agent scope (required for 'agent' or 'both' tiers).
            tier: One of 'global', 'agent', or 'both'.
            metadata: Base metadata (tier tag is added automatically).

        Returns:
            List of (tier_name, result_dict) tuples.
        """
        ns = self.namespace(workspace_id)
        base_meta = dict(metadata) if metadata else {}
        results: List[tuple] = []

        async def _store(user_id: str, tier_name: str) -> tuple:
            meta = {**base_meta, "tier": tier_name}
            try:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda uid=user_id, m=meta: self._mem0.add(
                        messages=messages, user_id=uid, metadata=m
                    ),
                )
                return (tier_name, result)
            except Exception:
                logger.error(
                    "[UnifiedMemoryService] store_two_tier %s failed user_id=%s",
                    tier_name,
                    user_id,
                    exc_info=True,
                )
                return (tier_name, {"error": f"store_{tier_name} failed"})

        tasks = []
        if tier in ("global", "both"):
            tasks.append(_store(ns.workspace(), "global"))
        if tier in ("agent", "both"):
            tasks.append(_store(ns.agent(agent_id) if agent_id else ns.workspace(), "agent"))

        if tasks:
            results = list(await asyncio.gather(*tasks))

        return results

    # ------------------------------------------------------------------
    # L2: Short-term Memory (Postgres) — stubbed for US-013
    # ------------------------------------------------------------------

    async def store_short_term(
        self,
        workspace_id: str,
        content: str,
        content_type: str = "exchange",
        agent_id: Optional[int] = None,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store content in L2 short-term memory (Postgres). Implemented in US-013."""
        pass

    async def search_short_term(
        self,
        workspace_id: str,
        query: str,
        days: int = 7,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search L2 short-term memory by text and time range. Implemented in US-013."""
        return []

    # ------------------------------------------------------------------
    # L1: Session Memory (Redis) — stubbed for US-009
    # ------------------------------------------------------------------

    async def get_session(
        self,
        workspace_id: str,
        conversation_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Retrieve L1 session from Redis. Implemented in US-009."""
        return None

    async def update_session(
        self,
        workspace_id: str,
        conversation_id: str,
        user_msg: str,
        assistant_msg: str,
    ) -> None:
        """Update L1 session in Redis. Implemented in US-009."""
        pass

    async def end_session(
        self,
        workspace_id: str,
        conversation_id: str,
    ) -> None:
        """End L1 session, set short TTL for consolidation window. Implemented in US-009."""
        pass

    # ------------------------------------------------------------------
    # Cross-layer — stubbed for later stories
    # ------------------------------------------------------------------

    async def retrieve_context(
        self,
        workspace_id: str,
        agent_id: int,
        query: str,
        conversation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Assemble context bundle across all layers. Implemented in US-017."""
        return {}

    async def store_exchange(
        self,
        workspace_id: str,
        agent_id: int,
        user_msg: str,
        assistant_msg: str,
        conversation_id: Optional[str] = None,
    ) -> None:
        """Store a chat exchange in L2 + L3. Implemented in US-014."""
        pass

    async def promote_to_long_term(self, memory_id: str) -> bool:
        """Promote an L2 item to L3. Implemented in US-021."""
        return False

    async def consolidate(self, workspace_id: str) -> None:
        """Run weekly consolidation for a workspace. Implemented later."""
        pass


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def get_unified_memory_service() -> UnifiedMemoryService:
    """Get the singleton UnifiedMemoryService instance."""
    return UnifiedMemoryService.get_instance()
