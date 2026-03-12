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
import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

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

    def recipe(self, recipe_id: Union[int, str]) -> str:
        """Recipe learnings (L3 per-recipe)."""
        return f"mem:{self.workspace_id}:recipe:{recipe_id}"

    def recipe_agent(self, recipe_id: Union[int, str], agent_id: Union[int, str]) -> str:
        """Per-agent step memories within a recipe (L3)."""
        return f"mem:{self.workspace_id}:recipe:{recipe_id}:agent:{agent_id}"

    def workflow(self, workflow_id: Union[int, str]) -> str:
        """Workflow execution memories (L3 per-workflow)."""
        return f"mem:{self.workspace_id}:workflow:{workflow_id}"

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
# SessionMemory — L1 working memory stored in Redis
# ---------------------------------------------------------------------------

@dataclass
class SessionMemory:
    """
    L1 session state stored in Redis per conversation.

    Persists across browser refreshes within a 24-hour window so agents
    remember what was just discussed. Consolidated into L2 after session ends.
    """

    summary: str = ""
    decisions: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    exchange_count: int = 0
    last_updated: str = ""  # ISO-8601 string for JSON serialisation
    ended: bool = False

    def to_json(self) -> str:
        """Serialise to JSON for Redis storage."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> "SessionMemory":
        """Deserialise from JSON string."""
        data = json.loads(raw)
        return cls(**data)


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
    # L3 Cache Helpers (Redis-backed, 5-min TTL)
    # ------------------------------------------------------------------

    async def _get_cached_search(self, cache_key: str) -> Optional[List[Dict[str, Any]]]:
        """Read cached Mem0 search results from Redis. Returns None on miss or error."""
        redis_client = self._get_redis()
        if redis_client is None:
            return None
        try:
            loop = asyncio.get_event_loop()
            conn = redis_client.get_redis()
            raw: Optional[str] = await loop.run_in_executor(None, conn.get, cache_key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception:
            logger.debug(
                "[UnifiedMemoryService] _get_cached_search failed key=%s",
                cache_key,
                exc_info=True,
            )
            return None

    async def _set_cached_search(self, cache_key: str, results: List[Dict[str, Any]]) -> None:
        """Write Mem0 search results to Redis with configured TTL."""
        from config import config

        redis_client = self._get_redis()
        if redis_client is None:
            return
        try:
            loop = asyncio.get_event_loop()
            conn = redis_client.get_redis()
            payload = json.dumps(results)
            ttl = config.MEMORY_CACHE_TTL_SECONDS
            await loop.run_in_executor(
                None,
                lambda: conn.setex(cache_key, ttl, payload),
            )
            logger.debug(
                "[UnifiedMemoryService] _set_cached_search key=%s ttl=%ds items=%d",
                cache_key,
                ttl,
                len(results),
            )
        except Exception:
            logger.debug(
                "[UnifiedMemoryService] _set_cached_search failed key=%s",
                cache_key,
                exc_info=True,
            )

    async def _invalidate_search_cache(self, workspace_id: str) -> None:
        """
        Delete all cached search results for a workspace.

        Uses SCAN (not KEYS) for production safety — avoids blocking Redis
        on large keyspaces.
        """
        redis_client = self._get_redis()
        if redis_client is None:
            return
        ns = self.namespace(workspace_id)
        pattern = ns.cache_pattern()
        try:
            loop = asyncio.get_event_loop()
            conn = redis_client.get_redis()

            def _scan_and_delete():
                deleted = 0
                cursor = 0
                while True:
                    cursor, keys = conn.scan(cursor=cursor, match=pattern, count=100)
                    if keys:
                        conn.delete(*keys)
                        deleted += len(keys)
                    if cursor == 0:
                        break
                return deleted

            deleted = await loop.run_in_executor(None, _scan_and_delete)
            if deleted > 0:
                logger.info(
                    "[UnifiedMemoryService] _invalidate_search_cache pattern=%s deleted=%d",
                    pattern,
                    deleted,
                )
        except Exception:
            logger.debug(
                "[UnifiedMemoryService] _invalidate_search_cache failed pattern=%s",
                pattern,
                exc_info=True,
            )

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
            # Invalidate search cache for this workspace (fire-and-forget)
            asyncio.ensure_future(self._invalidate_search_cache(workspace_id))
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

        Checks Redis cache first (5-min TTL). On cache miss, calls Mem0 and
        caches the result. Cache key:
        ``mem:cache:{workspace_id}:{agent_id|global}:{sha256(query)[:16]}``

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

        # --- Check Redis cache first ---
        query_hash = hashlib.sha256(query.encode()).hexdigest()[:16]
        cache_key = ns.cache_key(agent_id, query_hash)
        cached = await self._get_cached_search(cache_key)
        if cached is not None:
            logger.debug(
                "[UnifiedMemoryService] search_long_term CACHE HIT key=%s",
                cache_key,
            )
            return cached

        # --- Cache miss: call Mem0 ---
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
            # Cache the results (fire-and-forget — cache failure is non-fatal)
            asyncio.ensure_future(self._set_cached_search(cache_key, results))
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
    # L3: Scoped storage (for consumers with custom namespaces)
    # ------------------------------------------------------------------

    async def store_long_term_messages(
        self,
        user_id: str,
        messages: List[Dict[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Store messages in L3 long-term memory with a pre-built namespace user_id.

        Use MemoryNamespace to build the user_id. This supports custom message
        formats (e.g., conversational user+assistant pairs) for better Mem0
        fact extraction.

        Args:
            user_id: Pre-built user_id from MemoryNamespace (e.g., ns.recipe(id)).
            messages: Mem0-format messages list.
            metadata: Optional metadata dict.

        Returns:
            Mem0 response dict, or error dict on failure.
        """
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._mem0.add(messages=messages, user_id=user_id, metadata=metadata),
            )
            logger.info(
                "[UnifiedMemoryService] store_long_term_messages user_id=%s",
                user_id,
            )
            return result
        except Exception:
            logger.error(
                "[UnifiedMemoryService] store_long_term_messages failed for user_id=%s",
                user_id,
                exc_info=True,
            )
            return {"success": False, "error": "store_long_term_messages failed"}

    async def search_long_term_scoped(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search L3 long-term memory with a pre-built namespace user_id.

        Use MemoryNamespace to build the user_id (e.g., ns.recipe(id)).

        Args:
            user_id: Pre-built user_id from MemoryNamespace.
            query: Natural-language search query.
            limit: Maximum results to return.

        Returns:
            List of memory item dicts (may be empty on failure).
        """
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._mem0.search(query=query, user_id=user_id, limit=limit),
            )
            logger.debug(
                "[UnifiedMemoryService] search_long_term_scoped user_id=%s query=%r → %d results",
                user_id,
                query[:60],
                len(results),
            )
            return results
        except Exception:
            logger.error(
                "[UnifiedMemoryService] search_long_term_scoped failed for user_id=%s",
                user_id,
                exc_info=True,
            )
            return []

    async def get_all_memories_scoped(
        self,
        user_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all L3 long-term memories with a pre-built namespace user_id.

        Use MemoryNamespace to build the user_id (e.g., ns.recipe(id)).

        Args:
            user_id: Pre-built user_id from MemoryNamespace.
            limit: Maximum items.

        Returns:
            List of memory item dicts.
        """
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._mem0.get_all(user_id=user_id, limit=limit),
            )
            logger.debug(
                "[UnifiedMemoryService] get_all_memories_scoped user_id=%s → %d items",
                user_id,
                len(results),
            )
            return results
        except Exception:
            logger.error(
                "[UnifiedMemoryService] get_all_memories_scoped failed for user_id=%s",
                user_id,
                exc_info=True,
            )
            return []

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
    # L1: Session Memory (Redis)
    # ------------------------------------------------------------------

    async def get_session(
        self,
        workspace_id: str,
        conversation_id: str,
    ) -> Optional[SessionMemory]:
        """
        Retrieve L1 session from Redis.

        Returns None if no session exists or Redis is unavailable.
        Redis failures never break chat — they are logged and swallowed.
        """
        redis_client = self._get_redis()
        if redis_client is None:
            return None

        ns = self.namespace(workspace_id)
        key = ns.session(conversation_id)

        try:
            loop = asyncio.get_event_loop()
            conn = redis_client.get_redis()
            raw: Optional[str] = await loop.run_in_executor(None, conn.get, key)
            if raw is None:
                return None
            session = SessionMemory.from_json(raw)
            logger.debug(
                "[UnifiedMemoryService] get_session key=%s exchanges=%d",
                key,
                session.exchange_count,
            )
            return session
        except Exception:
            logger.error(
                "[UnifiedMemoryService] get_session failed for key=%s",
                key,
                exc_info=True,
            )
            return None

    async def update_session(
        self,
        workspace_id: str,
        conversation_id: str,
        user_msg: str,
        assistant_msg: str,
    ) -> None:
        """
        Update (or create) an L1 session in Redis after each exchange.

        Appends the exchange to the rolling summary (truncated to last
        500 chars for now — Phase 2 adds LLM summarisation). Resets the
        24-hour TTL on every update.

        Redis failures are logged but never break chat.
        """
        from config import config

        redis_client = self._get_redis()
        if redis_client is None:
            return

        ns = self.namespace(workspace_id)
        key = ns.session(conversation_id)
        ttl = config.MEMORY_SESSION_TTL_SECONDS

        try:
            loop = asyncio.get_event_loop()
            conn = redis_client.get_redis()

            # Fetch existing session or create new
            raw: Optional[str] = await loop.run_in_executor(None, conn.get, key)
            if raw is not None:
                session = SessionMemory.from_json(raw)
            else:
                session = SessionMemory()

            # Build exchange snippet and append to rolling summary
            exchange_snippet = f"User: {user_msg[:200]}\nAssistant: {assistant_msg[:200]}"
            if session.summary:
                combined = f"{session.summary}\n---\n{exchange_snippet}"
            else:
                combined = exchange_snippet

            # Naive truncation to last 500 chars (Phase 2 adds LLM rolling summary)
            session = SessionMemory(
                summary=combined[-500:],
                decisions=list(session.decisions),
                action_items=list(session.action_items),
                exchange_count=session.exchange_count + 1,
                last_updated=datetime.now(timezone.utc).isoformat(),
                ended=session.ended,
            )

            payload = session.to_json()
            await loop.run_in_executor(
                None,
                lambda: conn.setex(key, ttl, payload),
            )
            logger.debug(
                "[UnifiedMemoryService] update_session key=%s exchanges=%d",
                key,
                session.exchange_count,
            )
        except Exception:
            logger.error(
                "[UnifiedMemoryService] update_session failed for key=%s",
                key,
                exc_info=True,
            )

    async def end_session(
        self,
        workspace_id: str,
        conversation_id: str,
    ) -> None:
        """
        Mark session as ended and set a short TTL for the consolidation window.

        The session stays in Redis for MEMORY_SESSION_CONSOLIDATION_TTL_SECONDS
        (default 1 hour) so the hourly consolidation job can promote important
        decisions to L2 before the key expires.

        Redis failures are logged but never break chat.
        """
        from config import config

        redis_client = self._get_redis()
        if redis_client is None:
            return

        ns = self.namespace(workspace_id)
        key = ns.session(conversation_id)
        consolidation_ttl = config.MEMORY_SESSION_CONSOLIDATION_TTL_SECONDS

        try:
            loop = asyncio.get_event_loop()
            conn = redis_client.get_redis()

            raw: Optional[str] = await loop.run_in_executor(None, conn.get, key)
            if raw is None:
                logger.debug("[UnifiedMemoryService] end_session key=%s — no session found", key)
                return

            session = SessionMemory.from_json(raw)
            ended_session = SessionMemory(
                summary=session.summary,
                decisions=list(session.decisions),
                action_items=list(session.action_items),
                exchange_count=session.exchange_count,
                last_updated=datetime.now(timezone.utc).isoformat(),
                ended=True,
            )

            payload = ended_session.to_json()
            await loop.run_in_executor(
                None,
                lambda: conn.setex(key, consolidation_ttl, payload),
            )
            logger.info(
                "[UnifiedMemoryService] end_session key=%s ttl=%ds exchanges=%d",
                key,
                consolidation_ttl,
                ended_session.exchange_count,
            )
        except Exception:
            logger.error(
                "[UnifiedMemoryService] end_session failed for key=%s",
                key,
                exc_info=True,
            )

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
