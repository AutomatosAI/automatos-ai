"""
Unified Memory Service
======================

Single entry point for all memory operations across all consumers.
One shared service in front of every memory tier (PRD-79; durable tier un-split in PRD-187).

5-Layer Memory Stack:
  L0: Focus (context window — no code needed)
  L1: Working Memory (Redis session cache)
  L2: Short-term Memory (Postgres + time-based decay)
  L3: Long-term Memory (in-process durable store on Qdrant)
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
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# MemoryNamespace — builds standardised user_id strings for the durable store
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MemoryNamespace:
    """
    Builds standardised, scoped user_id strings for the durable store and Redis keys.

    All memory consumers MUST use this helper instead of raw string
    concatenation. This prevents the 5+ user_id format inconsistencies
    documented in PRD-79 § 1.2.
    """

    workspace_id: str

    # --- L3 Long-term (durable store) namespaces ---

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

    def l2(self) -> str:
        """Semantic mirror of L2 short-term rows (PRD-187 S3)."""
        return f"mem:{self.workspace_id}:l2"

    # --- L1 Session (Redis) namespaces ---

    def session(self, conversation_id: str) -> str:
        """Session cache key (L1 Redis)."""
        return f"mem:session:{self.workspace_id}:{conversation_id}"

    # --- L3 Cache (Redis) namespaces ---

    def cache_key(self, agent_id: Optional[int], query_hash: str) -> str:
        """Cache key for L3 durable-store search results cached in Redis."""
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

    # --- Resolve user_id for durable-store calls ---

    def resolve(self, agent_id: Optional[int] = None) -> str:
        """
        Resolve the correct durable-store user_id for a given scope.

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
    exchange_count: int = 0
    last_updated: str = ""  # ISO-8601 string for JSON serialisation
    ended: bool = False

    def to_json(self) -> str:
        """Serialise to JSON for Redis storage."""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> "SessionMemory":
        """Deserialise from JSON string.

        Tolerant of unknown keys so sessions written before PRD-159 removed the
        dead ``decisions``/``action_items`` fields still load cleanly.
        """
        data = json.loads(raw)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


# ---------------------------------------------------------------------------
# UnifiedMemoryService — singleton
# ---------------------------------------------------------------------------

class UnifiedMemoryService:
    """
    Single entry point for all memory operations across all consumers.

    Holds ONE shared DurableMemoryStore (in-process Qdrant, PRD-187 S1),
    ONE Redis client.
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
        # Shared in-process durable store (L3 long-term, PRD-187 S1)
        from modules.memory.durable_store import DurableMemoryStore

        self._durable = DurableMemoryStore()

        # Shared Redis client (L1 session + caching)
        from core.redis.client import get_redis_client

        self._redis_client_getter = get_redis_client
        logger.info("[UnifiedMemoryService] Initialised with shared DurableMemoryStore and Redis")

    @property
    def is_durable_configured(self) -> bool:
        """Check if the durable L3 backend (Qdrant) is configured."""
        from config import config

        return bool(config.QDRANT_URL)

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
        """Read cached L3 search results from Redis. Returns None on miss or error."""
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
        """Write L3 search results to Redis with configured TTL."""
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
    # L3: Long-term Memory (durable store)
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
        Store content in L3 long-term memory (durable store, verbatim).

        Args:
            workspace_id: Workspace scope.
            content: Text to store verbatim (writers curate before storing).
            agent_id: Optional agent scope.
            category: Optional category tag.
            metadata: Optional additional metadata.

        Returns:
            Store response dict, or error dict on failure.
        """
        ns = self.namespace(workspace_id)
        user_id = ns.resolve(agent_id)

        meta = dict(metadata) if metadata else {}
        if category:
            meta["category"] = category

        messages = [{"role": "user", "content": content}]

        try:
            result = await self._durable.add(messages=messages, user_id=user_id, metadata=meta or None, workspace_id=workspace_id)
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
        Search L3 long-term memory (durable store semantic search).

        Checks Redis cache first (5-min TTL). On cache miss, queries the
        durable store and caches the result. Cache key:
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

        # --- Cache miss: query the durable store ---
        try:
            results = await self._durable.search(query=query, user_id=user_id, limit=limit, workspace_id=workspace_id)
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
            results = await self._durable.get_all(user_id=user_id, limit=limit, workspace_id=workspace_id)
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

    async def delete_memory(
        self,
        memory_id: str,
        workspace_id: str,
        agent_id: Optional[int] = None,
    ) -> bool:
        """
        Delete a specific memory by ID from L3 (durable store).

        The delete requires the owning ``user_id`` — resolved here from the
        workspace/agent namespace — and only removes points whose namespace
        matches (the same ownership check the old bulk API enforced).

        Args:
            memory_id: The durable memory ID to delete.
            workspace_id: Workspace scope (used to resolve the namespace user_id).
            agent_id: Optional agent scope.

        Returns:
            True if deleted, False on failure.
        """
        try:
            user_id = self.namespace(workspace_id).resolve(agent_id)
            result = await self._durable.delete(
                memory_ids=[memory_id], user_id=user_id, workspace_id=workspace_id
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
    # GDPR erasure / export (PRD-181 S3/S4)
    # ------------------------------------------------------------------

    async def erase_workspace_memories(self, workspace_id: str, db: Any = None) -> int:
        """GDPR erasure — delete every durable (L3) memory for a workspace.

        One filter delete over the store's ``workspace_id`` payload index —
        every namespace (workspace-wide, per-agent, per-recipe, daily) carries
        the workspace tag, so no namespace enumeration is needed. ``db`` is
        accepted for signature stability with the GDPR service but unused.
        Returns the number of memories deleted.
        """
        try:
            total = await self._durable.erase_workspace(str(workspace_id))
        except Exception:
            logger.error(
                "[UnifiedMemoryService] durable erase failed for ws=%s",
                workspace_id, exc_info=True,
            )
            return 0
        logger.warning("[UnifiedMemoryService] GDPR erased %d durable memories for ws=%s", total, workspace_id)
        return total

    async def erase_subject_memories(self, workspace_id: str, subject_id: str, db: Any = None) -> int:
        """GDPR subject-level erasure for durable memories (see GDPR-GAP below)."""
        # GDPR-GAP: durable memories are namespaced by workspace / agent / recipe
        # (MemoryNamespace) and carry no data-subject tag in their payload, so a
        # single human's durable memories cannot be filtered from a shared
        # workspace/agent namespace. Until a subject tag is written at store
        # time, subject-level erasure here is not possible without erasing the
        # whole namespace. Returns 0; the gap is surfaced by the GDPR service.
        logger.warning(
            "[UnifiedMemoryService] GDPR subject erase requested subject=%s ws=%s "
            "but the durable store has no data-subject tag (GDPR-GAP) — 0 memories erased",
            subject_id, workspace_id,
        )
        return 0

    async def export_workspace_memories(self, workspace_id: str, db: Any = None) -> List[Dict[str, Any]]:
        """GDPR export — every durable memory for a workspace as plain dicts
        (one workspace-filter scroll; each item carries its ``namespace``).
        ``db`` is accepted for signature stability but unused."""
        try:
            return await self._durable.export_workspace(str(workspace_id))
        except Exception:
            logger.error(
                "[UnifiedMemoryService] durable export failed for ws=%s",
                workspace_id, exc_info=True,
            )
            return []

    # ------------------------------------------------------------------
    # L3: Scoped storage (for consumers with custom namespaces)
    # ------------------------------------------------------------------

    async def store_long_term_messages(
        self,
        user_id: str,
        messages: List[Dict[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
        workspace_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Store messages in L3 long-term memory with a pre-built namespace user_id.

        Use MemoryNamespace to build the user_id. Supports custom message
        formats (e.g., conversational user+assistant pairs); the store keeps
        the text verbatim.

        Args:
            user_id: Pre-built user_id from MemoryNamespace (e.g., ns.recipe(id)).
            messages: Messages list ([{"role": ..., "content": ...}]).
            metadata: Optional metadata dict.
            workspace_id: Stamped on the stored point for fail-closed tenancy
                and GDPR erasure (parsed from the namespace when omitted).

        Returns:
            Store response dict, or error dict on failure.
        """
        try:
            result = await self._durable.add(
                messages=messages, user_id=user_id, metadata=metadata,
                workspace_id=workspace_id,
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
        workspace_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search L3 long-term memory with a pre-built namespace user_id.

        Use MemoryNamespace to build the user_id (e.g., ns.recipe(id)).

        Args:
            user_id: Pre-built user_id from MemoryNamespace.
            query: Natural-language search query.
            limit: Maximum results to return.
            workspace_id: Tenancy scope (see store_long_term_messages).

        Returns:
            List of memory item dicts (may be empty on failure).
        """
        try:
            results = await self._durable.search(
                query=query, user_id=user_id, limit=limit, workspace_id=workspace_id
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
        workspace_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve all L3 long-term memories with a pre-built namespace user_id.

        Use MemoryNamespace to build the user_id (e.g., ns.recipe(id)).

        Args:
            user_id: Pre-built user_id from MemoryNamespace.
            limit: Maximum items.
            workspace_id: Tenancy scope (see store_long_term_messages).

        Returns:
            List of memory item dicts.
        """
        try:
            results = await self._durable.get_all(
                user_id=user_id, limit=limit, workspace_id=workspace_id
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
    # L3: Daily Logs (durable store, daily namespace)
    # ------------------------------------------------------------------

    async def store_daily_log(
        self,
        workspace_id: str,
        content: str,
        agent_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Store a daily log entry in the durable store under the daily namespace.

        Args:
            workspace_id: Workspace scope.
            content: Log entry text.
            agent_id: Optional agent that generated the log.
            metadata: Additional metadata (must include 'date' and 'type').

        Returns:
            Store response dict, or error dict on failure.
        """
        ns = self.namespace(workspace_id)
        user_id = ns.daily()
        messages = [{"role": "system", "content": content}]

        try:
            result = await self._durable.add(messages=messages, user_id=user_id, metadata=metadata, workspace_id=workspace_id)
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
        Retrieve all daily log entries from the durable store for a workspace.

        Args:
            workspace_id: Workspace scope.
            limit: Maximum items.

        Returns:
            List of memory item dicts.
        """
        ns = self.namespace(workspace_id)
        user_id = ns.daily()

        try:
            results = await self._durable.get_all(user_id=user_id, limit=limit, workspace_id=workspace_id)
            # PRD-159 S3: daily logs are time-ordered (newest first) and bounded,
            # not an arbitrary first-N slice — recall surfaces the most recent
            # operational activity deterministically.
            results.sort(key=lambda m: (m or {}).get("created_at") or "", reverse=True)
            results = results[:limit]
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
            messages: Messages list ([{"role": ..., "content": ...}]).
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
                result = await self._durable.add(messages=messages, user_id=user_id, metadata=meta, workspace_id=workspace_id)
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
    # L2: Short-term Memory (Postgres)
    # ------------------------------------------------------------------

    @staticmethod
    def _store_short_term_sync(
        workspace_id: str,
        content: str,
        content_type: str,
        agent_id: Optional[int],
        importance: float,
        metadata: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """
        Insert a row into memory_short_term (synchronous, runs in executor).

        Returns the new row's UUID as string, or None on failure.
        """
        from core.database.database import get_db_session
        from modules.memory.models import MemoryShortTerm

        with get_db_session() as db:
            row = MemoryShortTerm(
                workspace_id=workspace_id,
                agent_id=agent_id,
                content=content,
                content_type=content_type,
                importance=importance,
                metadata_=metadata or {},
            )
            db.add(row)
            db.flush()
            row_id = str(row.id)
            return row_id

    async def store_short_term(
        self,
        workspace_id: str,
        content: str,
        content_type: str = "exchange",
        agent_id: Optional[int] = None,
        importance: float = 0.5,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Store content in L2 short-term memory (Postgres).

        Args:
            workspace_id: Workspace scope (UUID string).
            content: Text content to store.
            content_type: One of exchange, recipe_summary, heartbeat_log,
                          tool_result, session_decision.
            agent_id: Optional agent scope.
            importance: Base importance score (0.0–1.0).
            metadata: Optional JSONB metadata dict.

        Returns:
            The new row's UUID as string, or None on failure.
        """
        try:
            loop = asyncio.get_event_loop()
            row_id = await loop.run_in_executor(
                None,
                self._store_short_term_sync,
                workspace_id,
                content,
                content_type,
                agent_id,
                importance,
                metadata,
            )
            logger.info(
                "[UnifiedMemoryService] store_short_term id=%s ws=%s type=%s",
                row_id,
                workspace_id,
                content_type,
            )
            # PRD-187 S3: mirror the row into the durable store's L2 namespace
            # so recall can match by MEANING, not ILIKE substring. Fire-and-
            # forget — the semantic mirror must never fail the L2 write.
            if row_id:
                asyncio.ensure_future(self._mirror_l2_to_durable(
                    row_id=row_id, workspace_id=workspace_id, content=content,
                    content_type=content_type, importance=importance,
                ))
            return row_id
        except Exception:
            logger.error(
                "[UnifiedMemoryService] store_short_term failed ws=%s type=%s",
                workspace_id,
                content_type,
                exc_info=True,
            )
            return None

    async def _mirror_l2_to_durable(
        self,
        *,
        row_id: str,
        workspace_id: str,
        content: str,
        content_type: str,
        importance: float,
    ) -> None:
        """Best-effort vector mirror of one L2 row (PRD-187 S3)."""
        try:
            await self._durable.add(
                messages=[{"role": "user", "content": content}],
                user_id=self.namespace(workspace_id).l2(),
                metadata={
                    "l2_id": str(row_id),
                    "content_type": content_type,
                    "importance": importance,
                },
                workspace_id=workspace_id,
            )
        except Exception:
            logger.error(
                "[UnifiedMemoryService] L2 semantic mirror failed id=%s ws=%s",
                row_id, workspace_id, exc_info=True,
            )

    async def search_short_term_semantic(
        self,
        workspace_id: str,
        query: str,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        """Semantic L2 recall (PRD-187 S3) — matches by meaning, every turn.

        Vector-searches the L2 mirror namespace, hydrates the LIVE Postgres
        rows (archived rows never recall), ranks with the shared
        ``field_scoring.resonance`` curve (similarity² × stability × recency —
        the same honest definition field memory uses), and touches each
        recalled row so ``access_count`` climbs (the recall that finally makes
        promotion's access signal real).

        A non-temporal query like "what did we learn about the Shopify sync?"
        — 0 rows by construction under the old ILIKE-behind-a-temporal-regex —
        now returns the relevant rows.
        """
        from config import config

        try:
            hits = await self._durable.search(
                query=query,
                user_id=self.namespace(workspace_id).l2(),
                limit=max(limit * 2, limit),
                workspace_id=workspace_id,
            )
        except Exception:
            logger.error(
                "[UnifiedMemoryService] semantic L2 search failed ws=%s",
                workspace_id, exc_info=True,
            )
            return []

        score_by_l2_id: Dict[str, float] = {}
        for h in hits:
            meta = h.get("metadata") or {}
            l2_id = meta.get("l2_id")
            if l2_id and l2_id not in score_by_l2_id:
                score_by_l2_id[str(l2_id)] = h.get("score") or 0.0
        if not score_by_l2_id:
            return []

        loop = asyncio.get_event_loop()
        rows = await loop.run_in_executor(
            None, self._hydrate_l2_rows_sync, list(score_by_l2_id),
        )

        from modules.context import field_scoring

        params = field_scoring.ScoringParams(
            decay_rate=config.FIELD_DECAY_RATE,
            reinforce_bonus=config.FIELD_REINFORCE_BONUS,
            reinforce_cap=config.FIELD_REINFORCE_CAP,
            archival_threshold=config.FIELD_ARCHIVAL_THRESHOLD,
            half_life_access_scale=config.FIELD_HALF_LIFE_ACCESS_SCALE,
        )
        now = datetime.now(timezone.utc)
        for row in rows:
            similarity = score_by_l2_id.get(row["id"], 0.0)
            try:
                created = datetime.fromisoformat(row["created_at"]) if row.get("created_at") else now
                age_hours = max(0.0, (now - created).total_seconds() / 3600)
            except (ValueError, TypeError):
                age_hours = 0.0
            row["score"] = field_scoring.resonance(
                similarity, row.get("importance") or 0.5, age_hours,
                row.get("access_count") or 0, params,
            )
            row["cosine_similarity"] = similarity

        rows.sort(key=lambda r: r["score"], reverse=True)
        rows = rows[:limit]

        # Recall bumps access_count — the promotion signal (PRD-154 S3 posture).
        if rows:
            await asyncio.gather(
                *(self.touch_short_term(r["id"]) for r in rows if r.get("id")),
                return_exceptions=True,
            )
        logger.debug(
            "[UnifiedMemoryService] semantic L2 recall ws=%s query=%r → %d rows",
            workspace_id, query[:60], len(rows),
        )
        return rows

    @staticmethod
    def _hydrate_l2_rows_sync(l2_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch LIVE (non-archived) L2 rows by id — archived rows never recall."""
        from core.database.database import get_db_session
        from modules.memory.models import MemoryShortTerm

        with get_db_session() as db:
            rows = (
                db.query(MemoryShortTerm)
                .filter(
                    MemoryShortTerm.id.in_(l2_ids),
                    MemoryShortTerm.archived_at.is_(None),
                )
                .all()
            )
            return [
                {
                    "id": str(row.id),
                    "content": row.content,
                    "content_type": row.content_type,
                    "importance": row.importance,
                    "decay_score": row.decay_score,
                    "access_count": row.access_count,
                    "metadata": row.metadata_ or {},
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                }
                for row in rows
            ]

    @staticmethod
    def _search_short_term_sync(
        workspace_id: str,
        query: str,
        days: int,
        limit: int,
        content_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search memory_short_term by text (ILIKE) within a time window (synchronous).

        ``content_types`` optionally restricts recall to typed records (e.g.
        ``mission_summary`` / ``task_failure`` — the PRD-164 planning pack
        recalls mission outcomes on this PRD-159 path).
        """
        from core.database.database import get_db_session
        from modules.memory.models import MemoryShortTerm
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)

        with get_db_session() as db:
            base = db.query(MemoryShortTerm).filter(
                MemoryShortTerm.workspace_id == workspace_id,
                MemoryShortTerm.created_at >= cutoff,
                MemoryShortTerm.archived_at.is_(None),
                MemoryShortTerm.content.ilike(f"%{query}%"),
            )
            if content_types:
                base = base.filter(
                    MemoryShortTerm.content_type.in_(list(content_types))
                )
            q = (
                base.order_by(MemoryShortTerm.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": str(row.id),
                    "content": row.content,
                    "content_type": row.content_type,
                    "importance": row.importance,
                    "decay_score": row.decay_score,
                    "access_count": row.access_count,
                    "metadata": row.metadata_ or {},
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "last_accessed_at": row.last_accessed_at.isoformat() if row.last_accessed_at else None,
                }
                for row in q
            ]

    async def search_short_term(
        self,
        workspace_id: str,
        query: str,
        days: int = 7,
        limit: int = 20,
        content_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search L2 short-term memory by text and time range.

        Uses ILIKE text search on content column within the specified
        day window. Results ordered by created_at DESC.

        Args:
            workspace_id: Workspace scope.
            query: Text to search for (case-insensitive substring match;
                empty string matches everything).
            days: Look-back window in days (default 7).
            limit: Maximum results (default 20).
            content_types: Optional content_type whitelist (typed recall —
                e.g. mission_summary/task_failure for the planning pack).

        Returns:
            List of memory item dicts.
        """
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self._search_short_term_sync,
                workspace_id,
                query,
                days,
                limit,
                content_types,
            )
            logger.debug(
                "[UnifiedMemoryService] search_short_term ws=%s query=%r days=%d → %d results",
                workspace_id,
                query[:60],
                days,
                len(results),
            )
            # PRD-154 S3: touch each recalled row so access_count climbs toward the
            # L2->L3 promotion threshold (touch_short_term had zero callers before).
            # Best-effort: a touch failure must never fail the recall.
            if results:
                await asyncio.gather(
                    *(self.touch_short_term(r["id"]) for r in results if r.get("id")),
                    return_exceptions=True,
                )
            return results
        except Exception:
            logger.error(
                "[UnifiedMemoryService] search_short_term failed ws=%s",
                workspace_id,
                exc_info=True,
            )
            return []

    @staticmethod
    def _get_short_term_by_time_sync(
        workspace_id: str,
        start_date: datetime,
        end_date: datetime,
        limit: int,
    ) -> List[Dict[str, Any]]:
        """
        Query memory_short_term by time range (synchronous).
        """
        from core.database.database import get_db_session
        from modules.memory.models import MemoryShortTerm

        with get_db_session() as db:
            rows = (
                db.query(MemoryShortTerm)
                .filter(
                    MemoryShortTerm.workspace_id == workspace_id,
                    MemoryShortTerm.created_at >= start_date,
                    MemoryShortTerm.created_at <= end_date,
                    MemoryShortTerm.archived_at.is_(None),
                )
                .order_by(MemoryShortTerm.created_at.desc())
                .limit(limit)
                .all()
            )
            return [
                {
                    "id": str(row.id),
                    "content": row.content,
                    "content_type": row.content_type,
                    "importance": row.importance,
                    "decay_score": row.decay_score,
                    "access_count": row.access_count,
                    "metadata": row.metadata_ or {},
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "last_accessed_at": row.last_accessed_at.isoformat() if row.last_accessed_at else None,
                }
                for row in rows
            ]

    async def get_short_term_by_time(
        self,
        workspace_id: str,
        start_date: datetime,
        end_date: datetime,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve L2 short-term memories within a time range.

        Used by the Context Router for temporal queries (e.g., "what did
        we discuss last week?").

        Args:
            workspace_id: Workspace scope.
            start_date: Start of time window (inclusive).
            end_date: End of time window (inclusive).
            limit: Maximum results (default 50).

        Returns:
            List of memory item dicts ordered by created_at DESC.
        """
        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                self._get_short_term_by_time_sync,
                workspace_id,
                start_date,
                end_date,
                limit,
            )
            logger.debug(
                "[UnifiedMemoryService] get_short_term_by_time ws=%s range=%s→%s → %d results",
                workspace_id,
                start_date.isoformat(),
                end_date.isoformat(),
                len(results),
            )
            return results
        except Exception:
            logger.error(
                "[UnifiedMemoryService] get_short_term_by_time failed ws=%s",
                workspace_id,
                exc_info=True,
            )
            return []

    @staticmethod
    def _touch_short_term_sync(memory_id: str) -> bool:
        """
        Increment access_count and update last_accessed_at (synchronous).
        """
        from core.database.database import get_db_session
        from modules.memory.models import MemoryShortTerm

        with get_db_session() as db:
            row = (
                db.query(MemoryShortTerm)
                .filter(MemoryShortTerm.id == memory_id)
                .first()
            )
            if row is None:
                return False
            row.access_count = (row.access_count or 0) + 1
            row.last_accessed_at = datetime.now(timezone.utc)
            return True

    async def touch_short_term(self, memory_id: str) -> bool:
        """
        Touch an L2 memory: increment access_count and update last_accessed_at.

        This boosts the item's retention score in the Ebbinghaus decay
        calculation, making frequently-accessed items persist longer.

        Args:
            memory_id: UUID of the memory_short_term row.

        Returns:
            True if the row was found and updated, False otherwise.
        """
        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._touch_short_term_sync,
                memory_id,
            )
            if result:
                logger.debug(
                    "[UnifiedMemoryService] touch_short_term id=%s",
                    memory_id,
                )
            return result
        except Exception:
            logger.error(
                "[UnifiedMemoryService] touch_short_term failed id=%s",
                memory_id,
                exc_info=True,
            )
            return False

    # ------------------------------------------------------------------
    # L2: Ebbinghaus Decay & Archival
    # ------------------------------------------------------------------

    @staticmethod
    def _run_decay_for_workspace_sync(
        workspace_id: str,
        decay_rate: float,
        archive_threshold: float,
        batch_size: int,
    ) -> Dict[str, int]:
        """
        Calculate Ebbinghaus retention scores and archive expired L2 items
        for a single workspace (synchronous, runs in executor).

        Formula:
            retention = exp(-decay_rate * hours_elapsed)
                        * (1 + 0.5*importance + 0.1*min(access_count, 10))

        Returns:
            {"decayed": N, "archived": M} counts.
        """
        import math
        from core.database.database import get_db_session
        from modules.memory.models import MemoryShortTerm

        now = datetime.now(timezone.utc)
        decayed = 0
        archived = 0

        with get_db_session() as db:
            # Process in batches using offset pagination.
            # The partial index ix_mem_st_ws_decay covers
            # (workspace_id, decay_score) WHERE archived_at IS NULL.
            offset = 0
            while True:
                rows = (
                    db.query(MemoryShortTerm)
                    .filter(
                        MemoryShortTerm.workspace_id == workspace_id,
                        MemoryShortTerm.archived_at.is_(None),
                    )
                    .order_by(MemoryShortTerm.created_at)
                    .offset(offset)
                    .limit(batch_size)
                    .all()
                )
                if not rows:
                    break

                for row in rows:
                    created = row.created_at
                    if created is None:
                        continue
                    # Ensure timezone-aware comparison
                    if created.tzinfo is None:
                        created = created.replace(tzinfo=timezone.utc)
                    hours_elapsed = max(
                        (now - created).total_seconds() / 3600.0, 0.0
                    )
                    importance = row.importance or 0.0
                    access_count = row.access_count or 0

                    retention = math.exp(-decay_rate * hours_elapsed) * (
                        1.0 + 0.5 * importance + 0.1 * min(access_count, 10)
                    )

                    row.decay_score = retention
                    decayed += 1

                    if retention < archive_threshold:
                        row.archived_at = now
                        archived += 1

                db.flush()
                offset += batch_size

        return {"decayed": decayed, "archived": archived}

    async def run_decay(self, workspace_id: str) -> Dict[str, int]:
        """
        Run Ebbinghaus decay scoring for a single workspace's L2 items.

        Updates decay_score on every non-archived row and archives items
        whose retention drops below the configured threshold.

        Args:
            workspace_id: Workspace to process.

        Returns:
            {"decayed": N, "archived": M} counts, or {"decayed": 0, "archived": 0}
            on failure.
        """
        from config import config

        try:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._run_decay_for_workspace_sync,
                workspace_id,
                config.MEMORY_DECAY_RATE,
                config.MEMORY_DECAY_ARCHIVE_THRESHOLD,
                config.MEMORY_DECAY_BATCH_SIZE,
            )
            logger.info(
                "[UnifiedMemoryService] run_decay ws=%s decayed=%d archived=%d",
                workspace_id,
                result["decayed"],
                result["archived"],
            )
            return result
        except Exception:
            logger.error(
                "[UnifiedMemoryService] run_decay failed ws=%s",
                workspace_id,
                exc_info=True,
            )
            return {"decayed": 0, "archived": 0}

    async def run_decay_all(self) -> Dict[str, Any]:
        """
        Run Ebbinghaus decay across ALL workspaces that have L2 items.

        Iterates distinct workspace_ids from memory_short_term and calls
        run_decay() per workspace. One workspace failure does not stop
        processing others.

        Returns:
            {"workspaces_processed": N, "total_decayed": M, "total_archived": K,
             "errors": E}
        """
        try:
            loop = asyncio.get_event_loop()
            workspace_ids = await loop.run_in_executor(
                None, self._get_active_workspace_ids_sync,
            )
        except Exception:
            logger.error(
                "[UnifiedMemoryService] run_decay_all failed to fetch workspace_ids",
                exc_info=True,
            )
            return {
                "workspaces_processed": 0,
                "total_decayed": 0,
                "total_archived": 0,
                "errors": 1,
            }

        total_decayed = 0
        total_archived = 0
        errors = 0

        for ws_id in workspace_ids:
            result = await self.run_decay(str(ws_id))
            if result["decayed"] == 0 and result["archived"] == 0:
                # Could be empty or error — don't count as error unless logged
                pass
            total_decayed += result["decayed"]
            total_archived += result["archived"]

        logger.info(
            "[UnifiedMemoryService] run_decay_all complete: "
            "workspaces=%d decayed=%d archived=%d errors=%d",
            len(workspace_ids),
            total_decayed,
            total_archived,
            errors,
        )
        return {
            "workspaces_processed": len(workspace_ids),
            "total_decayed": total_decayed,
            "total_archived": total_archived,
            "errors": errors,
        }

    @staticmethod
    def _get_active_workspace_ids_sync() -> List[str]:
        """
        Fetch distinct workspace_ids that have non-archived L2 rows
        (synchronous, runs in executor).
        """
        from core.database.database import get_db_session
        from modules.memory.models import MemoryShortTerm
        from sqlalchemy import distinct

        with get_db_session() as db:
            rows = (
                db.query(distinct(MemoryShortTerm.workspace_id))
                .filter(MemoryShortTerm.archived_at.is_(None))
                .all()
            )
            return [str(r[0]) for r in rows]

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

    async def run_sleep_time_consolidation(
        self, workspace_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """PRD-159 S4: contradiction-based consolidation — the primary lifecycle.

        For each workspace, fetch L3 memories, then plan + apply:
          - fold near-duplicates into one canonical (delete the rest),
          - resolve contradictions by recency+confidence (delete/archive the
            loser, reason logged).

        This replaces time-decay as the way memories leave the active set:
        memories are superseded by newer/contradicting facts or merged, not aged
        out. Promotion (L2→L3) stays in ``run_promotion_all``.
        """
        from modules.memory.operations.contradiction import plan_consolidation

        if workspace_id:
            workspace_ids: List[str] = [workspace_id]
        else:
            try:
                loop = asyncio.get_event_loop()
                workspace_ids = await loop.run_in_executor(
                    None, self._get_active_workspace_ids_sync
                )
            except Exception:
                logger.error(
                    "[UnifiedMemoryService] run_sleep_time_consolidation: "
                    "workspace fetch failed",
                    exc_info=True,
                )
                workspace_ids = []

        merged = 0
        superseded = 0
        errors = 0
        for ws_id in workspace_ids:
            try:
                memories = await self.get_all_memories(ws_id, limit=200)
                if not memories:
                    continue
                plan = plan_consolidation(memories)
                # Fold near-duplicates → delete the non-canonical members.
                for mg in plan.merges:
                    for dup_id in mg.merged_from:
                        if dup_id and await self.delete_memory(dup_id, ws_id):
                            merged += 1
                # Contradictions → the loser leaves the active set (reason logged).
                for s in plan.supersessions:
                    loser_id = str(s.loser.get("id") or "")
                    if loser_id and await self.delete_memory(loser_id, ws_id):
                        superseded += 1
                        logger.info(
                            "[UnifiedMemoryService] consolidation superseded "
                            "id=%s (%s)",
                            loser_id,
                            s.reason,
                        )
            except Exception:
                errors += 1
                logger.error(
                    "[UnifiedMemoryService] run_sleep_time_consolidation failed "
                    "for ws=%s",
                    ws_id,
                    exc_info=True,
                )

        logger.info(
            "[UnifiedMemoryService] run_sleep_time_consolidation complete: "
            "workspaces=%d merged=%d superseded=%d errors=%d",
            len(workspace_ids),
            merged,
            superseded,
            errors,
        )
        return {
            "workspaces_processed": len(workspace_ids),
            "merged": merged,
            "superseded": superseded,
            "errors": errors,
        }

    # ------------------------------------------------------------------
    # Cross-layer
    # ------------------------------------------------------------------

    async def retrieve_context(
        self,
        workspace_id: str,
        agent_id: int,
        query: str,
        conversation_id: Optional[str] = None,
        context_window: Optional[int] = None,
    ) -> "ContextBundle":
        """
        Assemble a budget-constrained context bundle across all memory layers.

        Delegates to ``ContextRouter.retrieve_context()`` which analyses the
        query for signals and fetches from L1/L2/L3 accordingly.

        Returns:
            A ContextBundle with session_summary, long_term_memories,
            temporal_results, daily_logs, knowledge_awareness, and
            total_tokens_estimate.
        """
        from modules.memory.context_router import ContextRouter, ContextBundle

        router = ContextRouter()
        try:
            return await router.retrieve_context(
                workspace_id=workspace_id,
                agent_id=agent_id,
                query=query,
                conversation_id=conversation_id,
                context_window=context_window,
            )
        except Exception:
            logger.error(
                "[UnifiedMemoryService] retrieve_context failed ws=%s agent=%s",
                workspace_id,
                agent_id,
                exc_info=True,
            )
            return ContextBundle()

    # PRD-142 W3-S7 — ``store_exchange`` retired (G12 collapse). The canonical
    # L2 write per chat turn is ``store_transcript`` (PRD-131d Phase 3); the
    # older ``content_type='exchange'`` row was a duplicate alongside the
    # transcript and is no longer written. No callers remain in production code.

    async def store_transcript(
        self,
        workspace_id: str,
        turns: List[Dict[str, str]],
        agent_id: Optional[int] = None,
        conversation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Store a raw multi-turn transcript in L2 short-term memory — PRD-131d Phase 3.

        The transcript is preserved verbatim (up to a generous char cap) so
        agents and users can read back the full conversation context, not just
        the distilled facts the L3 store carries.

        Args:
            workspace_id: Workspace scope (UUID string).
            turns: Ordered list of {"role": "user"|"assistant"|"system", "content": "..."}.
            agent_id: Optional agent scope.
            conversation_id: Optional chat session ID for grouping.
            metadata: Optional extra metadata (merged into the row).

        Returns:
            The L2 row UUID as string, or None on failure.
        """
        if not turns:
            return None

        # Serialize turns as readable multi-line text; cap at 20k chars total
        # (roughly 5k tokens — well above a distilled fact's size).
        lines: List[str] = []
        for turn in turns:
            role = str(turn.get("role", "user")).strip() or "user"
            content = str(turn.get("content", "")).strip()
            if not content:
                continue
            lines.append(f"{role.capitalize()}: {content}")
        body = "\n\n".join(lines)
        if not body:
            return None
        body = body[:20000]

        turn_count = len(lines)
        first_user = next(
            (t.get("content", "")[:200] for t in turns if t.get("role") == "user"),
            "",
        )
        last_assistant = next(
            (
                t.get("content", "")[:200]
                for t in reversed(turns)
                if t.get("role") == "assistant"
            ),
            "",
        )

        meta: Dict[str, Any] = {
            "conversation_id": conversation_id,
            "agent_id": agent_id,
            "turn_count": turn_count,
            "first_user_msg": first_user,
            "last_assistant_msg": last_assistant,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if metadata:
            meta.update(metadata)

        row_id = await self.store_short_term(
            workspace_id=workspace_id,
            content=body,
            content_type="transcript",
            agent_id=agent_id,
            importance=0.6,
            metadata=meta,
        )

        if row_id:
            logger.info(
                "[UnifiedMemoryService] store_transcript L2 id=%s ws=%s turns=%d chars=%d",
                row_id, workspace_id, turn_count, len(body),
            )
        return row_id

    @staticmethod
    def _list_short_term_by_type_sync(
        workspace_id: str,
        content_types: Optional[List[str]],
        limit: int,
        query: Optional[str],
    ) -> List[Dict[str, Any]]:
        """
        List L2 short-term memories filtered by content_type list (sync).

        If query is provided, performs case-insensitive substring match on
        content. Results ordered by created_at DESC.
        """
        from core.database.database import get_db_session
        from modules.memory.models import MemoryShortTerm

        with get_db_session() as db:
            q = (
                db.query(MemoryShortTerm)
                .filter(
                    MemoryShortTerm.workspace_id == workspace_id,
                    MemoryShortTerm.archived_at.is_(None),
                )
            )
            if content_types:
                q = q.filter(MemoryShortTerm.content_type.in_(content_types))
            if query:
                q = q.filter(MemoryShortTerm.content.ilike(f"%{query}%"))
            rows = q.order_by(MemoryShortTerm.created_at.desc()).limit(limit).all()
            return [
                {
                    "id": str(row.id),
                    "content": row.content,
                    "content_type": row.content_type,
                    "importance": row.importance,
                    "decay_score": row.decay_score,
                    "access_count": row.access_count,
                    "metadata": row.metadata_ or {},
                    "created_at": row.created_at.isoformat() if row.created_at else None,
                    "last_accessed_at": row.last_accessed_at.isoformat() if row.last_accessed_at else None,
                }
                for row in rows
            ]

    async def list_short_term_by_type(
        self,
        workspace_id: str,
        content_types: Optional[List[str]] = None,
        limit: int = 50,
        query: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        List L2 rows for a workspace, optionally filtered by content_type.

        Used by the Memory Explorer to surface raw transcripts, mission
        summaries, task failures, and retry recoveries that never make it to
        L3 — or whose durable version was distilled down to a fragment.
        """
        try:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None,
                self._list_short_term_by_type_sync,
                workspace_id,
                content_types,
                limit,
                query,
            )
        except Exception:
            logger.error(
                "[UnifiedMemoryService] list_short_term_by_type failed ws=%s",
                workspace_id,
                exc_info=True,
            )
            return []

    async def promote_to_long_term(self, memory_id: str) -> bool:
        """
        Promote a single L2 item to L3 long-term memory (durable store).

        Reads the L2 row, writes its content verbatim into the durable
        store (same-content writes dedup on content hash), then marks the
        row as promoted. The L2 row is NOT deleted — it stays until decay
        archives it (belt and suspenders).

        Args:
            memory_id: UUID string of the memory_short_term row.

        Returns:
            True if promotion succeeded, False otherwise.
        """
        try:
            loop = asyncio.get_event_loop()
            row_data = await loop.run_in_executor(
                None, self._read_l2_row_sync, memory_id,
            )
            if row_data is None:
                logger.warning(
                    "[UnifiedMemoryService] promote_to_long_term: row not found id=%s",
                    memory_id,
                )
                return False

            workspace_id = row_data["workspace_id"]
            agent_id = row_data.get("agent_id")
            content = row_data["content"]
            content_type = row_data.get("content_type", "exchange")
            metadata = row_data.get("metadata", {})

            # Store in L3 via the durable adapter (content-hash dedup applies)
            result = await self.store_long_term(
                workspace_id=workspace_id,
                content=content,
                agent_id=agent_id,
                category=content_type,
                metadata={
                    "promoted_from_l2": str(memory_id),
                    **(metadata if isinstance(metadata, dict) else {}),
                },
            )

            if isinstance(result, dict) and result.get("success") is False:
                logger.error(
                    "[UnifiedMemoryService] promote_to_long_term: L3 store failed id=%s",
                    memory_id,
                )
                return False

            # Mark as promoted in L2
            await loop.run_in_executor(
                None, self._mark_promoted_sync, memory_id,
            )
            logger.info(
                "[UnifiedMemoryService] promote_to_long_term success id=%s ws=%s",
                memory_id,
                workspace_id,
            )
            return True

        except Exception:
            logger.error(
                "[UnifiedMemoryService] promote_to_long_term failed id=%s",
                memory_id,
                exc_info=True,
            )
            return False

    @staticmethod
    def _read_l2_row_sync(memory_id: str) -> Optional[Dict[str, Any]]:
        """Read a single L2 row by ID (synchronous, runs in executor)."""
        from core.database.database import get_db_session
        from modules.memory.models import MemoryShortTerm

        with get_db_session() as db:
            row = db.query(MemoryShortTerm).filter(
                MemoryShortTerm.id == memory_id,
            ).first()
            if row is None:
                return None
            return {
                "workspace_id": str(row.workspace_id),
                "agent_id": row.agent_id,
                "content": row.content,
                "content_type": row.content_type,
                "importance": row.importance,
                "access_count": row.access_count,
                "metadata": row.metadata_ if row.metadata_ else {},
            }

    @staticmethod
    def _mark_promoted_sync(memory_id: str) -> None:
        """Mark an L2 row as promoted to L3 (synchronous, runs in executor)."""
        from core.database.database import get_db_session
        from modules.memory.models import MemoryShortTerm

        with get_db_session() as db:
            row = db.query(MemoryShortTerm).filter(
                MemoryShortTerm.id == memory_id,
            ).first()
            if row:
                row.promoted_to_l3 = True
                row.promoted_at = datetime.now(timezone.utc)
                db.flush()

    @staticmethod
    def _get_promotion_candidates_sync(
        workspace_id: str,
        batch_size: int,
    ) -> List[Dict[str, Any]]:
        """
        Fetch L2 rows eligible for promotion to L3 (synchronous, runs in executor).

        Eligibility is the shared type-aware policy (PRD-187 S4,
        ``modules/memory/promotion_policy``): distilled importance with a lower
        bar for high-signal types, noise types never — the old
        ``AND access_count > N`` conjunct (a bootstrap deadlock that produced
        zero promotions ever) is gone.
        """
        from core.database.database import get_db_session
        from modules.memory.models import MemoryShortTerm
        from modules.memory.promotion_policy import eligibility_conditions

        with get_db_session() as db:
            rows = (
                db.query(MemoryShortTerm)
                .filter(*eligibility_conditions(MemoryShortTerm, workspace_id))
                .order_by(MemoryShortTerm.importance.desc())
                .limit(batch_size)
                .all()
            )
            return [
                {
                    "id": str(row.id),
                    "workspace_id": str(row.workspace_id),
                    "agent_id": row.agent_id,
                    "content": row.content,
                    "content_type": row.content_type,
                    "importance": row.importance,
                    "access_count": row.access_count,
                    "metadata": row.metadata_ if row.metadata_ else {},
                }
                for row in rows
            ]

    async def run_promotion(self, workspace_id: str) -> Dict[str, int]:
        """
        Run L2→L3 promotion for a single workspace.

        Finds L2 items meeting the type-aware eligibility policy (PRD-187 S4:
        distilled importance, lower bar for high-signal types, noise types
        never, no access-count gate), then promotes each to L3 via the durable
        store.

        Args:
            workspace_id: Workspace to process.

        Returns:
            {"promoted": N, "failed": M} counts.
        """
        from config import config

        try:
            loop = asyncio.get_event_loop()
            candidates = await loop.run_in_executor(
                None,
                self._get_promotion_candidates_sync,
                workspace_id,
                config.MEMORY_PROMOTION_BATCH_SIZE,
            )
        except Exception:
            logger.error(
                "[UnifiedMemoryService] run_promotion: failed to fetch candidates ws=%s",
                workspace_id,
                exc_info=True,
            )
            return {"promoted": 0, "failed": 0}

        if not candidates:
            return {"promoted": 0, "failed": 0}

        promoted = 0
        failed = 0

        for candidate in candidates:
            success = await self.promote_to_long_term(candidate["id"])
            if success:
                promoted += 1
            else:
                failed += 1

        logger.info(
            "[UnifiedMemoryService] run_promotion ws=%s promoted=%d failed=%d",
            workspace_id,
            promoted,
            failed,
        )
        return {"promoted": promoted, "failed": failed}

    async def run_promotion_all(self) -> Dict[str, Any]:
        """
        Run L2→L3 promotion across ALL workspaces with eligible items.

        Iterates distinct workspace_ids from memory_short_term and calls
        run_promotion() per workspace. One workspace failure does not stop
        processing others.

        Returns:
            {"workspaces_processed": N, "total_promoted": M, "total_failed": K,
             "errors": E}
        """
        try:
            loop = asyncio.get_event_loop()
            workspace_ids = await loop.run_in_executor(
                None, self._get_active_workspace_ids_sync,
            )
        except Exception:
            logger.error(
                "[UnifiedMemoryService] run_promotion_all: failed to fetch workspace_ids",
                exc_info=True,
            )
            return {
                "workspaces_processed": 0,
                "total_promoted": 0,
                "total_failed": 0,
                "errors": 1,
            }

        total_promoted = 0
        total_failed = 0
        errors = 0

        for ws_id in workspace_ids:
            try:
                result = await self.run_promotion(str(ws_id))
                total_promoted += result["promoted"]
                total_failed += result["failed"]
            except Exception:
                logger.error(
                    "[UnifiedMemoryService] run_promotion_all: ws=%s failed",
                    ws_id,
                    exc_info=True,
                )
                errors += 1

        logger.info(
            "[UnifiedMemoryService] run_promotion_all complete: "
            "workspaces=%d promoted=%d failed=%d errors=%d",
            len(workspace_ids),
            total_promoted,
            total_failed,
            errors,
        )
        return {
            "workspaces_processed": len(workspace_ids),
            "total_promoted": total_promoted,
            "total_failed": total_failed,
            "errors": errors,
        }

    async def consolidate(self, workspace_id: str) -> None:
        """Run weekly consolidation for a workspace. Implemented later."""
        pass


# ---------------------------------------------------------------------------
# Module-level convenience function
# ---------------------------------------------------------------------------

def get_unified_memory_service() -> UnifiedMemoryService:
    """Get the singleton UnifiedMemoryService instance."""
    return UnifiedMemoryService.get_instance()
