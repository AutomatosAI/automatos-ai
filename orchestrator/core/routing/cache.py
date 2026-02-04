"""
Routing cache with TTL and workspace scoping (PRD-50: US-002).

Redis-backed cache for routing decisions keyed by (workspace_id, hash(content + source)).
Uses the existing RedisClient infrastructure (core.redis.client) with key prefixes
following the DatabaseCacheService pattern.

Falls back gracefully to cache-miss behaviour when Redis is unavailable.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import Dict, List, Optional
from uuid import UUID

from core.models.routing import ChannelSource, RoutingDecision

logger = logging.getLogger(__name__)

# Key prefixes (consistent with db:schema:, db:query:, etc.)
_PREFIX_DECISION = "routing:decision:"
_PREFIX_CORRECTION = "routing:corr:"
_PREFIX_STATS = "routing:stats:"

# Default TTL: 24 hours — overridden by config at init time
_DEFAULT_TTL_SECONDS = 24 * 3600


def _normalize_content(content: str) -> str:
    """Normalize content for consistent hashing (lowercase, strip whitespace)."""
    return content.strip().lower()


def _content_hash(workspace_id: UUID, content: str, source: ChannelSource) -> str:
    """Build a deterministic hash from workspace, content, and source."""
    normalized = _normalize_content(content) + "|" + source.value
    h = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{workspace_id}:{h}"


class RoutingCache:
    """Redis-backed routing decision cache with TTL and workspace scoping.

    Gracefully degrades — all public methods return None / no-op when Redis
    is unavailable, so the router simply skips Tier 1.
    """

    def __init__(self, ttl_seconds: Optional[int] = None) -> None:
        self._ttl = ttl_seconds or self._load_ttl()
        self._redis = self._get_redis()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self, workspace_id: UUID, content: str, source: ChannelSource
    ) -> Optional[RoutingDecision]:
        """Return a cached RoutingDecision or None."""
        redis_conn = self._get_redis()
        if redis_conn is None:
            return None

        key = _PREFIX_DECISION + _content_hash(workspace_id, content, source)
        try:
            raw = redis_conn.get(key)
            if raw is None:
                self._incr_stat("misses")
                return None

            self._incr_stat("hits")
            decision = RoutingDecision.model_validate_json(raw)
            decision.cached = True
            return decision
        except Exception as e:
            logger.warning(f"Routing cache get failed: {e}")
            return None

    def put(
        self,
        workspace_id: UUID,
        content: str,
        source: ChannelSource,
        decision: RoutingDecision,
    ) -> None:
        """Store a routing decision in the cache."""
        redis_conn = self._get_redis()
        if redis_conn is None:
            return

        key = _PREFIX_DECISION + _content_hash(workspace_id, content, source)
        try:
            redis_conn.setex(key, self._ttl, decision.model_dump_json())
        except Exception as e:
            logger.warning(f"Routing cache put failed: {e}")

    def record_correction(
        self,
        workspace_id: UUID,
        content: str,
        source: ChannelSource,
        correct_agent_id: int,
    ) -> None:
        """Record a user correction.

        If the same correction is made 2+ times for the same key,
        the cached decision is updated to reflect the corrected agent.
        """
        redis_conn = self._get_redis()
        if redis_conn is None:
            return

        cache_hash = _content_hash(workspace_id, content, source)
        correction_key = f"{_PREFIX_CORRECTION}{cache_hash}:{correct_agent_id}"
        decision_key = _PREFIX_DECISION + cache_hash

        try:
            count = redis_conn.incr(correction_key)
            # Expire correction counters with the same TTL as decisions
            redis_conn.expire(correction_key, self._ttl)

            if count >= 2:
                # Check if there's an existing cached decision to update
                raw = redis_conn.get(decision_key)
                if raw:
                    decision = RoutingDecision.model_validate_json(raw)
                    corrected = decision.model_copy(
                        update={
                            "agent_id": correct_agent_id,
                            "route_type": "agent",
                            "confidence": 1.0,
                            "reasoning": "Corrected by user (2+ corrections)",
                        }
                    )
                    redis_conn.setex(
                        decision_key, self._ttl, corrected.model_dump_json()
                    )
                    logger.info(
                        f"Routing cache updated via correction: "
                        f"workspace={workspace_id} agent={correct_agent_id}"
                    )
        except Exception as e:
            logger.warning(f"Routing cache record_correction failed: {e}")

    def clear(self, workspace_id: UUID) -> int:
        """Clear all cached entries for a specific workspace."""
        redis_conn = self._get_redis()
        if redis_conn is None:
            return 0

        try:
            pattern = f"{_PREFIX_DECISION}{workspace_id}:*"
            keys = list(redis_conn.scan_iter(pattern))

            corr_pattern = f"{_PREFIX_CORRECTION}{workspace_id}:*"
            keys.extend(redis_conn.scan_iter(corr_pattern))

            if keys:
                deleted = redis_conn.delete(*keys)
                logger.info(
                    f"Cleared {deleted} routing cache entries for workspace {workspace_id}"
                )
                return deleted
            return 0
        except Exception as e:
            logger.warning(f"Routing cache clear failed: {e}")
            return 0

    def stats(self) -> Dict:
        """Return cache statistics: size, hits, misses, hit_rate, top_routes."""
        redis_conn = self._get_redis()
        if redis_conn is None:
            return {"size": 0, "hits": 0, "misses": 0, "hit_rate": 0.0, "top_routes": []}

        try:
            hits = self._get_stat("hits")
            misses = self._get_stat("misses")
            total = hits + misses
            hit_rate = round(hits / total, 4) if total > 0 else 0.0

            # Count cached decisions and group by agent_id for top routes
            route_counts: Dict[Optional[int], int] = {}
            size = 0
            for key in redis_conn.scan_iter(f"{_PREFIX_DECISION}*"):
                size += 1
                raw = redis_conn.get(key)
                if raw:
                    try:
                        decision = RoutingDecision.model_validate_json(raw)
                        aid = decision.agent_id
                        route_counts[aid] = route_counts.get(aid, 0) + 1
                    except Exception:
                        pass

            top_routes = sorted(
                route_counts.items(), key=lambda x: x[1], reverse=True
            )[:10]

            return {
                "size": size,
                "hits": hits,
                "misses": misses,
                "hit_rate": hit_rate,
                "top_routes": [
                    {"agent_id": agent_id, "count": count}
                    for agent_id, count in top_routes
                ],
            }
        except Exception as e:
            logger.warning(f"Routing cache stats failed: {e}")
            return {"size": 0, "hits": 0, "misses": 0, "hit_rate": 0.0, "top_routes": []}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_ttl() -> int:
        """Load TTL from centralized config."""
        try:
            from config import config
            return config.ROUTING_CACHE_TTL_HOURS * 3600
        except Exception:
            return _DEFAULT_TTL_SECONDS

    @staticmethod
    def _get_redis():
        """Get a Redis connection via the centralized client. Returns None if unavailable."""
        try:
            from core.redis.client import get_redis_client
            client = get_redis_client()
            if client is None:
                return None
            return client.get_redis()
        except Exception:
            return None

    def _incr_stat(self, name: str) -> None:
        """Increment a routing stats counter in Redis."""
        redis_conn = self._get_redis()
        if redis_conn is None:
            return
        try:
            redis_conn.incr(f"{_PREFIX_STATS}{name}")
        except Exception:
            pass  # Stats are non-critical

    def _get_stat(self, name: str) -> int:
        """Read a routing stats counter from Redis."""
        redis_conn = self._get_redis()
        if redis_conn is None:
            return 0
        try:
            val = redis_conn.get(f"{_PREFIX_STATS}{name}")
            return int(val) if val else 0
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_routing_cache: Optional[RoutingCache] = None


def get_routing_cache() -> RoutingCache:
    """Get or create the singleton RoutingCache instance.

    All consumers (chat.py, composio.py, routing.py) share this one instance.
    """
    global _routing_cache
    if _routing_cache is None:
        _routing_cache = RoutingCache()
    return _routing_cache
