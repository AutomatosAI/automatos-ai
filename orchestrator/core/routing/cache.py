"""
Routing cache with TTL and workspace scoping (PRD-50: US-002).

In-memory cache for routing decisions keyed by (workspace_id, hash(content + source)).
Supports configurable TTL, lazy eviction, correction tracking, and cache statistics.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from uuid import UUID

from core.models.routing import ChannelSource, RoutingDecision


# Default TTL: 24 hours, configurable via env var
_CACHE_TTL_HOURS = int(os.getenv("ROUTING_CACHE_TTL_HOURS", "24"))
_CACHE_TTL_SECONDS = _CACHE_TTL_HOURS * 3600


def _normalize_content(content: str) -> str:
    """Normalize content for consistent hashing (lowercase, strip whitespace)."""
    return content.strip().lower()


def _cache_key(workspace_id: UUID, content: str, source: ChannelSource) -> str:
    """Build a deterministic cache key from workspace, content, and source."""
    normalized = _normalize_content(content) + "|" + source.value
    content_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"{workspace_id}:{content_hash}"


@dataclass
class _CacheEntry:
    """Internal cache entry with TTL tracking."""

    decision: RoutingDecision
    created_at: float
    correction_counts: Dict[int, int] = field(default_factory=dict)
    # Track how many times each agent_id has been suggested as correction


class RoutingCache:
    """In-memory routing decision cache with TTL and workspace scoping."""

    def __init__(self, ttl_seconds: Optional[int] = None) -> None:
        self._store: Dict[str, _CacheEntry] = {}
        self._ttl = ttl_seconds if ttl_seconds is not None else _CACHE_TTL_SECONDS
        self._hits = 0
        self._misses = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(
        self, workspace_id: UUID, content: str, source: ChannelSource
    ) -> Optional[RoutingDecision]:
        """Return a cached RoutingDecision or None.

        Performs lazy eviction — expired entries are removed on access.
        """
        key = _cache_key(workspace_id, content, source)
        entry = self._store.get(key)

        if entry is None:
            self._misses += 1
            return None

        # Lazy eviction
        if self._is_expired(entry):
            del self._store[key]
            self._misses += 1
            return None

        self._hits += 1
        decision = entry.decision.model_copy()
        decision.cached = True
        return decision

    def put(
        self,
        workspace_id: UUID,
        content: str,
        source: ChannelSource,
        decision: RoutingDecision,
    ) -> None:
        """Store a routing decision in the cache."""
        key = _cache_key(workspace_id, content, source)
        self._store[key] = _CacheEntry(
            decision=decision.model_copy(),
            created_at=time.monotonic(),
        )

    def record_correction(
        self,
        workspace_id: UUID,
        content: str,
        source: ChannelSource,
        correct_agent_id: int,
    ) -> None:
        """Record a user correction.

        If the same correction is made 2+ times for the same key,
        the cached entry is updated to reflect the corrected agent.
        """
        key = _cache_key(workspace_id, content, source)
        entry = self._store.get(key)

        if entry is None:
            return

        # Increment correction count for this agent_id
        entry.correction_counts[correct_agent_id] = (
            entry.correction_counts.get(correct_agent_id, 0) + 1
        )

        # If corrected 2+ times with the same agent, update the cached decision
        if entry.correction_counts[correct_agent_id] >= 2:
            entry.decision = entry.decision.model_copy(
                update={
                    "agent_id": correct_agent_id,
                    "route_type": "agent",
                    "confidence": 1.0,
                    "reasoning": "Corrected by user (2+ corrections)",
                }
            )
            # Reset the timer so the corrected entry stays fresh
            entry.created_at = time.monotonic()

    def clear(self, workspace_id: UUID) -> None:
        """Clear all cached entries for a specific workspace."""
        prefix = f"{workspace_id}:"
        keys_to_remove = [k for k in self._store if k.startswith(prefix)]
        for key in keys_to_remove:
            del self._store[key]

    def stats(self) -> Dict:
        """Return cache statistics.

        Returns dict with: size, hits, misses, hit_rate, top_routes.
        """
        # Evict expired entries first for accurate stats
        self._evict_expired()

        total_requests = self._hits + self._misses
        hit_rate = (self._hits / total_requests) if total_requests > 0 else 0.0

        # Compute top cached routes (agent_id -> count)
        route_counts: Dict[Optional[int], int] = {}
        for entry in self._store.values():
            agent_id = entry.decision.agent_id
            route_counts[agent_id] = route_counts.get(agent_id, 0) + 1

        top_routes = sorted(route_counts.items(), key=lambda x: x[1], reverse=True)[
            :10
        ]

        return {
            "size": len(self._store),
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 4),
            "top_routes": [
                {"agent_id": agent_id, "count": count}
                for agent_id, count in top_routes
            ],
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _is_expired(self, entry: _CacheEntry) -> bool:
        return (time.monotonic() - entry.created_at) > self._ttl

    def _evict_expired(self) -> None:
        """Remove all expired entries from the store."""
        expired_keys = [
            k for k, v in self._store.items() if self._is_expired(v)
        ]
        for key in expired_keys:
            del self._store[key]
