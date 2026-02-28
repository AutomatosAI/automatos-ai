"""
Neural Field Store
==================

PRD-59 Phase 4: Redis + pgvector storage for the shared neural field.

Each execution has its own field scope (no cross-execution contamination).
Agents write contributions (key-value + embedding) and read by key or
semantic similarity.

Storage layers:
  Phase 1: In-memory dict (same-process, no Redis needed)
  Phase 2: Redis HSET/HGET (cross-process sharing)
  Phase 3: Redis + pgvector (semantic retrieval)
"""

import logging
import time
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class FieldContribution:
    """A single agent's contribution to the neural field"""
    agent_id: int
    agent_name: str
    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    relevance_score: float = 1.0

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp


class NeuralFieldStore:
    """
    Storage backend for the neural field.

    Phase 1 (current): In-memory dict per execution_id.
    Phase 2: Redis-backed for cross-process sharing.
    Phase 3: pgvector for semantic retrieval.

    Usage:
        store = NeuralFieldStore()
        store.write("exec-123", contribution)
        items = store.read_by_key("exec-123", "agent_1")
        similar = store.read_by_similarity("exec-123", query_embedding, top_k=5)
    """

    def __init__(self, redis_client: Optional[Any] = None):
        """
        Args:
            redis_client: Optional Redis client for Phase 2+.
                          If None, uses in-memory storage.
        """
        self.redis = redis_client
        self._memory: Dict[str, List[FieldContribution]] = {}
        self._use_redis = redis_client is not None

        if self._use_redis:
            logger.info("NeuralFieldStore initialized with Redis backing")
        else:
            logger.info("NeuralFieldStore initialized (in-memory mode)")

    def write(
        self,
        execution_id: str,
        contribution: FieldContribution,
    ) -> None:
        """Write a contribution to the field"""
        if self._use_redis:
            self._write_redis(execution_id, contribution)
        else:
            self._write_memory(execution_id, contribution)

    def read_by_agent(
        self,
        execution_id: str,
        agent_id: int,
    ) -> List[FieldContribution]:
        """Read all contributions from a specific agent"""
        all_items = self._get_all(execution_id)
        return [c for c in all_items if c.agent_id == agent_id]

    def read_all(
        self,
        execution_id: str,
        exclude_agent_id: Optional[int] = None,
    ) -> List[FieldContribution]:
        """Read all contributions, optionally excluding one agent"""
        all_items = self._get_all(execution_id)
        if exclude_agent_id is not None:
            return [c for c in all_items if c.agent_id != exclude_agent_id]
        return all_items

    def read_by_similarity(
        self,
        execution_id: str,
        query_embedding: List[float],
        top_k: int = 5,
        exclude_agent_id: Optional[int] = None,
        min_similarity: float = 0.7,
    ) -> List[FieldContribution]:
        """Semantic retrieval: find contributions similar to query embedding"""
        candidates = self.read_all(execution_id, exclude_agent_id)

        # Filter to those with embeddings
        with_embeddings = [c for c in candidates if c.embedding is not None]
        if not with_embeddings:
            return candidates[:top_k]

        # Compute cosine similarity
        scored = []
        for contrib in with_embeddings:
            sim = self._cosine_similarity(query_embedding, contrib.embedding)
            if sim >= min_similarity:
                contrib.relevance_score = sim
                scored.append(contrib)

        # Sort by similarity, return top_k
        scored.sort(key=lambda c: c.relevance_score, reverse=True)
        return scored[:top_k]

    def clear(self, execution_id: str) -> None:
        """Clear all field data for an execution"""
        if self._use_redis:
            try:
                self.redis.delete(f"swarm:{execution_id}:field")
            except Exception as e:
                logger.warning(f"Redis clear failed: {e}")
        self._memory.pop(execution_id, None)

    def get_contribution_count(self, execution_id: str) -> int:
        """Count contributions in the field"""
        return len(self._get_all(execution_id))

    # ── Internal methods ──

    def _write_memory(self, execution_id: str, contribution: FieldContribution) -> None:
        if execution_id not in self._memory:
            self._memory[execution_id] = []
        self._memory[execution_id].append(contribution)

    def _write_redis(self, execution_id: str, contribution: FieldContribution) -> None:
        import json
        key = f"swarm:{execution_id}:field"
        field_key = f"{contribution.agent_id}:{contribution.timestamp}"
        data = {
            "agent_id": contribution.agent_id,
            "agent_name": contribution.agent_name,
            "content": contribution.content,
            "metadata": contribution.metadata,
            "timestamp": contribution.timestamp,
        }
        try:
            self.redis.hset(key, field_key, json.dumps(data))
            # TTL: 2 hours (execution duration + buffer)
            self.redis.expire(key, 7200)
        except Exception as e:
            logger.warning(f"Redis write failed, falling back to memory: {e}")
            self._write_memory(execution_id, contribution)

    def _get_all(self, execution_id: str) -> List[FieldContribution]:
        if self._use_redis:
            return self._get_all_redis(execution_id)
        return list(self._memory.get(execution_id, []))

    def _get_all_redis(self, execution_id: str) -> List[FieldContribution]:
        import json
        key = f"swarm:{execution_id}:field"
        try:
            raw = self.redis.hgetall(key)
            contributions = []
            for _field_key, value in raw.items():
                data = json.loads(value)
                contributions.append(FieldContribution(
                    agent_id=data["agent_id"],
                    agent_name=data["agent_name"],
                    content=data["content"],
                    metadata=data.get("metadata", {}),
                    timestamp=data.get("timestamp", time.time()),
                ))
            return contributions
        except Exception as e:
            logger.warning(f"Redis read failed: {e}")
            return list(self._memory.get(execution_id, []))

    @staticmethod
    def _cosine_similarity(a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        if len(a) != len(b) or len(a) == 0:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
