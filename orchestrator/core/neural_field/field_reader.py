"""
Neural Field Reader
====================

PRD-59 Phase 4: Semantic retrieval from the shared neural field.
Agents use FieldReader to get relevant context from OTHER agents'
contributions — coordinating via field dynamics rather than explicit messaging.

Features:
  - Semantic similarity search (cosine > 0.7 threshold)
  - Relevance × recency weighting
  - Token budget cap (2000 tokens max)
  - Excludes self-contributions (no echo)
"""

import logging
import time
from typing import Any, Dict, List, Optional

from .field_store import NeuralFieldStore, FieldContribution

logger = logging.getLogger(__name__)


class FieldReader:
    """
    Reads relevant context from the neural field for an agent.

    Usage:
        reader = FieldReader(field_store)
        context = await reader.read(
            execution_id="exec-123",
            agent_id=42,
            query_embedding=embedding,
        )
    """

    DEFAULT_TOP_K = 5
    DEFAULT_MIN_SIMILARITY = 0.7
    DEFAULT_MAX_TOKENS = 2000
    RECENCY_HALF_LIFE = 300  # 5 minutes half-life for temporal weighting

    def __init__(self, field_store: NeuralFieldStore):
        self.store = field_store

    async def read(
        self,
        execution_id: str,
        agent_id: int,
        query_embedding: Optional[List[float]] = None,
        query_text: Optional[str] = None,
        top_k: int = DEFAULT_TOP_K,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> List[FieldContribution]:
        """
        Read relevant contributions from other agents.

        Args:
            execution_id: Current execution scope
            agent_id: Current agent (excluded from results)
            query_embedding: Embedding of the current query/task
            query_text: Text query (if embedding not available)
            top_k: Max number of contributions to return
            min_similarity: Minimum cosine similarity threshold
            max_tokens: Max total tokens in returned contributions

        Returns:
            List of FieldContribution objects, ranked by relevance × recency
        """
        if query_embedding is not None:
            # Semantic retrieval
            contributions = self.store.read_by_similarity(
                execution_id=execution_id,
                query_embedding=query_embedding,
                top_k=top_k * 2,  # Over-fetch, then filter by token budget
                exclude_agent_id=agent_id,
                min_similarity=min_similarity,
            )
        else:
            # Fallback: return all contributions from other agents
            contributions = self.store.read_all(
                execution_id=execution_id,
                exclude_agent_id=agent_id,
            )

        if not contributions:
            return []

        # Apply recency weighting
        now = time.time()
        for c in contributions:
            age = now - c.timestamp
            recency_weight = 2 ** (-age / self.RECENCY_HALF_LIFE)
            c.relevance_score = c.relevance_score * 0.7 + recency_weight * 0.3

        # Sort by weighted score
        contributions.sort(key=lambda c: c.relevance_score, reverse=True)

        # Apply token budget
        result = []
        total_tokens = 0
        for c in contributions[:top_k]:
            estimated_tokens = len(c.content) // 4  # rough estimate
            if total_tokens + estimated_tokens > max_tokens:
                break
            result.append(c)
            total_tokens += estimated_tokens

        logger.debug(
            f"FieldReader: execution={execution_id}, agent={agent_id}, "
            f"returned {len(result)} contributions (~{total_tokens} tokens)"
        )

        return result

    async def read_as_context_string(
        self,
        execution_id: str,
        agent_id: int,
        query_embedding: Optional[List[float]] = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> str:
        """
        Read field contributions and format as a context string
        ready for injection into an agent's prompt.
        """
        contributions = await self.read(
            execution_id=execution_id,
            agent_id=agent_id,
            query_embedding=query_embedding,
            max_tokens=max_tokens,
        )

        if not contributions:
            return ""

        parts = ["[Neural Field — Other Agents' Findings]"]
        for c in contributions:
            parts.append(
                f"\n[{c.agent_name}] (relevance: {c.relevance_score:.0%}):\n"
                f"{c.content[:500]}"
            )

        return "\n".join(parts)
