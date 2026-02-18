"""
Field Coherence Metric
=======================

PRD-59 Phase 4: Measures whether agents in a multi-agent workflow
are converging (agreeing) or diverging (conflicting).

Coherence metric:
  C = 1 - (1/N²) × Σᵢ Σⱼ ||Ψᵢ - Ψⱼ||²

Where:
  C → 1 when all agents converge (agreement)
  C → 0 when agents diverge (conflict)
  Ψᵢ is agent i's contribution embedding

Use cases:
  - After each parallel execution group: check if agents agree
  - If coherence < 0.5: flag potential conflict
  - Include in workflow analytics
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from .field_store import NeuralFieldStore, FieldContribution

logger = logging.getLogger(__name__)


class FieldCoherence:
    """
    Measures field coherence — how well agents agree.

    Usage:
        coherence = FieldCoherence(field_store)
        score = coherence.compute("exec-123")
        if score < 0.5:
            logger.warning("Agents are diverging!")
    """

    CONFLICT_THRESHOLD = 0.5
    WARNING_THRESHOLD = 0.6

    def __init__(self, field_store: NeuralFieldStore):
        self.store = field_store

    def compute(self, execution_id: str) -> float:
        """
        Compute field coherence for an execution.

        C = 1 - (1/N²) × Σᵢ Σⱼ ||Ψᵢ - Ψⱼ||²

        Returns:
            Coherence score (0-1). Higher = more agreement.
            Returns 1.0 if fewer than 2 agents have contributions.
        """
        contributions = self.store.read_all(execution_id)

        # Group by agent — use the latest contribution per agent
        agent_contributions: Dict[int, FieldContribution] = {}
        for c in contributions:
            existing = agent_contributions.get(c.agent_id)
            if existing is None or c.timestamp > existing.timestamp:
                agent_contributions[c.agent_id] = c

        # Need at least 2 agents to measure coherence
        if len(agent_contributions) < 2:
            return 1.0

        agents = list(agent_contributions.values())

        # Check if embeddings are available
        with_embeddings = [a for a in agents if a.embedding is not None]
        if len(with_embeddings) < 2:
            # Fall back to text similarity if no embeddings
            return self._text_coherence(agents)

        return self._embedding_coherence(with_embeddings)

    def compute_detailed(
        self, execution_id: str
    ) -> Dict[str, Any]:
        """
        Compute detailed coherence report.

        Returns:
            Dict with coherence score, per-agent details, and conflict flags.
        """
        score = self.compute(execution_id)
        contributions = self.store.read_all(execution_id)

        # Group by agent
        agent_counts: Dict[str, int] = {}
        for c in contributions:
            agent_counts[c.agent_name] = agent_counts.get(c.agent_name, 0) + 1

        return {
            "coherence_score": score,
            "is_coherent": score >= self.WARNING_THRESHOLD,
            "has_conflict": score < self.CONFLICT_THRESHOLD,
            "num_agents": len(agent_counts),
            "total_contributions": len(contributions),
            "agent_contribution_counts": agent_counts,
            "threshold_warning": self.WARNING_THRESHOLD,
            "threshold_conflict": self.CONFLICT_THRESHOLD,
        }

    def _embedding_coherence(self, agents: List[FieldContribution]) -> float:
        """Compute coherence using embedding distances"""
        n = len(agents)
        if n < 2:
            return 1.0

        total_squared_dist = 0.0
        pairs = 0

        for i in range(n):
            for j in range(i + 1, n):
                dist_sq = self._squared_distance(
                    agents[i].embedding, agents[j].embedding
                )
                total_squared_dist += dist_sq
                pairs += 1

        if pairs == 0:
            return 1.0

        # Average squared distance, normalized
        avg_dist = total_squared_dist / pairs

        # Coherence: 1 - normalized distance
        # Embeddings are typically unit vectors, so max dist² ≈ 2
        coherence = max(0.0, min(1.0, 1.0 - avg_dist / 2.0))

        return coherence

    def _text_coherence(self, agents: List[FieldContribution]) -> float:
        """Fallback: rough coherence estimate from text overlap"""
        if len(agents) < 2:
            return 1.0

        # Simple word overlap as proxy
        word_sets = []
        for a in agents:
            words = set(a.content.lower().split())
            word_sets.append(words)

        total_overlap = 0.0
        pairs = 0
        for i in range(len(word_sets)):
            for j in range(i + 1, len(word_sets)):
                union = word_sets[i] | word_sets[j]
                if union:
                    intersection = word_sets[i] & word_sets[j]
                    total_overlap += len(intersection) / len(union)
                pairs += 1

        if pairs == 0:
            return 1.0

        return total_overlap / pairs

    @staticmethod
    def _squared_distance(a: List[float], b: List[float]) -> float:
        """Compute squared Euclidean distance between two vectors"""
        if len(a) != len(b):
            return 2.0  # Max distance
        return sum((x - y) ** 2 for x, y in zip(a, b))
