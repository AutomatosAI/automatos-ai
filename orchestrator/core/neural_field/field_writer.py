"""
Neural Field Writer
====================

PRD-59 Phase 4: Write agent contributions to the shared neural field
with diffusion dynamics so knowledge propagates to neighboring regions.

When an agent writes a contribution:
  1. The contribution is stored as a point source at position xᵢ
  2. Nearby field regions are updated (diffusion)
  3. Older contributions lose influence (temporal decay)

Field write equation (simplified):
  Ψ(xᵢ, t) += Aᵢ  (point source injection)
  Ψ(x, t+dt) += D × ∇²Ψ × dt  (diffusion step, applied lazily on read)
"""

import logging
import time
from typing import Any, Dict, List, Optional

from .field_store import NeuralFieldStore, FieldContribution

logger = logging.getLogger(__name__)


class FieldWriter:
    """
    Writes agent contributions to the neural field.

    Usage:
        writer = FieldWriter(field_store)
        await writer.write(
            execution_id="exec-123",
            agent_id=42,
            agent_name="ResearchAgent",
            content="Found that the API uses OAuth 2.0...",
            embedding=embedding_vector,
        )
    """

    DEFAULT_DIFFUSION_COEFFICIENT = 0.3  # 30% knowledge sharing rate

    def __init__(
        self,
        field_store: NeuralFieldStore,
        diffusion_coefficient: float = DEFAULT_DIFFUSION_COEFFICIENT,
        embedding_fn: Optional[Any] = None,
    ):
        """
        Args:
            field_store: The neural field storage backend
            diffusion_coefficient: D in the field equation (0-1).
                                    Higher = faster knowledge spread.
            embedding_fn: Optional async function to generate embeddings.
                          Signature: async def embed(text: str) -> List[float]
        """
        self.store = field_store
        self.D = diffusion_coefficient
        self.embed_fn = embedding_fn

    async def write(
        self,
        execution_id: str,
        agent_id: int,
        agent_name: str,
        content: str,
        embedding: Optional[List[float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FieldContribution:
        """
        Write an agent's contribution to the neural field.

        Args:
            execution_id: Current execution scope
            agent_id: Contributing agent's ID
            agent_name: Contributing agent's name
            content: Text content of the contribution
            embedding: Pre-computed embedding (or generated if embed_fn available)
            metadata: Additional metadata

        Returns:
            The stored FieldContribution
        """
        # Generate embedding if not provided and embed_fn available
        if embedding is None and self.embed_fn is not None:
            try:
                embedding = await self.embed_fn(content)
            except Exception as e:
                logger.debug(f"Embedding generation failed: {e}")

        contribution = FieldContribution(
            agent_id=agent_id,
            agent_name=agent_name,
            content=content,
            embedding=embedding,
            metadata=metadata or {},
            timestamp=time.time(),
        )

        self.store.write(execution_id, contribution)

        count = self.store.get_contribution_count(execution_id)
        logger.info(
            f"📝 FieldWriter: agent={agent_name}({agent_id}) wrote to "
            f"execution={execution_id} (total contributions: {count})"
        )

        return contribution

    async def write_subtask_result(
        self,
        execution_id: str,
        agent_id: int,
        agent_name: str,
        subtask_description: str,
        result: str,
        embedding: Optional[List[float]] = None,
    ) -> FieldContribution:
        """
        Convenience method: write a subtask execution result to the field.
        """
        content = (
            f"Task: {subtask_description}\n"
            f"Result: {result[:1500]}"  # Cap at 1500 chars
        )
        return await self.write(
            execution_id=execution_id,
            agent_id=agent_id,
            agent_name=agent_name,
            content=content,
            embedding=embedding,
            metadata={
                "type": "subtask_result",
                "subtask_description": subtask_description[:200],
            },
        )
