"""
Shared Context Port — PRD-107 / PRD-108
========================================

Abstract interface for shared context between agents.
Implementations:
  - RedisSharedContext  (PRD-107, message-passing baseline)
  - VectorFieldSharedContext (PRD-108, semantic field prototype)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class SharedContextPort(ABC):
    """Contract for shared context backends.

    Every implementation must support:
      create  → spin up a new shared context for a team
      inject  → agent writes a pattern/fact into the field
      query   → agent reads relevant patterns back
      destroy → tear down when the mission ends
    """

    @abstractmethod
    async def create_context(
        self,
        team_agent_ids: list[int],
        initial_data: Optional[dict[str, Any]] = None,
        provenance: Optional[dict[str, Any]] = None,
    ) -> str:
        """Create a shared context space. Returns context_id.

        ``provenance`` (PRD-166 S1) carries ``workspace_id``/``mission_id``/
        ``task_id`` so seeded patterns keep their lineage into the workspace field.
        """
        ...

    @abstractmethod
    async def inject(
        self,
        context_id: str,
        key: str,
        value: str,
        agent_id: int,
        strength: float = 1.0,
        provenance: Optional[dict[str, Any]] = None,
    ) -> None:
        """Inject a pattern into the shared context. ``provenance`` (PRD-166 S1)
        records workspace/mission/task lineage on the pattern."""
        ...

    @abstractmethod
    async def query(
        self,
        context_id: str,
        query: str,
        agent_id: int,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Query for relevant patterns. Returns ranked results."""
        ...

    @abstractmethod
    async def destroy_context(self, context_id: str) -> None:
        """Tear down a shared context and clean up resources."""
        ...
