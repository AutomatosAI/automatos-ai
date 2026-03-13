"""
Base section ABC and SectionContext dataclass.

All prompt sections inherit from BaseSection and share a consistent
render/priority/token interface. SectionContext is the mutable data bag
passed to every section's render() method.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from modules.context.estimator import TokenEstimator

logger = logging.getLogger(__name__)

# Shared estimator instance — stateless, safe to reuse.
_estimator = TokenEstimator()


@dataclass
class SectionContext:
    """Mutable data bag passed to every section's render() method.

    NOT frozen — used as mutable bag during assembly so the service
    and sections can annotate it (e.g. _memory_context).
    """

    agent: Any  # Agent DB record
    workspace_id: str
    workspace_name: Optional[str] = None
    db_session: Any = None
    messages: Optional[list[dict]] = None
    task_description: Optional[str] = None
    recipe_step: Optional[dict] = None
    complexity_assessment: Any = None
    tool_hints: Optional[list[str]] = None
    widget_mode: bool = False
    kwargs: dict = field(default_factory=dict)


class BaseSection(ABC):
    """Base class for composable prompt sections.

    Subclasses MUST set class-level ``name``, ``priority``, and
    ``max_tokens`` attributes before use.

    Priority:
        1 (highest / never dropped) → 10 (lowest / trimmed first).
    """

    name: str
    priority: int
    max_tokens: Optional[int] = None

    @abstractmethod
    async def render(self, ctx: SectionContext) -> str:
        """Render this section's content as a string.

        Must never raise — implementations should catch exceptions
        internally and return an empty string on failure.
        """
        ...

    def estimate_tokens(self, content: str) -> int:
        """Estimate token count for rendered content."""
        return _estimator.estimate(content)

    def truncate(self, content: str, max_tokens: int) -> str:
        """Truncate *content* so it fits within *max_tokens*.

        Uses the fast char-based heuristic (4 chars ≈ 1 token).
        """
        if not content:
            return content
        max_chars = max_tokens * 4
        if len(content) <= max_chars:
            return content
        return content[:max_chars]
