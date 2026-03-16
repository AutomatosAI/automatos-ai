"""
Token budget manager — priority-based section trimming.

Manages token allocation across sections. When rendered sections exceed
the available budget, the manager trims lowest-priority sections first
(highest priority number = lowest importance = trimmed first).

Priority 1-2 sections are NEVER dropped, even if over budget.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from modules.context.estimator import TokenEstimator
from modules.context.modes import ContextMode

logger = logging.getLogger(__name__)

_estimator = TokenEstimator()


@dataclass(frozen=True)
class TokenBudget:
    """Token budget constraints for a context build.

    ``available_for_sections`` is computed as
    ``total - reserved_for_response - reserved_for_messages``.
    """

    total: int
    reserved_for_response: int
    reserved_for_messages: int

    @property
    def available_for_sections(self) -> int:
        return self.total - self.reserved_for_response - self.reserved_for_messages


@dataclass(frozen=True)
class RenderedSection:
    """A section after rendering, before budget allocation."""

    name: str
    priority: int
    content: str
    token_estimate: int
    max_tokens: Optional[int] = None


class TokenBudgetManager:
    """Manages token allocation across rendered sections.

    Algorithm:
        1. Apply each section's ``max_tokens`` cap via truncation.
        2. If total still exceeds budget, drop sections from highest
           priority number (lowest importance) first.
        3. Priority 1-2 sections are NEVER dropped.
        4. Log warnings for every trim/drop.

    Returns ``(included_sections, trimmed_section_names)``.
    """

    def allocate(
        self,
        sections: list[RenderedSection],
        budget: TokenBudget,
    ) -> tuple[list[RenderedSection], list[str]]:
        """Allocate budget across *sections*, trimming as needed.

        Returns:
            A tuple of (included sections, names of trimmed/dropped sections).
        """
        available = budget.available_for_sections
        trimmed_names: list[str] = []

        # --- Step 1: Apply per-section max_tokens caps ---
        capped: list[RenderedSection] = []
        for section in sections:
            if section.max_tokens is not None and section.token_estimate > section.max_tokens:
                truncated_content = section.content[: section.max_tokens * 4]
                new_estimate = _estimator.estimate(truncated_content)
                saved = section.token_estimate - new_estimate
                logger.warning(
                    "[TokenBudgetManager] Capped section '%s' from %d to %d tokens (saved %d)",
                    section.name,
                    section.token_estimate,
                    new_estimate,
                    saved,
                )
                capped.append(
                    RenderedSection(
                        name=section.name,
                        priority=section.priority,
                        content=truncated_content,
                        token_estimate=new_estimate,
                        max_tokens=section.max_tokens,
                    )
                )
            else:
                capped.append(section)

        # --- Step 2: Check if within budget ---
        total_tokens = sum(s.token_estimate for s in capped)
        if total_tokens <= available:
            return capped, trimmed_names

        # --- Step 3: Drop sections lowest-importance-first ---
        # Sort by priority descending (highest number = least important = dropped first).
        # Stable sort preserves original order among equal priorities.
        droppable = sorted(
            capped, key=lambda s: s.priority, reverse=True
        )

        dropped_names: set[str] = set()
        for section in droppable:
            if total_tokens <= available:
                break
            # Never drop priority 1-2 (identity, task_context, recipe_context).
            if section.priority <= 2:
                continue
            dropped_names.add(section.name)
            trimmed_names.append(section.name)
            total_tokens -= section.token_estimate
            logger.warning(
                "[TokenBudgetManager] Dropped section '%s' (priority %d, %d tokens) — over budget",
                section.name,
                section.priority,
                section.token_estimate,
            )

        # Rebuild list preserving original order, excluding dropped.
        included = [s for s in capped if s.name not in dropped_names]

        if total_tokens > available:
            logger.warning(
                "[TokenBudgetManager] Still over budget after trimming (%d/%d tokens). "
                "Only priority 1-2 sections remain.",
                total_tokens,
                available,
            )

        return included, trimmed_names


# ---------------------------------------------------------------------------
# Default budgets per mode (from PRD §6.1)
# ---------------------------------------------------------------------------

DEFAULT_BUDGETS: dict[ContextMode, TokenBudget] = {
    ContextMode.CHATBOT: TokenBudget(
        total=128_000,
        reserved_for_response=4_096,
        reserved_for_messages=60_000,
    ),
    ContextMode.TASK_EXECUTION: TokenBudget(
        total=128_000,
        reserved_for_response=4_096,
        reserved_for_messages=20_000,
    ),
    ContextMode.HEARTBEAT_ORCHESTRATOR: TokenBudget(
        total=128_000,
        reserved_for_response=2_048,
        reserved_for_messages=0,
    ),
    ContextMode.HEARTBEAT_AGENT: TokenBudget(
        total=128_000,
        reserved_for_response=4_096,
        reserved_for_messages=0,
    ),
    ContextMode.RECIPE: TokenBudget(
        total=128_000,
        reserved_for_response=4_096,
        reserved_for_messages=10_000,
    ),
    ContextMode.NL2SQL: TokenBudget(
        total=128_000,
        reserved_for_response=2_048,
        reserved_for_messages=2_000,
    ),
    # Router and orchestrator stages use generous defaults.
    ContextMode.ROUTER: TokenBudget(
        total=128_000,
        reserved_for_response=4_096,
        reserved_for_messages=0,
    ),
    ContextMode.ORCHESTRATOR_STAGE: TokenBudget(
        total=128_000,
        reserved_for_response=4_096,
        reserved_for_messages=0,
    ),
}
