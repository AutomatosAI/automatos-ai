"""One LLM call's token counts and reported cost, read from the provider's ``usage`` dict.

The clients name the counts differently (``input_tokens`` or ``prompt_tokens``). Usage
tracking, the cost audit line and the call's GenAI span (PRD-256 O3) all read them here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class UsageCounts:
    """The tokens one call used, and the cost the provider reported for it (if any)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reported_cost: Optional[float] = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def usage_counts(response: Any) -> UsageCounts:
    """The counts on ``response.usage``; all zero when the call returned none (an error)."""
    usage = getattr(response, "usage", None) or {}
    cost = usage.get("cost")
    return UsageCounts(
        input_tokens=int(usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0) or 0),
        output_tokens=int(usage.get("output_tokens", 0) or usage.get("completion_tokens", 0) or 0),
        cache_read_tokens=int(usage.get("cache_read_tokens", 0) or 0),
        cache_write_tokens=int(usage.get("cache_write_tokens", 0) or 0),
        reported_cost=float(cost) if isinstance(cost, (int, float)) else None,
    )
