"""What one LLM call cost this turn — the provider's own figure when it gave one.

The chat lane's cost governor (PRD-223 S0.4) summed a static-map estimate per
call. On 2026-09-16 that map had no entry for the model in use, so a turn the
usage table recorded at $4.26 passed a $1.50 ceiling. OpenRouter returns the
credits it charged in ``usage.cost`` (requested with ``usage: {include: true}``
and copied into the platform's usage dict by the client); every other route
falls back to the estimate the manager already computes per route.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping, Optional


def call_cost_usd(usage: Optional[Mapping[str, Any]], estimate: Callable[[int, int], float]) -> float:
    """USD for one call: the reported cost when present and positive, else the
    route's estimate over the token counts, else 0 (nothing to bill)."""
    usage = usage or {}
    reported = usage.get("cost")
    if isinstance(reported, (int, float)) and not isinstance(reported, bool) and reported > 0:
        return float(reported)
    tokens_in = int(usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0) or 0)
    tokens_out = int(usage.get("output_tokens", 0) or usage.get("completion_tokens", 0) or 0)
    if not (tokens_in or tokens_out):
        return 0.0
    return float(estimate(tokens_in, tokens_out))
