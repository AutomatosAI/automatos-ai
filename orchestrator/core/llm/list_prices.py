"""Claude list prices: the one table (P256-FIX-T2, F399).

Anthropic's list prices per model (platform.claude.com/docs/en/about-claude/pricing,
read 9-10 Oct 2026), per 1K tokens. Three things vary by model and are held here,
never as per-model branches:

- the input and output price (Opus 5.5 is $4/$20, not Opus 5's $5/$25);
- the cache-read multiplier on the input price: 0.025 on Fable 5.1, 0.05 on
  Opus 5.5 and Sonnet 5.5, 0.1 on the rest (a cache write is 1.25x everywhere);
- Haiku 5.5's second tier: a prompt over 100,000 tokens (cache reads and writes
  count) pays $0.50/$2.50 instead of $0.10/$0.50, for the whole request.

The usage tracker reads the multiplier and the tier for every call; the audit
estimate and the tracker's last-resort rate read the prices; the model seed
writes the three 5.5 prices into ``llm_models``; the marketplace shows them on a
row that has none.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

DEFAULT_CACHE_READ_MULTIPLIER = 0.10
CACHE_WRITE_MULTIPLIER = 1.25
HAIKU_5_5_TIER_OVER_TOKENS = 100_000

# OpenRouter writes a version with a dot (anthropic/claude-opus-5.5); the table's keys use a hyphen.
_VERSION_DOT = re.compile(r"(?<=\d)\.(?=\d)")


@dataclass(frozen=True)
class PromptTier:
    """A price that applies to the whole request once its prompt is over ``over_tokens``."""

    over_tokens: int
    input_per_1k: float
    output_per_1k: float


@dataclass(frozen=True)
class ListPrice:
    """One model's list price per 1K tokens, its cache-read multiplier and any prompt-length tier."""

    input_per_1k: float
    output_per_1k: float
    cache_read_multiplier: float = DEFAULT_CACHE_READ_MULTIPLIER
    long_prompt: Optional[PromptTier] = None

    def rates(self, prompt_tokens: int) -> Tuple[float, float]:
        """(input, output) per 1K for a call whose prompt is ``prompt_tokens`` long."""
        tier = self.long_prompt
        if tier is not None and int(prompt_tokens or 0) > tier.over_tokens:
            return tier.input_per_1k, tier.output_per_1k
        return self.input_per_1k, self.output_per_1k


_LIST_PRICES: Dict[str, ListPrice] = {
    "claude-fable-5-1": ListPrice(0.010, 0.050, cache_read_multiplier=0.025),
    "claude-fable-5": ListPrice(0.010, 0.050),
    "claude-opus-5-5": ListPrice(0.004, 0.020, cache_read_multiplier=0.05),
    "claude-opus-5": ListPrice(0.005, 0.025),
    "claude-sonnet-5-5": ListPrice(0.002, 0.010, cache_read_multiplier=0.05),
    "claude-sonnet-5": ListPrice(0.002, 0.010),
    "claude-haiku-5-5": ListPrice(
        0.0001, 0.0005, long_prompt=PromptTier(HAIKU_5_5_TIER_OVER_TOKENS, 0.0005, 0.0025),
    ),
    "claude-haiku-4-5": ListPrice(0.001, 0.005),
}
# Substring-matched, so the longest key is tried first: "claude-opus-5-5" before "claude-opus-5".
LIST_PRICES: Dict[str, ListPrice] = dict(sorted(_LIST_PRICES.items(), key=lambda kv: -len(kv[0])))


def list_price(model_id: Optional[str]) -> Optional[ListPrice]:
    """The table's entry for ``model_id`` in any of its spellings (a direct Anthropic id,
    ``anthropic/claude-opus-5.5``, a dated id), or None for a model the table doesn't hold."""
    spelled = _VERSION_DOT.sub("-", (model_id or "").lower())
    for key, price in LIST_PRICES.items():
        if key in spelled:
            return price
    return None


def cache_read_multiplier(model_id: Optional[str], default: float) -> float:
    """The model's cache-read multiplier; ``default`` (the provider's) for a model the table doesn't hold."""
    price = list_price(model_id)
    return price.cache_read_multiplier if price is not None else default


def prompt_tier_factors(model_id: Optional[str], prompt_tokens: int) -> Tuple[float, float]:
    """How much the call's prompt length scales the model's base (input, output) price:
    (1, 1) below a tier, (5, 5) for a Haiku 5.5 prompt over 100k. A ratio, so a route's
    own price (a free route's 0, an operator's figure) is scaled, not replaced."""
    price = list_price(model_id)
    if price is None or not price.input_per_1k or not price.output_per_1k:
        return 1.0, 1.0
    rate_in, rate_out = price.rates(prompt_tokens)
    return rate_in / price.input_per_1k, rate_out / price.output_per_1k


def list_rates(model_id: Optional[str], prompt_tokens: int = 0) -> Optional[Tuple[float, float]]:
    """(input, output) per 1K at list price for a prompt of ``prompt_tokens``; None when unknown."""
    price = list_price(model_id)
    return price.rates(prompt_tokens) if price is not None else None


__all__ = [
    "CACHE_WRITE_MULTIPLIER", "DEFAULT_CACHE_READ_MULTIPLIER", "LIST_PRICES", "ListPrice", "PromptTier",
    "cache_read_multiplier", "list_price", "list_rates", "prompt_tier_factors",
]
