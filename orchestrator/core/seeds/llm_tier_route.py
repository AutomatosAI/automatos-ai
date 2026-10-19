"""The provider and model both LLM tiers' settings are seeded with (19 Oct 2026).

``seed_system_settings`` gave ``orchestrator_llm`` and ``system_llm`` their own copy of
``openrouter`` / ``google/gemini-2.5-flash``. Both now take core/llm/defaults' one
last-resort pair: the initial value on a fresh install. The seed never overwrites a
stored value, so this moves no existing deployment.
"""
from __future__ import annotations

import functools
from typing import Callable, Dict, List

from core.llm.defaults import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER

TierBuilder = Callable[[str, Dict[str, str]], List[dict]]


def on_the_default_route(build_tier: TierBuilder) -> TierBuilder:
    """Wrap a tier builder so its ``defaults`` start from the defaults pair; a tier
    that names its own provider or model still wins. A NEW dict is passed on."""
    @functools.wraps(build_tier)
    def build(category: str, defaults: Dict[str, str]) -> List[dict]:
        return build_tier(category, {"provider": DEFAULT_LLM_PROVIDER, "model": DEFAULT_LLM_MODEL, **defaults})
    return build


__all__ = ["on_the_default_route"]
