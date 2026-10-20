"""Auto's prompt and the LLM tier tooltips quote the public counts (facts sheet, 19 Oct 2026).

Auto told owners it had "100+ integrations via Composio", and the provider tooltip
on both LLM tiers said OpenRouter "covers 100+ models". The Composio catalogue is
over 1,000 apps and OpenRouter lists over 400 models (471 on 18 Oct 2026).

The tier builder was 119 code lines, over the code-shape limit, so its blocks moved
to ``core/seeds/llm_tier_settings.py`` in four builders. The tiers keep the same
nine settings, in the same order, from the same defaults.
PURE: no database.
"""
from __future__ import annotations

import pytest

from core.seeds.llm_tier_settings import (
    tier_call_settings,
    tier_penalty_settings,
    tier_route_settings,
    tier_sampling_settings,
)
from core.seeds.seed_system_settings import _llm_tier_settings

TIER_KEYS = [
    "provider", "model", "temperature", "max_tokens", "top_p",
    "frequency_penalty", "presence_penalty", "timeout_seconds", "max_retries",
]
DEFAULTS = {key: f"default-{key}" for key in TIER_KEYS}
TIERS = ("orchestrator_llm", "system_llm")


def _provider_description(category: str) -> str:
    return next(block["description"] for block in _llm_tier_settings(category, DEFAULTS)
                if block["key"] == "provider")


def test_auto_names_the_composio_catalogue_by_its_size():
    from consumers.chatbot.personality import AutomatosPersonality

    skill = AutomatosPersonality.get_platform_skill()
    assert "1,000+ apps through Composio" in skill
    assert "100+ integrations" not in skill


@pytest.mark.parametrize("category", TIERS)
def test_the_provider_tooltip_names_openrouters_catalogue_and_direct_keys(category):
    description = _provider_description(category)
    assert "400+ models" in description
    assert "Direct keys for OpenAI, Anthropic, Google and others" in description
    assert "100+" not in description


@pytest.mark.parametrize("category", TIERS)
def test_each_tier_keeps_its_nine_settings_in_order_from_its_defaults(category):
    blocks = _llm_tier_settings(category, DEFAULTS)
    assert [block["key"] for block in blocks] == TIER_KEYS
    assert all(block["category"] == category for block in blocks)
    assert [block["default_value"] for block in blocks] == [DEFAULTS[key] for key in TIER_KEYS]


def test_the_tier_is_the_four_builders_in_order():
    builders = (tier_route_settings, tier_sampling_settings, tier_penalty_settings, tier_call_settings)
    joined = [block for build in builders for block in build("system_llm", DEFAULTS)]
    assert joined == _llm_tier_settings("system_llm", DEFAULTS)
