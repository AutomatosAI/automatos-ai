"""The System LLM runs on OpenRouter when the Anthropic default has no key (19 Oct 2026).

A fresh install seeds both LLM tiers on core/llm/defaults' pair, now ``anthropic`` /
``claude-haiku-5-5``. With only an OpenRouter key (QUICKSTART's recommended path) the
Auto path already fell back to OpenRouter (agent_factory), but a manager built from
settings (planner, distiller, verifier) built an Anthropic client with no key.
``vendor_fit.keyless_via_openrouter`` gives it the same fallback, with the dotted
OpenRouter id. Also here: platform_create_agent's copy names the platform default,
not the last-resort literal. PURE: the loader and the key chain are faked.
"""
from __future__ import annotations

import json

import pytest

from core.llm import manager as llm_manager
from core.llm import vendor_fit
from core.llm.clients.base import LLMConfig, LLMProvider
from core.llm.defaults import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER

HAIKU_ON_OPENROUTER = "anthropic/claude-haiku-5.5"


class _Loader:
    """``_load_config_from_settings``: the settings tier on ``provider`` with ``key``
    for the providers in ``keyed``; records each call."""

    def __init__(self, provider=DEFAULT_LLM_PROVIDER, model=DEFAULT_LLM_MODEL, keyed=()):
        self.provider, self.model, self.keyed, self.calls = provider, model, keyed, []

    def __call__(self, service_name, provider=None, model=None):
        self.calls.append((service_name, provider, model))
        chosen = provider or self.provider
        return LLMConfig(provider=LLMProvider(chosen), model=model or self.model,
                         api_key=f"key-{chosen}" if chosen in self.keyed else None)


@pytest.fixture
def keys(monkeypatch):
    def install(*keyed):
        monkeypatch.setattr(vendor_fit, "_has_key", lambda slug, service_name: slug in keyed)
    return install


def test_an_anthropic_tier_with_only_an_openrouter_key_runs_on_openrouter_dotted(keys):
    keys("openrouter")
    load = _Loader(keyed=("openrouter",))
    config = vendor_fit.keyless_via_openrouter(load, "planner")
    assert config.provider == LLMProvider.OPENROUTER and config.model == HAIKU_ON_OPENROUTER
    assert config.api_key == "key-openrouter"
    assert load.calls == [("planner", None, None), ("planner", "openrouter", HAIKU_ON_OPENROUTER)]


def test_a_tier_with_its_own_key_is_left_alone(keys):
    keys("openrouter", "anthropic")
    load = _Loader(keyed=("anthropic", "openrouter"))
    config = vendor_fit.keyless_via_openrouter(load, "planner")
    assert (config.provider, config.model) == (LLMProvider.ANTHROPIC, DEFAULT_LLM_MODEL)
    assert len(load.calls) == 1


def test_with_no_openrouter_key_either_the_keyless_config_is_returned_as_before(keys):
    keys()
    config = vendor_fit.keyless_via_openrouter(_Loader(), "planner")
    assert config.provider == LLMProvider.ANTHROPIC and config.api_key is None


def test_a_keyless_openrouter_tier_is_not_rerouted_to_itself(keys):
    keys("openrouter")
    load = _Loader(provider="openrouter", model="google/gemini-2.5-flash")
    vendor_fit.keyless_via_openrouter(load, "planner")
    assert len(load.calls) == 1


def test_a_settings_built_manager_takes_the_fallback(monkeypatch, keys):
    keys("openrouter")
    monkeypatch.setattr(llm_manager.LLMManager, "_load_config_from_settings",
                        lambda self, *a: _Loader(keyed=("openrouter",))(*a))
    monkeypatch.setattr(llm_manager.output_budget, "manager_budgets", lambda *a, **k: (None, None))
    manager = llm_manager.LLMManager(service_name="planner")
    assert manager.config.provider == LLMProvider.OPENROUTER and manager.config.model == HAIKU_ON_OPENROUTER


def test_the_create_copy_names_the_platform_default_not_the_literal():
    """Omitting model_id gives the agent the platform default (new_agent_model), so
    the copy no longer says the default model is the last-resort literal."""
    from modules.tools.discovery.action_registry import get_action_registry
    from modules.tools.discovery.actions_agent_create import OMIT_MODEL_ID

    action = get_action_registry().get("platform_create_agent")
    copy = json.dumps(action.parameters)
    assert OMIT_MODEL_ID in copy and f"Omit it for the default model, {DEFAULT_LLM_MODEL}" not in copy
    assert "runtime" in action.parameters["properties"]          # still registered through takes_the_runtime
