"""P256-FIX-T1 (F397/F400): a service's model and its provider come from one vendor.

Night of 9 Oct, System LLM = anthropic/claude-sonnet-5-5: the distiller and the
thread checkpoint sent ``google/gemini-2.5-flash`` (MEMORY_DISTILL_MODEL's default)
to the Anthropic client, and the cross-model verifier ``openai/gpt-4o-mini``: a 404
on every turn, finished runs unjudged. These tests build the real manager from
settings (keys and settings faked) and read the config it would send.
"""
from __future__ import annotations

import logging

import pytest

from core.llm import create_llm_manager, output_budget, vendor_fit
from core.llm import manager as llm_manager
from core.llm import workspace_keys
from core.llm.clients.base import LLMProvider
from core.llm.defaults import DEFAULT_LLM_MODEL

ANTHROPIC_SYSTEM = {("system_llm", "provider"): "anthropic", ("system_llm", "model"): "claude-sonnet-5-5"}
OPENROUTER_SYSTEM = {("system_llm", "provider"): "openrouter", ("system_llm", "model"): "google/gemini-2.5-flash"}
ENV_OVERRIDES = ("MEMORY_DISTILL_MODEL", "COORDINATOR_VERIFIER_MODEL_MAPPING", "COORDINATOR_VERIFIER_FALLBACK_MODEL")


@pytest.fixture
def platform(monkeypatch):
    """``platform(settings, keyed)``: these settings rows, keys for these providers only."""
    for name in ENV_OVERRIDES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(output_budget, "manager_budgets", lambda *a, **k: (None, None))
    monkeypatch.setattr(workspace_keys, "get_platform_workspace_key", lambda provider: None)
    monkeypatch.setattr(vendor_fit, "env_api_key", lambda provider: None)
    monkeypatch.setattr(llm_manager, "env_api_key", lambda provider: None)
    vendor_fit._warn_once.cache_clear()

    def install(settings, keyed):
        rows = dict(settings)
        monkeypatch.setattr(llm_manager, "get_system_setting", lambda c, k, d=None: rows.get((c, k), d))
        monkeypatch.setattr(llm_manager, "get_credential_data",
                            lambda provider, environment=None, service_name="orchestrator":
                            {"api_key": f"key-{provider}"} if provider in keyed else {})
    return install


def _sent(service_name, model):
    mgr = create_llm_manager(service_name=service_name, model=model)
    return mgr.config.provider, mgr.config.model


def _verifier_for(executor_model):
    from modules.coordination.verification import _select_verifier_model
    return _select_verifier_model(executor_model)


# ── the distiller ───────────────────────────────────────────────────────────

def test_the_distiller_on_an_anthropic_system_llm_runs_a_claude_model_on_anthropic(platform):
    from config import config

    platform(ANTHROPIC_SYSTEM, keyed={"anthropic"})
    assert config.MEMORY_DISTILL_MODEL == "claude-sonnet-5-5"               # was google/gemini-2.5-flash
    provider, model = _sent("memory_integration", config.MEMORY_DISTILL_MODEL)
    assert provider is LLMProvider.ANTHROPIC
    assert model.startswith("claude") and "/" not in model


def test_a_distill_model_setting_still_wins(platform):
    from config import config

    platform({**ANTHROPIC_SYSTEM, ("memory", "distill_model"): "claude-haiku-5-5"}, keyed={"anthropic"})
    assert config.MEMORY_DISTILL_MODEL == "claude-haiku-5-5"


# ── the verifier ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("executor, verifier", [
    ("claude-opus-5-5", "claude-sonnet-5-5"),
    ("anthropic/claude-fable-5-1", "claude-sonnet-5-5"),
    ("claude-sonnet-5-5", "claude-haiku-5-5"),
    ("claude-haiku-5-5", "claude-sonnet-5-5"),
    ("google/gemini-2.5-flash", "claude-sonnet-5-5"),
    (None, "claude-sonnet-5-5"),
])
def test_the_verifier_on_an_anthropic_system_llm_is_another_claude_on_anthropic(platform, executor, verifier):
    platform(ANTHROPIC_SYSTEM, keyed={"anthropic"})
    chosen = _verifier_for(executor)
    assert chosen == verifier and chosen != executor                         # was openai/gpt-4o-mini
    assert _sent("verifier", chosen) == (LLMProvider.ANTHROPIC, verifier)


# ── an OpenRouter System LLM is unchanged ───────────────────────────────────

def test_an_openrouter_system_llm_keeps_todays_models_and_routes(platform):
    from config import config

    platform(OPENROUTER_SYSTEM, keyed={"openrouter"})
    assert config.MEMORY_DISTILL_MODEL == DEFAULT_LLM_MODEL == "google/gemini-2.5-flash"
    assert _sent("memory_integration", config.MEMORY_DISTILL_MODEL) == (LLMProvider.OPENROUTER, "google/gemini-2.5-flash")
    assert _verifier_for("claude-opus-5-5") == "openai/gpt-4o-mini"
    assert _verifier_for("gpt-4o") == "anthropic/claude-haiku-4-5"
    assert _verifier_for(None) == "openai/gpt-4o-mini"
    assert _sent("verifier", "openai/gpt-4o-mini") == (LLMProvider.OPENROUTER, "openai/gpt-4o-mini")


# ── a vendor-prefixed id never reaches the Anthropic client ────────────────

@pytest.mark.parametrize("keyed, expected", [
    ({"anthropic"}, (LLMProvider.ANTHROPIC, "claude-sonnet-5-5")),           # the category's own model
    ({"anthropic", "openrouter"}, (LLMProvider.OPENROUTER, "google/gemini-2.5-flash")),
    ({"anthropic", "google"}, (LLMProvider.GOOGLE, "gemini-2.5-flash")),
])
def test_a_vendor_prefixed_id_on_anthropic_goes_where_it_is_served(platform, keyed, expected):
    platform(ANTHROPIC_SYSTEM, keyed=keyed)
    provider, model = _sent("memory_integration", "google/gemini-2.5-flash")
    assert (provider, model) == expected
    assert not (provider is LLMProvider.ANTHROPIC and "/" in model)


def test_anthropics_own_prefix_is_dropped_not_rerouted(platform):
    platform(ANTHROPIC_SYSTEM, keyed={"anthropic", "openrouter"})
    assert _sent("verifier", "anthropic/claude-haiku-5-5") == (LLMProvider.ANTHROPIC, "claude-haiku-5-5")


def test_with_no_serving_key_one_warning_names_service_model_and_provider(platform, caplog):
    platform(ANTHROPIC_SYSTEM, keyed={"anthropic"})
    with caplog.at_level(logging.WARNING, logger=vendor_fit.__name__):
        for _ in range(3):                                                   # three turns
            assert _sent("verifier", "openai/gpt-4o-mini") == (LLMProvider.ANTHROPIC, "claude-sonnet-5-5")
    warnings = [r.getMessage() for r in caplog.records
                if r.name == vendor_fit.__name__ and r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert all(word in warnings[0] for word in ("verifier", "openai/gpt-4o-mini", "anthropic"))


# ── a bare Claude id on a provider that isn't Anthropic ────────────────────

@pytest.mark.parametrize("keyed, expected", [
    ({"openrouter", "anthropic"}, (LLMProvider.ANTHROPIC, "claude-haiku-5-5")),
    ({"openrouter"}, (LLMProvider.OPENROUTER, "anthropic/claude-haiku-5-5")),
])
def test_a_bare_claude_id_on_openrouter_runs_where_it_is_served(platform, keyed, expected):
    platform(OPENROUTER_SYSTEM, keyed=keyed)
    assert _sent("verifier", "claude-haiku-5-5") == expected


# ── what the check leaves alone ─────────────────────────────────────────────

def test_a_named_provider_is_never_second_guessed(platform):
    platform(ANTHROPIC_SYSTEM, keyed={"anthropic"})
    mgr = create_llm_manager(service_name="verifier", provider="openrouter", model="openai/gpt-4o-mini")
    assert (mgr.config.provider, mgr.config.model) == (LLMProvider.OPENROUTER, "openai/gpt-4o-mini")


def test_an_azure_deployment_name_is_not_read_as_a_vendor(platform):
    platform({("system_llm", "provider"): "azure", ("system_llm", "model"): "my-deployment"}, keyed={"azure"})
    assert _sent("verifier", "gpt-4o") == (LLMProvider.AZURE, "gpt-4o")


def test_unreadable_settings_leave_the_call_as_it_was(platform):
    platform({}, keyed=set())
    assert vendor_fit.fitted("verifier", "openai/gpt-4o-mini") == (None, "openai/gpt-4o-mini")
