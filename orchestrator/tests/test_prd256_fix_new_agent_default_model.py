"""P256-FIX-N13-8 (F415): an agent Auto creates with no model takes the workspace's default.

Night 13 (11–12 Oct): INVOICE DESK (#368), created by Auto at the owner's ask, got
``google/gemini-2.5-flash`` on OpenRouter while the workspace ran Auto on
``anthropic/claude-haiku-5-5``: create_agent seeded ``model_config`` from
``core/llm/defaults``. It now takes the ``orchestrator_llm`` provider and model from
system settings (both set, fitted by P256-FIX-T1's rule), else the defaults' pair.
PURE: settings, keys and the session are faked.
"""
from __future__ import annotations

import asyncio
from uuid import UUID

import pytest

from core.llm import manager as llm_manager
from core.llm import vendor_fit, workspace_keys
from core.llm.defaults import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER, DEFAULT_MAX_OUTPUT_TOKENS
from core.models import Agent

WS = UUID("00000000-0000-0000-0000-0000000000c1")
HAIKU = {("orchestrator_llm", "provider"): "anthropic", ("orchestrator_llm", "model"): "claude-haiku-5-5"}


class _Rows:
    def filter(self, *args, **kwargs):
        return self

    def order_by(self, *args):
        return self

    def first(self):
        return None

    def all(self):
        return []

    def count(self):
        return 0


class _Lock:
    def scalar(self):
        return True


class _Db:
    """No namesake, no plan limit; records what is added."""

    def __init__(self):
        self.added = []

    def query(self, *entities):
        return _Rows()

    def add(self, obj):
        self.added.append(obj)

    def flush(self):
        pass

    def execute(self, *args, **kwargs):  # F200's agent-count lock
        return _Lock()


@pytest.fixture
def settings(monkeypatch):
    """``settings(rows, keyed=())``: these system-settings rows; keys for these providers only."""
    monkeypatch.setattr(workspace_keys, "get_platform_workspace_key", lambda provider: None)
    monkeypatch.setattr(vendor_fit, "env_api_key", lambda provider: None)
    vendor_fit._warn_once.cache_clear()

    def install(rows, keyed=()):
        stored = dict(rows)
        monkeypatch.setattr(llm_manager, "get_system_setting", lambda c, k, d=None: stored.get((c, k), d))
        monkeypatch.setattr(llm_manager, "get_credential_data",
                            lambda provider, environment=None, service_name="orchestrator":
                            {"api_key": f"key-{provider}"} if provider in keyed else {})
    return install


def _create(params):
    from modules.tools.discovery import handlers_agents

    db = _Db()
    reply = asyncio.run(handlers_agents.create_agent(db, WS, params))
    assert reply["success"] is True, reply
    made = [obj for obj in db.added if isinstance(obj, Agent)]
    assert len(made) == 1
    return reply, made[0].model_config


def _route(model_config):
    return model_config["provider"], model_config["model_id"]


# ── create_agent with no model ──────────────────────────────────────────────

def test_a_new_agent_takes_the_workspaces_orchestrator_model(settings):
    settings(HAIKU)
    reply, model_config = _create({"name": "Invoice Desk"})
    assert _route(model_config) == ("anthropic", "claude-haiku-5-5")
    assert (reply["agent"]["provider"], reply["agent"]["model_id"]) == ("anthropic", "claude-haiku-5-5")
    assert model_config["max_tokens"] == DEFAULT_MAX_OUTPUT_TOKENS and model_config["temperature"] == 0.7


def test_with_no_setting_the_defaults_module_decides(settings):
    settings({})
    _, model_config = _create({"name": "Invoice Desk"})
    assert _route(model_config) == (DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL)


@pytest.mark.parametrize("half", [
    {("orchestrator_llm", "provider"): "anthropic"},
    {("orchestrator_llm", "model"): "claude-haiku-5-5"},
])
def test_half_a_setting_never_pairs_with_half_the_defaults(settings, half):
    settings(half)
    _, model_config = _create({"name": "Invoice Desk"})
    assert _route(model_config) == (DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL)


def test_the_legacy_setting_keys_are_read_too(settings):
    settings({("orchestrator_llm", "llm_provider"): "anthropic", ("orchestrator_llm", "llm_model"): "claude-haiku-5-5"})
    _, model_config = _create({"name": "Invoice Desk"})
    assert _route(model_config) == ("anthropic", "claude-haiku-5-5")


def test_a_temperature_given_still_applies_on_the_default_model(settings):
    settings(HAIKU)
    _, model_config = _create({"name": "Invoice Desk", "temperature": 0.2})
    assert _route(model_config) == ("anthropic", "claude-haiku-5-5") and model_config["temperature"] == 0.2


# ── the model-fits-provider rule (P256-FIX-T1) ──────────────────────────────

def test_a_setting_whose_model_its_provider_does_not_serve_moves_to_a_keyed_provider(settings):
    settings({("orchestrator_llm", "provider"): "anthropic", ("orchestrator_llm", "model"): "google/gemini-2.5-flash"},
             keyed=("openrouter",))
    _, model_config = _create({"name": "Invoice Desk"})
    assert _route(model_config) == ("openrouter", "google/gemini-2.5-flash")


def test_a_vendor_prefixed_claude_on_anthropic_runs_as_anthropics_own_id(settings):
    settings({("orchestrator_llm", "provider"): "anthropic", ("orchestrator_llm", "model"): "anthropic/claude-haiku-5-5"},
             keyed=("anthropic", "openrouter"))
    _, model_config = _create({"name": "Invoice Desk"})
    assert _route(model_config) == ("anthropic", "claude-haiku-5-5")


# ── the factory's fallbacks read the same default ───────────────────────────

def test_the_factorys_default_and_a_config_with_no_model_take_the_workspaces_default(settings):
    from modules.agents.factory.agent_factory import ModelConfiguration

    settings(HAIKU)
    default = ModelConfiguration.get_default()
    assert (default.provider, default.model_id) == ("anthropic", "claude-haiku-5-5")
    no_model = ModelConfiguration.from_dict({"temperature": 0.3})
    assert (no_model.provider, no_model.model_id, no_model.temperature) == ("anthropic", "claude-haiku-5-5", 0.3)


def test_a_config_that_names_its_model_keeps_it(settings):
    from modules.agents.factory.agent_factory import ModelConfiguration

    settings(HAIKU)
    kept = ModelConfiguration.from_dict({"provider": "openrouter", "model_id": "openai/gpt-4o-mini"})
    assert (kept.provider, kept.model_id) == ("openrouter", "openai/gpt-4o-mini")
    bare = ModelConfiguration.from_dict({"model_id": "openai/gpt-4o-mini"})
    assert (bare.provider, bare.model_id) == (DEFAULT_LLM_PROVIDER, "openai/gpt-4o-mini")
