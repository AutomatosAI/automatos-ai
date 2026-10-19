"""A new Auto takes the platform default from system settings (Gerard, 19 Oct 2026).

18 of 21 production Auto rows carried ``openrouter`` / ``google/gemini-2.5-flash``:
the seed stamped core/llm/defaults' literal pair and never read the admin's
``orchestrator_llm`` setting. It now builds the row from
``new_agent_model_config`` (the setting, fitted; else the defaults pair whole) and
marks it as inheriting, so an admin's later change of the default moves it.
PURE: settings, keys and the session are faked.
"""
from __future__ import annotations

import uuid
from unittest.mock import MagicMock

import pytest

import core.seeds.seed_auto_agent as auto_seed
from core.llm import auto_model_sync, vendor_fit, workspace_keys
from core.llm import manager as llm_manager
from core.llm.defaults import DEFAULT_LLM_MODEL, DEFAULT_LLM_PROVIDER

SONNET = {("orchestrator_llm", "provider"): "anthropic", ("orchestrator_llm", "model"): "claude-sonnet-5-5"}


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


def _seed_new_auto(monkeypatch):
    monkeypatch.setattr(auto_seed, "ensure_builtin_skill", lambda db, name: None)
    monkeypatch.setattr(auto_seed, "seed_brand_designer", lambda db, ws: None)
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = None  # no Auto yet
    return auto_seed.seed_auto_agent(db, uuid.uuid4()).model_config


def _route(model_config):
    return model_config["provider"], model_config["model_id"]


def test_a_new_auto_takes_the_orchestrator_llm_setting_and_is_marked(settings, monkeypatch):
    settings(SONNET, keyed=("anthropic",))
    model_config = _seed_new_auto(monkeypatch)
    assert _route(model_config) == ("anthropic", "claude-sonnet-5-5")
    assert model_config["source"] == auto_model_sync.PLATFORM_DEFAULT_SOURCE
    assert model_config[auto_model_sync.STAMPED_ROUTE_KEY] == {"provider": "anthropic", "model_id": "claude-sonnet-5-5"}
    assert model_config["max_tokens"] == auto_seed.AUTO_MAX_TOKENS == 4000
    assert auto_model_sync.is_inheriting(model_config)


def test_with_no_setting_the_defaults_pair_decides(settings, monkeypatch):
    settings({})
    model_config = _seed_new_auto(monkeypatch)
    assert _route(model_config) == (DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL)
    assert model_config["source"] == auto_model_sync.PLATFORM_DEFAULT_SOURCE


@pytest.mark.parametrize("half", [
    {("orchestrator_llm", "provider"): "openrouter"},
    {("orchestrator_llm", "model"): "openai/gpt-5.5"},
])
def test_half_a_setting_falls_back_to_the_defaults_pair_whole(settings, monkeypatch, half):
    settings(half)
    model_config = _seed_new_auto(monkeypatch)
    assert _route(model_config) == (DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL)


def test_the_new_default_is_haiku_5_5_on_anthropic():
    assert (DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL) == ("anthropic", "claude-haiku-5-5")


# ── the settings seed names the defaults pair once ──────────────────────────

class _SettingsQuery:
    def __init__(self, stored):
        self._stored, self._pair = stored, (None, None)

    def filter(self, *clauses):
        values = [getattr(getattr(c, "right", None), "value", None) for c in clauses]
        self._pair = tuple(values[:2]) if len(values) >= 2 else (None, None)
        return self

    def first(self):
        return self._stored.get(self._pair)

    def all(self):
        return []

    def delete(self, *args, **kwargs):
        return 0

    def update(self, *args, **kwargs):
        return 0


class _SettingsSession:
    def __init__(self, stored=None):
        self.stored, self.added = dict(stored or {}), []

    def query(self, model):
        return _SettingsQuery(self.stored)

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        pass

    def rollback(self):
        pass


@pytest.mark.parametrize("tier", ["orchestrator_llm", "system_llm"])
def test_a_fresh_install_seeds_both_tiers_with_the_defaults_pair(tier):
    from core.seeds.seed_system_settings import seed_system_settings

    db = _SettingsSession()
    seed_system_settings(db)
    seeded = {s.key: (s.value, s.default_value) for s in db.added if s.category == tier}
    assert seeded["provider"] == (DEFAULT_LLM_PROVIDER, DEFAULT_LLM_PROVIDER)
    assert seeded["model"] == (DEFAULT_LLM_MODEL, DEFAULT_LLM_MODEL)


def test_a_stored_orchestrator_model_is_never_overwritten_by_the_seed():
    from core.models.system_settings import SystemSetting
    from core.seeds.seed_system_settings import seed_system_settings

    chosen = SystemSetting(category="orchestrator_llm", key="model", value="openai/gpt-5.5",
                           default_value="google/gemini-2.5-flash", value_type="string", description="",
                           is_required=True, is_sensitive=False, validation_rules=None)
    seed_system_settings(_SettingsSession({("orchestrator_llm", "model"): chosen}))
    assert chosen.value == "openai/gpt-5.5" and chosen.default_value == DEFAULT_LLM_MODEL
