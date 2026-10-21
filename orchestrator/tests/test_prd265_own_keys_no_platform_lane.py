"""PRD-265 US-004 (FR-3) — without credits, an agent runs on the workspace's own keys or not at all.

Key resolution used to fall back to the platform's key for any provider the
workspace had no key for — so the $9 Own keys plan (and every workspace whose trial
was spent) ran on Automatos's keys unmetered (leaks 1 and 2). Now, when a
workspace's credits can't pay for a platform-key call, the agent factory resolves
the workspace's OWN keys only — the provider's own key, then the workspace's own
OpenRouter key — and stops with ``CreditsExhaustedError`` when it has neither. The
key resolver's ``own_keys_only`` never reaches the platform's keys.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from core.llm.key_resolver import ResolvedKey, resolve_provider_key
from modules.agents.factory.agent_factory import AgentFactory
from services.credits import asks
from services.credits.routing import (
    FUNDED_BY_ALLOWANCE,
    FUNDED_BY_CREDITS,
    CreditsExhaustedError,
    Funding,
)

AUTO_ON_ANTHROPIC = {"provider": "anthropic", "model": "claude-haiku-5-5"}
UNFUNDED_OWN_KEYS = Funding(FUNDED_BY_CREDITS, ("trial",), 0, own_keys_plan=True)


def _factory(monkeypatch, *, unfunded, keys):
    """An AgentFactory whose credit check and key lookups are given."""
    factory = AgentFactory(db_session=MagicMock())
    calls = []

    async def _resolve(provider, agent_name="", workspace_id=None, own_keys_only=False):
        calls.append((provider, own_keys_only))
        return keys.get((provider, own_keys_only))

    monkeypatch.setattr(factory, "_platform_unfunded", lambda ws: unfunded)
    monkeypatch.setattr(factory, "_resolve_api_key", _resolve)
    return factory, calls


def _key(provider, byok):
    return ResolvedKey(api_key="sk-test", source="byok" if byok else "platform", is_byok=byok, provider=provider)


def test_own_keys_without_an_anthropic_key_gets_add_a_key_not_a_platform_call(monkeypatch):
    factory, calls = _factory(monkeypatch, unfunded=UNFUNDED_OWN_KEYS, keys={})
    with pytest.raises(CreditsExhaustedError) as stop:
        asyncio.run(factory._activation_key(AUTO_ON_ANTHROPIC, "Auto", "ws-1"))
    assert stop.value.own_keys_plan is True and "add your" in stop.value.message.lower()
    assert calls and all(own_only for _, own_only in calls)  # never asked for a platform key


def test_own_keys_with_only_an_openrouter_key_runs_on_it(monkeypatch):
    factory, calls = _factory(monkeypatch, unfunded=UNFUNDED_OWN_KEYS,
                              keys={("openrouter", True): _key("openrouter", byok=True)})
    provider, model, resolved = asyncio.run(factory._activation_key(AUTO_ON_ANTHROPIC, "Auto", "ws-1"))
    assert provider == "openrouter" and resolved.is_byok is True
    assert calls == [("anthropic", True), ("openrouter", True)]


def test_a_funded_workspace_resolves_as_before(monkeypatch):
    factory, calls = _factory(monkeypatch, unfunded=None,
                              keys={("anthropic", False): _key("anthropic", byok=False)})
    provider, _, resolved = asyncio.run(factory._activation_key(AUTO_ON_ANTHROPIC, "Auto", "ws-1"))
    assert provider == "anthropic" and resolved.is_byok is False
    assert calls == [("anthropic", False)]


def test_the_creation_path_applies_the_same_rule(monkeypatch):
    from modules.agents.factory.agent_factory import ModelConfiguration

    factory, _ = _factory(monkeypatch, unfunded=UNFUNDED_OWN_KEYS, keys={})
    model = ModelConfiguration(provider="anthropic", model_id="claude-haiku-5-5")
    with pytest.raises(CreditsExhaustedError):
        asyncio.run(factory._creation_key(model, "Helper", "ws-1"))


# --------------------------------------------------------------------------- #
# _platform_unfunded — when may the platform's keys pay?
# --------------------------------------------------------------------------- #


def _unfunded(monkeypatch, funding, background_seen=None):
    def _funding(db, ws, background=False):
        if background_seen is not None:
            background_seen.append(background)
        return funding

    monkeypatch.setattr("services.credits.routing.workspace_funding", _funding)
    return AgentFactory(db_session=MagicMock())._platform_unfunded("ws-1")


def test_any_credits_left_let_the_platform_pay(monkeypatch):
    assert _unfunded(monkeypatch, Funding(FUNDED_BY_CREDITS, ("trial",), 1)) is None


def test_no_credits_means_own_keys_only(monkeypatch):
    assert _unfunded(monkeypatch, Funding(FUNDED_BY_CREDITS, ("trial",), 0)) is not None


def test_an_allowance_plan_lets_the_platform_pay(monkeypatch):
    assert _unfunded(monkeypatch, Funding(FUNDED_BY_ALLOWANCE)) is None


def test_a_background_run_is_judged_without_the_trial(monkeypatch):
    seen = []
    with asks.ask_scope("heartbeat", workspace_id="ws-1", background=True):
        _unfunded(monkeypatch, Funding(FUNDED_BY_CREDITS, (), 0), background_seen=seen)
    assert seen == [True]


def test_no_workspace_is_not_gated():
    assert AgentFactory(db_session=MagicMock())._platform_unfunded(None) is None


# --------------------------------------------------------------------------- #
# key_resolver — own_keys_only stops at the workspace's own key
# --------------------------------------------------------------------------- #


def test_own_keys_only_never_reaches_the_platforms_key(monkeypatch):
    platform_asked = []
    monkeypatch.setattr("core.llm.key_resolver._byok_key", lambda db, provider, ws: None)
    monkeypatch.setattr("core.llm.key_resolver._platform_key", lambda provider, name: platform_asked.append(provider))
    assert resolve_provider_key(MagicMock(), "anthropic", workspace_id="ws-1", own_keys_only=True) is None
    assert platform_asked == []


def test_own_keys_only_still_returns_the_workspaces_own_key(monkeypatch):
    own = _key("anthropic", byok=True)
    monkeypatch.setattr("core.llm.key_resolver._byok_key", lambda db, provider, ws: own)
    assert resolve_provider_key(MagicMock(), "anthropic", workspace_id="ws-1", own_keys_only=True) is own
