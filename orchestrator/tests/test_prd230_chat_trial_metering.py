"""PRD-230 US-001 (W0, fix/prd-222) — chat-path credit gating.

Originally: the trial gate lived ONLY in ``AgentFactory._create_llm_manager``
(the mission factory path); the chatbot activated Auto through
``activate_agent``, which built the ``LLMManager`` directly and never called
the gate, so chat ran unmetered and unpinned. That wave threaded the
CONVERSATION's workspace into ``activate_agent`` and shared one gate
(``_resolve_trial_decision``) across both surfaces.

PRD-265 W1 replaced the $5 trial ledger and ``_resolve_trial_decision`` with
the credit gate: ``AgentFactory._platform_unfunded`` (PRD-265 US-004) is now
the shared own-keys-only check, called by both ``_creation_key`` (mission
path) and ``_activation_key`` (chat path). A workspace whose credits can't
pay for the platform's keys runs on its own keys or not at all —
``CreditsExhaustedError`` when it has none. These are unit tests of
``_activation_key`` with ``_platform_unfunded`` and ``_resolve_api_key``
monkeypatched (no DB, no live LLM), plus the real ``key_resolver`` proving
BYOK bypasses ``own_keys_only``, and the established source-grep wiring
guards (``test_prd222_trial_enforcement`` style).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from modules.agents.factory.agent_factory import AgentFactory
from services.credits.routing import FUNDED_BY_CREDITS, CreditsExhaustedError, Funding

REPO = Path(__file__).resolve().parents[1]
WS_ID = "11111111-1111-1111-1111-111111111111"

UNFUNDED = Funding(FUNDED_BY_CREDITS, spendable=0, own_keys_plan=False)


def _factory():
    # _activation_key never touches the session directly when _platform_unfunded
    # and _resolve_api_key are monkeypatched — any object will do.
    return AgentFactory(db_session=object())


class _Resolved:
    def __init__(self, is_byok=False, api_key="key"):
        self.is_byok = is_byok
        self.api_key = api_key
        self.provider = "anthropic"
        self.source = "byok" if is_byok else "platform"
        self.base_url = None


# --------------------------------------------------------------------------- #
# _activation_key — the chat path's own-keys-only gate (PRD-265 US-004)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_funded_workspace_resolves_the_platform_key(monkeypatch):
    f = _factory()
    monkeypatch.setattr(f, "_platform_unfunded", lambda ws: None)
    own_keys_only_seen = []

    async def fake_resolve(provider, agent_name="", workspace_id=None, own_keys_only=False):
        own_keys_only_seen.append(own_keys_only)
        return _Resolved(is_byok=False)

    monkeypatch.setattr(f, "_resolve_api_key", fake_resolve)
    _provider, _model, resolved = await f._activation_key(
        {"provider": "anthropic", "model": "claude-x"}, "Auto", WS_ID,
    )
    assert resolved.is_byok is False
    assert own_keys_only_seen == [False]  # funded — nothing restricts the platform's keys


@pytest.mark.asyncio
async def test_unfunded_workspace_resolves_own_keys_only(monkeypatch):
    f = _factory()
    monkeypatch.setattr(f, "_platform_unfunded", lambda ws: UNFUNDED)
    own_keys_only_seen = []

    async def fake_resolve(provider, agent_name="", workspace_id=None, own_keys_only=False):
        own_keys_only_seen.append(own_keys_only)
        return _Resolved(is_byok=True)

    monkeypatch.setattr(f, "_resolve_api_key", fake_resolve)
    _provider, _model, resolved = await f._activation_key(
        {"provider": "anthropic", "model": "claude-x"}, "Auto", WS_ID,
    )
    assert resolved.is_byok is True
    assert own_keys_only_seen == [True]  # unfunded — only the workspace's own keys may run it


@pytest.mark.asyncio
async def test_unfunded_workspace_with_no_own_key_raises_credits_exhausted(monkeypatch):
    f = _factory()
    monkeypatch.setattr(f, "_platform_unfunded", lambda ws: UNFUNDED)

    async def fake_resolve(provider, agent_name="", workspace_id=None, own_keys_only=False):
        return None  # no BYOK, and the OpenRouter fallback finds nothing either

    monkeypatch.setattr(f, "_resolve_api_key", fake_resolve)
    with pytest.raises(CreditsExhaustedError) as ei:
        await f._activation_key({"provider": "anthropic", "model": "claude-x"}, "Auto", WS_ID)
    assert ei.value.own_keys_plan is False


# --------------------------------------------------------------------------- #
# BYOK resolves through — key_resolver itself, independent of own_keys_only
# --------------------------------------------------------------------------- #


def test_byok_resolves_through_even_when_own_keys_only(monkeypatch):
    from core.llm import key_resolver

    sentinel = key_resolver.ResolvedKey(api_key="sk-byok", source="byok", is_byok=True, provider="anthropic")
    monkeypatch.setattr(key_resolver, "_byok_key", lambda db, provider, ws: sentinel)
    result = key_resolver.resolve_provider_key(object(), "anthropic", workspace_id=WS_ID, own_keys_only=True)
    assert result is sentinel  # checked before own_keys_only is ever consulted


# --------------------------------------------------------------------------- #
# Wiring — the chat activation path shares the gate; the conversation
# workspace is threaded in; nothing duplicates the check (AC1/AC2)
# --------------------------------------------------------------------------- #


def test_gate_is_shared_not_duplicated():
    # _creation_key (mission path) and _activation_key (chat path) must call
    # the SAME helper — no parallel credit-funding check to drift.
    src = (REPO / "modules" / "agents" / "factory" / "agent_factory.py").read_text()
    assert src.count("def _platform_unfunded(") == 1
    assert src.count("_platform_unfunded(") >= 3  # defined once, called by creation + activation (+ cached-runtime refresh)


def test_activate_agent_reraises_credits_exhausted_not_swallowed():
    src = (REPO / "modules" / "agents" / "factory" / "agent_factory.py").read_text()
    assert "_platform_unfunded" in src
    assert "except CreditsExhaustedError:" in src  # not swallowed into a generic None


def test_chatbot_threads_conversation_workspace_into_activation():
    src = (REPO / "consumers" / "chatbot" / "service.py").read_text()
    assert "activate_agent(" in src
    assert "workspace_id=self.workspace_id" in src  # grep-proof call site (AC1), unchanged by PRD-265
