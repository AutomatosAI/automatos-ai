"""Spoken turns answer on ``SPOKEN_MODE_MODEL``; typed turns keep the agent's own model.

28 Oct 2026: a spoken quick ask took 26-32 s a turn on Auto's claude-opus-5-5. The 11 Oct
benchmark scored claude-haiku-5-5 the same at about 3% of the cost.

* A spoken turn's agent answers on the spoken model: a copy of the agent's runtime whose manager
  names that model and its provider, on the key that pays. Only the model changes: persona, tools
  and the rest are the agent's, and the agent the factory holds keeps its own model.
* A typed turn answers on the agent's model, the very runtime.
* The setting empty: spoken turns answer on the agent's model too.
* No key for the provider (none at all, a BYO-key-only provider in this edition, or a workspace
  on its own keys only without one), a policy block, or an agent on the workspace's own key that
  would move to the platform's: the agent's model, said once at info, never the person's words.
  A lookup that fails keeps the agent's model too.
* The call is booked, and priced, on the model that answered it.
* Every turn logs the model it answers on, and the service's own factory is back after the turn.
* ``StreamingChatService._stream_response_with_agent_scoped`` is wrapped.

DB-free: no database, model or socket.
"""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest

from config import config
from consumers.chatbot import spoken_model as sm
from core.llm.clients.base import LLMConfig, LLMProvider
from core.llm.key_resolver import ResolvedKey
from core.llm.manager import LLMManager, estimate_cost_usd
from modules.agents.factory.agent_factory import AgentFactory, AgentLifecycle, AgentMetadata, AgentRuntime

WS = "6a0f3c1e-2b4d-4e8f-9a1b-5b0e40000001"
AUTO = 337
OPUS = "claude-opus-5-5"
HAIKU = "claude-haiku-5-5"
ANTHROPIC = "anthropic"
AGENT_BUDGET = 8000
HAIKU_CEILING = 64_000
ASKED = "Read me what needs me today"
TOOLS = ({"app": "gmail"},)
SPOKEN_LOGGER = "consumers.chatbot.spoken_model"


@pytest.fixture(autouse=True)
def _spoken_model_on(monkeypatch):
    monkeypatch.setattr(config, "SPOKEN_MODE_MODEL", HAIKU)
    monkeypatch.setattr(config, "SPOKEN_MODE_PROVIDER", ANTHROPIC)


def _runtime(*, is_byok=False):
    """Auto on claude-opus-5-5, as the factory activates it."""
    manager = LLMManager(config=LLMConfig(provider=LLMProvider.ANTHROPIC, model=OPUS, max_tokens=AGENT_BUDGET,
                                          api_key="agent-key", temperature=0.7),
                         workspace_id=WS, agent_id=AUTO, is_byok=is_byok)
    return AgentRuntime(agent_id=AUTO, metadata=AgentMetadata(name="Auto", agent_type="orchestrator", persona="Auto"),
                        llm_manager=manager, lifecycle_state=AgentLifecycle.ACTIVE, created_at=datetime.now(),
                        tools=list(TOOLS), is_byok=is_byok, resolved_provider=ANTHROPIC)


def _key(*, byok):
    return ResolvedKey(api_key="own-key" if byok else "platform-key", source="byok" if byok else "env",
                       is_byok=byok, provider=ANTHROPIC)


def _factory(monkeypatch, *, keys=None, unfunded=None, allowed=True, ceiling=HAIKU_CEILING):
    """An AgentFactory whose credit check, key lookup, model ceiling and policy are given."""
    factory = AgentFactory(db_session=MagicMock())
    asked = NS(keys=[], policy=[])

    async def _resolve(provider, agent_name="", workspace_id=None, own_keys_only=False):
        asked.keys.append((provider, own_keys_only))
        return (keys or {}).get((provider, own_keys_only))

    def _policy(db, workspace_id, model_id, *, orchestrator_seat, provider=None):
        asked.policy.append((model_id, provider, orchestrator_seat))
        return (True, "allowed") if allowed else (False, f"model '{model_id}' is quarantined in this workspace")

    monkeypatch.setattr(factory, "_platform_unfunded", lambda ws: unfunded)
    monkeypatch.setattr(factory, "_resolve_api_key", _resolve)
    monkeypatch.setattr(factory, "_model_ceiling", lambda model_id: ceiling)
    monkeypatch.setattr("core.llm.model_policy.check_model_for_agent", _policy)
    return factory, asked


def _spoken(factory, runtime):
    return asyncio.run(sm.spoken_turn_runtime(factory, runtime, WS))


def _turn(factory, runtime, monkeypatch, *, spoken):
    """One turn through the wrapper: the stand-in turn activates the agent the way the real one
    does and yields the model it answers on. Returns (models, the chat)."""
    async def _activate(agent_id, **kwargs):
        return runtime

    monkeypatch.setattr(factory, "activate_agent", _activate)

    async def turn(chat, chat_id, messages, agent_id, user_id, **kwargs):
        answering = await chat.agent_factory.activate_agent(agent_id, use_orchestrator_llm=False, workspace_id=WS)
        yield sm.model_of(answering)

    chat = NS(agent_factory=factory)
    wrapped = sm.answers_on_the_turns_model(turn)

    async def run():
        messages = [{"role": "user", "content": ASKED}]
        return [chunk async for chunk in wrapped(chat, "chat-1", messages, AUTO, 7, spoken_mode=spoken)]

    return asyncio.run(run()), chat


# --------------------------------------------------------------------------- the spoken model


def test_a_spoken_turn_answers_on_the_spoken_model(monkeypatch):
    factory, asked = _factory(monkeypatch, keys={(ANTHROPIC, False): _key(byok=False)})

    spoken = _spoken(factory, _runtime())

    assert sm.model_of(spoken) == HAIKU
    assert spoken.llm_manager.config.provider == LLMProvider.ANTHROPIC   # the direct client, never OpenRouter
    assert spoken.llm_manager.config.api_key == "platform-key"
    assert (spoken.is_byok, spoken.resolved_provider) == (False, ANTHROPIC)
    assert asked.keys == [(ANTHROPIC, False)]
    assert asked.policy == [(HAIKU, ANTHROPIC, True)]                     # held to Auto's seat


def test_only_the_model_changes_and_the_agent_keeps_its_own(monkeypatch):
    factory, _ = _factory(monkeypatch, keys={(ANTHROPIC, False): _key(byok=False)})
    runtime = _runtime()

    spoken = _spoken(factory, runtime)

    assert spoken is not runtime
    assert (spoken.agent_id, spoken.metadata, spoken.tools, spoken.tool_executor) == (
        runtime.agent_id, runtime.metadata, runtime.tools, runtime.tool_executor)
    assert spoken.llm_manager.config.temperature == runtime.llm_manager.config.temperature
    assert spoken.llm_manager.config.max_tokens == AGENT_BUDGET
    assert sm.model_of(runtime) == OPUS                                    # the factory's agent is untouched


def test_the_agents_budget_is_capped_by_the_spoken_models_ceiling(monkeypatch):
    factory, _ = _factory(monkeypatch, keys={(ANTHROPIC, False): _key(byok=False)}, ceiling=4096)

    spoken = _spoken(factory, _runtime())

    assert (spoken.llm_manager.config.max_tokens, spoken.llm_manager.config.output_ceiling) == (4096, 4096)


def test_a_workspace_key_of_its_own_carries_the_spoken_model(monkeypatch):
    factory, _ = _factory(monkeypatch, keys={(ANTHROPIC, False): _key(byok=True)})

    spoken = _spoken(factory, _runtime(is_byok=True))

    assert (sm.model_of(spoken), spoken.is_byok, spoken.llm_manager.config.api_key) == (HAIKU, True, "own-key")


# --------------------------------------------------------------------------- typed and off


def test_a_typed_turn_answers_on_the_agents_model(monkeypatch):
    factory, asked = _factory(monkeypatch, keys={(ANTHROPIC, False): _key(byok=False)})

    models, chat = _turn(factory, _runtime(), monkeypatch, spoken=False)

    assert models == [OPUS]
    assert asked.keys == [] and asked.policy == []
    assert chat.agent_factory is factory


def test_a_spoken_turn_through_the_wrapper_answers_on_the_spoken_model(monkeypatch):
    factory, _ = _factory(monkeypatch, keys={(ANTHROPIC, False): _key(byok=False)})
    runtime = _runtime()

    models, chat = _turn(factory, runtime, monkeypatch, spoken=True)

    assert models == [HAIKU]
    assert chat.agent_factory is factory                                   # the service's own factory is back
    assert sm.model_of(runtime) == OPUS


@pytest.mark.parametrize("setting", ["MODEL", "PROVIDER"])
def test_the_setting_empty_keeps_the_agents_model(monkeypatch, setting):
    monkeypatch.setattr(config, f"SPOKEN_MODE_{setting}", "")
    factory, asked = _factory(monkeypatch, keys={(ANTHROPIC, False): _key(byok=False)})
    runtime = _runtime()

    assert _spoken(factory, runtime) is runtime
    assert asked.keys == []


def test_an_agent_already_on_the_spoken_model_keeps_its_runtime(monkeypatch):
    monkeypatch.setattr(config, "SPOKEN_MODE_MODEL", OPUS)
    factory, asked = _factory(monkeypatch, keys={(ANTHROPIC, False): _key(byok=False)})
    runtime = _runtime()

    assert _spoken(factory, runtime) is runtime
    assert asked.keys == []


# --------------------------------------------------------------------------- the fallbacks


def _stays_lines(caplog):
    return [r for r in caplog.records if r.name == SPOKEN_LOGGER and r.getMessage().startswith("[spoken]")]


def test_no_key_for_the_provider_keeps_the_agents_model_and_says_so_once(monkeypatch, caplog):
    factory, asked = _factory(monkeypatch, keys={})
    caplog.set_level(logging.INFO, logger=SPOKEN_LOGGER)

    models, _ = _turn(factory, _runtime(), monkeypatch, spoken=True)

    assert models == [OPUS]
    said = _stays_lines(caplog)
    assert len(said) == 1 and said[0].levelno == logging.INFO
    assert HAIKU in said[0].getMessage() and OPUS in said[0].getMessage()
    assert "no anthropic key" in said[0].getMessage()
    assert ASKED not in caplog.text                                        # the ids, never the person's words


def test_a_byo_key_only_provider_in_this_edition_keeps_the_agents_model(monkeypatch):
    """The real resolver: no key of the workspace's own, and no platform lane for the provider."""
    factory = AgentFactory(db_session=MagicMock())
    monkeypatch.setattr(factory, "_platform_unfunded", lambda ws: None)
    monkeypatch.setattr(factory, "_model_ceiling", lambda model_id: HAIKU_CEILING)
    monkeypatch.setattr("core.llm.model_policy.check_model_for_agent", lambda *a, **k: (True, "allowed"))
    monkeypatch.setattr("core.llm.key_resolver._byok_key", lambda db, provider, ws: None)
    monkeypatch.setattr("core.llm.providers.platform_key_allowed", lambda provider, edition=None: False)
    runtime = _runtime()

    assert _spoken(factory, runtime) is runtime


def test_a_workspace_on_its_own_keys_only_without_an_anthropic_key_keeps_the_agents_model(monkeypatch):
    factory, asked = _factory(monkeypatch, keys={(ANTHROPIC, False): _key(byok=False)}, unfunded=object())
    runtime = _runtime()

    assert _spoken(factory, runtime) is runtime
    assert asked.keys == [(ANTHROPIC, True)]                               # never the platform's key


def test_an_agent_on_the_workspaces_own_key_is_never_moved_onto_the_platforms(monkeypatch, caplog):
    factory, _ = _factory(monkeypatch, keys={(ANTHROPIC, False): _key(byok=False)})
    caplog.set_level(logging.INFO, logger=SPOKEN_LOGGER)
    runtime = _runtime(is_byok=True)

    assert _spoken(factory, runtime) is runtime
    assert "own key" in _stays_lines(caplog)[0].getMessage()


def test_a_model_the_policy_blocks_keeps_the_agents_model(monkeypatch):
    factory, asked = _factory(monkeypatch, keys={(ANTHROPIC, False): _key(byok=False)}, allowed=False)
    runtime = _runtime()

    assert _spoken(factory, runtime) is runtime
    assert asked.keys == []


def test_a_lookup_that_fails_keeps_the_turn_on_the_agents_model(monkeypatch, caplog):
    factory, _ = _factory(monkeypatch)

    async def _down(*args, **kwargs):
        raise RuntimeError("credential store unreachable")

    monkeypatch.setattr(factory, "_resolve_api_key", _down)
    caplog.set_level(logging.INFO, logger=SPOKEN_LOGGER)

    models, _ = _turn(factory, _runtime(), monkeypatch, spoken=True)

    assert models == [OPUS]
    assert any(r.levelno == logging.ERROR and r.exc_info for r in caplog.records if r.name == SPOKEN_LOGGER)


# --------------------------------------------------------------------------- booking and the log


def test_a_spoken_call_is_booked_and_priced_on_the_model_that_answered(monkeypatch):
    from core.llm import usage_tracker

    booked = []
    monkeypatch.setattr(usage_tracker.UsageTracker, "track", staticmethod(lambda **kw: booked.append(kw)))
    factory, _ = _factory(monkeypatch, keys={(ANTHROPIC, False): _key(byok=False)})
    runtime = _runtime()
    spoken = _spoken(factory, runtime)

    spoken.llm_manager._track_usage(NS(usage={"input_tokens": 12_000, "output_tokens": 300}), time.monotonic())

    assert [(b["model_id"], b["provider"], b["agent_id"], b["is_byok"]) for b in booked] == [
        (HAIKU, ANTHROPIC, AUTO, False)]
    spoken_cost, _ = spoken.llm_manager.call_cost(12_000, 300)
    assert spoken_cost == pytest.approx(estimate_cost_usd(HAIKU, 12_000, 300))
    assert spoken_cost < runtime.llm_manager.call_cost(12_000, 300)[0]


@pytest.mark.parametrize("spoken,model", [(True, HAIKU), (False, OPUS)])
def test_every_turn_logs_the_model_it_answers_on(monkeypatch, caplog, spoken, model):
    factory, _ = _factory(monkeypatch, keys={(ANTHROPIC, False): _key(byok=False)})
    caplog.set_level(logging.INFO, logger=SPOKEN_LOGGER)

    _turn(factory, _runtime(), monkeypatch, spoken=spoken)

    lines = [r.getMessage() for r in caplog.records if r.name == SPOKEN_LOGGER and r.levelno == logging.INFO]
    assert f"[chat] turn model={model} spoken={str(spoken).lower()}" in lines


def test_the_chat_turn_is_wrapped():
    from consumers.chatbot.service import StreamingChatService

    async def _turn_stub(chat, *args, **kwargs):
        yield None

    codes, fn = [], StreamingChatService._stream_response_with_agent_scoped
    while fn is not None:
        codes.append(fn.__code__)
        fn = getattr(fn, "__wrapped__", None)

    assert sm.answers_on_the_turns_model(_turn_stub).__code__ in codes
