"""#1678: a chat with no key for Auto's model said "OpenRouter API key not configured".

A self-hoster with only an OpenAI key asked Auto a question. Auto's model, Anthropic's
claude-haiku-5-5 as a fresh install seeds it, had no key, so the turn fell back to OpenRouter,
which had none either, and the error named OpenRouter. Now a fallback happens only when OpenRouter
has a key. Every client with no key fails with one sentence (``core.llm.no_key``): the provider and
model that need the key, and both ways out, as Settings → Orchestrator says. The chat turn shows
that sentence as it is.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from consumers.chatbot.turn_errors import CODE_NO_API_KEY, RAW_MESSAGE_CHARS, describe_turn_error
from core.llm.key_resolver import ResolvedKey
from core.llm.no_key import NO_KEY_LEAD, no_key_sentence
from modules.agents.factory.agent_factory import AgentFactory

AUTO = {"provider": "anthropic", "model": "claude-haiku-5-5"}  # a fresh install's Auto (core/llm/defaults.py)
SENTENCE = ("No API key for Anthropic claude-haiku-5-5: add an Anthropic or OpenRouter key in "
            "Settings → Model keys, or choose a model whose provider has a key.")


def _factory(monkeypatch, keys) -> AgentFactory:
    """A factory whose key lookups answer from ``keys`` (provider -> ResolvedKey), funded as the
    local edition is."""
    factory = AgentFactory(db_session=MagicMock())
    monkeypatch.setattr(factory, "_platform_unfunded", lambda _ws: None)

    async def resolve(provider, agent_name="", workspace_id=None, own_keys_only=False):
        return keys.get(provider)

    monkeypatch.setattr(factory, "_resolve_api_key", resolve)
    return factory


def test_the_sentence_names_the_model_and_both_ways_out():
    assert no_key_sentence("anthropic", "claude-haiku-5-5") == SENTENCE


@pytest.mark.parametrize(("provider", "model", "keys"), [
    ("openai", "gpt-4o-mini", "an OpenAI or OpenRouter key"),
    ("google", "gemini-2.5-flash", "a Google or OpenRouter key"),
    ("openrouter", "anthropic/claude-haiku-5.5", "an OpenRouter key in"),  # already on OpenRouter
    ("aws_bedrock", "amazon.nova-pro-v1:0", "an AWS Bedrock key in"),  # not a model OpenRouter runs
])
def test_openrouter_is_offered_only_where_it_can_run_the_model(provider, model, keys):
    sentence = no_key_sentence(provider, model)
    assert sentence.startswith(NO_KEY_LEAD) and model in sentence and keys in sentence
    assert len(sentence) <= RAW_MESSAGE_CHARS  # a turn error keeps it whole


def test_with_no_openrouter_key_auto_stays_on_its_own_model(monkeypatch):
    assert asyncio.run(_factory(monkeypatch, {})._activation_key(AUTO, "Auto", "ws-1")) == (
        "anthropic", "claude-haiku-5-5", None)


def test_with_an_openrouter_key_auto_still_runs_on_it(monkeypatch):
    routed = ResolvedKey(api_key="sk-or-test", source="platform", is_byok=False, provider="openrouter")
    provider, model, resolved = asyncio.run(
        _factory(monkeypatch, {"openrouter": routed})._activation_key(AUTO, "Auto", "ws-1"))
    assert (provider, resolved) == ("openrouter", routed) and model.startswith("anthropic/")


def test_the_chat_turn_shows_the_sentence_whole():
    error = describe_turn_error(ValueError(SENTENCE), agent_name="Auto")
    assert (error.code, error.message) == (CODE_NO_API_KEY, f"Auto can't answer. {SENTENCE}")


def test_a_providers_own_no_key_wording_still_reads_as_no_key():
    error = describe_turn_error(ValueError("OpenAI API key not configured."), agent_name="Auto")
    assert error.code == CODE_NO_API_KEY and error.message.startswith("Auto has no API key for its provider.")


def test_auto_with_only_an_openai_key_fails_its_first_call_naming_its_own_model(monkeypatch):
    """The chat path end to end at the model: Auto's manager as activation builds it, its first
    call, and what the turn shows. The OpenAI key is the workspace's, and can't run Anthropic."""
    openai_only = {"openai": ResolvedKey(api_key="sk-test", source="byok", is_byok=True, provider="openai")}
    factory = _factory(monkeypatch, openai_only)
    auto = type("Auto", (), {"name": "Auto"})()
    manager, resolved, provider = asyncio.run(factory._activation_llm_manager(dict(AUTO), auto, 1, None))
    assert (provider, manager.config.model, resolved) == ("anthropic", "claude-haiku-5-5", None)

    with pytest.raises(ValueError) as failed:
        asyncio.run(manager.generate_response(messages=[{"role": "user", "content": "What is two plus three?"}]))
    assert str(failed.value) == SENTENCE
    assert describe_turn_error(failed.value, agent_name="Auto").message == f"Auto can't answer. {SENTENCE}"


def test_an_initialism_takes_its_article_from_the_letters_name():
    assert "add an NVIDIA key" in no_key_sentence("nvidia", "m")
    assert "add an AWS Bedrock key" in no_key_sentence("aws_bedrock", "m")
    assert "add a HuggingFace key" in no_key_sentence("huggingface", "m")


@pytest.mark.parametrize(("edition", "runs_as"), [("saas", "openrouter"), ("local", "nvidia")])
def test_an_own_keys_only_provider_never_stays_where_the_edition_bills(monkeypatch, edition, runs_as):
    """NVIDIA's key is the workspace's own where Automatos bills (byok_only), and the resolver refuses
    the server's. Left on NVIDIA with no key, its client would read that env key itself; it goes to
    OpenRouter as before instead. Where nobody bills, the operator's key is theirs, so it stays."""
    from config import config

    monkeypatch.setattr(config, "AUTH_EDITION", edition)
    provider, _, resolved = asyncio.run(
        _factory(monkeypatch, {})._activation_key({"provider": "nvidia", "model": "m"}, "agent", "ws-1"))
    assert (provider, resolved) == (runs_as, None)
