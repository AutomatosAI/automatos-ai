"""PRD-278 US-103 — a Bedrock model resolves from the pod's own role, with no key.

What treated a missing key as "provider not set up", above the client:

* ``core.llm.key_resolver._platform_key`` ends at the env tier, and Bedrock's env
  key is ``AWS_ACCESS_KEY_ID`` (``core.llm.providers``, ``env_key``). No key → None.
* The agent factory reads None as "not set up": the creation path reroutes to
  OpenRouter or raises "No API key available"; the activation path, and the spoken
  model, also take an empty ``api_key`` for none.
* The provider lists (``services.ai_providers``, the marketplace) offer a platform
  provider only when its env key is present.

With ``AWS_USE_WORKLOAD_IDENTITY`` on, the role answers each of them for Bedrock:
the resolver gives a key with no value and source ``workload_identity``, and the
client is built on its default-chain branch, with no key arguments. Off, nothing
changes.

Pure tests: the credential store and the operator workspace are stubbed, boto3 is
replaced at ``boto3.client``. No database, no AWS.
"""
from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import core.llm.clients.bedrock_client as bedrock_mod
import modules.agents.factory.agent_factory as factory_mod
from api.llm_marketplace import _get_available_providers
from config import config
from consumers.chatbot.spoken_model import _key_refusal
from core.llm import providers as registry
from core.llm.clients.base import LLMProvider
from core.llm import key_resolver
from core.llm.key_resolver import (
    SOURCE_WORKLOAD_IDENTITY,
    OwnKeyLookupFailed,
    ResolvedKey,
    carries_calls,
    resolve_provider_key,
)
from core.llm.manager import LLMManager
from modules.agents.factory.agent_factory import AgentFactory, ModelConfiguration
from services import ai_providers

NOVA = "amazon.nova-pro-v1:0"


@pytest.fixture(autouse=True)
def no_platform_keys(monkeypatch):
    """No workspace key, no credential-store key, no env key: only the role can answer."""
    monkeypatch.setitem(sys.modules, "core.credentials.resolver", types.SimpleNamespace(
        get_credential_resolver=lambda: SimpleNamespace(get_credential_field=lambda *a, **k: None)))
    monkeypatch.setitem(sys.modules, "core.llm.workspace_keys",
                        types.SimpleNamespace(get_platform_workspace_key=lambda *_a, **_k: None))
    monkeypatch.setattr(config, "AWS_ACCESS_KEY_ID", None)
    monkeypatch.setattr(config, "AWS_SECRET_ACCESS_KEY", None)
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(config, "AWS_REGION", "eu-west-1")
    monkeypatch.setattr(config, "AUTH_EDITION", "local")


def _identity(monkeypatch, on: bool) -> None:
    monkeypatch.setattr(config, "AWS_USE_WORKLOAD_IDENTITY", on)


def _factory(monkeypatch) -> AgentFactory:
    factory = AgentFactory(db_session=MagicMock())
    monkeypatch.setattr(factory, "_platform_unfunded", lambda _ws: None)
    return factory


# --------------------------------------------------------------------------- #
# the registry and the resolver
# --------------------------------------------------------------------------- #


def test_only_bedrock_runs_on_the_role_and_only_with_the_setting_on(monkeypatch):
    _identity(monkeypatch, True)
    assert registry.runs_on_workload_identity("bedrock") is True
    assert registry.runs_on_workload_identity("aws_bedrock") is True
    assert [slug for slug in registry.all_slugs() if registry.runs_on_workload_identity(slug)] == ["bedrock"]
    _identity(monkeypatch, False)
    assert registry.runs_on_workload_identity("bedrock") is False


def test_on_with_no_key_bedrock_resolves_to_the_role(monkeypatch):
    _identity(monkeypatch, True)
    resolved = resolve_provider_key(None, "bedrock", agent_name="agent")
    assert resolved == ResolvedKey(api_key="", source=SOURCE_WORKLOAD_IDENTITY, is_byok=False, provider="bedrock")
    assert carries_calls(resolved) is True


def test_on_another_provider_with_no_key_still_resolves_to_nothing(monkeypatch):
    _identity(monkeypatch, True)
    assert resolve_provider_key(None, "anthropic", agent_name="agent") is None


def test_off_with_no_key_bedrock_resolves_to_nothing_as_today(monkeypatch):
    _identity(monkeypatch, False)
    assert resolve_provider_key(None, "bedrock", agent_name="agent") is None
    assert carries_calls(None) is False
    assert carries_calls(ResolvedKey(api_key="", source="env", is_byok=False)) is False


# --------------------------------------------------------------------------- #
# a workspace's own key that could not be read never ends on the role
# --------------------------------------------------------------------------- #

WORKSPACE = "ws-1"


def _own_key_lookup(monkeypatch, answer) -> None:
    monkeypatch.setattr(key_resolver, "_byok_key", lambda db, provider, ws: answer)


def test_on_after_an_own_key_lookup_error_the_role_is_not_returned(monkeypatch):
    _identity(monkeypatch, True)
    _own_key_lookup(monkeypatch, OwnKeyLookupFailed(provider="bedrock"))
    assert resolve_provider_key(MagicMock(), "bedrock", workspace_id=WORKSPACE, agent_name="agent") is None


def test_on_with_no_own_key_saved_the_role_is_returned(monkeypatch):
    _identity(monkeypatch, True)
    _own_key_lookup(monkeypatch, None)
    resolved = resolve_provider_key(MagicMock(), "bedrock", workspace_id=WORKSPACE, agent_name="agent")
    assert resolved is not None and resolved.source == SOURCE_WORKLOAD_IDENTITY


@pytest.mark.parametrize("on", [True, False])
def test_after_an_own_key_lookup_error_the_platform_keys_are_still_tried(monkeypatch, on):
    _identity(monkeypatch, on)
    _own_key_lookup(monkeypatch, OwnKeyLookupFailed(provider="bedrock"))
    platform = ResolvedKey(api_key="platform-key", source="env", is_byok=False, provider="bedrock")
    monkeypatch.setattr(key_resolver, "_platform_key", lambda provider, name: platform)
    assert resolve_provider_key(MagicMock(), "bedrock", workspace_id=WORKSPACE, agent_name="agent") == platform


def test_off_after_an_own_key_lookup_error_nothing_resolves_as_today(monkeypatch):
    _identity(monkeypatch, False)
    _own_key_lookup(monkeypatch, OwnKeyLookupFailed(provider="bedrock"))
    assert resolve_provider_key(MagicMock(), "bedrock", workspace_id=WORKSPACE, agent_name="agent") is None


def test_on_the_creation_path_after_an_own_key_lookup_error_finds_no_key(monkeypatch):
    _identity(monkeypatch, True)
    _own_key_lookup(monkeypatch, OwnKeyLookupFailed(provider="bedrock"))
    with pytest.raises(ValueError, match="No API key available for bedrock"):
        asyncio.run(_factory(monkeypatch)._creation_key(ModelConfiguration("bedrock", NOVA), "agent", WORKSPACE))


def test_a_lookup_that_raises_is_reported_as_a_failed_lookup_not_as_no_key():
    db = MagicMock()
    db.query.side_effect = RuntimeError("database unavailable")
    assert key_resolver._byok_key(db, "bedrock", WORKSPACE) == OwnKeyLookupFailed(provider="bedrock")


# --------------------------------------------------------------------------- #
# the agent factory: creation and activation
# --------------------------------------------------------------------------- #


def test_on_the_creation_path_keeps_bedrock_and_builds_the_client_on_the_role_branch(monkeypatch):
    _identity(monkeypatch, True)
    factory = _factory(monkeypatch)
    built = []
    monkeypatch.setattr(factory_mod, "LLMManager", lambda config, **_kw: SimpleNamespace(config=config))
    monkeypatch.setattr(bedrock_mod.boto3, "client",
                        lambda service, **kwargs: built.append((service, kwargs)) or MagicMock(name=service))

    manager, resolved = asyncio.run(factory._create_llm_manager(ModelConfiguration("bedrock", NOVA), "agent"))

    assert manager.config.provider == LLMProvider.AWS_BEDROCK and manager.config.model == NOVA
    assert resolved.source == SOURCE_WORKLOAD_IDENTITY

    provider = LLMManager._create_provider(manager.config)
    assert isinstance(provider, bedrock_mod.BedrockProvider)
    [(service, kwargs)] = built
    assert service == "bedrock-runtime"
    assert set(kwargs) == {"config"}, "no key arguments: the SDK's chain finds the role"
    assert kwargs["config"].region_name == "eu-west-1"


def test_off_the_creation_path_still_finds_no_key(monkeypatch):
    _identity(monkeypatch, False)
    with pytest.raises(ValueError, match="No API key available for bedrock"):
        asyncio.run(_factory(monkeypatch)._creation_key(ModelConfiguration("bedrock", NOVA), "agent", None))


def test_on_the_activation_path_keeps_bedrock(monkeypatch):
    _identity(monkeypatch, True)
    provider, model, resolved = asyncio.run(
        _factory(monkeypatch)._activation_key({"provider": "aws_bedrock", "model": NOVA}, "agent", None))
    assert (provider, model, resolved.source) == ("aws_bedrock", NOVA, SOURCE_WORKLOAD_IDENTITY)


def test_off_the_activation_path_falls_back_to_openrouter_as_today(monkeypatch):
    _identity(monkeypatch, False)
    factory = _factory(monkeypatch)
    routed = ResolvedKey(api_key="sk-or-test", source="platform", is_byok=False, provider="openrouter")

    async def resolve(provider, agent_name="", workspace_id=None, own_keys_only=False):
        return routed if provider == "openrouter" else None

    monkeypatch.setattr(factory, "_resolve_api_key", resolve)
    provider, _, resolved = asyncio.run(
        factory._activation_key({"provider": "aws_bedrock", "model": NOVA}, "agent", None))
    assert provider == "openrouter" and resolved is routed


def test_off_with_no_openrouter_key_either_it_stays_on_its_own_model(monkeypatch):
    """#1678: a fallback with no key can't run it either; the agent keeps the model the person chose,
    so its error names that model, not OpenRouter's id for it."""
    _identity(monkeypatch, False)
    assert asyncio.run(_factory(monkeypatch)._activation_key(
        {"provider": "aws_bedrock", "model": NOVA}, "agent", None)) == ("aws_bedrock", NOVA, None)


def test_on_the_spoken_model_takes_the_role_as_a_key():
    role = ResolvedKey(api_key="", source=SOURCE_WORKLOAD_IDENTITY, is_byok=False, provider="bedrock")
    assert _key_refusal(SimpleNamespace(is_byok=False), "bedrock", role) is None
    keyless = ResolvedKey(api_key="", source="env", is_byok=False, provider="bedrock")
    assert _key_refusal(SimpleNamespace(is_byok=False), "bedrock", keyless) == "no bedrock key is open to this workspace"


# --------------------------------------------------------------------------- #
# offered: the provider lists
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("on", [True, False])
def test_the_platform_routes_offer_bedrock_only_with_the_setting_on(monkeypatch, on):
    _identity(monkeypatch, on)
    for spec in registry.REGISTRY.values():
        if spec.env_key:
            monkeypatch.setattr(config, spec.env_key, "", raising=False)
    monkeypatch.setattr(ai_providers, "_operator_workspace_slugs", lambda _db: [])
    monkeypatch.setattr(ai_providers, "_credential_store_slugs", lambda _db, _slugs: [])

    routes = ai_providers._platform_routes(None, None)

    assert ("bedrock" in routes) is on
    assert set(routes) <= {"bedrock"}


@pytest.mark.parametrize("on", [True, False])
def test_the_marketplace_counts_bedrock_as_available_only_with_the_setting_on(monkeypatch, on):
    _identity(monkeypatch, on)
    for spec in registry.REGISTRY.values():
        if spec.env_key:
            monkeypatch.setattr(config, spec.env_key, "", raising=False)

    available = _get_available_providers(None, None)

    assert ("bedrock" in available) is on
    assert available <= {"bedrock"}
