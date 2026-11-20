"""PRD-278 US-105 — a Bedrock model is called on Bedrock, in the install's region.

With ``AWS_USE_WORKLOAD_IDENTITY`` on, an agent whose provider is stored as ``bedrock`` or
``aws_bedrock`` is called on Bedrock when it is created, when it is activated for chat and in
the spoken model, and so is a settings tier that names either: in ``AWS_REGION`` on every path,
signed by the install's role when no key is set, and never moved to another provider. With the
setting off, each path behaves as on main.

The one name rule is ``providers.stored_provider_enum`` (the registry's ``enum_for`` for Bedrock
with the setting on, the strict enum value otherwise); the region is
``providers.workload_identity_region``.

botocore's real request pipeline runs and stops just before sending (as in
``test_prd278_w1_bedrock_credential_sources.py``): each test sees the request's endpoint and
how it was signed. The SDK's chain is given a key of its own through the environment, standing
in for the install's role. ``AWS_REGION`` in the environment is removed, so a client left to
its own default would call ``us-east-1``.
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import types
from datetime import datetime
from types import SimpleNamespace
from urllib.parse import urlsplit

import boto3
import pytest
from botocore.awsrequest import AWSResponse
from unittest.mock import MagicMock

import core.llm.manager as llm_manager
import modules.agents.factory.agent_factory as factory_mod
from config import config
from consumers.chatbot import spoken_model
from core.llm import key_resolver
from core.llm import providers as registry
from core.llm.clients.base import LLMConfig, LLMProvider
from core.llm.clients.bedrock_client import BedrockProvider
from core.llm.key_resolver import OwnKeyLookupFailed
from modules.agents.factory.agent_factory import (
    AgentFactory, AgentLifecycle, AgentMetadata, AgentRuntime, ModelConfiguration,
)

NOVA = "amazon.nova-pro-v1:0"
# An application inference profile is named by its ARN, which carries a slash (AWS's example account).
PROFILE_ARN = "arn:aws:bedrock:eu-west-1:123456789012:application-inference-profile/example"
INSTALL_REGION = "eu-west-1"
CLIENT_DEFAULT_REGION = "us-east-1"
STORED_REGION = "ap-southeast-2"
BEDROCK_NAMES = ("bedrock", "aws_bedrock")
ROLE_KEY_ID, ROLE_SECRET = "AKIAI44QH8DHBEXAMPLE", "je7MtGbClwBF/2Zp9Utk/h3yCo8nvbEXAMPLEKEY"
BEDROCK_API_KEY = "ABSK-example-long-term"
OPENROUTER_KEY = "or-example-key"
WORKSPACE = "ws-1"
REPLY = {"output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}}, "stopReason": "end_turn",
         "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}}


class _Raw:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def stream(self, **_kwargs):
        yield self._body


@pytest.fixture(autouse=True)
def install(monkeypatch, tmp_path):
    """The chain answers with the role's key; no stored, operator or env key; the install's region
    is ``eu-west-1``. ``install.credentials`` is what the credential store holds, by provider."""
    for name in ("AWS_SESSION_TOKEN", "AWS_PROFILE", "AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION", "AWS_DEFAULT_REGION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", ROLE_KEY_ID)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", ROLE_SECRET)
    monkeypatch.setattr(boto3, "DEFAULT_SESSION", None)
    monkeypatch.setitem(sys.modules, "core.credentials.resolver", types.SimpleNamespace(
        get_credential_resolver=lambda: SimpleNamespace(get_credential_field=lambda *a, **k: None)))
    monkeypatch.setitem(sys.modules, "core.llm.workspace_keys",
                        types.SimpleNamespace(get_platform_workspace_key=lambda *_a, **_k: None))
    for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"):
        monkeypatch.setattr(config, name, None)
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    monkeypatch.setattr(config, "AWS_REGION", INSTALL_REGION)
    monkeypatch.setattr(config, "AUTH_EDITION", "local")
    held = SimpleNamespace(credentials={})
    monkeypatch.setattr(llm_manager, "get_credential_data",
                        lambda provider, environment=None, service_name="orchestrator":
                        dict(held.credentials.get(provider, {})))
    monkeypatch.setattr(llm_manager.output_budget, "manager_budgets", lambda *a, **k: (None, None))
    monkeypatch.setattr(llm_manager, "get_system_setting", lambda c, k, d=None: d)  # no settings rows
    return held


def _identity(monkeypatch, on: bool) -> None:
    monkeypatch.setattr(config, "AWS_USE_WORKLOAD_IDENTITY", on)


def _sent(llm_config: LLMConfig):
    """The one Converse request a client built from this config sends (provider checked)."""
    provider = llm_manager.LLMManager._create_provider(llm_config)
    assert isinstance(provider, BedrockProvider), "called on Bedrock"
    seen = []

    def send(request, **_kwargs):
        seen.append(request)
        return AWSResponse(request.url, 200, {"content-type": "application/json"}, _Raw(json.dumps(REPLY).encode()))

    provider.client.meta.events.register("before-send.bedrock-runtime.Converse", send)
    provider.generate_response_sync([{"role": "user", "content": "hi"}])
    return seen[0]


def _endpoint_region(request) -> str:
    return urlsplit(request.url).hostname.split(".")[1]


def _signed_by_the_role_in(llm_config: LLMConfig) -> str:
    """The region a role-signed request goes to; fails unless the role signed it, in that region."""
    request = _sent(llm_config)
    credential = request.headers["Authorization"].decode().split("Credential=")[1].split(",")[0]
    key_id, _date, scope_region, service, _end = credential.split("/")
    assert (key_id, service) == (ROLE_KEY_ID, "bedrock"), "signed by the install's role"
    assert scope_region == _endpoint_region(request)
    return scope_region


def _factory(monkeypatch) -> AgentFactory:
    """A factory whose managers are captured (their config is what a client is built from)."""
    factory = AgentFactory(db_session=MagicMock())
    monkeypatch.setattr(factory, "_platform_unfunded", lambda _ws: None)
    monkeypatch.setattr(factory, "_model_ceiling", lambda _model: None)
    monkeypatch.setattr(factory_mod, "LLMManager", lambda config, **_kw: SimpleNamespace(config=config))
    return factory


def _activated(monkeypatch, name: str, model: str = NOVA):
    return asyncio.run(_factory(monkeypatch)._activation_llm_manager(
        {"provider": name, "model": model}, SimpleNamespace(name="agent"), 1, None))


def _spoken(monkeypatch, name: str):
    """An agent on another model, moved to a Bedrock spoken model for one turn."""
    monkeypatch.setattr(config, "SPOKEN_MODE_MODEL", NOVA)
    monkeypatch.setattr(config, "SPOKEN_MODE_PROVIDER", name)
    monkeypatch.setattr("core.llm.model_policy.check_model_for_agent", lambda *a, **k: (True, "allowed"))
    agent = SimpleNamespace(config=LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-opus-5-5", api_key="k"))
    runtime = AgentRuntime(agent_id=1, metadata=AgentMetadata(name="Auto", agent_type="orchestrator", persona="Auto"),
                           llm_manager=agent, lifecycle_state=AgentLifecycle.ACTIVE, created_at=datetime.now())
    return asyncio.run(spoken_model.spoken_turn_runtime(_factory(monkeypatch), runtime, None))


def _settings_tier(monkeypatch, name: str, model: str = NOVA) -> LLMConfig:
    rows = {("system_llm", "provider"): name, ("system_llm", "model"): model}
    monkeypatch.setattr(llm_manager, "get_system_setting", lambda c, k, d=None: rows.get((c, k), d))
    return llm_manager.LLMManager(service_name="planner").config


def _loaded(provider: str, model: str) -> LLMConfig:
    """The settings loader given a provider and a model (as ``vendor_fit.fitted`` hands them on)."""
    return llm_manager.LLMManager._load_config_from_settings(SimpleNamespace(), "planner", provider, model)


def _warnings_of(logger_name: str, run):
    """``run()``'s result and the warnings ``logger_name`` logged while it ran."""
    records = []
    handler = logging.Handler(level=logging.WARNING)
    handler.emit = records.append
    logger = logging.getLogger(logger_name)
    logger.addHandler(handler)
    try:
        return run(), [record.getMessage() for record in records]
    finally:
        logger.removeHandler(handler)


def _own_key_lookup_fails_for_bedrock(monkeypatch) -> None:
    monkeypatch.setattr(key_resolver, "_byok_key", lambda db, provider, ws:
                        OwnKeyLookupFailed(provider=provider) if provider in BEDROCK_NAMES else None)


# --------------------------------------------------------------------------- #
# the name rule
# --------------------------------------------------------------------------- #


def test_on_either_bedrock_name_runs_as_bedrock_and_no_other_name_changes(monkeypatch):
    _identity(monkeypatch, True)
    assert [registry.stored_provider_enum(n) for n in BEDROCK_NAMES] == [LLMProvider.AWS_BEDROCK] * 2
    assert registry.stored_provider_enum("Bedrock") is LLMProvider.AWS_BEDROCK
    for name in ("xai", "x-ai", "azure_openai", "cohere", "aiml", None):
        assert registry.stored_provider_enum(name) is None, name
    _identity(monkeypatch, False)
    assert registry.stored_provider_enum("bedrock") is None
    assert registry.stored_provider_enum("aws_bedrock") is LLMProvider.AWS_BEDROCK
    assert registry.workload_identity_region("bedrock") is None


# --------------------------------------------------------------------------- #
# on: every path calls Bedrock, in AWS_REGION, on the role
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", BEDROCK_NAMES)
def test_on_an_agent_is_called_on_bedrock_in_the_install_region_when_created_and_activated(monkeypatch, name):
    _identity(monkeypatch, True)
    created, _ = asyncio.run(_factory(monkeypatch)._create_llm_manager(ModelConfiguration(name, NOVA), "agent"))
    activated, resolved, provider = _activated(monkeypatch, name)

    assert provider == name and resolved.source == key_resolver.SOURCE_WORKLOAD_IDENTITY
    for manager in (created, activated):
        assert manager.config.provider is LLMProvider.AWS_BEDROCK and manager.config.workload_identity is True
        assert _signed_by_the_role_in(manager.config) == INSTALL_REGION


@pytest.mark.parametrize("name", BEDROCK_NAMES)
def test_on_the_spoken_model_is_called_on_bedrock_in_the_install_region(monkeypatch, name):
    _identity(monkeypatch, True)
    runtime = _spoken(monkeypatch, name)

    assert runtime.llm_manager.config.model == NOVA
    assert _signed_by_the_role_in(runtime.llm_manager.config) == INSTALL_REGION


@pytest.mark.parametrize("name", BEDROCK_NAMES)
def test_on_a_keyless_settings_tier_runs_on_bedrock_in_the_install_region(monkeypatch, name):
    _identity(monkeypatch, True)
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", OPENROUTER_KEY)  # a key another provider could use

    tier = _settings_tier(monkeypatch, name)

    assert tier.provider is LLMProvider.AWS_BEDROCK and tier.workload_identity is True
    assert _signed_by_the_role_in(tier) == INSTALL_REGION


def test_on_a_stored_credentials_region_gives_way_to_the_install_region(monkeypatch, install):
    _identity(monkeypatch, True)
    install.credentials["aws_bedrock"] = {"bedrock_api_key": BEDROCK_API_KEY, "aws_region": STORED_REGION}

    tier = _settings_tier(monkeypatch, "aws_bedrock")

    assert (tier.api_key, tier.base_url, tier.workload_identity) == (BEDROCK_API_KEY, INSTALL_REGION, False)
    assert _endpoint_region(_sent(tier)) == INSTALL_REGION


def test_on_an_inference_profile_arn_stays_on_bedrock_on_every_path(monkeypatch):
    _identity(monkeypatch, True)
    created, _ = asyncio.run(_factory(monkeypatch)._create_llm_manager(
        ModelConfiguration("bedrock", PROFILE_ARN), "agent"))
    activated, _, _ = _activated(monkeypatch, "bedrock", PROFILE_ARN)
    tier = _settings_tier(monkeypatch, "bedrock", PROFILE_ARN)
    handed_on = _loaded("bedrock", PROFILE_ARN)

    for llm_config in (created.config, activated.config, tier, handed_on):
        assert (llm_config.provider, llm_config.model) == (LLMProvider.AWS_BEDROCK, PROFILE_ARN)
        assert _signed_by_the_role_in(llm_config) == INSTALL_REGION


# --------------------------------------------------------------------------- #
# on: a Bedrock agent with no usable key is never moved to OpenRouter
# --------------------------------------------------------------------------- #


def test_on_creation_never_moves_a_bedrock_agent_to_openrouter(monkeypatch):
    _identity(monkeypatch, True)
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", OPENROUTER_KEY)
    _own_key_lookup_fails_for_bedrock(monkeypatch)

    with pytest.raises(ValueError, match="No API key available for bedrock"):
        asyncio.run(_factory(monkeypatch)._creation_key(ModelConfiguration("bedrock", NOVA), "agent", WORKSPACE))


@pytest.mark.parametrize("name", BEDROCK_NAMES)
def test_on_activation_never_moves_a_bedrock_agent_to_openrouter(monkeypatch, name):
    _identity(monkeypatch, True)
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", OPENROUTER_KEY)
    _own_key_lookup_fails_for_bedrock(monkeypatch)

    provider, model, resolved = asyncio.run(
        _factory(monkeypatch)._activation_key({"provider": name, "model": NOVA}, "agent", WORKSPACE))

    assert (provider, model, resolved) == (name, NOVA, None)


# --------------------------------------------------------------------------- #
# off: every path as on main
# --------------------------------------------------------------------------- #


def test_off_an_agent_stored_as_bedrock_is_activated_on_openrouter_as_on_main(monkeypatch):
    _identity(monkeypatch, False)
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", OPENROUTER_KEY)

    (manager, resolved, provider), warnings = _warnings_of(
        factory_mod.__name__, lambda: _activated(monkeypatch, "bedrock"))

    assert (provider, manager.config.provider, manager.config.model) == (
        "openrouter", LLMProvider.OPENROUTER, f"bedrock/{NOVA}")
    assert (resolved.api_key, manager.config.base_url) == (OPENROUTER_KEY, None)
    assert "Provider 'bedrock' is not a valid LLMProvider enum; forcing OpenRouter" in warnings


def test_off_keyless_bedrock_agents_move_to_openrouter_as_on_main(monkeypatch):
    _identity(monkeypatch, False)
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", OPENROUTER_KEY)
    factory = _factory(monkeypatch)

    enum, provider, model, resolved = asyncio.run(factory._creation_key(ModelConfiguration("bedrock", NOVA), "agent", None))
    assert (enum, provider, model, resolved.api_key) == (LLMProvider.OPENROUTER, "openrouter", NOVA, OPENROUTER_KEY)
    assert asyncio.run(factory._activation_key({"provider": "aws_bedrock", "model": NOVA}, "agent", None))[:2] == (
        "openrouter", NOVA)
    assert factory._resolve_provider_for_model("bedrock", PROFILE_ARN) == ("openrouter", PROFILE_ARN)


def test_off_activation_and_the_spoken_model_leave_the_region_to_the_client_as_on_main(monkeypatch):
    _identity(monkeypatch, False)
    monkeypatch.setattr(config, "AWS_ACCESS_KEY_ID", BEDROCK_API_KEY)  # an env key, so the spoken model moves

    activated, _, _ = _activated(monkeypatch, "aws_bedrock")
    spoken = _spoken(monkeypatch, "aws_bedrock").llm_manager.config

    for llm_config in (activated.config, spoken):
        assert (llm_config.provider, llm_config.base_url) == (LLMProvider.AWS_BEDROCK, None)
        assert _endpoint_region(_sent(llm_config)) == CLIENT_DEFAULT_REGION


def test_off_the_settings_tier_reads_names_and_regions_as_on_main(monkeypatch, install):
    _identity(monkeypatch, False)
    assert _settings_tier(monkeypatch, "aws_bedrock").base_url == CLIENT_DEFAULT_REGION
    install.credentials["aws_bedrock"] = {"bedrock_api_key": BEDROCK_API_KEY, "aws_region": STORED_REGION}
    assert _settings_tier(monkeypatch, "aws_bedrock").base_url == STORED_REGION
    with pytest.raises(ValueError, match="Unknown LLM provider: 'bedrock'"):
        _settings_tier(monkeypatch, "bedrock")
    assert _loaded("bedrock", PROFILE_ARN).provider is LLMProvider.OPENROUTER


@pytest.mark.parametrize("provider, credential, expected", [
    ("openai", {"api_key": "k", "base_url": "https://o", "organization_id": "org"}, ("k", "https://o", "org", None)),
    ("anthropic", {"api_key": "k", "base_url": "https://a"}, ("k", "https://a", None, None)),
    ("google", {"api_key": "k", "base_url": "ignored"}, ("k", None, None, None)),
    ("azure", {"api_key": "k", "endpoint_url": "https://r"}, ("k", "https://r", None, None)),
    ("huggingface", {"api_token": "t"}, ("t", None, None, None)),
    ("grok", {"api_token": "t"}, ("t", None, None, None)),
    ("openrouter", {"api_key": "k", "base_url": ""}, ("k", None, None, None)),
    ("aws_bedrock", {"aws_access_key_id": "id", "aws_secret_access_key": "s", "aws_region": STORED_REGION},
     ("id", STORED_REGION, None, "s")),
])
def test_the_settings_loader_reads_each_providers_credential_as_on_main(monkeypatch, install, provider, credential,
                                                                         expected):
    _identity(monkeypatch, False)
    install.credentials[provider] = credential
    loaded = _loaded(provider, "m")
    assert (loaded.api_key, loaded.base_url, loaded.organization_id, loaded.secret_key) == expected


def test_the_settings_loader_takes_the_operator_key_over_the_stored_one_and_needs_a_huggingface_token(monkeypatch):
    _identity(monkeypatch, False)
    monkeypatch.setitem(sys.modules, "core.llm.workspace_keys",
                        types.SimpleNamespace(get_platform_workspace_key=lambda provider: "operator-key"))
    loaded = _loaded("openai", "m")
    assert (loaded.api_key, loaded.base_url, loaded.organization_id) == ("operator-key", None, None)
    monkeypatch.setitem(sys.modules, "core.llm.workspace_keys",
                        types.SimpleNamespace(get_platform_workspace_key=lambda provider: None))
    with pytest.raises(ValueError, match="HuggingFace credential not found"):
        _loaded("huggingface", "m")
