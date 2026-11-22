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
# AWS's two documented example access key IDs: the chain's (the install's role) and a key of
# its own. Every other secret-shaped value is put together at run time.
ROLE_KEY_ID, OWN_KEY_ID = "AKIAI44QH8DHBEXAMPLE", "AKIAIOSFODNN7EXAMPLE"
ROLE_SECRET, OWN_SECRET = "example-" + "role-secret", "example-" + "own-secret"
BEDROCK_API_KEY = "ABSK" + "-example-long-term"
WORKSPACE_API_KEY = "ABSK" + "-example-workspace"
OPENROUTER_KEY = "or-" + "example"
WORKSPACE = "6a0f3c1e-2b4d-4e8f-9a1b-5b0e40000001"
BARE_OTHER_VENDOR_IDS = ("claude-opus-5-5", "gemini-2.5-flash")
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


def _sent(llm_config: LLMConfig, status: int = 200, body: dict = None, error_type: str = None):
    """The one Converse request a client built from this config sends (provider checked)."""
    provider = llm_manager.LLMManager._create_provider(llm_config)
    assert isinstance(provider, BedrockProvider), "called on Bedrock"
    seen = []

    def send(request, **_kwargs):
        seen.append(request)
        headers = {"content-type": "application/json", **({"x-amzn-ErrorType": error_type} if error_type else {})}
        return AWSResponse(request.url, status, headers, _Raw(json.dumps(body or REPLY).encode()))

    provider.client.meta.events.register("before-send.bedrock-runtime.Converse", send)
    provider.generate_response_sync([{"role": "user", "content": "hi"}])
    return seen[0]


def _endpoint_region(request) -> str:
    return urlsplit(request.url).hostname.split(".")[1]


def _signed_with(llm_config: LLMConfig) -> tuple:
    """(the access key ID or Bedrock API key the request carries, the region it goes to)."""
    request = _sent(llm_config)
    authorization = request.headers["Authorization"].decode()
    if authorization.startswith("Bearer "):
        return authorization[len("Bearer "):], _endpoint_region(request)
    key_id, _date, scope_region, service, _end = authorization.split("Credential=")[1].split(",")[0].split("/")
    assert service == "bedrock" and scope_region == _endpoint_region(request)
    return key_id, scope_region


def _signed_by_the_role_in(llm_config: LLMConfig) -> str:
    """The region a role-signed request goes to; fails unless the role signed it."""
    key_id, region = _signed_with(llm_config)
    assert key_id == ROLE_KEY_ID, "signed by the install's role"
    return region


def _factory(monkeypatch) -> AgentFactory:
    """A factory whose managers are captured (their config is what a client is built from)."""
    factory = AgentFactory(db_session=MagicMock())
    monkeypatch.setattr(factory, "_platform_unfunded", lambda _ws: None)
    monkeypatch.setattr(factory, "_model_ceiling", lambda _model: None)
    monkeypatch.setattr(factory_mod, "LLMManager", lambda config, **_kw: SimpleNamespace(config=config))
    return factory


def _activated(monkeypatch, name: str, model: str = NOVA, workspace=None):
    return asyncio.run(_factory(monkeypatch)._activation_llm_manager(
        {"provider": name, "model": model}, SimpleNamespace(name="agent"), 1, workspace))


def _spoken(monkeypatch, name: str, workspace=None):
    """An agent on another model, moved to a Bedrock spoken model for one turn."""
    monkeypatch.setattr(config, "SPOKEN_MODE_MODEL", NOVA)
    monkeypatch.setattr(config, "SPOKEN_MODE_PROVIDER", name)
    monkeypatch.setattr("core.llm.model_policy.check_model_for_agent", lambda *a, **k: (True, "allowed"))
    agent = SimpleNamespace(config=LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-opus-5-5", api_key="k"))
    runtime = AgentRuntime(agent_id=1, metadata=AgentMetadata(name="Auto", agent_type="orchestrator", persona="Auto"),
                           llm_manager=agent, lifecycle_state=AgentLifecycle.ACTIVE, created_at=datetime.now())
    return asyncio.run(spoken_model.spoken_turn_runtime(_factory(monkeypatch), runtime, workspace))


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


def _own_key_saved_under(monkeypatch, saved_as: str) -> list:
    """The workspace's own Bedrock API key, saved under ``saved_as`` (the database matches the
    name exactly). Returns the names it was looked up under."""
    asked = []

    def _byok_key(db, provider, ws):
        asked.append(provider)
        if provider != saved_as:
            return None
        return key_resolver.ResolvedKey(api_key=WORKSPACE_API_KEY, source="byok", is_byok=True, provider=provider)

    monkeypatch.setattr(key_resolver, "_byok_key", _byok_key)
    return asked


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


@pytest.mark.parametrize("credential, signed_with", [
    ({"aws_region": STORED_REGION}, (ROLE_KEY_ID, INSTALL_REGION)),
    ({"aws_access_key_id": OWN_KEY_ID, "aws_secret_access_key": OWN_SECRET, "aws_region": STORED_REGION},
     (OWN_KEY_ID, STORED_REGION)),
    ({"bedrock_api_key": BEDROCK_API_KEY, "aws_region": STORED_REGION}, (BEDROCK_API_KEY, STORED_REGION)),
    ({"aws_access_key_id": OWN_KEY_ID, "aws_secret_access_key": OWN_SECRET}, (OWN_KEY_ID, INSTALL_REGION)),
    ({"bedrock_api_key": BEDROCK_API_KEY}, (BEDROCK_API_KEY, INSTALL_REGION)),
], ids=["the-role-in-the-install-region", "a-key-pair-in-its-own-region", "an-api-key-in-its-own-region",
        "a-key-pair-naming-no-region", "an-api-key-naming-no-region"])
def test_on_the_role_uses_the_install_region_and_a_key_keeps_the_region_it_names(monkeypatch, install,
                                                                                 credential, signed_with):
    _identity(monkeypatch, True)
    install.credentials["aws_bedrock"] = credential

    tier = _settings_tier(monkeypatch, "aws_bedrock")

    assert tier.workload_identity is (signed_with[0] == ROLE_KEY_ID)
    assert _signed_with(tier) == signed_with


def test_on_a_workspaces_own_key_saved_as_bedrock_pays_for_an_agent_stored_as_aws_bedrock(monkeypatch):
    _identity(monkeypatch, True)
    asked = _own_key_saved_under(monkeypatch, "bedrock")

    created, _ = asyncio.run(_factory(monkeypatch)._create_llm_manager(
        ModelConfiguration("aws_bedrock", NOVA), "agent", WORKSPACE))
    activated, resolved, _ = _activated(monkeypatch, "aws_bedrock", workspace=WORKSPACE)
    spoken = _spoken(monkeypatch, "aws_bedrock", workspace=WORKSPACE)

    assert resolved.is_byok and spoken.is_byok
    assert set(asked) == {"bedrock"}, "found under the slug, first"
    for llm_config in (created.config, activated.config, spoken.llm_manager.config):
        assert llm_config.workload_identity is False, "the workspace's key, not the install's role"
        assert _signed_with(llm_config) == (WORKSPACE_API_KEY, INSTALL_REGION)


def test_off_a_workspaces_own_key_is_looked_up_under_the_stored_name_only_as_on_main(monkeypatch):
    _identity(monkeypatch, False)
    asked = _own_key_saved_under(monkeypatch, "bedrock")

    assert key_resolver.resolve_provider_key(MagicMock(), "aws_bedrock", workspace_id=WORKSPACE) is None
    assert asked == ["aws_bedrock"]


def test_on_an_inference_profile_arn_stays_on_bedrock_on_every_path(monkeypatch):
    _identity(monkeypatch, True)
    created, _ = asyncio.run(_factory(monkeypatch)._create_llm_manager(
        ModelConfiguration("bedrock", PROFILE_ARN), "agent"))
    activated, _, _ = _activated(monkeypatch, "bedrock", PROFILE_ARN)
    tier = _settings_tier(monkeypatch, "bedrock", PROFILE_ARN)
    # A provider and model handed to a manager (create_llm_manager → vendor_fit.fitted): the
    # manager builds the tier's config, and with_workload_identity marks it for the role.
    handed_on = llm_manager.LLMManager(service_name="planner", provider="bedrock", model=PROFILE_ARN).config

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
def test_on_activation_with_no_usable_key_fails_clearly_and_never_moves_to_openrouter(monkeypatch, name):
    _identity(monkeypatch, True)
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", OPENROUTER_KEY)
    _own_key_lookup_fails_for_bedrock(monkeypatch)
    factory = _factory(monkeypatch)
    real_resolve, asked = factory._resolve_api_key, []

    async def _resolve(provider, *args, **kwargs):
        asked.append(provider)
        return await real_resolve(provider, *args, **kwargs)

    monkeypatch.setattr(factory, "_resolve_api_key", _resolve)

    with pytest.raises(ValueError, match=f"No API key available for {name}"):
        asyncio.run(factory._activation_key({"provider": name, "model": NOVA}, "agent", WORKSPACE))
    assert asked == [name], "no other provider is tried"


@pytest.mark.parametrize("model", BARE_OTHER_VENDOR_IDS)
def test_on_a_bare_id_naming_another_vendor_stays_on_bedrock_and_fails_in_bedrocks_words(monkeypatch, model):
    _identity(monkeypatch, True)
    factory = _factory(monkeypatch)
    assert factory._resolve_provider_for_model("bedrock", model) == ("bedrock", model)

    activated, _, provider = _activated(monkeypatch, "aws_bedrock", model)

    assert (provider, activated.config.provider) == ("aws_bedrock", LLMProvider.AWS_BEDROCK)
    with pytest.raises(ValueError, match=f"AWS Bedrock won't serve {model}"):
        _sent(activated.config, status=400, error_type="ValidationException",
              body={"message": "The provided model identifier is invalid."})


@pytest.mark.parametrize("model, vendor", list(zip(BARE_OTHER_VENDOR_IDS, ("anthropic", "google"))))
def test_off_a_bare_id_naming_another_vendor_moves_to_that_vendor_as_on_main(monkeypatch, model, vendor):
    _identity(monkeypatch, False)
    assert _factory(monkeypatch)._resolve_provider_for_model("bedrock", model) == (vendor, model)


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
