"""PRD-278 US-103 — with workload identity on, Bedrock uses the install's role only when told to.

The default AWS credential chain is the install's own role once ``AWS_USE_WORKLOAD_IDENTITY``
is on. A Bedrock client may use it only when the key resolver answered with that role:
``ResolvedKey.source == "workload_identity"`` becomes ``LLMConfig.workload_identity``, which
``bedrock_client.runtime_client`` reads as ``use_role``. It is never inferred from an empty
secret.

  (a) the install's own Bedrock use (an agent's platform route, a settings tier) signs with
      the role, and the same config without the signal is refused;
  (b) a key ID with its own secret signs with those two parts, never the chain;
  (c) a key ID with no secret (a workspace's own key, a stored credential) is refused, and
      the server's AWS secret is never put beside a workspace's key;
  (d) with the setting off, every shape behaves as on main.

botocore's real request pipeline runs and stops just before sending (as in
``test_1170_bedrock_converse.py``). The SDK's chain is given a key of its own through the
environment, standing in for the install's role, so a request that fell through to the chain
is seen signed with it.
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
import types
from types import SimpleNamespace
from unittest.mock import MagicMock

import boto3
import pytest
from botocore.awsrequest import AWSResponse

import core.llm.clients.bedrock_client as bedrock_mod
import core.llm.manager as llm_manager
import modules.agents.factory.agent_factory as factory_mod
from config import config
from core.llm.clients.base import LLMConfig, LLMProvider
from core.llm.clients.bedrock_client import BedrockProvider
from core.llm.key_resolver import ResolvedKey, with_workload_identity
from modules.agents.factory.agent_factory import AgentFactory, ModelConfiguration

NOVA = "amazon.nova-pro-v1:0"
# AWS's documented example keys: the chain's (the install's role) and a workspace's own.
ROLE_KEY_ID, ROLE_SECRET = "AKIAI44QH8DHBEXAMPLE", "je7MtGbClwBF/2Zp9Utk/h3yCo8nvbEXAMPLEKEY"
OWN_KEY_ID, OWN_SECRET = "AKIAIOSFODNN7EXAMPLE", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
SERVER_SECRET = "server-secret-example"
REPLY = {"output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}}, "stopReason": "end_turn",
         "usage": {"inputTokens": 1, "outputTokens": 1, "totalTokens": 2}}


class _Raw:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def stream(self, **_kwargs):
        yield self._body


@pytest.fixture(autouse=True)
def chain_is_the_install_role(monkeypatch, tmp_path):
    """The SDK's chain answers with the role's key; no profile, no store, no platform key."""
    for name in ("AWS_SESSION_TOKEN", "AWS_PROFILE", "AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION", "AWS_DEFAULT_REGION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", ROLE_KEY_ID)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", ROLE_SECRET)
    monkeypatch.setattr(boto3, "DEFAULT_SESSION", None)  # no credentials cached by another test
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


def _signed_by(llm_config: LLMConfig) -> str:
    """The access key ID one Converse call of this config is signed with."""
    provider = BedrockProvider(llm_config)
    seen = []

    def send(request, **_kwargs):
        seen.append(request)
        return AWSResponse(request.url, 200, {"content-type": "application/json"},
                           _Raw(json.dumps(REPLY).encode()))

    provider.client.meta.events.register("before-send.bedrock-runtime.Converse", send)
    provider.generate_response_sync([{"role": "user", "content": "hi"}])
    authorization = seen[0].headers["Authorization"].decode()
    return authorization.split("Credential=")[1].split("/")[0]


def _bedrock_config(**fields) -> LLMConfig:
    return LLMConfig(provider=LLMProvider.AWS_BEDROCK, model=NOVA, base_url="eu-west-1", **fields)


def _factory(monkeypatch, resolved=None) -> tuple:
    """A factory whose managers are captured; ``resolved`` replaces the key lookup when given."""
    factory = AgentFactory(db_session=MagicMock())
    monkeypatch.setattr(factory, "_platform_unfunded", lambda _ws: None)
    if resolved is not None:
        async def _resolve(*_a, **_k):
            return resolved

        monkeypatch.setattr(factory, "_resolve_api_key", _resolve)
    monkeypatch.setattr(factory_mod, "LLMManager", lambda config, **_kw: SimpleNamespace(config=config))
    return factory


def _refused_without_a_client(monkeypatch, llm_config: LLMConfig, match: str) -> None:
    built = []
    monkeypatch.setattr(bedrock_mod.boto3, "client", lambda *a, **kw: built.append(kw) or MagicMock())
    with pytest.raises(ValueError, match=match):
        llm_manager.LLMManager._create_provider(llm_config)
    assert built == [], "no client: the request never reaches the default chain"


def _workspace_key() -> ResolvedKey:
    return ResolvedKey(api_key=OWN_KEY_ID, source="byok", is_byok=True, provider="bedrock")


# --------------------------------------------------------------------------- #
# (a) the install's own Bedrock use runs on the role, because the resolver said so
# --------------------------------------------------------------------------- #


def test_on_an_agents_platform_route_signs_with_the_role_and_only_because_of_the_signal(monkeypatch):
    _identity(monkeypatch, True)
    factory = _factory(monkeypatch)
    manager, resolved = asyncio.run(factory._create_llm_manager(ModelConfiguration("bedrock", NOVA), "agent"))

    assert resolved.source == "workload_identity" and manager.config.workload_identity is True
    assert _signed_by(manager.config) == ROLE_KEY_ID
    _refused_without_a_client(monkeypatch, dataclasses.replace(manager.config, workload_identity=False),
                              "not configured")


def test_on_the_activation_path_carries_the_signal(monkeypatch):
    _identity(monkeypatch, True)
    factory = _factory(monkeypatch)
    manager, resolved, provider = asyncio.run(factory._activation_llm_manager(
        {"provider": "aws_bedrock", "model": NOVA}, SimpleNamespace(name="agent"), 1, None))

    assert (provider, resolved.source, manager.config.workload_identity) == ("aws_bedrock", "workload_identity", True)
    assert _signed_by(dataclasses.replace(manager.config, base_url="eu-west-1")) == ROLE_KEY_ID


def test_on_a_keyless_settings_tier_runs_on_the_role(monkeypatch):
    _identity(monkeypatch, True)
    monkeypatch.setattr(llm_manager.LLMManager, "_load_config_from_settings",
                        lambda self, *a, **k: _bedrock_config())
    monkeypatch.setattr(llm_manager.output_budget, "manager_budgets", lambda *a, **k: (None, None))

    manager = llm_manager.LLMManager(service_name="planner")

    assert manager.config.workload_identity is True
    assert _signed_by(manager.config) == ROLE_KEY_ID
    _refused_without_a_client(monkeypatch, _bedrock_config(), "not configured")


def test_the_settings_signal_needs_the_setting_bedrock_and_no_key(monkeypatch):
    _identity(monkeypatch, False)
    assert with_workload_identity(_bedrock_config()).workload_identity is False
    _identity(monkeypatch, True)
    assert with_workload_identity(_bedrock_config(api_key=OWN_KEY_ID)).workload_identity is False
    other = LLMConfig(provider=LLMProvider.ANTHROPIC, model="claude-haiku-5-5")
    assert with_workload_identity(other).workload_identity is False


# --------------------------------------------------------------------------- #
# (b) a key with both parts signs with those parts
# --------------------------------------------------------------------------- #


def test_on_a_key_with_its_secret_signs_with_that_key_not_the_role(monkeypatch):
    _identity(monkeypatch, True)
    assert _signed_by(_bedrock_config(api_key=OWN_KEY_ID, secret_key=OWN_SECRET)) == OWN_KEY_ID
    assert _signed_by(_bedrock_config(api_key=OWN_KEY_ID, secret_key=OWN_SECRET, workload_identity=True)) == OWN_KEY_ID


# --------------------------------------------------------------------------- #
# (c) a key ID with no secret is refused, never served by the role
# --------------------------------------------------------------------------- #


def test_on_a_workspaces_key_id_alone_is_refused_on_the_creation_path(monkeypatch):
    _identity(monkeypatch, True)
    monkeypatch.setattr(config, "AWS_SECRET_ACCESS_KEY", SERVER_SECRET)  # even if one were present
    factory = _factory(monkeypatch, resolved=_workspace_key())

    manager, _ = asyncio.run(factory._create_llm_manager(ModelConfiguration("bedrock", NOVA), "agent", "ws-1"))

    assert manager.config.secret_key is None, "the server's secret never sits beside a workspace's key"
    assert manager.config.workload_identity is False
    _refused_without_a_client(monkeypatch, manager.config, "without its secret")


def test_on_a_workspaces_key_id_alone_is_refused_on_the_activation_path(monkeypatch):
    _identity(monkeypatch, True)
    factory = _factory(monkeypatch, resolved=_workspace_key())

    manager, resolved, _ = asyncio.run(factory._activation_llm_manager(
        {"provider": "aws_bedrock", "model": NOVA}, SimpleNamespace(name="agent"), 1, "ws-1"))

    assert resolved.is_byok and manager.config.workload_identity is False
    _refused_without_a_client(monkeypatch, manager.config, "without its secret")


def test_on_the_signal_never_launders_a_key_id_without_its_secret(monkeypatch):
    _identity(monkeypatch, True)
    _refused_without_a_client(monkeypatch, _bedrock_config(api_key=OWN_KEY_ID, workload_identity=True),
                              "without its secret")


def test_on_no_key_and_no_signal_is_refused(monkeypatch):
    _identity(monkeypatch, True)
    _refused_without_a_client(monkeypatch, _bedrock_config(), "not configured")


# --------------------------------------------------------------------------- #
# (d) off: every shape as on main
# --------------------------------------------------------------------------- #


def test_off_every_client_shape_is_as_on_main(monkeypatch):
    _identity(monkeypatch, False)
    assert _signed_by(_bedrock_config(api_key=OWN_KEY_ID, secret_key=OWN_SECRET)) == OWN_KEY_ID
    assert _signed_by(_bedrock_config(api_key=OWN_KEY_ID)) == ROLE_KEY_ID
    assert _signed_by(_bedrock_config()) == ROLE_KEY_ID


def test_off_the_creation_path_builds_bedrock_configs_as_on_main(monkeypatch):
    _identity(monkeypatch, False)
    monkeypatch.setattr(config, "AWS_SECRET_ACCESS_KEY", SERVER_SECRET)
    factory = _factory(monkeypatch, resolved=_workspace_key())

    manager, _ = asyncio.run(factory._create_llm_manager(ModelConfiguration("bedrock", NOVA), "agent", "ws-1"))

    assert (manager.config.api_key, manager.config.secret_key) == (OWN_KEY_ID, SERVER_SECRET)
    assert (manager.config.base_url, manager.config.workload_identity) == ("eu-west-1", False)
