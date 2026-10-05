"""#873: a workspace's Azure key is saved with its own endpoint and reaches it.

The Add API Key dialog took only the key, and ``user_api_keys`` had no URL column,
so a workspace's Azure key could only call the API service's
``AZURE_OPENAI_ENDPOINT``: the operator's resource on a self-hosted install, and
never the customer's on the hosted edition. The endpoint is now saved with the key
(``core.llm.byok_endpoint``). On saas it is checked against private and metadata
addresses when saved, and every call to it is pinned to the address that was checked,
with redirects off (``core.security.pinned_http``). A BYOK ``LLMManager`` hands it to
the client, so the mission and chat paths both get it.

Pure tests: DNS, the OpenAI SDK, the database and encryption are all stubbed.
"""
from __future__ import annotations

import asyncio
import socket
import sys
import types
from datetime import datetime
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

import api.user_api_keys as uak
import core.llm.byok_endpoint as be
import core.llm.key_resolver as kr
import core.security.web_access as web_access
from api.user_api_keys import ApiKeyCreate, ApiKeyValidation, _validate_provider_key

RESOURCE = "https://contoso.openai.azure.com"
V1 = "https://contoso.openai.azure.com/openai/v1/"
PUBLIC_IP = "20.50.2.10"
PRIVATE_IP = "10.1.2.3"


@pytest.fixture
def resolves_to(monkeypatch):
    """Point every DNS lookup the outbound check makes at one address."""
    def _set(ip):
        def _getaddrinfo(host, port, *args, **kwargs):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port))]
        monkeypatch.setattr(web_access, "_getaddrinfo", _getaddrinfo)
    return _set


@pytest.fixture
def saas(monkeypatch):
    monkeypatch.setattr(be.config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(be.config, "AZURE_OPENAI_ENDPOINT", None)


@pytest.fixture
def local(monkeypatch):
    monkeypatch.setattr(be.config, "AUTH_EDITION", "local")


# ── what is saved ──────────────────────────────────────────────────────────


def test_the_endpoint_is_saved_as_its_v1_base_url(saas, resolves_to):
    resolves_to(PUBLIC_IP)
    assert be.clean_endpoint("azure", f"  {RESOURCE}/  ") == V1
    assert be.clean_endpoint("azure", f"{RESOURCE}/openai/deployments/gpt5?api-version=x") == V1


def test_a_blank_endpoint_is_none():
    assert be.clean_endpoint("azure", "   ") is None
    assert be.clean_endpoint("azure", None) is None


def test_a_provider_with_a_fixed_address_takes_no_endpoint():
    assert be.takes_endpoint("azure") and be.takes_endpoint("azure_openai")
    assert not be.takes_endpoint("openai")
    with pytest.raises(be.EndpointRefused):
        be.clean_endpoint("openai", RESOURCE)


# ── what the hosted edition refuses ────────────────────────────────────────


def test_saas_refuses_an_endpoint_that_resolves_to_a_private_address(saas, resolves_to):
    resolves_to(PRIVATE_IP)
    with pytest.raises(be.EndpointRefused, match="can't be used"):
        be.clean_endpoint("azure", RESOURCE)


def test_saas_refuses_the_cloud_metadata_address(saas):
    with pytest.raises(be.EndpointRefused):
        be.clean_endpoint("azure", "https://169.254.169.254")


def test_saas_refuses_plain_http(saas, resolves_to):
    resolves_to(PUBLIC_IP)
    with pytest.raises(be.EndpointRefused, match="https"):
        be.clean_endpoint("azure", "http://contoso.openai.azure.com")


def test_self_hosted_keeps_a_private_endpoint(local, resolves_to):
    # A bank's Azure private endpoint resolves to a 10.x address on purpose.
    resolves_to(PRIVATE_IP)
    assert be.clean_endpoint("azure", RESOURCE) == V1


def test_saas_needs_the_endpoint_and_self_hosted_can_use_the_env_one(monkeypatch):
    monkeypatch.setattr(be.config, "AZURE_OPENAI_ENDPOINT", RESOURCE)
    assert be.endpoint_required("azure", "saas") is True
    assert be.endpoint_required("azure", "local") is False
    monkeypatch.setattr(be.config, "AZURE_OPENAI_ENDPOINT", None)
    assert be.endpoint_required("azure", "local") is True
    assert be.endpoint_required("openai", "saas") is False


def test_the_save_answers_400_with_the_reason(saas, resolves_to):
    resolves_to(PRIVATE_IP)
    with pytest.raises(HTTPException) as refused:
        uak._key_endpoint("azure", RESOURCE)
    assert refused.value.status_code == 400 and "can't be used" in refused.value.detail
    with pytest.raises(HTTPException) as missing:
        uak._key_endpoint("azure", "")
    assert missing.value.status_code == 400 and "endpoint" in missing.value.detail.lower()


# ── the live key check ─────────────────────────────────────────────────────


def _fake_openai(monkeypatch, *, status=None):
    calls = []

    class _Refused(Exception):
        def __init__(self, code):
            super().__init__(f"Error code: {code}")
            self.status_code = code

    class _Models:
        def list(self):
            if status:
                raise _Refused(status)
            return SimpleNamespace(data=[])

    class OpenAI:  # noqa: N801 - mirrors the SDK's name
        def __init__(self, **kwargs):
            calls.append(kwargs)
            self.models = _Models()

    mod = types.ModuleType("openai")
    mod.OpenAI = OpenAI
    monkeypatch.setitem(sys.modules, "openai", mod)
    return calls


def test_the_key_is_checked_against_its_own_endpoint(monkeypatch):
    calls = _fake_openai(monkeypatch)
    result = asyncio.run(_validate_provider_key("azure", "azure-key-0001", V1))
    assert result.valid is True and result.message == "API key is valid"
    assert calls[0]["base_url"] == V1 and calls[0]["api_key"] == "azure-key-0001"
    assert calls[0]["max_retries"] == 0 and calls[0]["timeout"] == uak.AZURE_CHECK_TIMEOUT_SECONDS


def test_a_refused_key_is_saved_inactive(monkeypatch):
    _fake_openai(monkeypatch, status=401)
    result = asyncio.run(_validate_provider_key("azure", "azure-key-bad", V1))
    assert result.valid is False and "401" in result.message


def test_an_endpoint_without_a_models_list_says_the_key_was_not_checked(monkeypatch):
    _fake_openai(monkeypatch, status=404)
    result = asyncio.run(_validate_provider_key("azure", "azure-key-0001", V1))
    assert result.valid is True and "not checked" in result.message


def test_no_endpoint_anywhere_never_claims_a_check(monkeypatch):
    monkeypatch.setattr(uak.config, "AZURE_OPENAI_ENDPOINT", None)
    result = asyncio.run(_validate_provider_key("azure", "azure-key-0001"))
    assert result.valid is True and "no endpoint" in result.message


# ── the save stores it ─────────────────────────────────────────────────────


class _KeysDB:
    def __init__(self):
        self.added = []

    def query(self, _model):
        return SimpleNamespace(get=lambda _id: None)

    def add(self, row):
        row.id, row.created_at = 1, datetime.utcnow()
        self.added.append(row)

    def commit(self):
        pass

    def refresh(self, _row):
        pass


def test_add_api_key_stores_the_endpoint_and_checks_the_key_there(monkeypatch, local, resolves_to):
    resolves_to(PUBLIC_IP)
    seen = {}

    async def _check(provider, key, base_url=None):
        seen.update(provider=provider, base_url=base_url)
        return ApiKeyValidation(valid=False, message="Invalid key: 401", tested_at=datetime.utcnow())

    monkeypatch.setattr(uak, "_validate_provider_key", _check)
    monkeypatch.setattr(uak, "get_encryption_service", lambda: SimpleNamespace(
        encrypt=lambda s: f"enc::{s}", decrypt=lambda s: s[5:]))
    db = _KeysDB()
    body = ApiKeyCreate(provider="azure", api_key="azure-key-0001", base_url=RESOURCE)

    asyncio.run(uak.add_api_key(body, ctx=SimpleNamespace(workspace_id="ws-1"), db=db))

    assert db.added[0].base_url == V1
    assert seen == {"provider": "azure", "base_url": V1}


# ── every call to it is pinned on saas ─────────────────────────────────────


def test_saas_calls_a_key_endpoint_through_the_pinned_client(saas):
    from core.security.pinned_http import PinnedTransport

    client = be.endpoint_http_client(15.0)
    assert client.follow_redirects is False
    assert isinstance(client._transport, PinnedTransport)


def test_self_hosted_keeps_the_sdk_client(local):
    assert be.endpoint_http_client(15.0) is None


def test_a_request_goes_to_the_checked_address_with_the_name_as_host_and_sni(monkeypatch, resolves_to):
    import httpx
    from core.security.pinned_http import PinnedTransport

    resolves_to(PUBLIC_IP)
    sent = []
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda self, req: sent.append(req) or httpx.Response(200))

    PinnedTransport().handle_request(httpx.Request("GET", f"{V1}models"))

    assert sent[0].url.host == PUBLIC_IP and sent[0].url.path == "/openai/v1/models"
    assert sent[0].headers["Host"] == "contoso.openai.azure.com"
    assert sent[0].extensions["sni_hostname"] == "contoso.openai.azure.com"


def test_a_request_whose_name_now_resolves_privately_never_leaves(monkeypatch, resolves_to):
    # DNS can change after the save; the transport checks again on every request.
    import httpx
    from core.security.pinned_http import PinnedTransport

    resolves_to(PRIVATE_IP)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda *a: pytest.fail("connected"))
    with pytest.raises(httpx.ConnectError, match="blocked range"):
        PinnedTransport().handle_request(httpx.Request("GET", f"{V1}models"))


def test_the_saas_key_check_uses_the_pinned_client(monkeypatch, saas):
    calls = _fake_openai(monkeypatch)
    asyncio.run(_validate_provider_key("azure", "azure-key-0001", V1))
    assert calls[0]["http_client"].follow_redirects is False


def test_the_saas_azure_client_uses_the_pinned_client(monkeypatch, saas):
    from core.llm.clients import azure_client
    from core.llm.clients.base import LLMConfig, LLMProvider

    built = {}
    monkeypatch.setattr(azure_client, "OpenAI", lambda **kw: built.update(kw) or SimpleNamespace())
    azure_client.AzureProvider(LLMConfig(provider=LLMProvider.AZURE, model="prod-chat", api_key="k", base_url=V1))
    assert built["base_url"] == V1 and built["http_client"].follow_redirects is False


# ── the key takes it to the client ─────────────────────────────────────────


def _byok_db(row):
    workspace = SimpleNamespace(settings={"byok_overrides": {"azure": True}})

    class _Query:
        def __init__(self, model):
            self._model = model

        def get(self, _id):
            return workspace

        def filter(self, *_a):
            return self

        def order_by(self, *_a):
            return self

        def first(self):
            return row

    return SimpleNamespace(query=_Query)


@pytest.fixture
def plain_encryption(monkeypatch):
    import core.credentials.encryption as enc

    monkeypatch.setattr(enc, "get_encryption_service", lambda: SimpleNamespace(decrypt=lambda s: s))


def test_the_resolved_byok_key_carries_its_endpoint(plain_encryption):
    row = SimpleNamespace(encrypted_key="azure-key-0001", base_url=V1)
    resolved = kr._byok_key(_byok_db(row), "azure", "ws-1")
    assert (resolved.api_key, resolved.base_url, resolved.is_byok) == ("azure-key-0001", V1, True)


def test_a_key_without_an_endpoint_resolves_as_before(plain_encryption):
    row = SimpleNamespace(encrypted_key="sk-0001", base_url=None)
    assert kr._byok_key(_byok_db(row), "azure", "ws-1").base_url is None


def _manager(provider, *, is_byok, base_url=None):
    from core.llm.clients.base import LLMConfig
    from core.llm.manager import LLMManager

    config = LLMConfig(provider=provider, model="prod-chat", api_key="k", base_url=base_url)
    return LLMManager(config=config, workspace_id="ws-1", is_byok=is_byok)


def test_a_byok_azure_manager_gets_the_endpoint_saved_with_the_key(monkeypatch):
    # The mission and the chat paths both build the config from the key alone.
    from core.llm.clients.base import LLMProvider

    asked = []
    monkeypatch.setattr(kr, "byok_endpoint", lambda provider, ws: asked.append((provider, ws)) or V1)
    assert _manager(LLMProvider.AZURE, is_byok=True).config.base_url == V1
    assert asked == [("azure", "ws-1")]


def test_only_a_byok_azure_manager_without_an_endpoint_asks(monkeypatch):
    from core.llm.clients.base import LLMProvider

    monkeypatch.setattr(kr, "byok_endpoint", lambda *a: pytest.fail("asked"))
    assert _manager(LLMProvider.AZURE, is_byok=False).config.base_url is None
    assert _manager(LLMProvider.AZURE, is_byok=True, base_url=RESOURCE).config.base_url == RESOURCE
    assert _manager(LLMProvider.OPENAI, is_byok=True).config.base_url is None


# ── a deployment named after its model stays on Azure ──────────────────────


@pytest.mark.parametrize("deployment", ["gpt-4o", "o3-mini", "claude-sonnet-5", "prod-chat"])
def test_an_azure_deployment_named_after_its_model_stays_on_azure(deployment):
    import logging
    import modules.agents.factory.agent_factory as af

    factory = af.AgentFactory.__new__(af.AgentFactory)
    factory.logger = logging.getLogger("test.873")
    assert factory._resolve_provider_for_model("azure", deployment) == ("azure", deployment)
    # Other providers keep the mismatch rule.
    assert factory._resolve_provider_for_model("anthropic", "gpt-4o") == ("openai", "gpt-4o")
