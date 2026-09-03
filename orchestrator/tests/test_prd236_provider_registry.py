"""PRD-236 W0 — one provider registry; NVIDIA/DeepSeek through the generic adapter.

Pure tests: no Postgres, no live network. The OpenAI SDK is stubbed into
``sys.modules`` so the adapter and the BYOK validator exercise their real
branches; the factory's routing helpers run on a bare instance; the usage
tracker runs against a fake session. Real-Postgres coverage is CI's job.

What is pinned here (PRD-236 §Design):
- the registry and the ``LLMProvider`` enum agree in both directions (S0.1);
- the adapter builds the NVIDIA client against the registry base URL with a
  Bearer key and NO attribution headers, keeps OpenRouter's headers, and turns
  a 429 into ``ProviderRateLimitError`` carrying the spec's note (S0.2);
- the BYOK list, the workspace allowlist and the marketplace probe list are
  the registry, and ``_validate_provider_key`` live-tests NVIDIA (S0.3);
- routing keeps a vendor-prefixed id on a provider that hosts them, the
  legacy mismatch rules are unchanged, and a byok_only provider never
  resolves from the platform tiers in saas (S0.4);
- a call served by a free route books zero cost (S0.5);
- the providers endpoint exposes no secret and reports the edition (S0.6).
"""
from __future__ import annotations

import asyncio
import sys
import types
from types import SimpleNamespace

import pytest

from config import config
from core.llm import providers as reg
from core.llm.clients.base import LLMConfig, LLMProvider
from core.llm.clients import openai_compatible_client as oc
from core.llm.clients.openai_compatible_client import (
    OpenAICompatibleProvider,
    ProviderRateLimitError,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #


class _FakeModels:
    def __init__(self, log):
        self._log = log

    def list(self):
        self._log.append("models.list")
        return SimpleNamespace(data=[])


class _FakeCompletions:
    def __init__(self, raise_exc=None):
        self._raise = raise_exc
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._raise:
            raise self._raise
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="hi", tool_calls=None), finish_reason="stop")],
            usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4),
            model=kwargs["model"],
        )


class _FakeOpenAI:
    created: list = []
    raise_exc = None

    def __init__(self, **kwargs):
        type(self).created.append(kwargs)
        self.log = []
        self.models = _FakeModels(self.log)
        self.chat = SimpleNamespace(completions=_FakeCompletions(type(self).raise_exc))


@pytest.fixture
def fake_openai(monkeypatch):
    _FakeOpenAI.created = []
    _FakeOpenAI.raise_exc = None
    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_FakeOpenAI))
    monkeypatch.setattr(oc, "OpenAI", _FakeOpenAI)
    return _FakeOpenAI


def _cfg(provider: LLMProvider, model: str, **kw) -> LLMConfig:
    return LLMConfig(provider=provider, model=model, api_key=kw.pop("api_key", "k-test"), **kw)


# --------------------------------------------------------------------------- #
# S0.1 — the registry
# --------------------------------------------------------------------------- #


def test_every_enum_member_has_a_spec_and_every_chat_spec_has_an_enum():
    for member in LLMProvider:
        spec = reg.get_spec(member.value)
        assert spec is not None, f"LLMProvider.{member.name} has no registry spec"
        assert spec.enum_value == member.value
    for spec in reg.REGISTRY.values():
        if spec.chat:
            assert spec.enum_value, f"{spec.slug} has a chat adapter but no enum value"
            assert LLMProvider(spec.enum_value)


def test_aliases_and_enum_values_normalise_to_the_slug():
    assert reg.normalize_slug("aws_bedrock") == "bedrock"
    assert reg.normalize_slug("azure_openai") == "azure"
    assert reg.normalize_slug("x-ai") == "grok"
    assert reg.normalize_slug("NVIDIA") == "nvidia"
    assert reg.normalize_slug("aiml") is None
    assert reg.enum_for("bedrock") is LLMProvider.AWS_BEDROCK
    assert reg.enum_for("cohere") is None  # key-only provider


def test_nvidia_spec_is_free_byok_only_and_hosts_vendor_models():
    spec = reg.get_spec("nvidia")
    assert spec.adapter == reg.ADAPTER_OPENAI_COMPATIBLE
    assert spec.byok_only and spec.free and spec.hosts_vendor_models
    assert spec.price_multiplier == 0.0
    assert "trial" in spec.terms_note.lower()
    assert "40 requests" in spec.rate_limit_note
    assert reg.base_url_for("nvidia") == config.NVIDIA_BASE_URL == "https://integrate.api.nvidia.com/v1"
    assert reg.hosts_vendor_models("nvidia") and reg.hosts_vendor_models("openrouter")
    assert not reg.hosts_vendor_models("deepseek")


def test_platform_key_allowed_depends_on_edition(monkeypatch):
    monkeypatch.setattr(reg.config, "AUTH_EDITION", "saas")
    assert not reg.platform_key_allowed("nvidia")
    assert reg.platform_key_allowed("openrouter")
    assert "nvidia" not in reg.platform_key_slugs()
    monkeypatch.setattr(reg.config, "AUTH_EDITION", "local")
    assert reg.platform_key_allowed("nvidia")
    assert "nvidia" in reg.platform_key_slugs()


def test_routable_names_exclude_key_only_providers_but_keep_aliases():
    routable = reg.routable_provider_names()
    assert {"openai", "openrouter", "nvidia", "deepseek", "bedrock", "aws_bedrock", "azure_openai", "x-ai"} <= routable
    assert "cohere" not in routable
    assert "cohere" in reg.known_provider_names()


def test_price_multiplier_is_zero_only_for_free_routes():
    assert reg.price_multiplier_for("nvidia") == 0.0
    assert reg.price_multiplier_for("openrouter") == 1.0
    assert reg.price_multiplier_for("aws_bedrock") == 1.0
    assert reg.price_multiplier_for("something-unknown") == 1.0


def test_public_registry_carries_no_secret_or_config_attribute(monkeypatch):
    monkeypatch.setattr(reg.config, "AUTH_EDITION", "saas")
    public = reg.public_registry()
    assert public["edition"] == "saas"
    slugs = [p["slug"] for p in public["providers"]]
    assert slugs == reg.all_slugs()
    for entry in public["providers"]:
        assert "env_key" not in entry and "base_url_key" not in entry
        assert not any("api_key" in str(v).lower() for v in entry.values() if isinstance(v, str) and "sk-" in v)
    nvidia = next(p for p in public["providers"] if p["slug"] == "nvidia")
    assert nvidia["platform_key"] is False and nvidia["byok"] is True and nvidia["free"] is True


# --------------------------------------------------------------------------- #
# S0.2 — the generic adapter
# --------------------------------------------------------------------------- #


def test_nvidia_client_uses_registry_base_url_and_no_attribution_headers(fake_openai):
    provider = OpenAICompatibleProvider(_cfg(LLMProvider.NVIDIA, "moonshotai/kimi-k3", api_key="nvapi-x"))
    assert provider.client is not None
    kwargs = fake_openai.created[-1]
    assert kwargs["api_key"] == "nvapi-x"
    assert kwargs["base_url"] == "https://integrate.api.nvidia.com/v1"
    assert "default_headers" not in kwargs


def test_openrouter_client_keeps_attribution_headers(fake_openai):
    OpenAICompatibleProvider(_cfg(LLMProvider.OPENROUTER, "moonshotai/kimi-k3", api_key="sk-or-x"))
    kwargs = fake_openai.created[-1]
    assert kwargs["base_url"] == config.OPENROUTER_BASE_URL
    assert kwargs["default_headers"]["X-Title"] == "Automatos AI"
    assert kwargs["default_headers"]["HTTP-Referer"] == config.OPENROUTER_SITE_URL


def test_explicit_base_url_override_wins(fake_openai):
    OpenAICompatibleProvider(_cfg(LLMProvider.DEEPSEEK, "deepseek-chat", base_url="https://proxy.example/v1"))
    assert fake_openai.created[-1]["base_url"] == "https://proxy.example/v1"


def test_adapter_refuses_a_non_openai_compatible_provider(fake_openai):
    with pytest.raises(ValueError):
        OpenAICompatibleProvider(_cfg(LLMProvider.ANTHROPIC, "claude-x"))


def test_429_raises_rate_limit_error_with_the_spec_note(fake_openai):
    fake_openai.raise_exc = Exception("Error code: 429 - {'error': 'rate limit exceeded'}")
    provider = OpenAICompatibleProvider(_cfg(LLMProvider.NVIDIA, "moonshotai/kimi-k3", api_key="nvapi-x"))
    with pytest.raises(ProviderRateLimitError) as exc_info:
        provider.generate_response_sync([{"role": "user", "content": "hi"}])
    assert "NVIDIA rate limit" in str(exc_info.value)
    assert "40 requests" in str(exc_info.value)
    assert "never rerouted" in str(exc_info.value)
    # the async path takes the same exit
    with pytest.raises(ProviderRateLimitError):
        asyncio.run(provider.generate_response([{"role": "user", "content": "hi"}]))


def test_response_is_tagged_with_the_serving_provider(fake_openai):
    provider = OpenAICompatibleProvider(_cfg(LLMProvider.NVIDIA, "moonshotai/kimi-k3", api_key="nvapi-x"))
    resp = asyncio.run(provider.generate_response([{"role": "user", "content": "hi"}]))
    assert resp.provider == "nvidia"
    assert resp.content == "hi"
    assert resp.usage["total_tokens"] == 4


def test_manager_builds_the_generic_adapter_for_registry_providers(fake_openai):
    from core.llm.manager import LLMManager

    for member in (LLMProvider.NVIDIA, LLMProvider.DEEPSEEK, LLMProvider.OPENROUTER):
        built = LLMManager._create_provider(_cfg(member, "m", api_key="k"))
        assert isinstance(built, OpenAICompatibleProvider)
        assert built.spec.slug == member.value


# --------------------------------------------------------------------------- #
# S0.3 — the lists ARE the registry; NVIDIA is live-validated
# --------------------------------------------------------------------------- #


def test_the_provider_lists_read_the_registry():
    import api.user_api_keys as uak
    from api.workspaces import _ALLOWED_PROVIDERS

    assert uak.SUPPORTED_PROVIDERS == reg.byok_slugs()
    assert "nvidia" in uak.SUPPORTED_PROVIDERS and "deepseek" in uak.SUPPORTED_PROVIDERS
    assert _ALLOWED_PROVIDERS == frozenset(reg.byok_slugs())


def test_validate_provider_key_live_tests_nvidia_against_its_base_url(fake_openai):
    from api.user_api_keys import _validate_provider_key

    result = asyncio.run(_validate_provider_key("nvidia", "nvapi-test-key"))
    assert result.valid is True and result.message == "API key is valid"
    kwargs = fake_openai.created[-1]
    assert kwargs["base_url"] == "https://integrate.api.nvidia.com/v1"
    assert kwargs["api_key"] == "nvapi-test-key"
    assert "default_headers" not in kwargs


def test_validate_provider_key_reports_a_rejected_nvidia_key(fake_openai):
    from api.user_api_keys import _validate_provider_key

    class _Rejecting(fake_openai):
        def __init__(self, **kwargs):
            raise RuntimeError("401 Unauthorized: invalid api key")

    sys.modules["openai"].OpenAI = _Rejecting
    result = asyncio.run(_validate_provider_key("nvidia", "nvapi-bad"))
    assert result.valid is False and "401" in result.message


def test_validate_provider_key_still_reports_no_live_check_for_key_only_providers():
    from api.user_api_keys import _validate_provider_key

    result = asyncio.run(_validate_provider_key("cohere", "co-key-123456"))
    assert result.valid is True and "not available" in result.message


# --------------------------------------------------------------------------- #
# S0.4 — routing
# --------------------------------------------------------------------------- #


def _factory():
    import logging
    from modules.agents.factory.agent_factory import AgentFactory

    f = AgentFactory.__new__(AgentFactory)
    f.logger = logging.getLogger("test.prd236")
    f.db_session = None
    return f


@pytest.mark.parametrize(
    "provider,model,expected",
    [
        ("nvidia", "moonshotai/kimi-k3", ("nvidia", "moonshotai/kimi-k3")),
        ("openrouter", "moonshotai/kimi-k3", ("openrouter", "moonshotai/kimi-k3")),
        ("deepseek", "deepseek-chat", ("deepseek", "deepseek-chat")),
        # legacy rules, unchanged
        ("openai", "moonshotai/kimi-k3", ("openrouter", "moonshotai/kimi-k3")),
        ("aiml", "deepseek-ai/DeepSeek-R1", ("openrouter", "deepseek-ai/DeepSeek-R1")),
        ("aiml", "llama-3-70b", ("openrouter", "meta-llama/llama-3-70b")),
        ("anthropic", "gpt-4o", ("openai", "gpt-4o")),
        ("openai", "gpt-4o", ("openai", "gpt-4o")),
        ("bedrock", "anthropic.claude-3", ("bedrock", "anthropic.claude-3")),
    ],
)
def test_resolve_provider_for_model(provider, model, expected):
    assert _factory()._resolve_provider_for_model(provider, model) == expected


def test_key_only_provider_still_takes_the_unknown_provider_path():
    provider, model = _factory()._resolve_provider_for_model("cohere", "command-r")
    assert provider == "openrouter"


def test_openrouter_model_id_uses_registry_prefixes():
    f = _factory()
    assert f._openrouter_model_id("deepseek", "deepseek-chat") == "deepseek/deepseek-chat"
    assert f._openrouter_model_id("openai", "gpt-4o") == "openai/gpt-4o"
    assert f._openrouter_model_id("grok", "grok-3") == "x-ai/grok-3"
    assert f._openrouter_model_id("nvidia", "moonshotai/kimi-k3") == "moonshotai/kimi-k3"
    assert f._openrouter_model_id("nvidia", "bare-id") == "bare-id"


@pytest.fixture
def platform_tiers_stubbed(monkeypatch):
    """No workspace key, no credential-store key: only the env tier can answer."""
    monkeypatch.setitem(
        sys.modules,
        "core.credentials.resolver",
        types.SimpleNamespace(
            get_credential_resolver=lambda: SimpleNamespace(get_credential_field=lambda *a, **k: None)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "core.llm.workspace_keys",
        types.SimpleNamespace(get_platform_workspace_key=lambda *_a, **_k: None),
    )


def test_byok_only_provider_never_resolves_from_platform_tiers_in_saas(monkeypatch, platform_tiers_stubbed):
    monkeypatch.setattr(reg.config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "NVIDIA_API_KEY", "nvapi-platform-should-be-ignored")
    resolved = asyncio.run(_factory()._resolve_api_key("nvidia", "agent"))
    assert resolved is None


def test_byok_only_provider_resolves_from_env_in_local_edition(monkeypatch, platform_tiers_stubbed):
    monkeypatch.setattr(reg.config, "AUTH_EDITION", "local")
    monkeypatch.setattr(config, "NVIDIA_API_KEY", "nvapi-operator")
    resolved = asyncio.run(_factory()._resolve_api_key("nvidia", "agent"))
    assert resolved is not None
    assert resolved.api_key == "nvapi-operator" and resolved.source == "env" and not resolved.is_byok


def test_env_tier_reads_the_registry_for_every_provider(monkeypatch, platform_tiers_stubbed):
    monkeypatch.setattr(reg.config, "AUTH_EDITION", "saas")
    monkeypatch.setattr(config, "DEEPSEEK_API_KEY", "sk-ds")
    resolved = asyncio.run(_factory()._resolve_api_key("deepseek", "agent"))
    assert resolved is not None and resolved.api_key == "sk-ds" and resolved.source == "env"


# --------------------------------------------------------------------------- #
# S0.5 — honest cost
# --------------------------------------------------------------------------- #


class _FakeQuery:
    def __init__(self, row):
        self._row = row

    def filter(self, *_a, **_k):
        return self

    def first(self):
        return self._row


class _FakeSession:
    added: list = []

    def __init__(self, row):
        self._row = row

    def query(self, _model):
        return _FakeQuery(self._row)

    def add(self, obj):
        type(self).added.append(obj)

    def commit(self):
        pass

    def close(self):
        pass


def _track(monkeypatch, provider: str):
    from core.llm.usage_tracker import UsageTracker
    from uuid import uuid4

    row = SimpleNamespace(
        input_cost_per_1k_tokens=0.003, output_cost_per_1k_tokens=0.015,
        sourcing="aggregator", serving_provider="openrouter", model_id="moonshotai/kimi-k3",
    )
    _FakeSession.added = []
    monkeypatch.setitem(
        sys.modules, "core.database.database", types.SimpleNamespace(SessionLocal=lambda: _FakeSession(row))
    )
    UsageTracker.track(
        workspace_id=uuid4(), model_id="moonshotai/kimi-k3", provider=provider,
        input_tokens=1000, output_tokens=1000,
    )
    assert len(_FakeSession.added) == 1, "usage row was not recorded"
    return _FakeSession.added[0]


def test_a_free_route_books_zero_cost(monkeypatch):
    usage = _track(monkeypatch, "nvidia")
    assert usage.provider == "nvidia"
    assert usage.total_cost == 0.0 and usage.input_cost == 0.0 and usage.output_cost == 0.0
    assert usage.total_tokens == 2000


def test_a_paid_route_still_books_the_row_price(monkeypatch):
    usage = _track(monkeypatch, "openrouter")
    assert usage.provider == "openrouter"
    assert usage.total_cost == pytest.approx(0.018)


# --------------------------------------------------------------------------- #
# S0.6 — the providers endpoint
# --------------------------------------------------------------------------- #


def test_providers_endpoint_returns_the_public_registry(monkeypatch):
    import api.user_api_keys as uak

    monkeypatch.setattr(reg.config, "AUTH_EDITION", "saas")
    payload = asyncio.run(uak.list_providers(ctx=None))
    assert payload["edition"] == "saas"
    nvidia = next(p for p in payload["providers"] if p["slug"] == "nvidia")
    assert nvidia["platform_key"] is False and nvidia["terms_note"]
    assert set(payload["providers"][0].keys()) == {
        "slug", "label", "kind", "chat", "embeddings", "byok", "platform_key",
        "hosts_vendor_models", "free", "key_placeholder", "docs_url", "terms_note", "rate_limit_note",
    }


def test_route_manifest_lists_the_providers_endpoint():
    import json
    from pathlib import Path

    manifest = json.loads((Path(__file__).resolve().parents[1] / "reports" / "route-manifest.json").read_text())
    assert {"method": "GET", "path": "/api/keys/providers"} in manifest["routes"]
    assert manifest["route_count"] == len(manifest["routes"])
