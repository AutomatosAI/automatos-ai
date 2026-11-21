"""PRD-278 US-502 and US-503 — sizes that agree or no start, and no stand-in for a real index.

US-502 (``core.llm.embedding_sizes``):
  * with a real provider set, a model smaller than a size setting in use refuses the start,
    naming each setting, both numbers and the value to set, and no credential;
  * a model larger than the settings starts (its vectors are cut to fit: the provider tests);
  * a Bedrock model ID that is not supported refuses the start, with the list;
  * the default configuration starts unchanged: no provider, and the seeded OpenRouter one;
  * the check reads settings only (no client is built, no call made), in a worker thread,
    after ``_boot_core`` and before the server listens.

US-503 (``EmbeddingManager._load_provider``):
  * a provider set and unusable (no key, no client built for it, a failed set-up) gives a
    provider that names it and raises on every call, never the deterministic stand-in; it
    still reports itself degraded, so readiness says what it said before;
  * with no provider set at all, the stand-in behaves as today.

No network, no database: settings are a dict, keys are absent.
"""
from __future__ import annotations

import ast
import asyncio
import sys
import threading
import types
from pathlib import Path

import pytest

import core.llm.clients.bedrock_client as bedrock_client
import core.llm.embedding_manager as em
import core.llm.manager as llm_manager
from config import config
from core.llm import embedding_sizes
from core.llm.clients.base import (
    DeterministicEmbeddingProvider,
    EmbeddingUnavailableError,
    UnusableEmbeddingProvider,
)
from core.llm.embedding_sizes import EmbeddingSettingsError, check_embedding_sizes
from core.llm.embedding_status import provider_ready

ORCHESTRATOR = Path(__file__).resolve().parents[1]
TITAN = "amazon.titan-embed-text-v2:0"
QWEN = "qwen/qwen3-embedding-8b"
STORED_SECRET = "stored-secret-for-tests"


@pytest.fixture(autouse=True)
def no_keys_and_default_sizes(monkeypatch):
    """No key anywhere, the three sizes at their 2,048 defaults, both indexes in use."""
    monkeypatch.setitem(sys.modules, "core.llm.workspace_keys",
                        types.SimpleNamespace(get_platform_workspace_key=lambda *_a, **_k: None))
    monkeypatch.setattr(em, "get_credential_field", lambda *_a, **_k: None)
    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "COHERE_API_KEY", "EMBEDDING_PROVIDER", "EMBEDDING_MODEL"):
        monkeypatch.setattr(config, name, None)
    monkeypatch.setattr(config, "AWS_SECRET_ACCESS_KEY", STORED_SECRET)
    monkeypatch.setattr(config, "VECTOR_STORE_DIMENSIONS", 2048)
    monkeypatch.setattr(config, "S3_VECTORS_DIMENSION", 2048)
    monkeypatch.setattr(config, "FIELD_EMBEDDING_DIM", 2048)
    monkeypatch.setattr(config, "S3_VECTORS_ENABLED", True)
    monkeypatch.setattr(config, "QDRANT_URL", "http://qdrant.invalid:6333")
    _settings(monkeypatch, {})


def _settings(monkeypatch, rows: dict) -> None:
    """The ``embeddings`` system settings: ``rows`` maps a key to its stored value."""
    monkeypatch.setattr(llm_manager, "get_system_setting",
                        lambda category, key, default=None: rows.get(key) if category == "embeddings"
                        and rows.get(key) else default)


def _sizes(monkeypatch, vector: int, s3: int, field: int) -> None:
    monkeypatch.setattr(config, "VECTOR_STORE_DIMENSIONS", vector)
    monkeypatch.setattr(config, "S3_VECTORS_DIMENSION", s3)
    monkeypatch.setattr(config, "FIELD_EMBEDDING_DIM", field)


def _no_client_built(monkeypatch) -> None:
    def refuse(*_a, **_k):
        raise AssertionError("the size check built an AWS client")

    monkeypatch.setattr(bedrock_client, "runtime_client", refuse)
    monkeypatch.setattr(bedrock_client.boto3, "client", refuse)


# --------------------------------------------------------------------------- US-502


def test_bedrock_with_the_default_sizes_refuses_the_start_naming_each_setting(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "bedrock")
    _no_client_built(monkeypatch)

    with pytest.raises(EmbeddingSettingsError) as refused:
        check_embedding_sizes()

    message = str(refused.value)
    for setting in ("VECTOR_STORE_DIMENSIONS is 2048", "S3_VECTORS_DIMENSION is 2048", "FIELD_EMBEDDING_DIM is 2048"):
        assert setting in message
    assert "produces at most 1024 dimensions" in message and "Set each to 1024" in message
    assert TITAN in message and STORED_SECRET not in message


def test_only_the_settings_larger_than_the_model_are_named(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "bedrock")
    _sizes(monkeypatch, vector=1024, s3=1024, field=2048)

    with pytest.raises(EmbeddingSettingsError) as refused:
        check_embedding_sizes()

    assert "FIELD_EMBEDDING_DIM is 2048" in str(refused.value)
    assert "S3_VECTORS_DIMENSION" not in str(refused.value) and "VECTOR_STORE_DIMENSIONS" not in str(refused.value)


def test_the_dimensions_system_setting_is_named_when_it_gives_the_size(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "bedrock")
    _sizes(monkeypatch, vector=1024, s3=1024, field=1024)
    _settings(monkeypatch, {"dimensions": "1536"})

    with pytest.raises(EmbeddingSettingsError, match="'dimensions' system setting .* is 1536"):
        check_embedding_sizes()


@pytest.mark.parametrize("size", [1024, 512, 256, 300])
def test_settings_the_model_can_fill_start(monkeypatch, size):
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "bedrock")
    _sizes(monkeypatch, vector=size, s3=size, field=size)
    _no_client_built(monkeypatch)

    check_embedding_sizes()


def test_an_index_that_is_off_is_not_compared(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "bedrock")
    _sizes(monkeypatch, vector=1024, s3=2048, field=2048)
    monkeypatch.setattr(config, "S3_VECTORS_ENABLED", False)
    monkeypatch.setattr(config, "QDRANT_URL", "")

    check_embedding_sizes()


def test_a_bedrock_model_it_does_not_support_refuses_the_start_with_the_list(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "bedrock")
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "amazon.titan-embed-text-v1")
    _sizes(monkeypatch, vector=1024, s3=1024, field=1024)

    with pytest.raises(EmbeddingSettingsError, match=r"not supported.*amazon\.titan-embed-text-v2:0 \(1024"):
        check_embedding_sizes()


def test_the_default_configuration_starts_unchanged(monkeypatch):
    """No provider set: nothing is compared. The seeded OpenRouter model (4,096) fills 2,048."""
    _no_client_built(monkeypatch)
    check_embedding_sizes()
    _settings(monkeypatch, {"provider": "disabled"})
    check_embedding_sizes()
    _settings(monkeypatch, {"provider": "openrouter", "model": QWEN, "dimensions": "2048"})
    check_embedding_sizes()


def test_another_providers_model_smaller_than_the_index_refuses_too(monkeypatch):
    _settings(monkeypatch, {"provider": "openrouter", "model": "baai/bge-m3"})

    with pytest.raises(EmbeddingSettingsError, match="produces at most 1024 dimensions"):
        check_embedding_sizes()


def test_a_model_whose_size_is_not_known_here_is_not_checked(monkeypatch):
    _settings(monkeypatch, {"provider": "openrouter", "model": "someone/new-embedding-model"})
    check_embedding_sizes()
    _settings(monkeypatch, {"provider": "huggingface_local", "model": "BAAI/bge-m3"})
    check_embedding_sizes()


def test_the_check_runs_in_a_worker_thread(monkeypatch):
    seen = []
    monkeypatch.setattr(embedding_sizes, "check_embedding_sizes", lambda: seen.append(threading.get_ident()))

    asyncio.run(embedding_sizes.verify_embedding_sizes())

    assert seen and seen[0] != threading.get_ident()


def _lifespan_calls() -> list:
    tree = ast.parse((ORCHESTRATOR / "main.py").read_text(encoding="utf-8"))
    lifespan = next(node for node in tree.body if isinstance(node, ast.AsyncFunctionDef) and node.name == "lifespan")
    calls = sorted((node for node in ast.walk(lifespan) if isinstance(node, ast.Call)),
                   key=lambda node: (node.lineno, node.col_offset))
    return [getattr(node.func, "attr", None) or getattr(node.func, "id", None) for node in calls]


def test_the_lifespan_checks_after_the_leader_seeds_and_before_it_serves():
    calls = _lifespan_calls()
    order = [calls.index(name) for name in ("_boot_core", "verify_embedding_sizes", "_start_serving")]
    assert order == sorted(order), calls


# --------------------------------------------------------------------------- US-503


def _loaded(monkeypatch, rows: dict) -> em.EmbeddingManager:
    _settings(monkeypatch, rows)
    manager = em.EmbeddingManager()
    manager._ensure_provider()
    return manager


def _raises_naming(manager: em.EmbeddingManager, provider: str, reason: str) -> None:
    with pytest.raises(EmbeddingUnavailableError) as refused:
        asyncio.run(manager.generate_embedding("hello"))
    assert f"'{provider}' is set but cannot be used" in str(refused.value) and reason in str(refused.value)
    with pytest.raises(EmbeddingUnavailableError):
        asyncio.run(manager.generate_embeddings_batch(["a", "b"]))


def test_a_set_provider_with_no_key_raises_and_is_not_the_stand_in(monkeypatch):
    manager = _loaded(monkeypatch, {"provider": "openrouter", "model": QWEN})

    assert isinstance(manager.provider, UnusableEmbeddingProvider)
    assert not isinstance(manager.provider, DeterministicEmbeddingProvider)
    _raises_naming(manager, "openrouter", "no API key was found for it")
    assert provider_ready(manager) is False, "readiness still says memory and knowledge search are off"


def test_a_set_provider_whose_set_up_fails_raises_naming_it(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "bedrock")
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "not-a-bedrock-embedding-model")
    manager = _loaded(monkeypatch, {})

    assert isinstance(manager.provider, UnusableEmbeddingProvider)
    _raises_naming(manager, "bedrock", "it could not be set up (UnsupportedBedrockEmbeddingModel)")


@pytest.mark.parametrize("provider,reason", [("cohere", "no embedding client is built for it"),
                                             ("not-a-provider", "it could not be set up (ValueError)")])
def test_a_set_provider_no_client_is_built_for_raises_naming_it(monkeypatch, provider, reason):
    monkeypatch.setattr(em, "get_credential_field", lambda *_a, **_k: "key-for-tests")
    manager = _loaded(monkeypatch, {"provider": provider})

    assert isinstance(manager.provider, UnusableEmbeddingProvider)
    _raises_naming(manager, provider, reason)


@pytest.mark.parametrize("rows", [{}, {"provider": "disabled"}])
def test_with_no_provider_set_the_stand_in_behaves_as_today(monkeypatch, rows):
    manager = _loaded(monkeypatch, rows)

    assert type(manager.provider) is DeterministicEmbeddingProvider
    assert manager.provider.is_degraded is True and manager.get_dimension() == 2048
    with pytest.raises(EmbeddingUnavailableError, match="refusing to return random vectors"):
        asyncio.run(manager.generate_embedding("hello"))
    assert provider_ready(manager) is False
    assert manager.get_provider_info()["provider"] == "disabled"
