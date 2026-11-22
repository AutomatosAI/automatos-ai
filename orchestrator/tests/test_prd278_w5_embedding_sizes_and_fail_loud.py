"""PRD-278 US-502 and US-503 — sizes that agree at start-up, and no stand-in for a real index.

US-502 (``core.llm.embedding_sizes``, as the Wave 5 review settled it):
  * Bedrock refuses the start when an index in use expects another size than its vectors
    have, and the message gives the fix in the environment, which wins for Bedrock;
  * Bedrock with sizes that agree starts; an index not in use (S3 Vectors off, a memory store
    that does not answer) is not compared, and the store is asked only when its size disagrees;
  * any other provider never refuses: a known model smaller than an index in use starts and
    warns once; the seeded default starts with no warning;
  * a stored size that is not a number never stops a start;
  * a Bedrock model ID that is not supported refuses the start, with the list;
  * the settings read runs in a worker thread, after ``_boot_core`` and before the server
    listens, and no AWS client is built.

US-503 (``EmbeddingManager._load_provider``):
  * a provider set and unusable (no key, no client built for it, a failed set-up) gives a
    provider that names it and raises on every call, never the deterministic stand-in; it
    still reports itself degraded, so readiness says what it said before;
  * with no provider set at all, the stand-in behaves as today.

No network, no database: settings are a dict, keys are absent, the memory store is a stub.
"""
from __future__ import annotations

import ast
import asyncio
import logging
import sys
import threading
import types
from pathlib import Path

import httpx
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
from core.llm.embedding_sizes import EmbeddingSettingsError
from core.llm.embedding_status import provider_ready

ORCHESTRATOR = Path(__file__).resolve().parents[1]
TITAN = "amazon.titan-embed-text-v2:0"
QWEN = "qwen/qwen3-embedding-8b"
STORED_SECRET = "stored-secret-for-tests"
PROBE = embedding_sizes.memory_store_answers  # the real probe, before the fixture stubs it


@pytest.fixture(autouse=True)
def no_keys_and_default_sizes(monkeypatch):
    """No key anywhere, the three sizes at their 2,048 defaults, both indexes in use."""
    monkeypatch.setitem(sys.modules, "core.llm.workspace_keys",
                        types.SimpleNamespace(get_platform_workspace_key=lambda *_a, **_k: None))
    monkeypatch.setattr(em, "get_credential_field", lambda *_a, **_k: None)
    for name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "COHERE_API_KEY", "EMBEDDING_PROVIDER", "EMBEDDING_MODEL"):
        monkeypatch.setattr(config, name, None)
    monkeypatch.setattr(config, "AWS_SECRET_ACCESS_KEY", STORED_SECRET)
    _sizes(monkeypatch, vector=2048, s3=2048, field=2048)
    monkeypatch.setattr(config, "S3_VECTORS_ENABLED", True)
    monkeypatch.setattr(config, "QDRANT_URL", "http://qdrant.invalid:6333")
    monkeypatch.setattr(config, "QDRANT_API_KEY", "")
    em._unreadable_size.cache_clear()
    _store(monkeypatch, answers=True)
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


def _store(monkeypatch, *, answers: bool) -> list:
    """The memory store answers (in use) or not; returns the list of times it was asked."""
    asked = []

    async def probe(transport=None):
        asked.append(True)
        return answers

    monkeypatch.setattr(embedding_sizes, "memory_store_answers", probe)
    return asked


def _no_client_built(monkeypatch) -> None:
    def refuse(*_a, **_k):
        raise AssertionError("the size check built an AWS client")

    monkeypatch.setattr(bedrock_client, "runtime_client", refuse)
    monkeypatch.setattr(bedrock_client.boto3, "client", refuse)


def _check() -> None:
    asyncio.run(embedding_sizes.verify_embedding_sizes())


def _bedrock(monkeypatch) -> None:
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "bedrock")


def _warnings(caplog) -> list:
    return [r for r in caplog.records if r.name == embedding_sizes.__name__ and r.levelno == logging.WARNING]


# --------------------------------------------------------------------------- US-502: Bedrock refuses


def test_bedrock_with_the_default_sizes_refuses_the_start_with_the_fix(monkeypatch):
    _bedrock(monkeypatch)
    _no_client_built(monkeypatch)

    with pytest.raises(EmbeddingSettingsError) as refused:
        _check()

    message = str(refused.value)
    assert "S3_VECTORS_DIMENSION is 2048" in message and "FIELD_EMBEDDING_DIM is 2048" in message
    assert "produces at most 1024 dimensions" in message and "vectors here are 1024 long" in message
    assert "Set each of them to 1024 in the environment" in message
    assert "VECTOR_STORE_DIMENSIONS wins over the stored 'dimensions' system setting" in message
    assert "EMBEDDING_PROVIDER and EMBEDDING_MODEL over the stored 'provider' and 'model'" in message
    assert embedding_sizes.PAGE in message
    assert TITAN in message and STORED_SECRET not in message


def test_bedrock_names_only_the_indexes_that_disagree(monkeypatch):
    _bedrock(monkeypatch)
    _sizes(monkeypatch, vector=2048, s3=1024, field=2048)

    with pytest.raises(EmbeddingSettingsError) as refused:
        _check()

    assert "FIELD_EMBEDDING_DIM is 2048" in str(refused.value)
    assert "S3_VECTORS_DIMENSION is" not in str(refused.value)


def test_bedrock_refuses_an_index_larger_than_the_size_its_vectors_are_cut_to(monkeypatch):
    _bedrock(monkeypatch)
    _sizes(monkeypatch, vector=512, s3=1024, field=512)

    with pytest.raises(EmbeddingSettingsError, match="vectors here are 512 long.*S3_VECTORS_DIMENSION is 1024"):
        _check()


@pytest.mark.parametrize("vector,index", [(1024, 1024), (512, 512), (256, 256), (300, 300), (2048, 1024)])
def test_bedrock_with_sizes_that_agree_starts_and_asks_no_store(monkeypatch, vector, index):
    """The vectors are VECTOR_STORE_DIMENSIONS long, or the model's 1,024 when that is larger."""
    _bedrock(monkeypatch)
    _sizes(monkeypatch, vector=vector, s3=index, field=index)
    _no_client_built(monkeypatch)
    asked = _store(monkeypatch, answers=True)

    _check()

    assert asked == [], "an agreeing memory store size needs no request"


def test_bedrock_reads_every_size_from_the_environment_never_the_stored_setting(monkeypatch):
    _bedrock(monkeypatch)
    _sizes(monkeypatch, vector=1024, s3=1024, field=1024)
    _settings(monkeypatch, {"provider": "openrouter", "model": QWEN, "dimensions": "1536"})

    _check()

    choice = em.read_embedding_choice()
    assert (choice.provider, choice.model, choice.dimension) == ("bedrock", TITAN, 1024)
    assert choice.dimension_setting == em.DIMENSION_FROM_CONFIG


@pytest.mark.parametrize("s3_on,store_answers", [(False, False), (False, True), (True, False)])
def test_an_index_not_in_use_is_not_compared(monkeypatch, s3_on, store_answers):
    _bedrock(monkeypatch)
    _sizes(monkeypatch, vector=1024, s3=1024 if s3_on else 2048, field=1024 if store_answers else 2048)
    monkeypatch.setattr(config, "S3_VECTORS_ENABLED", s3_on)
    _store(monkeypatch, answers=store_answers)

    _check()


def test_a_bedrock_model_it_does_not_support_refuses_the_start_with_the_list(monkeypatch):
    _bedrock(monkeypatch)
    monkeypatch.setattr(config, "EMBEDDING_MODEL", "amazon.titan-embed-text-v1")
    _sizes(monkeypatch, vector=1024, s3=1024, field=1024)

    with pytest.raises(EmbeddingSettingsError, match=r"not supported.*amazon\.titan-embed-text-v2:0 \(1024"):
        _check()


# --------------------------------------------------------------------------- US-502: the others start


def test_the_default_configuration_starts_unchanged(monkeypatch, caplog):
    """No provider set, disabled, and the seeded OpenRouter model (4,096, cut to 2,048): each
    starts, with no warning and no request to the memory store."""
    caplog.set_level(logging.INFO, logger=embedding_sizes.__name__)
    _no_client_built(monkeypatch)
    asked = _store(monkeypatch, answers=True)
    for rows in ({}, {"provider": "disabled"}, {"provider": "openrouter", "model": QWEN, "dimensions": "2048"}):
        _settings(monkeypatch, rows)
        _check()

    assert _warnings(caplog) == [] and asked == []


@pytest.mark.parametrize("provider,model", [("openai", "text-embedding-3-small"),
                                            ("openrouter", "openai/text-embedding-3-small")])
def test_a_smaller_model_of_another_provider_starts_and_warns_once(monkeypatch, caplog, provider, model):
    """A 1,536-dimension model with the default sizes worked before Wave 5 (an index not in use
    takes nothing) and starts now; with both indexes in use it says why writes to them fail."""
    caplog.set_level(logging.INFO, logger=embedding_sizes.__name__)
    _settings(monkeypatch, {"provider": provider, "model": model})

    _check()

    (warning,) = _warnings(caplog)
    message = warning.getMessage()
    assert "produces at most 1536 dimensions" in message
    assert "S3_VECTORS_DIMENSION is 2048" in message and "FIELD_EMBEDDING_DIM is 2048" in message
    assert "set to 1536 in the environment" in message and "Starting anyway" in message


def test_a_smaller_model_of_another_provider_with_no_index_in_use_starts_silently(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=embedding_sizes.__name__)
    monkeypatch.setattr(config, "S3_VECTORS_ENABLED", False)
    _store(monkeypatch, answers=False)
    _settings(monkeypatch, {"provider": "openai", "model": "text-embedding-3-small"})

    _check()

    assert _warnings(caplog) == []


def test_a_model_whose_size_is_not_known_here_is_not_checked(monkeypatch):
    _settings(monkeypatch, {"provider": "openrouter", "model": "someone/new-embedding-model"})
    _check()
    _settings(monkeypatch, {"provider": "huggingface_local", "model": "BAAI/bge-m3"})
    _check()


@pytest.mark.parametrize("rows", [{"dimensions": "not-a-size"},
                                  {"provider": "openrouter", "model": QWEN, "dimensions": "not-a-size"}])
def test_a_stored_size_that_is_not_a_number_never_stops_a_start(monkeypatch, caplog, rows):
    caplog.set_level(logging.WARNING, logger=em.__name__)
    _settings(monkeypatch, rows)

    _check()
    _check()

    assert em.read_embedding_choice().dimension == 2048, "VECTOR_STORE_DIMENSIONS is used"
    logged = [r for r in caplog.records if r.name == em.__name__ and "not a number" in r.getMessage()]
    assert len(logged) == 1, "logged once"


def test_the_settings_read_runs_in_a_worker_thread(monkeypatch):
    seen = []
    real = em.read_embedding_choice
    monkeypatch.setattr(em, "read_embedding_choice", lambda: seen.append(threading.get_ident()) or real())

    _check()

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


# --------------------------------------------------------------------------- the memory store probe


def _probe(handler) -> bool:
    return asyncio.run(PROBE(transport=httpx.MockTransport(handler)))


def test_a_memory_store_that_answers_is_in_use():
    seen = []

    def answer(request):
        seen.append(request.url.path)
        return httpx.Response(200, json={"result": {"collections": []}, "status": "ok"})

    assert _probe(answer) is True and seen == ["/collections"]


def test_a_memory_store_that_does_not_answer_is_not_in_use():
    def refuse(request):
        raise httpx.ConnectError("refused", request=request)

    assert _probe(refuse) is False


def test_no_memory_store_address_is_not_in_use_and_sends_nothing(monkeypatch):
    monkeypatch.setattr(config, "QDRANT_URL", "")

    def never(request):
        raise AssertionError("no address: nothing is sent")

    assert _probe(never) is False


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
    _bedrock(monkeypatch)
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
