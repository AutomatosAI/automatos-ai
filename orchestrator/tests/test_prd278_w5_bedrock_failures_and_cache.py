"""PRD-278 Wave 5 review — Bedrock embeddings under load and under refusal, and what is cached.

  (a) every Bedrock embedding call in the process runs in the provider's own pool: two batches
      at once have at most ``CALLS_IN_FLIGHT`` calls in flight between them, none in the
      default executor;
  (b) a refusal that will not get better (``AccessDeniedException``) raises an
      ``EmbeddingUnavailableError`` with the code and the model only, cancels the batch's calls
      not yet made, marks every Bedrock provider degraded (readiness says off), holds further
      calls back for ``REFUSAL_HOLD_S`` without a request, and clears on the next success;
  (c) the ingestion batch does not try such a refusal again one text at a time; any other
      failure keeps its sequential pass;
  (d) the ingestion cache is keyed by the embedding space, so another model's cached vector is
      never served for Bedrock's;
  (e) the code index's column follows Bedrock's real vector length, never the stored size.

No network, no database: the Bedrock client is a stand-in, the cache is a dict.
"""
from __future__ import annotations

import asyncio
import io
import json
import sys
import threading
import time
import types
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import boto3
import pytest
from botocore.exceptions import ClientError

import core.cache
import core.llm.clients.bedrock_embedding as bedrock_embedding
import core.llm.clients.bedrock_embedding_calls as calls
import core.llm.embedding_manager as em
import core.llm.manager as llm_manager
from config import config
from core.llm.clients.base import EmbeddingConfig, EmbeddingProvider, EmbeddingUnavailableError
from core.llm.clients.bedrock_embedding import TITAN_TEXT_EMBEDDINGS_V2, BedrockEmbeddingProvider
from core.llm.embedding_status import provider_ready

SPACE = f"bedrock:{TITAN_TEXT_EMBEDDINGS_V2}:1024"
DENIED = ClientError({"Error": {"Code": "AccessDeniedException", "Message": "the role used in tests may not"}},
                     "InvokeModel")


@pytest.fixture(autouse=True)
def bedrock_on_the_role(monkeypatch, tmp_path):
    """A Bedrock provider on the install's role (the SDK's chain answers from the environment
    with an obviously fake key, so no metadata service is asked), no stored credential."""
    for name in ("AWS_SESSION_TOKEN", "AWS_PROFILE", "AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION", "AWS_DEFAULT_REGION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "role-key-id-for-tests")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "role-secret-for-tests")
    monkeypatch.setattr(boto3, "DEFAULT_SESSION", None)
    monkeypatch.setitem(sys.modules, "core.llm.workspace_keys",
                        types.SimpleNamespace(get_platform_workspace_key=lambda *_a, **_k: None))
    monkeypatch.setattr(llm_manager, "get_system_setting", lambda _c, _k, default=None: default)
    monkeypatch.setattr(llm_manager, "get_credential_data", lambda *_a, **_k: {})
    monkeypatch.setattr(config, "AWS_USE_WORKLOAD_IDENTITY", True)
    monkeypatch.setattr(config, "AWS_ACCESS_KEY_ID", None)
    monkeypatch.setattr(config, "AWS_SECRET_ACCESS_KEY", None)
    monkeypatch.setattr(config, "AWS_REGION", "eu-west-1")
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "bedrock")
    monkeypatch.setattr(config, "EMBEDDING_MODEL", None)
    monkeypatch.setattr(config, "VECTOR_STORE_DIMENSIONS", 1024)
    monkeypatch.setattr(config, "EMBEDDING_BATCH_TIMEOUT_S", 60.0)
    monkeypatch.setattr(bedrock_embedding, "record_embedding_usage", lambda *a, **k: None)
    calls.HEALTH.answered()
    yield
    calls.HEALTH.answered()


class _Runtime:
    """A bedrock-runtime stand-in: counts calls and how many are in flight, notes each call's
    thread, sleeps ``delay``; ``fail(n)`` says whether call ``n`` (from 0) raises ``error``, or,
    with ``fail_text``, only the call for that text does."""

    def __init__(self, delay: float = 0.0, error: Exception = None, fail=lambda _n: True, fail_text: str = None):
        self.delay, self.error, self.fail, self.fail_text = delay, error, fail, fail_text
        self.lock = threading.Lock()
        self.calls = self.in_flight = self.peak = 0
        self.thread_names = set()

    def invoke_model(self, **kwargs):
        with self.lock:
            number = self.calls
            self.calls += 1
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            self.thread_names.add(threading.current_thread().name)
        try:
            time.sleep(self.delay)
            text = json.loads(kwargs["body"])["inputText"]
            failing = self.fail(number) if self.fail_text is None else text == self.fail_text
            if self.error is not None and failing:
                raise self.error
            body = {"embedding": [0.25] * 1024, "inputTextTokenCount": 1}
            return {"body": io.BytesIO(json.dumps(body).encode())}
        finally:
            with self.lock:
                self.in_flight -= 1


def _manager(runtime: _Runtime) -> em.EmbeddingManager:
    provider = BedrockEmbeddingProvider(EmbeddingConfig(provider=EmbeddingProvider.BEDROCK,
                                                        model=TITAN_TEXT_EMBEDDINGS_V2, dimension=1024))
    provider.client = runtime
    manager = em.EmbeddingManager()
    manager.provider, manager._provider_loaded = provider, True
    return manager


# --------------------------------------------------------------------------- (a) the process-wide bound


def test_two_batches_at_once_share_the_process_wide_bound_in_their_own_threads(monkeypatch):
    monkeypatch.setattr(calls, "_CALL_POOL", ThreadPoolExecutor(max_workers=3, thread_name_prefix="bedrock-embedding"))
    runtime = _Runtime(delay=0.05)
    manager = _manager(runtime)

    async def both():
        texts = [f"text {n}" for n in range(10)]
        return await asyncio.gather(manager.generate_embeddings_batch(texts, max_concurrent=5),
                                    manager.generate_embeddings_batch(texts, max_concurrent=5))

    first, second = asyncio.run(both())

    assert len(first) == len(second) == 10 and runtime.calls == 20
    assert 1 < runtime.peak <= 3, "ten could be in flight per the two batches' own bounds"
    assert all(name.startswith("bedrock-embedding") for name in runtime.thread_names), runtime.thread_names


def test_the_bound_is_a_named_constant_and_the_pool_has_that_many_threads():
    assert calls.CALLS_IN_FLIGHT == calls._CALL_POOL._max_workers


# --------------------------------------------------------------------------- (b) refusals that will not get better


def test_a_refusal_that_will_not_get_better_raises_the_code_and_marks_it_degraded(monkeypatch):
    manager = _manager(_Runtime(error=DENIED))

    with pytest.raises(EmbeddingUnavailableError) as raised:
        asyncio.run(manager.generate_embedding("hello"))

    assert isinstance(raised.value, calls.BedrockEmbeddingUnavailable)
    assert "AccessDeniedException" in str(raised.value) and TITAN_TEXT_EMBEDDINGS_V2 in str(raised.value)
    assert "the role used in tests" not in str(raised.value) and raised.value.__cause__ is DENIED
    assert manager.provider.is_degraded is True and manager.provider.reason == "Bedrock refused it (AccessDeniedException)"
    assert provider_ready(manager) is False, "readiness says off, as with no provider"
    assert _manager(_Runtime()).provider.is_degraded is True, "every Bedrock provider in the process"


def test_it_stops_the_batch_and_cancels_the_calls_not_yet_made(monkeypatch):
    runtime = _Runtime(delay=0.05, error=DENIED, fail=lambda n: n == 0)
    manager = _manager(runtime)

    with pytest.raises(calls.BedrockEmbeddingUnavailable):
        asyncio.run(manager.generate_embeddings_batch([f"text {n}" for n in range(20)], max_concurrent=2))

    assert runtime.calls <= 3, f"{runtime.calls} of 20 texts were sent after the refusal"


def test_calls_are_held_back_after_it_and_the_next_success_clears_it(monkeypatch):
    runtime = _Runtime(error=DENIED, fail=lambda n: n == 0)
    manager = _manager(runtime)
    with pytest.raises(calls.BedrockEmbeddingUnavailable):
        asyncio.run(manager.generate_embedding("first"))
    assert manager.provider.is_degraded is True, "access denied marks it degraded"

    with pytest.raises(calls.BedrockEmbeddingUnavailable, match="AccessDeniedException"):
        asyncio.run(manager.generate_embedding("held"))
    assert runtime.calls == 1, "held back: no request"

    monkeypatch.setattr(calls, "REFUSAL_HOLD_S", 0.0)
    assert len(asyncio.run(manager.generate_embedding("after the hold"))) == 1024
    assert runtime.calls == 2 and manager.provider.is_degraded is False and provider_ready(manager) is True


def _validation(message: str) -> ClientError:
    return ClientError({"Error": {"Code": "ValidationException", "Message": message}}, "InvokeModel")


TOO_LONG = _validation("Input is too long for requested model.")
MODEL_NOT_OFFERED = _validation("The provided model identifier is invalid.")


async def _all_calls_made(runtime: _Runtime, wanted: int, within_s: float = 5.0) -> None:
    """Wait, in the batch's own loop, until ``wanted`` calls have been made, or ``within_s``."""
    deadline = time.monotonic() + within_s
    while runtime.calls < wanted and time.monotonic() < deadline:
        await asyncio.sleep(0.01)


def test_a_validation_error_on_one_text_of_a_batch_leaves_the_other_texts_calls_made():
    texts = ["a text past the model's input limit"] + [f"text {n}" for n in range(1, 8)]
    runtime = _Runtime(delay=0.05, error=TOO_LONG, fail_text=texts[0])
    manager = _manager(runtime)

    async def batch():
        with pytest.raises(calls.BedrockEmbeddingError) as raised:
            await manager.generate_embeddings_batch(texts, max_concurrent=2)
        await _all_calls_made(runtime, len(texts))
        return raised.value

    error = asyncio.run(batch())

    assert not isinstance(error, EmbeddingUnavailableError) and error.code == "ValidationException"
    assert runtime.calls == len(texts), "the batch's other texts were still sent: none cancelled"
    assert manager.provider.is_degraded is False and calls.HEALTH.holding() is None


def test_a_validation_error_on_one_text_leaves_the_provider_ready_and_starts_no_hold():
    runtime = _Runtime(error=TOO_LONG, fail_text="a text past the model's input limit")
    manager = _manager(runtime)
    with pytest.raises(calls.BedrockEmbeddingError):
        asyncio.run(manager.generate_embedding("a text past the model's input limit"))

    assert manager.provider.is_degraded is False and calls.HEALTH.holding() is None
    assert len(asyncio.run(manager.generate_embedding("the next text"))) == 1024 and runtime.calls == 2


def test_the_model_identifier_is_invalid_form_still_marks_it_degraded_and_holds():
    runtime = _Runtime(error=MODEL_NOT_OFFERED)
    manager = _manager(runtime)
    with pytest.raises(calls.BedrockEmbeddingUnavailable, match="ValidationException"):
        asyncio.run(manager.generate_embedding("hello"))

    assert manager.provider.is_degraded is True
    with pytest.raises(calls.BedrockEmbeddingUnavailable):
        asyncio.run(manager.generate_embedding("held"))
    assert runtime.calls == 1, "held back: no request"


@pytest.mark.parametrize("message", ["The provided model identifier is invalid.",
                                     "This model version has reached the end of its life.",
                                     "This model is marked by provider as Legacy."])
def test_each_form_the_chat_client_reads_as_a_model_not_served_is_final(message):
    assert isinstance(calls.refusal(_validation(message), TITAN_TEXT_EMBEDDINGS_V2), calls.BedrockEmbeddingUnavailable)


@pytest.mark.parametrize("code", sorted(calls.WILL_NOT_GET_BETTER))
def test_each_code_that_will_not_get_better_is_final(code):
    error = ClientError({"Error": {"Code": code, "Message": "x"}}, "InvokeModel")
    assert isinstance(calls.refusal(error, TITAN_TEXT_EMBEDDINGS_V2), calls.BedrockEmbeddingUnavailable)


@pytest.mark.parametrize("code", ["ThrottlingException", "ModelNotReadyException", "ModelTimeoutException",
                                  "ModelErrorException", "ServiceUnavailableException", "InternalServerException",
                                  "ServiceQuotaExceededException", "ValidationException"])
def test_each_code_that_may_get_better_is_not_final_and_leaves_it_ready(code):
    error = ClientError({"Error": {"Code": code, "Message": "x"}}, "InvokeModel")
    raised = calls.refusal(error, TITAN_TEXT_EMBEDDINGS_V2)
    assert type(raised) is calls.BedrockEmbeddingError and raised.code == code
    assert calls.HEALTH.code is None


# --------------------------------------------------------------------------- (c), (d) ingestion


class _Cache:
    """The embedding cache's four calls over a dict keyed by (model, text)."""

    def __init__(self, rows=None):
        self.rows = dict(rows or {})

    def get_embedding(self, text, model="default"):
        return self.rows.get((model, text))

    def set_embedding(self, text, embedding, model="default"):
        self.rows[(model, text)] = embedding

    def get_embeddings_batch(self, texts, model="default"):
        return {text: self.rows.get((model, text)) for text in texts}

    def set_embeddings_batch(self, embeddings, model="default"):
        self.rows.update({(model, text): emb for text, emb in embeddings.items()})


def _ingestion(monkeypatch, runtime: _Runtime, cache: _Cache):
    from modules.rag.ingestion.manager import DocumentManager

    monkeypatch.setattr(core.cache, "get_cache_service", lambda: cache)
    manager = DocumentManager.__new__(DocumentManager)
    manager.embedding_manager = _manager(runtime)
    one_at_a_time = []
    real = manager._generate_embedding

    async def sequential(text):
        one_at_a_time.append(text)
        return await real(text)

    monkeypatch.setattr(manager, "_generate_embedding", sequential)
    return manager, one_at_a_time


def test_ingestion_does_not_try_a_refusal_that_will_not_get_better_one_text_at_a_time(monkeypatch):
    manager, one_at_a_time = _ingestion(monkeypatch, _Runtime(error=DENIED), _Cache())

    with pytest.raises(EmbeddingUnavailableError):
        asyncio.run(manager._generate_embeddings_batch(["one chunk of text", "another chunk of text"]))

    assert one_at_a_time == []


def test_ingestion_keeps_its_sequential_pass_for_a_failure_that_may_get_better(monkeypatch):
    throttled = ClientError({"Error": {"Code": "ThrottlingException", "Message": "slow down"}}, "InvokeModel")
    manager, one_at_a_time = _ingestion(monkeypatch, _Runtime(error=throttled, fail=lambda n: n == 0), _Cache())

    vectors = asyncio.run(manager._generate_embeddings_batch(["one chunk of text"]))

    assert one_at_a_time == ["one chunk of text"] and len(vectors[0]) == 1024


def test_another_models_cached_vector_is_never_served_for_bedrocks(monkeypatch):
    text, query = "a chunk cached under the model used before the switch", "a query cached the same way"
    cache = _Cache({("default", text): [0.9] * 2048, ("openrouter:qwen/qwen3-embedding-8b:2048", text): [0.8] * 2048,
                    ("default", query): [0.9] * 2048})
    runtime = _Runtime()
    manager, _ = _ingestion(monkeypatch, runtime, cache)

    (vector,) = asyncio.run(manager._generate_embeddings_batch([text]))
    single = asyncio.run(manager._generate_embedding(query))

    assert vector == [0.25] * 1024 and single == [0.25] * 1024, "embedded by Bedrock, not cut from another model's"
    assert runtime.calls == 2
    assert cache.rows[(SPACE, text)] == [0.25] * 1024 and cache.rows[(SPACE, query)] == [0.25] * 1024
    assert cache.rows[("default", text)] == [0.9] * 2048, "the old entry is left alone, never read"


def test_a_vector_cached_in_bedrocks_own_space_is_served(monkeypatch):
    text = "a chunk Bedrock embedded before"
    runtime = _Runtime()
    manager, _ = _ingestion(monkeypatch, runtime, _Cache({(SPACE, text): [0.5] * 1024}))

    assert asyncio.run(manager._generate_embeddings_batch([text])) == [[0.5] * 1024] and runtime.calls == 0


def test_the_space_key_names_the_provider_model_and_size():
    assert _manager(_Runtime()).space_key() == SPACE


# --------------------------------------------------------------------------- (e) the code index


@pytest.mark.parametrize("provider,expected", [("bedrock", 1024), ("openrouter", 2048)])
def test_the_code_indexs_column_follows_bedrocks_vectors_not_the_stored_size(monkeypatch, provider, expected):
    from modules.codegraph.codegraph_service import CodeGraphService

    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", provider)
    monkeypatch.setattr(llm_manager, "get_system_setting",
                        lambda category, key, default=None: "2048" if (category, key) == ("embeddings", "dimensions")
                        else ("openrouter" if (category, key) == ("embeddings", "provider") else default))
    service = CodeGraphService.__new__(CodeGraphService)
    service.embedding_manager = SimpleNamespace(get_dimension=lambda: 1024)

    assert service._wanted_embedding_dimension() == expected
