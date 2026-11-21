"""PRD-278 US-501 — the Bedrock embedding provider.

``EMBEDDING_PROVIDER=bedrock`` selects ``BedrockEmbeddingProvider``, which embeds on
``bedrock-runtime`` (``InvokeModel``) with the Bedrock chat client's own credential handling:

  (a) the manager selects it, also over a seeded ``embeddings.provider`` row, and
      ``get_dimension()`` returns its dimension;
  (b) it signs with the pod's role in ``AWS_REGION``, or with a stored Bedrock key in the
      key's region, and its client is built by ``bedrock_client.runtime_client`` with exactly
      the credentials and region the chat client's settings tier gets: the embeddings tier's
      credential when it names one, else the one chat's tier finds, else the role;
  (c) the request is Titan Text Embeddings V2's documented body; the answer is checked; an
      empty or whitespace-only text is refused before any call;
  (d) a call runs in a worker thread, so a coroutine's loop keeps running, and a batch has
      at most ``max_concurrent`` calls in flight, each inside the embedding time limits;
  (e) a refusal that may get better (``ThrottlingException``) raises the provider's own error,
      with the code and the model only; nothing stands in for it.

``test_prd278_w5_bedrock_failures_and_cache.py`` holds the rest of the Wave 5 review: the
process-wide bound, refusals that will not get better, readiness, and the ingestion cache.

botocore's real request pipeline runs and stops just before sending (as in
``test_prd278_w1_bedrock_credential_sources.py``). The SDK's chain is given a key of its own
through the environment, standing in for the pod's role. No network, no database.
"""
from __future__ import annotations

import asyncio
import io
import json
import sys
import threading
import time
import types
import uuid
from types import SimpleNamespace

import boto3
import pytest
from botocore.awsrequest import AWSResponse
from botocore.exceptions import ClientError

import core.llm.clients.bedrock_client as bedrock_client
import core.llm.clients.bedrock_embedding as bedrock_embedding
import core.llm.clients.bedrock_embedding_calls as calls
import core.llm.embedding_manager as em
import core.llm.manager as llm_manager
from config import config
from core.llm.clients.base import EmbeddingConfig, EmbeddingProvider, EmbeddingUnavailableError
from core.llm.clients.bedrock_client import BedrockProvider
from core.llm.clients.bedrock_embedding import (
    TITAN_TEXT_EMBEDDINGS_V2,
    BedrockEmbeddingProvider,
    UnsupportedBedrockEmbeddingModel,
)
from core.llm.key_resolver import with_workload_identity

# Obviously fake credentials: the chain's (standing in for the pod's role) and a stored Bedrock API key.
ROLE_KEY_ID, ROLE_SECRET = "role-key-id-for-tests", "role-secret-for-tests"
STORED_API_KEY = "bedrock-api-key-for-tests"
INSTALL_REGION, KEY_REGION = "eu-west-1", "eu-central-1"


@pytest.fixture(autouse=True)
def chain_is_the_pod_role(monkeypatch, tmp_path):
    """The SDK's chain answers with the role's key; no profile, no stored key, no platform key."""
    for name in ("AWS_SESSION_TOKEN", "AWS_PROFILE", "AWS_BEARER_TOKEN_BEDROCK", "AWS_REGION", "AWS_DEFAULT_REGION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "none"))
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", ROLE_KEY_ID)
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", ROLE_SECRET)
    monkeypatch.setattr(boto3, "DEFAULT_SESSION", None)
    monkeypatch.setitem(sys.modules, "core.llm.workspace_keys",
                        types.SimpleNamespace(get_platform_workspace_key=lambda *_a, **_k: None))
    monkeypatch.setattr(llm_manager, "get_system_setting", lambda _c, _k, default=None: default)
    monkeypatch.setattr(config, "AWS_ACCESS_KEY_ID", None)
    monkeypatch.setattr(config, "AWS_SECRET_ACCESS_KEY", None)
    monkeypatch.setattr(config, "AWS_REGION", INSTALL_REGION)
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", None)
    monkeypatch.setattr(config, "EMBEDDING_MODEL", None)
    monkeypatch.setattr(config, "EMBEDDING_CONNECT_TIMEOUT_S", 5.0)
    monkeypatch.setattr(config, "EMBEDDING_QUERY_TIMEOUT_S", 15.0)
    monkeypatch.setattr(config, "EMBEDDING_BATCH_TIMEOUT_S", 60.0)
    monkeypatch.setattr(config, "EMBEDDING_MAX_RETRIES", 1)
    monkeypatch.setattr(bedrock_embedding, "record_embedding_usage", lambda *a, **k: None)
    calls.HEALTH.answered()
    yield
    calls.HEALTH.answered()


def _stored(monkeypatch, credential=None, *, identity: bool) -> None:
    """The stored Bedrock credential the settings tier finds (none: an empty dict)."""
    monkeypatch.setattr(config, "AWS_USE_WORKLOAD_IDENTITY", identity)
    monkeypatch.setattr(llm_manager, "get_credential_data", lambda *_a, **_k: dict(credential or {}))


def _provider(dimension: int = 1024, model: str = TITAN_TEXT_EMBEDDINGS_V2) -> BedrockEmbeddingProvider:
    return BedrockEmbeddingProvider(EmbeddingConfig(provider=EmbeddingProvider.BEDROCK, model=model,
                                                    dimension=dimension))


def _answer(size: int) -> dict:
    return {"embedding": [0.5] * size, "inputTextTokenCount": 2, "embeddingsByType": {"float": [0.5] * size}}


def _sent(provider: BedrockEmbeddingProvider, answer: dict):
    """One real InvokeModel request, stopped before it leaves: (the request, the vector)."""
    seen = []

    def send(request, **_kwargs):
        seen.append(request)
        return AWSResponse(request.url, 200, {"content-type": "application/json"},
                           io.BytesIO(json.dumps(answer).encode()))

    provider.client.meta.events.register("before-send.bedrock-runtime.InvokeModel", send)
    vector = asyncio.run(provider.generate_embedding("hello world"))
    return seen[0], vector


class _FakeRuntime:
    """A bedrock-runtime stand-in: each call sleeps ``delay`` in its thread and counts how many
    are in flight. The vector's numbers are the text's length, so order can be checked."""

    def __init__(self, delay: float = 0.0, size: int = 1024, error: Exception = None):
        self.delay, self.size, self.error = delay, size, error
        self.lock = threading.Lock()
        self.in_flight = self.peak = 0
        self.threads = set()

    def invoke_model(self, **kwargs):
        with self.lock:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            self.threads.add(threading.get_ident())
        try:
            time.sleep(self.delay)
            if self.error is not None:
                raise self.error
            text = json.loads(kwargs["body"])["inputText"]
            body = {"embedding": [float(len(text))] * self.size, "inputTextTokenCount": 1}
            return {"body": io.BytesIO(json.dumps(body).encode())}
        finally:
            with self.lock:
                self.in_flight -= 1


# --------------------------------------------------------------------------- (a) selection


def test_embedding_provider_bedrock_selects_the_bedrock_provider_over_a_seeded_row(monkeypatch):
    _stored(monkeypatch, identity=True)
    seeded = {("embeddings", "provider"): "openrouter", ("embeddings", "model"): "qwen/qwen3-embedding-8b",
              ("embeddings", "dimensions"): "2048"}
    monkeypatch.setattr(llm_manager, "get_system_setting", lambda c, k, default=None: seeded.get((c, k)) or default)
    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "bedrock")
    monkeypatch.setattr(config, "VECTOR_STORE_DIMENSIONS", 1024)

    manager = em.EmbeddingManager()
    manager._ensure_provider()

    assert isinstance(manager.provider, BedrockEmbeddingProvider)
    assert manager.provider.config.model == TITAN_TEXT_EMBEDDINGS_V2, "EMBEDDING_MODEL unset: Titan V2"
    assert manager.provider.client.meta.service_model.service_name == "bedrock-runtime"
    assert manager.get_dimension() == 1024
    assert em.configured_provider() == "bedrock"


@pytest.mark.parametrize("configured,asked,returned", [(1024, 1024, 1024), (512, 512, 512), (256, 256, 256),
                                                         (768, 1024, 768), (2048, 1024, 1024)])
def test_it_reports_its_dimension_and_asks_for_the_size_that_holds_it(monkeypatch, configured, asked, returned):
    _stored(monkeypatch, identity=True)
    provider = _provider(dimension=configured)

    request, vector = _sent(provider, _answer(asked))

    assert json.loads(request.body)["dimensions"] == asked
    assert provider.get_dimension() == returned == len(vector)


def test_a_model_it_does_not_support_is_refused_with_the_list(monkeypatch):
    _stored(monkeypatch, identity=True)

    with pytest.raises(UnsupportedBedrockEmbeddingModel) as refused:
        _provider(model="cohere.embed-english-v3")

    assert "cohere.embed-english-v3" in str(refused.value) and TITAN_TEXT_EMBEDDINGS_V2 in str(refused.value)


def test_the_consent_list_names_aws_bedrock(monkeypatch):
    from services import ai_providers

    monkeypatch.setattr(config, "EMBEDDING_PROVIDER", "bedrock")

    assert ai_providers._embeddings(None, uuid.uuid4()) == {"bedrock": "AWS Bedrock"}


# --------------------------------------------------------------------------- (b) credentials and region


def test_on_the_pods_role_it_signs_with_the_chain_in_the_installs_region(monkeypatch):
    _stored(monkeypatch, identity=True)

    request, _ = _sent(_provider(), _answer(1024))

    assert f"bedrock-runtime.{INSTALL_REGION}.amazonaws.com" in request.url
    assert f"Credential={ROLE_KEY_ID}/" in request.headers["Authorization"].decode()


@pytest.mark.parametrize("identity", [True, False])
def test_with_a_stored_bedrock_key_it_sends_the_key_in_the_keys_region(monkeypatch, identity):
    _stored(monkeypatch, {"bedrock_api_key": STORED_API_KEY, "aws_region": KEY_REGION}, identity=identity)

    request, _ = _sent(_provider(), _answer(1024))

    assert f"bedrock-runtime.{KEY_REGION}.amazonaws.com" in request.url
    assert request.headers["Authorization"].decode() == f"Bearer {STORED_API_KEY}"


def _runtime_calls(monkeypatch) -> list:
    calls = []
    monkeypatch.setattr(bedrock_client, "runtime_client",
                        lambda *args, **kwargs: calls.append((args, kwargs)) or SimpleNamespace())
    return calls


@pytest.mark.parametrize("identity,credential", [
    (True, None),
    (True, {"bedrock_api_key": STORED_API_KEY, "aws_region": KEY_REGION}),
    (True, {"bedrock_api_key": STORED_API_KEY}),
    (False, {"bedrock_api_key": STORED_API_KEY, "aws_region": KEY_REGION}),
    (False, None),
])
def test_its_client_gets_exactly_what_the_chat_clients_settings_tier_gets(monkeypatch, identity, credential):
    """The same settings read and the same client factory as chat: no second way to find either."""
    _stored(monkeypatch, credential, identity=identity)
    built = _runtime_calls(monkeypatch)

    chat_config = with_workload_identity(llm_manager.LLMManager._load_config_from_settings(
        SimpleNamespace(), "orchestrator", "aws_bedrock", TITAN_TEXT_EMBEDDINGS_V2))
    BedrockProvider(chat_config)
    _provider()

    (chat_args, chat_kwargs), (own_args, own_kwargs) = built
    assert own_args == chat_args, "api key, secret and region"
    assert own_kwargs["use_role"] == chat_kwargs.get("use_role", False)
    assert own_kwargs["extra"] is not None and "extra" not in chat_kwargs, "only the time limits are its own"


def _tiers(monkeypatch, mappings: dict, found: dict, *, identity: bool) -> list:
    """Each tier's credential mapping (``credential_name_aws_bedrock``) and what each tier's lookup
    finds; returns the tiers the lookup was asked for."""
    asked = []
    monkeypatch.setattr(config, "AWS_USE_WORKLOAD_IDENTITY", identity)
    monkeypatch.setattr(llm_manager, "get_system_setting",
                        lambda category, key, default=None: mappings.get(category, default)
                        if key == "credential_name_aws_bedrock" else default)

    def lookup(provider, environment=None, service_name="orchestrator"):
        asked.append(service_name)
        return dict(found.get(service_name) or {})

    monkeypatch.setattr(llm_manager, "get_credential_data", lookup)
    return asked


EMBEDDINGS_KEY = {"bedrock_api_key": "embeddings-tier-key-for-tests", "aws_region": "eu-north-1"}
CHAT_KEY = {"bedrock_api_key": STORED_API_KEY, "aws_region": KEY_REGION}


@pytest.mark.parametrize("identity", [True, False])
def test_a_credential_the_embeddings_tier_names_comes_first(monkeypatch, identity):
    asked = _tiers(monkeypatch, {"embeddings": "bedrock-for-embeddings", "orchestrator_llm": "bedrock-for-chat"},
                   {"embeddings": EMBEDDINGS_KEY, "orchestrator": CHAT_KEY}, identity=identity)
    built = _runtime_calls(monkeypatch)

    _provider()

    assert asked == ["embeddings"]
    assert built[0][0] == ("embeddings-tier-key-for-tests", None, "eu-north-1")


@pytest.mark.parametrize("identity", [True, False])
def test_with_no_mapping_of_its_own_it_uses_the_credential_chats_tier_finds(monkeypatch, identity):
    asked = _tiers(monkeypatch, {"orchestrator_llm": "bedrock-for-chat"},
                   {"embeddings": EMBEDDINGS_KEY, "orchestrator": CHAT_KEY}, identity=identity)
    built = _runtime_calls(monkeypatch)

    _provider()

    assert asked == ["orchestrator"], "chat's lookup, not a guess of its own"
    assert built[0][0] == (STORED_API_KEY, None, KEY_REGION)


def test_with_no_mapping_and_no_credential_found_on_the_role_it_uses_the_role(monkeypatch):
    asked = _tiers(monkeypatch, {}, {}, identity=True)
    built = _runtime_calls(monkeypatch)

    _provider()

    assert asked == ["orchestrator"]
    assert built[0][0] == (None, None, INSTALL_REGION) and built[0][1]["use_role"] is True


def test_with_no_mapping_and_the_role_off_it_runs_chats_lookup(monkeypatch):
    """Chat's tier's lookup: its mapping, then the shared name guesses and the first credential
    of the type (``get_credential_data``), exactly as chat runs it."""
    asked = _tiers(monkeypatch, {}, {"orchestrator": CHAT_KEY}, identity=False)
    built = _runtime_calls(monkeypatch)

    _provider()

    assert asked == ["orchestrator"]
    assert built[0][0] == (STORED_API_KEY, None, KEY_REGION) and built[0][1]["use_role"] is False


def test_its_client_carries_the_embedding_time_limits(monkeypatch):
    _stored(monkeypatch, identity=True)

    client_config = _provider().client.meta.config

    assert client_config.connect_timeout == 5.0 and client_config.read_timeout == 15.0
    assert client_config.retries["total_max_attempts"] == 2 and client_config.retries["mode"] == "adaptive"
    assert client_config.region_name == INSTALL_REGION


# --------------------------------------------------------------------------- (c) the request and the answer


def test_the_request_is_titan_v2s_documented_body(monkeypatch):
    _stored(monkeypatch, identity=True)

    request, vector = _sent(_provider(), _answer(1024))

    assert "/model/amazon.titan-embed-text-v2" in request.url and request.url.endswith("/invoke")
    assert json.loads(request.body) == {"inputText": "hello world", "dimensions": 1024, "normalize": True}
    assert request.headers["Content-Type"].decode() == "application/json"
    assert vector == [0.5] * 1024


def test_text_past_the_models_limit_is_cut_before_the_call(monkeypatch):
    _stored(monkeypatch, identity=True)
    provider = _provider()
    provider.client = _FakeRuntime()
    sent = []
    monkeypatch.setattr(provider.client, "invoke_model",
                        lambda **kw: sent.append(json.loads(kw["body"])["inputText"]) or
                        {"body": io.BytesIO(json.dumps(_answer(1024)).encode())})

    asyncio.run(provider.generate_embedding("x" * 40_000))

    # 8,192 tokens less a margin, counted as characters: safe where a character is a token.
    assert len(sent[0]) == 8_000 == provider.model.max_chars < provider.model.max_input_tokens


@pytest.mark.parametrize("answer", [{}, {"embedding": [0.5] * 10}, {"embedding": ["a"] * 1024}, []])
def test_an_answer_without_the_asked_embedding_raises(monkeypatch, answer):
    _stored(monkeypatch, identity=True)

    with pytest.raises(ValueError, match="without an embedding of 1024 numbers"):
        _sent(_provider(), answer)


@pytest.mark.parametrize("text", ["", "   ", "\n\t "])
def test_an_empty_or_blank_text_is_refused_without_a_call(monkeypatch, text):
    _stored(monkeypatch, identity=True)
    provider = _provider()
    provider.client = _FakeRuntime()

    with pytest.raises(ValueError, match="empty or whitespace-only"):
        asyncio.run(provider.generate_embedding(text))
    with pytest.raises(ValueError, match=r"positions \[1\]"):
        asyncio.run(provider.generate_embeddings_batch(["a text", text, "another"]))
    assert provider.client.threads == set(), "no call for any text of the batch"


# --------------------------------------------------------------------------- (d) off the loop, bounded


def test_a_call_from_a_coroutine_runs_in_a_thread_and_the_loop_keeps_running(monkeypatch):
    _stored(monkeypatch, identity=True)
    provider = _provider()
    provider.client = _FakeRuntime(delay=0.3)

    async def scenario():
        ticks, done = [], asyncio.Event()

        async def ticker():
            while not done.is_set():
                ticks.append(time.monotonic())
                await asyncio.sleep(0.01)

        task = asyncio.create_task(ticker())
        vector = await provider.generate_embedding("hello")
        done.set()
        await task
        return ticks, vector

    ticks, vector = asyncio.run(scenario())

    assert len(ticks) >= 10, "the loop was held while the call ran"
    assert threading.get_ident() not in provider.client.threads
    assert len(vector) == 1024


def test_a_batch_has_at_most_max_concurrent_calls_in_flight_and_keeps_its_order(monkeypatch):
    _stored(monkeypatch, identity=True)
    provider = _provider()
    provider.client = _FakeRuntime(delay=0.1)
    manager = em.EmbeddingManager()
    manager.provider, manager._provider_loaded = provider, True
    texts = ["x" * n for n in range(1, 13)]

    vectors = asyncio.run(manager.generate_embeddings_batch(texts, max_concurrent=3))

    assert 1 < provider.client.peak <= 3
    assert [vector[0] for vector in vectors] == [float(len(text)) for text in texts]


def test_a_call_that_outlasts_the_batch_limit_raises(monkeypatch):
    _stored(monkeypatch, identity=True)
    monkeypatch.setattr(config, "EMBEDDING_BATCH_TIMEOUT_S", 0.05)
    provider = _provider()
    provider.client = _FakeRuntime(delay=0.5)

    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(provider.generate_embedding("hello"))


# --------------------------------------------------------------------------- (e) a refusal raises


def test_a_throttled_call_raises_and_nothing_stands_in(monkeypatch):
    _stored(monkeypatch, identity=True)
    provider = _provider()
    throttled = ClientError({"Error": {"Code": "ThrottlingException", "Message": "Too many requests"}}, "InvokeModel")
    provider.client = _FakeRuntime(error=throttled)
    manager = em.EmbeddingManager()
    manager.provider, manager._provider_loaded = provider, True

    with pytest.raises(calls.BedrockEmbeddingError) as raised:
        asyncio.run(manager.generate_embedding("hello"))
    assert str(raised.value) == f"Bedrock embedding with '{TITAN_TEXT_EMBEDDINGS_V2}' failed: ThrottlingException"
    assert raised.value.__cause__ is throttled, "AWS's own error is kept as the cause, not in the message"
    assert not isinstance(raised.value, EmbeddingUnavailableError), "it may get better"
    with pytest.raises(calls.BedrockEmbeddingError):
        asyncio.run(manager.generate_embeddings_batch(["a", "b"]))
    assert provider.is_degraded is False
