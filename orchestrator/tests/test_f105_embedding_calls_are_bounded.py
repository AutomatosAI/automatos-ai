"""F105 (night 3) — an embedding call has a bound.

Night 3: both embedding clients ran on the OpenAI SDK defaults (600 s read, 2
retries), so a stalled provider call could hold a search for minutes — "The
Salt Loft" spent 63 s in three attempts before failing. The clients now take
their bounds from config, one text gets the short read bound, and the log line
says which bound was hit and how long the call really took.
"""
from __future__ import annotations

import asyncio
import logging
import time
from types import SimpleNamespace

import pytest

from config import config
from core.llm.clients.base import EmbeddingConfig, EmbeddingProvider


def _openrouter(dimension=8):
    from core.llm.clients.openrouter_embedding import OpenRouterEmbeddingProvider

    return OpenRouterEmbeddingProvider(EmbeddingConfig(
        provider=EmbeddingProvider.OPENROUTER, model="qwen/qwen3-embedding-8b", dimension=dimension,
        api_key="test-key"))


def test_both_clients_are_built_with_the_configured_bounds():
    from core.llm.clients.openai_embedding import OpenAIEmbeddingProvider

    async def openrouter_client():
        return _openrouter()._client_for_loop()

    openai_client = OpenAIEmbeddingProvider(EmbeddingConfig(
        provider=EmbeddingProvider.OPENAI, model="text-embedding-3-small", dimension=8, api_key="test-key")).client
    for client in (asyncio.run(openrouter_client()), openai_client):
        assert client.timeout.read == config.EMBEDDING_BATCH_TIMEOUT_S
        assert client.timeout.connect == config.EMBEDDING_CONNECT_TIMEOUT_S
        assert client.max_retries == config.EMBEDDING_MAX_RETRIES == 1


def test_one_text_carries_the_short_bound():
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1] * 8, index=0)])

    provider = _openrouter()
    provider._client_for_loop = lambda: SimpleNamespace(embeddings=SimpleNamespace(create=create))
    asyncio.run(provider.generate_embedding("The Salt Loft"))
    assert calls[0]["timeout"].read == config.EMBEDDING_QUERY_TIMEOUT_S < config.EMBEDDING_BATCH_TIMEOUT_S


def test_a_stalled_provider_is_cut_off_at_the_bound_and_the_log_names_it(monkeypatch, caplog):
    """A provider that accepts the call and never answers: two attempts (one
    retry) at the read bound, then an error that says so."""
    from openai import APITimeoutError

    monkeypatch.setattr(config, "EMBEDDING_QUERY_TIMEOUT_S", 0.3)
    monkeypatch.setattr(config, "EMBEDDING_CONNECT_TIMEOUT_S", 0.3)
    requests = []

    async def never_answers(reader, writer):
        requests.append(await reader.read(65536))
        await asyncio.sleep(30)

    async def main():
        server = await asyncio.start_server(never_answers, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        monkeypatch.setattr(config, "OPENROUTER_BASE_URL", f"http://127.0.0.1:{port}/api/v1")
        started = time.monotonic()
        try:
            with pytest.raises(APITimeoutError):
                await _openrouter().generate_embedding("moreish")
            return time.monotonic() - started
        finally:
            server.close()

    with caplog.at_level(logging.ERROR, logger="core.llm.clients.openrouter_embedding"):
        elapsed = asyncio.run(main())
    assert len(requests) == 2                                    # the call and its one retry
    assert elapsed < 3                                           # not 3 x 600 s
    assert "timed out after" in caplog.text and "read 0.3s, 1 retry" in caplog.text
