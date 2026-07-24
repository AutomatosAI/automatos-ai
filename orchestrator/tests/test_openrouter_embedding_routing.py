"""OpenRouter embedding provider-routing preferences (latency sort).

OpenRouter routes embeddings by PRICE by default, which let the slowest
upstream serve qwen3-embedding-8b at 37-67s/call (measured 2026-07-09).
These tests pin the fix: every embeddings.create call carries
``extra_body={"provider": {"sort": <config>}}``, and an empty config value
omits the preference entirely (extra_body=None).

Pure unit tests — the AsyncOpenAI client is replaced with a capture fake;
no network. The real ``config`` singleton attribute is snapshotted and
restored around every test (never leave real-module attrs clobbered).
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from config import config
from core.llm.clients.base import EmbeddingConfig, EmbeddingProvider
from core.llm.clients.openrouter_embedding import OpenRouterEmbeddingProvider

_SORT_ATTR = "OPENROUTER_EMBEDDING_PROVIDER_SORT"
_SENTINEL = object()


@pytest.fixture
def sort_setting():
    """Set config.OPENROUTER_EMBEDDING_PROVIDER_SORT for one test, restore after."""
    prior = getattr(config, _SORT_ATTR, _SENTINEL)

    def _set(value):
        setattr(config, _SORT_ATTR, value)

    yield _set
    if prior is _SENTINEL:
        if hasattr(config, _SORT_ATTR):
            delattr(config, _SORT_ATTR)
    else:
        setattr(config, _SORT_ATTR, prior)


class _CaptureEmbeddings:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        inputs = kwargs["input"]
        n = len(inputs) if isinstance(inputs, list) else 1
        return SimpleNamespace(
            data=[SimpleNamespace(embedding=[0.1] * 8, index=i) for i in range(n)]
        )


def _make_provider() -> tuple[OpenRouterEmbeddingProvider, _CaptureEmbeddings]:
    provider = OpenRouterEmbeddingProvider(
        EmbeddingConfig(
            provider=EmbeddingProvider.OPENROUTER,
            model="qwen/qwen3-embedding-8b",
            dimension=8,
            api_key="test-key",
        )
    )
    capture = _CaptureEmbeddings()
    provider._client_for_loop = lambda: SimpleNamespace(embeddings=capture)
    return provider, capture


def test_single_embed_carries_latency_sort(sort_setting):
    sort_setting("latency")
    provider, capture = _make_provider()
    vec = asyncio.run(provider.generate_embedding("hello"))
    assert len(vec) == 8
    assert capture.calls[0]["extra_body"] == {"provider": {"sort": "latency"}}


def test_batch_embed_carries_latency_sort(sort_setting):
    sort_setting("latency")
    provider, capture = _make_provider()
    vecs = asyncio.run(provider.generate_embeddings_batch(["a", "b"]))
    assert len(vecs) == 2
    assert capture.calls[0]["extra_body"] == {"provider": {"sort": "latency"}}


def test_empty_sort_omits_provider_preferences(sort_setting):
    sort_setting("")
    provider, capture = _make_provider()
    asyncio.run(provider.generate_embedding("hello"))
    assert capture.calls[0]["extra_body"] is None


def test_custom_sort_value_passes_through(sort_setting):
    sort_setting("throughput")
    provider, capture = _make_provider()
    asyncio.run(provider.generate_embedding("hello"))
    assert capture.calls[0]["extra_body"] == {"provider": {"sort": "throughput"}}
