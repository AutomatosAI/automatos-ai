"""PRD-185 S3: embeddings fail loud instead of returning random vectors.

``DeterministicEmbeddingProvider`` used to return hash-seeded random vectors when
no real provider was configured/reachable, so similarity search ran over noise
and returned confident-but-meaningless matches — plausibly since the ~06-16
OpenRouter outage, with nothing on any dashboard. It now raises
``EmbeddingUnavailableError`` so selection paths return a typed EMPTY result
(the RAG candidate path already returns [] on exception) instead of noise.

Pure unit test — no DB / network.
"""
import pytest


def _load():
    try:
        from core.llm.clients.base import (
            DeterministicEmbeddingProvider,
            EmbeddingUnavailableError,
        )
    except Exception as e:  # env without core.llm deps
        pytest.skip(f"core.llm.clients.base not importable in this env: {e}")
    return DeterministicEmbeddingProvider, EmbeddingUnavailableError


def test_degraded_provider_is_marked():
    Det, _ = _load()
    assert getattr(Det(dimension=8), "is_degraded", False) is True


@pytest.mark.asyncio
async def test_generate_embedding_raises_not_random():
    Det, EmbeddingUnavailableError = _load()
    p = Det(dimension=8)
    with pytest.raises(EmbeddingUnavailableError):
        await p.generate_embedding("hello world")
