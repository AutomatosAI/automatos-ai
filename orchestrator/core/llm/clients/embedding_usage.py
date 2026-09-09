"""Book an embeddings call in ``llm_usage`` (2026-09-09 analytics).

Embeddings were the highest-volume UNTRACKED calls in the platform: the
tool-routing embed on every chat turn, every RAG query, every ingested chunk,
every memory write. The API reports ``usage.prompt_tokens``; when a provider
omits it the call is still recorded, with the ~4-chars-per-token estimate the
clients already use for truncation (so a row is never silently missing).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Sequence

logger = logging.getLogger(__name__)

CHARS_PER_TOKEN_ESTIMATE = 4


def _prompt_tokens(response: Any, texts: Sequence[str]) -> int:
    usage = getattr(response, "usage", None)
    reported = int(getattr(usage, "prompt_tokens", 0) or getattr(usage, "total_tokens", 0) or 0) if usage else 0
    if reported > 0:
        return reported
    return sum(len(t) for t in texts) // CHARS_PER_TOKEN_ESTIMATE


def record_embedding_usage(provider: str, model_id: str, response: Any, texts: Sequence[str], started: float) -> None:
    """Never raises — a usage row is never worth failing the embedding."""
    try:
        from core.llm.usage_tracker import UsageTracker

        UsageTracker.track_embedding(
            provider=provider,
            model_id=model_id,
            prompt_tokens=_prompt_tokens(response, texts),
            latency_ms=int((time.monotonic() - started) * 1000),
            inputs=len(texts),
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("embedding usage not recorded: %s", exc)
