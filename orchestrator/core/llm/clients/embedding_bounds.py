"""F105 (night 3): an embedding call has a bound.

Both OpenAI-SDK embedding clients were built with the SDK's defaults — a
600-second read timeout and 2 retries — so a stalled call to the provider could
hold a search, and whatever it had checked out, for up to half an hour. The
clients now take their connect and read bounds and their retry count from
config: a single text (a query) gets the short read bound, a batch the long
one. When a bound is hit, the log line names it and how long the call took —
far longer than the bound means the event loop itself was frozen.
"""
from __future__ import annotations

import time
from typing import Any, Dict

import httpx

from config import config


def client_kwargs() -> Dict[str, Any]:
    """Timeout and retries for an AsyncOpenAI embedding client — the batch bound."""
    return {
        "timeout": httpx.Timeout(config.EMBEDDING_BATCH_TIMEOUT_S, connect=config.EMBEDDING_CONNECT_TIMEOUT_S),
        "max_retries": config.EMBEDDING_MAX_RETRIES,
    }


def single_text_timeout() -> httpx.Timeout:
    """The per-request bound for embedding one text."""
    return httpx.Timeout(config.EMBEDDING_QUERY_TIMEOUT_S, connect=config.EMBEDDING_CONNECT_TIMEOUT_S)


def timeout_note(started: float, *, single: bool) -> str:
    """How long a timed-out call took against its bound, for the log line."""
    read = config.EMBEDDING_QUERY_TIMEOUT_S if single else config.EMBEDDING_BATCH_TIMEOUT_S
    retries = config.EMBEDDING_MAX_RETRIES
    return (f"timed out after {time.monotonic() - started:.1f}s (bound: connect "
            f"{config.EMBEDDING_CONNECT_TIMEOUT_S:g}s, read {read:g}s, {retries} "
            f"{'retry' if retries == 1 else 'retries'})")
