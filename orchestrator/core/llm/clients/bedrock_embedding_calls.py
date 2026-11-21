"""How Bedrock embedding calls run, fail and recover (PRD-278 US-501, Wave 5 review).

- **Their own threads, a process-wide bound.** Every Bedrock embedding call in this process
  runs in one pool of ``CALLS_IN_FLIGHT`` threads (``in_pool``), never in the default
  executor that every other blocking call shares. However many documents ingest at once, and
  while start-up warms an index, at most ``CALLS_IN_FLIGHT`` calls are in flight. A call
  waiting for a thread has not started: its time limit starts with it.
- **Two kinds of refusal.** An error code that will not get better without a change to the
  role, the key, the region or the model (``WILL_NOT_GET_BETTER``) raises
  ``BedrockEmbeddingUnavailable``, an ``EmbeddingUnavailableError``: the batch it is part of
  stops, and the ingestion fallbacks do not try again. Any other (throttling, a model or
  service error, a lost connection) raises ``BedrockEmbeddingError``, as a provider's error
  raised before.
- **Messages carry the code and the model only.** AWS's own message can name a role and an
  account, so it stays on the original exception, kept as the cause.
- **Readiness.** After a refusal that will not get better, every Bedrock embedding provider in
  this process reports itself degraded, with the code (``BedrockEmbeddingHealth``): readiness
  and memory say what they say with no provider. For ``REFUSAL_HOLD_S`` its calls fail at once,
  without a request; the first call that succeeds after that clears it.
"""
from __future__ import annotations

import asyncio
import contextvars
import functools
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Optional, TypeVar

from .base import EmbeddingUnavailableError

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

# At most this many Bedrock embedding calls in flight in one process, across every batch.
CALLS_IN_FLIGHT = 10
# After a refusal that will not get better, calls fail at once for this long, then try again.
REFUSAL_HOLD_S = 30.0
# InvokeModel's errors that the same request will meet again until an operator changes the
# role, the key, the region or the model (docs.aws.amazon.com/bedrock/latest/APIReference/
# API_runtime_InvokeModel.html, and userguide/troubleshooting-api-error-codes.html).
# ValidationException is here: the request itself is refused, also for a model ID the region
# does not offer. Throttling, ModelNotReady, ModelTimeout, ModelError, ServiceUnavailable,
# InternalServer and ServiceQuotaExceeded ("resubmit later") may get better and are not.
WILL_NOT_GET_BETTER = frozenset({
    "AccessDeniedException", "ResourceNotFoundException", "ValidationException",
    "UnrecognizedClientException", "InvalidClientTokenId", "InvalidSignatureException",
    "IncompleteSignature", "NotAuthorized", "ExpiredTokenException",
})

_CALL_POOL = ThreadPoolExecutor(max_workers=CALLS_IN_FLIGHT, thread_name_prefix="bedrock-embedding")


class BedrockEmbeddingError(RuntimeError):
    """A Bedrock embedding call failed in a way that may get better: the code and the model."""

    def __init__(self, code: str, model_id: str):
        super().__init__(f"Bedrock embedding with '{model_id}' failed: {code}")
        self.code = code


class BedrockEmbeddingUnavailable(EmbeddingUnavailableError):
    """Bedrock refused an embedding call in a way that will not get better: the code and the model."""

    def __init__(self, code: str, model_id: str):
        super().__init__(f"Bedrock refused embedding with '{model_id}': {code}. It will not get better "
                         f"until the role, the key, the region or the model is changed.")
        self.code = code


class BedrockEmbeddingHealth:
    """The last refusal that will not get better, for every Bedrock embedding provider in this
    process: they share the settings, so they share the refusal. Thread-safe."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._code: Optional[str] = None
        self._at = 0.0

    def refused(self, code: str) -> None:
        with self._lock:
            self._code, self._at = code, time.monotonic()

    def answered(self) -> None:
        with self._lock:
            self._code = None

    @property
    def code(self) -> Optional[str]:
        """The refusal's code while it stands, else None."""
        with self._lock:
            return self._code

    def holding(self) -> Optional[str]:
        """The refusal's code while calls are held back (``REFUSAL_HOLD_S`` after it), else None."""
        with self._lock:
            held = self._code is not None and time.monotonic() - self._at < REFUSAL_HOLD_S
            return self._code if held else None


HEALTH = BedrockEmbeddingHealth()


def error_code(exc: BaseException) -> str:
    """AWS's error code for a refusal (never its message), else the exception's class."""
    response = getattr(exc, "response", None)
    code = response.get("Error", {}).get("Code") if isinstance(response, dict) else None
    return str(code) if code else type(exc).__name__


def refusal(exc: BaseException, model_id: str) -> RuntimeError:
    """The error to raise for a failed call: ``BedrockEmbeddingUnavailable`` (and the shared
    health marked) for a code that will not get better, else ``BedrockEmbeddingError``."""
    code = error_code(exc)
    if code in WILL_NOT_GET_BETTER:
        HEALTH.refused(code)
        return BedrockEmbeddingUnavailable(code, model_id)
    return BedrockEmbeddingError(code, model_id)


async def in_pool(fn: Callable[..., _T], *args: Any, timeout: float) -> _T:
    """``fn(*args)`` in the Bedrock embedding pool, in a copy of the caller's context, awaited
    for at most ``timeout`` seconds from the moment a thread takes it."""
    loop = asyncio.get_running_loop()
    started = asyncio.Event()

    def run() -> _T:
        loop.call_soon_threadsafe(started.set)
        return fn(*args)

    future = loop.run_in_executor(_CALL_POOL, functools.partial(contextvars.copy_context().run, run))
    try:
        await started.wait()
    except asyncio.CancelledError:
        future.cancel()  # not started: it never will be
        raise
    return await asyncio.wait_for(future, timeout=timeout)


async def cancel_all(tasks: list) -> None:
    """Cancel ``tasks`` and wait for them to finish; their results and errors are dropped."""
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


__all__ = [
    "CALLS_IN_FLIGHT", "HEALTH", "REFUSAL_HOLD_S", "WILL_NOT_GET_BETTER", "BedrockEmbeddingError",
    "BedrockEmbeddingHealth", "BedrockEmbeddingUnavailable", "cancel_all", "error_code", "in_pool", "refusal",
]
