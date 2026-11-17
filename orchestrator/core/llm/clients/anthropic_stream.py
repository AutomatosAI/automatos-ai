"""The worker-thread reader of one Anthropic Messages stream (voice latency B-2).

``AnthropicProvider.stream_response`` runs :func:`read_stream` in a worker thread: it opens the
SDK's stream, hands each text delta to ``emit`` as it arrives, and builds the reply from the
message the SDK accumulated, with the same function the whole (non-streamed) call uses. A stream
that opened and then broke is one typed error, :class:`AnthropicStreamInterrupted`, whose message
is a fixed sentence: it is shown to the owner and read aloud on a spoken turn, so it never carries
the provider's text.

Nothing here imports the SDK: the stream is whatever ``open_stream()`` returns, so the reader is
tested with fakes. A refusal before the stream opens (a 4xx, a 429, a 5xx) is the SDK's own error
and is not wrapped.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Optional, Tuple

from .base import LLMResponse

logger = logging.getLogger(__name__)

# What the owner reads, and a spoken turn says, when a stream that had opened broke.
INTERRUPTED = "The AI provider's stream ended early. Ask again."
# The log's ``why`` for a stream that ended with no stop reason (its body was cut short).
NO_STOP_REASON = "no stop reason"
# The SDK's stream event that carries a text delta (``anthropic.lib.streaming.TextEvent``).
TEXT_EVENT = "text"
# The packages whose errors are a provider's or a transport's own words: only theirs are logged
# as ``detail``. Any other error (the SDK's "unable to parse tool JSON" is a plain ValueError)
# can quote what the model wrote, so it is logged by its class name only.
_PROVIDER_ERROR_PACKAGES = frozenset({"anthropic", "httpx", "httpcore"})
_DETAIL_CHARS = 300
_NO_DETAIL = "-"


class AnthropicStreamInterrupted(RuntimeError):
    """The stream opened (the request was accepted) and then broke: an ``error`` event, a dropped
    connection, or an end with no stop reason. Its message is the fixed sentence INTERRUPTED: it
    is shown to the owner and read aloud on a spoken turn, so it never carries the provider's
    text. ``why`` is the cause in a few words for the log; ``partial_response`` is what had
    arrived (text, input and cache token counts), or None."""

    def __init__(self, why: str, *, partial_response: Optional[LLMResponse] = None) -> None:
        super().__init__(INTERRUPTED)
        self.why, self.partial_response = why, partial_response


def spent(partial: Optional[LLMResponse]) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """The fresh input, cache-read and cache-write token counts a broken stream had reported,
    for the log; three Nones when nothing had arrived."""
    usage = getattr(partial, "usage", None) or {}
    if not usage:
        return None, None, None
    cache_read, cache_write = usage.get("cache_read_tokens", 0), usage.get("cache_write_tokens", 0)
    return usage.get("prompt_tokens", 0) - cache_read - cache_write, cache_read, cache_write


def _partial(stream: Any, to_response: Callable[[Any], LLMResponse]) -> Optional[LLMResponse]:
    """What had arrived when the stream broke; None when that cannot be read (the SDK's snapshot
    asserts before its first event), so a bad snapshot never masks the interruption."""
    try:
        return to_response(stream.current_message_snapshot)
    except Exception as exc:  # noqa: BLE001 — logged; the interruption itself is what is raised
        logger.debug("[anthropic] a broken stream left no readable snapshot (%s)", type(exc).__name__)
        return None


def _detail(cause: Optional[BaseException]) -> str:
    """The cause's own text for the log, cut short: only for a provider's or a transport's error."""
    if cause is None or type(cause).__module__.split(".")[0] not in _PROVIDER_ERROR_PACKAGES:
        return _NO_DETAIL
    return str(cause)[:_DETAIL_CHARS]


def _interrupted(why: str, cause: Optional[BaseException], stream: Any,
                 to_response: Callable[[Any], LLMResponse]) -> AnthropicStreamInterrupted:
    """The typed error for a stream that broke, logged once with what it had spent. The
    provider's raw text lives in this log line and in ``__cause__``, nowhere else."""
    partial = _partial(stream, to_response)
    fresh, cache_read, cache_write = spent(partial)
    logger.warning("[anthropic] stream interrupted why=%s detail=%s input=%s cache_read=%s cache_write=%s",
                   why, _detail(cause), fresh, cache_read, cache_write)
    return AnthropicStreamInterrupted(why, partial_response=partial)


def read_stream(open_stream: Callable[[], Any], emit: Callable[[str], None], stopped: Callable[[], bool],
                to_response: Callable[[Any], LLMResponse]) -> Optional[LLMResponse]:
    """Read one Messages stream to its end in the calling (worker) thread.

    ``open_stream()`` returns the SDK's stream manager; entering it makes the request, and an
    error there is the SDK's own (as ``create`` raises it). Each text delta goes to ``emit``.
    Returns ``to_response(final message)``; None when ``stopped()`` said to stop. Leaving the
    ``with`` closes the response (which ends the generation), never the client: the client may
    be shared between calls (``LLM_CLIENT_REUSE``).
    """
    with open_stream() as stream:
        try:
            for event in stream:
                if stopped():
                    return None
                if getattr(event, "type", None) == TEXT_EVENT and event.text:
                    emit(event.text)
            final = stream.get_final_message()
        except Exception as exc:  # noqa: BLE001 — re-raised as the one typed error, with what arrived
            raise _interrupted(type(exc).__name__, exc, stream, to_response) from exc
        if getattr(final, "stop_reason", None) is None:
            raise _interrupted(NO_STOP_REASON, None, stream, to_response)
        return to_response(final)


__all__ = ["INTERRUPTED", "NO_STOP_REASON", "AnthropicStreamInterrupted", "read_stream", "spent"]
