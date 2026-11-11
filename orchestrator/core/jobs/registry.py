"""Job kinds and their handlers (PRD-276 §6.2).

A kind names one sort of work and carries everything the queue needs to run it: the handler, its
backoff, attempt limit, lease, and how many may run at once (in one worker, and per workspace across
all workers). ``@job_handler`` registers a kind when its module is imported; ``enqueue`` refuses a
kind nobody registered, so a typo never queues work no worker will claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Tuple

DEFAULT_BACKOFF = (5, 30, 120, 600)
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_LEASE_SECONDS = 60
DEFAULT_CONCURRENCY = 4
DEFAULT_WORKSPACE_CONCURRENCY = 2

Handler = Callable[[Any], Awaitable[None]]


@dataclass(frozen=True)
class JobKind:
    """One kind of work and how the queue runs it."""

    kind: str
    handler: Handler
    backoff: Tuple[int, ...] = DEFAULT_BACKOFF
    max_attempts: int = DEFAULT_MAX_ATTEMPTS
    lease_seconds: int = DEFAULT_LEASE_SECONDS
    concurrency: int = DEFAULT_CONCURRENCY
    workspace_concurrency: int = DEFAULT_WORKSPACE_CONCURRENCY

    def delay_after(self, attempts: int) -> int:
        """Seconds to wait after the ``attempts``-th attempt failed: ``backoff[n - 1]``, the last
        delay repeating when the tuple is shorter (§6.2)."""
        if not self.backoff:
            return 0
        return int(self.backoff[min(max(attempts, 1), len(self.backoff)) - 1])


_KINDS: Dict[str, JobKind] = {}


def job_handler(kind: str, **options: Any) -> Callable[[Handler], Handler]:
    """Registers the decorated coroutine as ``kind``'s handler, with ``options`` (``backoff``,
    ``max_attempts``, ``lease_seconds``, ``concurrency``, ``workspace_concurrency``)."""

    def register(handler: Handler) -> Handler:
        spec = JobKind(kind=kind, handler=handler, **options)
        if spec.max_attempts < 1 or spec.lease_seconds < 1 or spec.concurrency < 1:
            raise ValueError(f"job kind {kind}: max_attempts, lease_seconds and concurrency must be at least 1")
        _KINDS[kind] = spec
        return handler

    return register


def kind_of(kind: str) -> JobKind:
    """The registered kind; ``KeyError`` when no handler registered it."""
    return _KINDS[kind]


def registered() -> Tuple[str, ...]:
    """The kinds a worker may claim."""
    return tuple(sorted(_KINDS))


def forget(kind: str) -> None:
    """Removes ``kind`` from the registry (tests)."""
    _KINDS.pop(kind, None)
