"""Job kinds and their handlers (PRD-276 §6.2).

A kind names one sort of work and carries everything the queue needs to run it: the handler, its
backoff, attempt limit, lease, how many may run at once (in one worker, and per workspace across
all workers), and which payload keys are removed from the row the moment a job ends (``scrub``). ``@job_handler`` registers a kind when its module is imported; ``enqueue`` refuses a
kind nobody registered, so a typo never queues work no worker will claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Tuple

DEFAULT_BACKOFF = (5, 30, 120, 600)
DEFAULT_MAX_ATTEMPTS = 5
DEFAULT_LEASE_SECONDS = 60
DEFAULT_CONCURRENCY = 4
DEFAULT_WORKSPACE_CONCURRENCY = 2
# The worker renews a lease every third of its length, waking at least once a second, so a lease
# shorter than this could lapse between renewals and be swept from a job that is still running.
MIN_LEASE_SECONDS = 10
# What a schedule of this kind queues for slots missed while no scheduler ran (§6.4, Wave 2b):
# every missed slot (capped), the most recent one only, or none.
CATCH_UP_ALL, CATCH_UP_LATEST, CATCH_UP_NONE = "all", "latest", "none"
CATCH_UPS = (CATCH_UP_ALL, CATCH_UP_LATEST, CATCH_UP_NONE)

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
    scrub: Tuple[str, ...] = ()  # payload keys held only until the job ends, done or dead
    catch_up: str = CATCH_UP_LATEST  # a scheduled kind's missed slots: all, latest or none

    def delay_after(self, attempts: int) -> int:
        """Seconds to wait after the ``attempts``-th attempt failed: ``backoff[n - 1]``, the last
        delay repeating when the tuple is shorter (§6.2)."""
        if not self.backoff:
            return 0
        return int(self.backoff[min(max(attempts, 1), len(self.backoff)) - 1])


_KINDS: Dict[str, JobKind] = {}


def job_handler(kind: str, **options: Any) -> Callable[[Handler], Handler]:
    """Registers the decorated coroutine as ``kind``'s handler, with ``options`` (``backoff``,
    ``max_attempts``, ``lease_seconds``, ``concurrency``, ``workspace_concurrency``, ``scrub``, ``catch_up``)."""

    def register(handler: Handler) -> Handler:
        spec = JobKind(kind=kind, handler=handler, **options)
        if min(spec.max_attempts, spec.concurrency, spec.workspace_concurrency) < 1:
            raise ValueError(f"job kind {kind}: max_attempts, concurrency and workspace_concurrency must be at least 1")
        if spec.lease_seconds < MIN_LEASE_SECONDS:
            raise ValueError(f"job kind {kind}: lease_seconds must be at least {MIN_LEASE_SECONDS}")
        if spec.catch_up not in CATCH_UPS:
            raise ValueError(f"job kind {kind}: catch_up must be one of {CATCH_UPS}")
        _KINDS[kind] = spec
        return handler

    return register


def kind_of(kind: str) -> JobKind:
    """The registered kind; ``KeyError`` when no handler registered it."""
    return _KINDS[kind]


def scrub_of(kind: str) -> List[str]:
    """The payload keys ``kind`` removes when a job ends; none for a kind no longer registered."""
    spec = _KINDS.get(kind)
    return list(spec.scrub) if spec else []


def registered() -> Tuple[str, ...]:
    """The kinds a worker may claim."""
    return tuple(sorted(_KINDS))


def forget(kind: str) -> None:
    """Removes ``kind`` from the registry (tests)."""
    _KINDS.pop(kind, None)
