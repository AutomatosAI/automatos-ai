"""PRD-256 O4 (#847): background work linked to the request that asked for it.

A ticket, a Mission, a Run Now or a plan approval is asked for in one request
and run later by a background loop (the board dispatcher, the coordinator's
tick), usually in another worker process, after the request's trace has ended.
So the run is not the request's child: it is a trace of its own, whose root span
(the agent run's ``invoke_agent``, ``genai.traced_agent_run``) carries a **span
link** to the request's span.

* **Stored on the work item.** A ``BoardTask`` or ``OrchestrationRun`` inserted
  while a span is current (a request, an agent run, a heartbeat) is stamped with
  that span's ``traceparent`` (``planning_data`` / ``config``, key ``trace_links``)
  by a SQLAlchemy listener, registered only when tracing is on. A caller's own
  ``trace_links`` never survive the insert. Run Now and plan approval add
  theirs (:func:`with_link`).
* **Linked at the run.** The loop reads the row's links (:func:`links_of`) and
  starts the work inside :func:`linked` / :func:`run_linked`; the agent run's
  span starts with them as links. Work filed from inside (a Mission's Claude
  Code step files a ticket) is stamped with the same links when no span is
  current, so the chain holds.

A traceparent is only IDs and the sampled flag. Off: nothing is stamped or
imported, and every helper passes straight through.
"""
from __future__ import annotations

import contextlib
import re
from contextvars import ContextVar
from typing import Any, Awaitable, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

from core.observability.otel import tracing_enabled

LINKS_KEY = "trace_links"
MAX_LINKS = 8  # a ticket run again and again keeps its latest few
ATTR_LINK_KIND = "automatos.link"
LINK_REQUESTED_BY = "requested_by"

_TRACEPARENT = re.compile(r"00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}")
_pending: ContextVar[Tuple[str, ...]] = ContextVar("automatos_trace_links", default=())


def current_traceparent() -> Optional[str]:
    """The current span's W3C ``traceparent``; None when tracing is off or no span is current."""
    if not tracing_enabled():
        return None
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    carrier: Dict[str, str] = {}
    TraceContextTextMapPropagator().inject(carrier)
    return carrier.get("traceparent")


def pending_links() -> Tuple[str, ...]:
    """The links the work running here carries (set by :func:`linked`)."""
    return _pending.get()


def links_of(data: Any) -> Tuple[str, ...]:
    """The well-formed traceparents stored on a row's JSON (``planning_data`` / ``config``)."""
    raw = data.get(LINKS_KEY) if isinstance(data, Mapping) else None
    if not isinstance(raw, list):
        return ()
    return tuple(value for value in raw if isinstance(value, str) and _TRACEPARENT.fullmatch(value))[-MAX_LINKS:]


def _here() -> Tuple[str, ...]:
    """What work asked for here links to: the current span, else the links this work itself carries."""
    traceparent = current_traceparent()
    return (traceparent,) if traceparent else _pending.get()


def stamped(data: Optional[Mapping[str, Any]], *, keep: bool = False) -> Dict[str, Any]:
    """``data`` with this context's links: replacing any it had (a new item, which a
    caller can't seed), or added to them (``keep``: a re-run, an approval)."""
    base = dict(data or {})
    links: List[str] = list(dict.fromkeys((links_of(base) if keep else ()) + _here()))[-MAX_LINKS:]
    base.pop(LINKS_KEY, None)
    if links:
        base[LINKS_KEY] = links
    return base


def with_link(data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """``data`` with this context's link added; ``data`` itself when there is none (tracing off)."""
    return stamped(data, keep=True) if _here() else data


def _stamp_new(target: Any, column: str) -> None:
    data = getattr(target, column)
    fresh = stamped(data)
    if fresh != dict(data or {}):
        setattr(target, column, fresh)


def _stamp_ticket(_mapper: Any, _connection: Any, ticket: Any) -> None:
    _stamp_new(ticket, "planning_data")


def _stamp_mission(_mapper: Any, _connection: Any, run: Any) -> None:
    _stamp_new(run, "config")


def register_work_links(_: Any = None) -> None:
    """Stamp every new ticket and Mission with the links of where it was asked for, once per process."""
    from sqlalchemy import event

    from core.models.core import BoardTask
    from core.models.orchestration import OrchestrationRun

    for model, listener in ((BoardTask, _stamp_ticket), (OrchestrationRun, _stamp_mission)):
        if not event.contains(model, "before_insert", listener):
            event.listen(model, "before_insert", listener)


@contextlib.contextmanager
def linked(traceparents: Optional[Iterable[str]]) -> Iterator[None]:
    """Work started inside carries these links; a task created inside copies them."""
    token = _pending.set(tuple(traceparents or ()))
    try:
        yield
    finally:
        _pending.reset(token)


async def run_linked(work: Awaitable[Any], traceparents: Optional[Iterable[str]]) -> Any:
    """Await ``work`` carrying these links (a Mission step, in its own task under ``gather``)."""
    with linked(traceparents):
        return await work


def span_links(traceparents: Iterable[str]) -> List[Any]:
    """OpenTelemetry ``Link``s to these traceparents (tracing on only)."""
    from opentelemetry import trace
    from opentelemetry.trace import Link
    from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

    links = []
    for traceparent in traceparents:
        context = TraceContextTextMapPropagator().extract({"traceparent": traceparent})
        span_context = trace.get_current_span(context).get_span_context()
        if span_context.is_valid:
            links.append(Link(span_context, {ATTR_LINK_KIND: LINK_REQUESTED_BY}))
    return links
