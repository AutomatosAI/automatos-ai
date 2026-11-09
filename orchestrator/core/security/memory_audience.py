"""Need-to-know: a fact limited to the agents allowed to know it (PRD draft "Automatos for products", §5.8).

A durable fact may carry ``metadata.audience``: a short list of labels (``["finance"]``,
``["operator"]``). An agent may carry ``configuration.audience``. An agent reads a labelled fact
only when their labels meet; an unlabelled fact is everyone's, as before.

The rule is held at the durable store, by the run's **reader**, not path by path:

- An agent's run (a chat turn, ``StreamingChatService``; a task, channel message, heartbeat,
  schedule or webhook, ``AgentFactory.execute_with_prompt``) sets its reader, a context
  variable, for everything it does: every read the turn, its tools and the work it starts make
  (``for_its_reader`` on the store's ``search`` and ``_scroll``) leaves out what the reader may
  not see, and the two memory caches key on the reader too (``reader_key``).
- A system agent (Auto, the CTO) reads everything, as the owner's own pages do; so does any read
  with no reader set: the owner's memory pages, exports, GDPR, and the platform's jobs.
- A fact written during a labelled agent's run carries that agent's labels unless the write
  names its own audience (``labels_what_its_reader_writes`` on the store's ``add``), so what a
  finance agent learns stays with the finance agents.

Off unless ``MEMORY_AUDIENCE_ENABLED``: then no reader is set and nothing is filtered or labelled.
The widget lane (F155) keeps its own, stricter rule (a visitor's turn reads only its agent's
memories); the labels apply on top of it.
"""
from __future__ import annotations

import dataclasses
import functools
import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Awaitable, Callable, Dict, FrozenSet, Iterable, Iterator, List, Mapping, Optional

AUDIENCE_KEY = "audience"
LABEL_MAX = 64
LABELS_MAX = 10

Labels = FrozenSet[str]
Read = Callable[..., Awaitable[List[Dict[str, Any]]]]

# The run's reader: its labels; None when the run reads everything (no agent, a system agent, the dial off).
_reader: ContextVar[Optional[Labels]] = ContextVar("memory_reader", default=None)


def audience_errors(value: Any, where: str = AUDIENCE_KEY) -> List[str]:
    """Errors (empty when valid) for an audience: absent, or a list of up to ``LABELS_MAX``
    non-empty labels of at most ``LABEL_MAX`` characters."""
    if value is None:
        return []
    ok = (isinstance(value, list) and len(value) <= LABELS_MAX
          and all(isinstance(v, str) and v.strip() and len(v.strip()) <= LABEL_MAX for v in value))
    return [] if ok else [f"{where} must be a list of up to {LABELS_MAX} labels, such as [\"finance\"]"]


def labels(value: Any) -> Labels:
    """``value`` as a set of labels, case and surrounding spaces ignored; empty when it is none."""
    if not isinstance(value, (list, tuple, set, frozenset)):
        return frozenset()
    return frozenset(str(v).strip().lower() for v in value if isinstance(v, str) and v.strip())


def audience_of(row: Mapping[str, Any]) -> Labels:
    """A stored fact's audience (its ``metadata.audience``)."""
    metadata = row.get("metadata") if isinstance(row, Mapping) else None
    return labels(metadata.get(AUDIENCE_KEY)) if isinstance(metadata, Mapping) else frozenset()


def visible_to(row: Mapping[str, Any], reader: Optional[Labels]) -> bool:
    """``reader`` may read ``row``: it reads everything, or the fact is unlabelled, or they meet."""
    audience = audience_of(row)
    return reader is None or not audience or bool(audience & reader)


def reader_for(agent: Any) -> Optional[Labels]:
    """The reader an agent's run is: its labels; None (everything) for a system agent or the dial off."""
    from config import config

    if not getattr(config, "MEMORY_AUDIENCE_ENABLED", False) or agent is None:
        return None
    if getattr(agent, "is_system_agent", False):
        return None
    configuration = getattr(agent, "configuration", None)
    return labels(configuration.get(AUDIENCE_KEY)) if isinstance(configuration, Mapping) else frozenset()


def reader_for_agent_id(db: Any, agent_id: Any) -> Optional[Labels]:
    """``reader_for`` the agent with ``agent_id``; None when the dial is off or there is no such agent."""
    from config import config

    if not getattr(config, "MEMORY_AUDIENCE_ENABLED", False) or db is None or agent_id is None:
        return None
    from core.models import Agent

    return reader_for(db.query(Agent).filter(Agent.id == agent_id).first())


def current_reader() -> Optional[Labels]:
    return _reader.get()


@contextmanager
def memory_reader(reader: Optional[Labels]) -> Iterator[None]:
    """Run the block as ``reader`` (None: reads everything)."""
    token = _reader.set(reader)
    try:
        yield
    finally:
        try:
            _reader.reset(token)
        except ValueError:  # closed from another context: left as it is (core.security.surface)
            pass


def reader_key() -> str:
    """The run's reader for a cache key: empty when it reads everything."""
    reader = _reader.get()
    return "" if reader is None else "aud=" + ",".join(sorted(reader))


def for_its_reader(read: Read) -> Read:
    """Wrap a durable-store read: the rows it returns, less what the run's reader may not see."""
    @functools.wraps(read)
    async def wrapped(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        rows = await read(*args, **kwargs)
        reader = _reader.get()
        return rows if reader is None else [row for row in rows if visible_to(row, reader)]
    return wrapped


def with_readers_audience(metadata: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """``metadata`` labelled with the run's reader's labels when it names no audience of its own
    and the reader has some; a new dict, or ``metadata`` as it is."""
    reader = _reader.get()
    if not reader or (isinstance(metadata, Mapping) and metadata.get(AUDIENCE_KEY)):
        return dict(metadata) if metadata is not None else None
    return {**(metadata or {}), AUDIENCE_KEY: sorted(reader)}


def labels_what_its_reader_writes(add: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Wrap the durable store's ``add``: a write during a labelled agent's run carries its labels."""
    signature = inspect.signature(add)

    @functools.wraps(add)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        if _reader.get():
            bound = signature.bind(*args, **kwargs)
            bound.arguments["metadata"] = with_readers_audience(bound.arguments.get("metadata"))
            return await add(*bound.args, **bound.kwargs)
        return await add(*args, **kwargs)
    return wrapped


def reads_memory_as_its_agent(execute: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Wrap ``AgentFactory.execute_with_prompt``: the run reads memory as its agent (its first
    argument, an id or a runtime)."""
    @functools.wraps(execute)
    async def wrapped(factory: Any, agent: Any, *args: Any, **kwargs: Any) -> Any:
        agent_id = getattr(agent, "agent_id", agent)
        with memory_reader(reader_for_agent_id(getattr(factory, "db_session", None), agent_id)):
            return await execute(factory, agent, *args, **kwargs)
    return wrapped


def by_audience(rows: Iterable[Mapping[str, Any]]) -> Dict[Labels, List[Mapping[str, Any]]]:
    """``rows`` grouped by their audience, in their order."""
    groups: Dict[Labels, List[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(audience_of(row), []).append(row)
    return groups


def plans_each_audience_apart(plan: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``plan_consolidation``: each audience's facts are planned on their own, so a fact
    for some agents is never merged into, or superseded by, a fact for others (or everyone's).
    One audience (every unlabelled workspace today) is planned exactly as before."""
    @functools.wraps(plan)
    def wrapped(memories: List[Dict[str, Any]], *args: Any, **kwargs: Any) -> Any:
        groups = by_audience(memories)
        if len(groups) <= 1:
            return plan(memories, *args, **kwargs)
        plans = [plan(list(group), *args, **kwargs) for group in groups.values()]
        joined = {f.name: [item for p in plans for item in getattr(p, f.name)] for f in dataclasses.fields(plans[0])}
        return type(plans[0])(**joined)
    return wrapped


def one_audience_only(find: Callable[..., Awaitable[List[Dict[str, Any]]]]) -> Callable[..., Awaitable[Any]]:
    """Wrap the memory page's consolidate lookup: memories for different audiences are never
    merged into one (a ValueError the route reports)."""
    @functools.wraps(find)
    async def wrapped(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
        targets = await find(*args, **kwargs)
        if len(by_audience(targets)) > 1:
            raise ValueError("these memories are for different audiences; consolidate each audience's on its own")
        return targets
    return wrapped


def shared_audience(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """The metadata a merge of ``rows`` keeps: their one audience, when they have one."""
    audiences = {audience_of(row) for row in rows}
    only = next(iter(audiences)) if len(audiences) == 1 else frozenset()
    return {AUDIENCE_KEY: sorted(only)} if only else {}
