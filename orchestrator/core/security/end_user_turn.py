"""An end user's chat turn: what it writes stays theirs (PRD-274 "Automatos for products", Wave 3, §5.5).

A product's server key may name the end user a turn is for (§8-6, §8-17). That turn is theirs: what it
remembers belongs in their namespace, ``mem:{ws}:end_user:{id}``, and nowhere another end user, the
owner's Auto or the workspace's other conversations would read it.

The rule is held where memory is written, by the turn's **end user**, a context variable, not path by
path (as ``core.security.memory_audience`` holds the reader):

- **A write is made only while their consent counts** (``writes_to_its_end_user``): it is checked
  before the write, and again after it. An end user whose erase has started (``end_users.erasing_at``)
  gets no new write, and a write that landed while the erase was deleting is taken back, so a turn still
  running when they are erased leaves nothing behind.
- **Every durable write lands in the end user's namespace** (``writes_to_its_end_user`` on the durable
  store's ``add``). The distilled facts of the turn, a tool's outcome, the context guard's flush: each
  names the workspace's namespace or an agent's, and each is written to the end user's instead, tagged
  with their subject (``end_user:{id}``) so their erasure finds it, and a fact with an owner is theirs.
  A write into another workspace during the turn is refused.
- **Nothing goes where the whole workspace reads** (``shared_with_no_end_user``): no L2 transcript
  (``memory_short_term``) and so no ``mem:{ws}:l2`` mirror, and no daily log. The Context Router reads
  those on every chat turn of the workspace. The end user's conversation stays in their chat.

What the turn **reads** is theirs and the workspace's, never another end user's:

- **The turn's subject is theirs** (``turn_subject``): the private-row guard and the standing rules
  read as the end user, not as the key's service user, whom every end user of the key shares.
- **Wherever it reads the workspace tier, it reads theirs with it** (``reads_its_end_users_memory`` on
  the durable store's ``search``): the two-tier recall's workspace search, and any search that widens
  to the workspace. An agent's own namespace is read as it is.
- **What they own is in their namespace** (``owned_reads_go_to_its_end_user`` on ``get_where_any``):
  a read of the rows the end user owns (their standing rules) reads their namespace.
- The memory caches key on the end user too (``end_user_key``), and the memory section reads no
  other conversation and no daily log on their turn (``modules.context.sections.memory``).

**Who sets it:** a server key's chat turn that names an end user whose consent is on record
(``endUserId`` on ``POST /api/chat``, ``api.chat_server_key``), for the whole turn
(``StreamingChatService.stream_response_with_agent``), the work it starts included. No other turn is an
end user's, and every read and write of it goes where it always has.
"""
from __future__ import annotations

import asyncio
import functools
import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Iterator, Optional

END_USER_ID_KEY = "end_user_id"
OWNER_KEY = "owner"  # a fact's owner (modules.memory.write_contract): who may see it when private
SUBJECT_PREFIX = "end_user:"
NAMESPACE_PREFIX = "mem:"
NOT_SHARED = "an end user's turn: nothing is shared with the workspace"
ERASED = "the end user is erased, or their erase has started: nothing more is written for them"


@dataclass(frozen=True)
class EndUser:
    """The end user a turn is for: ``(workspace, end_user_id)`` (§8-17)."""

    workspace_id: str
    end_user_id: str
    # The consent record the turn was admitted under: it keys the caches, and a write lands only while it
    # still stands. After an erase, a consent recorded again is a new record, so nothing cached or admitted
    # before the erase is read or written under the new one.
    record_id: Optional[str] = None

    @property
    def namespace(self) -> str:
        from modules.memory.unified_memory_service import MemoryNamespace

        return MemoryNamespace(self.workspace_id).end_user(self.end_user_id)

    @property
    def subject(self) -> str:
        """Their GDPR data-subject tag, as ``user:{users.id}`` is a person's (PRD-196 S6)."""
        return f"{SUBJECT_PREFIX}{self.end_user_id}"


# The turn's end user; None on every turn that is not an end user's.
_end_user: ContextVar[Optional[EndUser]] = ContextVar("end_user_turn", default=None)


def current_end_user() -> Optional[EndUser]:
    """The turn's end user; None on a turn that is no end user's."""
    return _end_user.get()


@contextmanager
def end_user_turn(end_user: Optional[EndUser]) -> Iterator[None]:
    """Run the block as ``end_user``'s turn (None: no end user's)."""
    token = _end_user.set(end_user)
    try:
        yield
    finally:
        try:
            _end_user.reset(token)
        except ValueError:  # closed from another context: left as it is (core.security.surface)
            pass


def turn_subject(person: Optional[str]) -> Optional[str]:
    """The turn's data subject: the end user's on their turn, else ``person`` (``user:{users.id}``)."""
    end_user = _end_user.get()
    return person if end_user is None else end_user.subject


def end_user_key() -> str:
    """The turn's end user for a cache key: empty on a turn that is no end user's."""
    end_user = _end_user.get()
    if end_user is None:
        return ""
    return f"eu={end_user.end_user_id}" + (f":{end_user.record_id}" if end_user.record_id else "")


def _in_workspace(namespace: str, workspace_id: str) -> bool:
    own = f"{NAMESPACE_PREFIX}{workspace_id}"
    return namespace == own or namespace.startswith(own + ":")


def routed(end_user: EndUser, user_id: str, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Where a write during ``end_user``'s turn goes: their namespace, their subject, and
    ``metadata`` naming them (a new dict). A fact with an owner is theirs, not the key's service
    user's, whom every end user of the key shares. A namespace of another workspace is refused."""
    if not _in_workspace(user_id, end_user.workspace_id):
        raise ValueError(f"an end user's turn writes only to their own workspace, not {user_id}")
    named = {**(metadata or {}), END_USER_ID_KEY: end_user.end_user_id}
    if named.get(OWNER_KEY):
        named[OWNER_KEY] = end_user.subject
    return {"user_id": end_user.namespace, "subject_id": end_user.subject, "metadata": named}


async def _consented(end_user: EndUser) -> bool:
    """Their consent is on record, no erase of them has started, and it is the record the turn was
    admitted under when it names one (a read on its own session)."""
    from core.database.database import SessionLocal
    from core.models.end_users import consent_record_id

    def check() -> bool:
        db = SessionLocal()
        try:
            record_id = consent_record_id(db, end_user.workspace_id, end_user.end_user_id)
        finally:
            db.close()
        return record_id is not None and end_user.record_id in (None, record_id)

    return await asyncio.to_thread(check)


def writes_to_its_end_user(add: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Wrap the durable store's ``add``: during an end user's turn the write lands in their namespace,
    and only while their consent counts, checked before and after it (an erase that started meanwhile
    takes the write back). An erase deletes their namespace after marking them, so a write that landed
    first is deleted by the erase, and one that landed after is taken back here."""
    signature = inspect.signature(add)

    @functools.wraps(add)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        end_user = _end_user.get()
        if end_user is None:
            return await add(*args, **kwargs)
        if not await _consented(end_user):
            return {"success": False, "skipped": ERASED}
        bound = signature.bind(*args, **kwargs)
        bound.arguments.update(routed(end_user, bound.arguments["user_id"], bound.arguments.get("metadata")))
        result = await add(*bound.args, **bound.kwargs)
        if await _consented(end_user):
            return result
        await _take_back(bound.arguments["self"], end_user, result)
        return {"success": False, "skipped": ERASED}

    return wrapped


async def _take_back(store: Any, end_user: EndUser, result: Any) -> None:
    """Deletes the point a write just made, when the end user's erase started while it was made."""
    point_id = result.get("id") if isinstance(result, dict) else None
    if point_id:
        await store.delete([point_id], user_id=end_user.namespace)


def reads_its_end_users_memory(search: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Wrap the durable store's ``search``: during an end user's turn, a search that reads the
    workspace's namespace reads theirs with it (one query, one ranked list)."""
    signature = inspect.signature(search)

    @functools.wraps(search)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        end_user = _end_user.get()
        if end_user is None:
            return await search(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        workspace = f"{NAMESPACE_PREFIX}{end_user.workspace_id}"
        also = tuple(bound.arguments.get("also_namespaces") or ())
        if workspace in (bound.arguments["user_id"], *also) and end_user.namespace not in also:
            bound.arguments["also_namespaces"] = (*also, end_user.namespace)
        return await search(*bound.args, **bound.kwargs)

    return wrapped


def owned_reads_go_to_its_end_user(read: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Wrap the durable store's ``get_where_any``: during an end user's turn, a read of the rows the
    end user owns (``where`` names their subject as owner) reads their namespace, where those rows are."""
    signature = inspect.signature(read)

    @functools.wraps(read)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        end_user = _end_user.get()
        if end_user is None:
            return await read(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        where = bound.arguments.get("where") or {}
        if any(key.endswith(OWNER_KEY) and value == end_user.subject for key, value in where.items()):
            if not _in_workspace(bound.arguments["user_id"], end_user.workspace_id):
                raise ValueError(f"an end user's turn reads only its own workspace, not {bound.arguments['user_id']}")
            bound.arguments["user_id"] = end_user.namespace
        return await read(*bound.args, **bound.kwargs)

    return wrapped


def shared_with_no_end_user(skipped: Any) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Wrap a write the whole workspace reads (an L2 row, a daily log): during an end user's turn it
    is not made, and the call answers ``skipped`` (what the write answers when it stores nothing)."""
    def decorate(write: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        @functools.wraps(write)
        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            if _end_user.get() is not None:
                return skipped() if callable(skipped) else skipped
            return await write(*args, **kwargs)

        return wrapped

    return decorate


def skipped_log() -> Dict[str, Any]:
    """A daily log's answer when none is written: no error, nothing stored."""
    return {"success": False, "skipped": NOT_SHARED}
