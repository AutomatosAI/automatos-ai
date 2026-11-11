"""An end user's chat turn: what it writes stays theirs (PRD-274 "Automatos for products", Wave 3, §5.5).

A product's server key may name the end user a turn is for (§8-6, §8-17). That turn is theirs: what it
remembers belongs in their namespace, ``mem:{ws}:end_user:{id}``, and nowhere another end user, the
owner's Auto or the workspace's other conversations would read it.

The rule is held where memory is written, by the turn's **end user**, a context variable, not path by
path (as ``core.security.memory_audience`` holds the reader):

- **Every durable write lands in the end user's namespace** (``writes_to_its_end_user`` on the durable
  store's ``add``). The distilled facts of the turn, a tool's outcome, the context guard's flush: each
  names the workspace's namespace or an agent's, and each is written to the end user's instead, tagged
  with their subject (``end_user:{id}``) so their erasure finds it. A write into another workspace
  during the turn is refused.
- **Nothing goes where the whole workspace reads** (``shared_with_no_end_user``): no L2 transcript
  (``memory_short_term``) and so no ``mem:{ws}:l2`` mirror, and no daily log. The Context Router reads
  those on every chat turn of the workspace. The end user's conversation stays in their chat.

Nothing sets the end user yet: until a server key's chat turn names one, no turn is an end user's and
every write goes where it always has.
"""
from __future__ import annotations

import functools
import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Iterator, Optional

END_USER_ID_KEY = "end_user_id"
SUBJECT_PREFIX = "end_user:"
NAMESPACE_PREFIX = "mem:"
NOT_SHARED = "an end user's turn: nothing is shared with the workspace"


@dataclass(frozen=True)
class EndUser:
    """The end user a turn is for: ``(workspace, end_user_id)`` (§8-17)."""

    workspace_id: str
    end_user_id: str

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


def end_user_key() -> str:
    """The turn's end user for a cache key: empty on a turn that is no end user's."""
    end_user = _end_user.get()
    return "" if end_user is None else f"eu={end_user.end_user_id}"


def _in_workspace(namespace: str, workspace_id: str) -> bool:
    own = f"{NAMESPACE_PREFIX}{workspace_id}"
    return namespace == own or namespace.startswith(own + ":")


def routed(end_user: EndUser, user_id: str, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Where a write during ``end_user``'s turn goes: their namespace, their subject, and
    ``metadata`` naming them (a new dict). A namespace of another workspace is refused."""
    if not _in_workspace(user_id, end_user.workspace_id):
        raise ValueError(f"an end user's turn writes only to their own workspace, not {user_id}")
    return {"user_id": end_user.namespace, "subject_id": end_user.subject,
            "metadata": {**(metadata or {}), END_USER_ID_KEY: end_user.end_user_id}}


def writes_to_its_end_user(add: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Wrap the durable store's ``add``: during an end user's turn the write lands in their namespace."""
    signature = inspect.signature(add)

    @functools.wraps(add)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        end_user = _end_user.get()
        if end_user is None:
            return await add(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.arguments.update(routed(end_user, bound.arguments["user_id"], bound.arguments.get("metadata")))
        return await add(*bound.args, **bound.kwargs)

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
