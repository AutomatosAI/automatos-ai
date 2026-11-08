"""Lean turns: the actions an agent's turn may call, and the refusal for anything else.

A lean turn (``consumers.chatbot.lean_turn``) is offered only what the agent's session
tool groups grant. Offering is not enough on its own: with the dispatcher's enum kept
cache-stable (``TOOL_ENUM_CACHE_STABLE``) a model can still name any action, so the
executor judges every call against the turn's grants here, as it judges a widget turn
against its key's scopes (``core.security.widget_scopes``). The grants are a context
variable, so concurrent turns never see each other's, and tasks a turn starts inherit them.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import FrozenSet, Iterable, Iterator, Optional

LEAN_REFUSAL = "This agent is not given that tool. Answer with what you have, or say what you would need."

_granted: ContextVar[Optional[FrozenSet[str]]] = ContextVar("lean_grants", default=None)


def lean_turn() -> bool:
    """The current turn is a lean agent's."""
    return _granted.get() is not None


def lean_may_call(name: str) -> bool:
    """True on any turn that is not lean, and on a lean turn for a granted action or tool."""
    granted = _granted.get()
    return granted is None or name in granted


@contextmanager
def lean_grants(names: Optional[Iterable[str]]) -> Iterator[None]:
    """Mark every tool call made inside the block with the lean turn's grants;
    ``None`` (a turn that is not lean) marks nothing."""
    token = _granted.set(frozenset(names) if names is not None else None)
    try:
        yield
    finally:
        try:
            _granted.reset(token)
        except ValueError:
            # Closed from another context (see core.security.surface.turn_surface): left as it is.
            pass
