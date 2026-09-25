"""F155 — the surface a chat turn runs on.

A public widget turn is an anonymous visitor's. The chat service marks the
turn for its duration, and the predicates every gate shares read the mark, so
it holds whatever caller context a tool call built: a widget turn is made for
nobody (core.security.driving_user), never autonomous (the full-autonomy dial,
core.services.auto_autonomy, is the owner's grant to Auto, not to the site's
visitors), never an admin or super admin and never the human whose
instruction approves a card (platform_executor), and is offered no admin tier
(tool_router). A widget turn also carries its key's scopes, which decide what
it may call at all (core.security.widget_scopes), and its key's team lock,
which scopes every document it reads (core.team_access.retrieval_team). The mark is a context
variable, not state on the process-wide tool router, so concurrent turns never
see each other's; tasks a turn starts inherit it.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import FrozenSet, Iterable, Iterator, NamedTuple, Optional

WIDGET = "widget"


class _Turn(NamedTuple):
    surface: Optional[str]
    scopes: FrozenSet[str]
    team: Optional[str]


_surface_var: ContextVar[_Turn] = ContextVar("turn_surface", default=_Turn(None, frozenset(), None))


def widget_turn() -> bool:
    """The current turn is a public widget visitor's."""
    return _surface_var.get().surface == WIDGET


def widget_scopes() -> FrozenSet[str]:
    """The widget key's scopes on a widget turn; empty on any other turn."""
    turn = _surface_var.get()
    return turn.scopes if turn.surface == WIDGET else frozenset()


def widget_team() -> Optional[str]:
    """The widget key's team lock on a widget turn; None on any other turn or
    for a key without one."""
    turn = _surface_var.get()
    return turn.team if turn.surface == WIDGET else None


@contextmanager
def turn_surface(surface: Optional[str], scopes: Iterable[str] = (), team: Optional[str] = None) -> Iterator[None]:
    """Mark every tool call made inside the block with ``surface`` and, on a
    widget turn, its key's ``scopes`` and team lock."""
    from core.team_access import normalize_team

    lock = normalize_team(team) if team and team.strip() else None
    token = _surface_var.set(_Turn(surface, frozenset(scopes or ()), lock))
    try:
        yield
    finally:
        try:
            _surface_var.reset(token)
        except ValueError:
            # Closed from another context (a finalizer, or a task the turn
            # started). That context is not this block's to change: writing to
            # it could clear a mark it inherited, so it is left as it is.
            pass
