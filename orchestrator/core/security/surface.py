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
which scopes every document it reads (core.team_access.retrieval_team), and
the agent it is locked to, if any (whose own plugins it may use). The mark is a context
variable, not state on the process-wide tool router, so concurrent turns never
see each other's; tasks a turn starts inherit it. Work a widget turn starts
that runs later, outside the turn (a mission's tasks on the coordinator tick,
a playbook's steps), carries the turn's origin on its config (stamp_origin)
and runs under it again (origin_surface); a retry or rerun of that work
carries it on (origin_of).
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Dict, FrozenSet, Iterable, Iterator, Mapping, NamedTuple, Optional

WIDGET = "widget"
ORIGIN_SURFACE = "origin_surface"
ORIGIN_SCOPES = "origin_scopes"
ORIGIN_TEAM = "origin_team"
ORIGIN_AGENT_LOCK = "origin_agent_lock"
_ORIGIN_KEYS = (ORIGIN_SURFACE, ORIGIN_SCOPES, ORIGIN_TEAM, ORIGIN_AGENT_LOCK)


class _Turn(NamedTuple):
    surface: Optional[str]
    scopes: FrozenSet[str]
    team: Optional[str]
    agent_lock: Optional[int] = None


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


def widget_agent_lock() -> Optional[int]:
    """The agent the widget key is locked to, on a widget turn; None on any
    other turn or for a key any agent may answer."""
    turn = _surface_var.get()
    return turn.agent_lock if turn.surface == WIDGET else None


@contextmanager
def turn_surface(surface: Optional[str], scopes: Iterable[str] = (), team: Optional[str] = None,
                 agent_lock: Optional[int] = None) -> Iterator[None]:
    """Mark every tool call made inside the block with ``surface`` and, on a
    widget turn, its key's ``scopes``, team lock and agent lock."""
    from core.team_access import normalize_team

    lock = normalize_team(team) if team and team.strip() else None
    token = _surface_var.set(_Turn(surface, frozenset(scopes or ()), lock,
                                   int(agent_lock) if agent_lock is not None else None))
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


def stamp_origin(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """A copy of ``config`` saying where the work was started: on a widget turn
    its surface, key scopes and team lock; otherwise nothing. Server-set: a
    caller's values are dropped either way."""
    stamped = {key: value for key, value in dict(config or {}).items() if key not in _ORIGIN_KEYS}
    turn = _surface_var.get()
    if turn.surface == WIDGET:
        stamped.update({ORIGIN_SURFACE: WIDGET, ORIGIN_SCOPES: sorted(turn.scopes), ORIGIN_TEAM: turn.team})
        if turn.agent_lock is not None:
            stamped[ORIGIN_AGENT_LOCK] = turn.agent_lock
    return stamped


def origin_of(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """The origin ``config`` was stamped with, for work that continues it (a
    retry, a rerun); empty unless a widget turn started it."""
    if not widget_born(config):
        return {}
    return {key: config[key] for key in _ORIGIN_KEYS if key in config}


def widget_born(config: Optional[Mapping[str, Any]]) -> bool:
    """The work ``config`` describes was started on a widget turn."""
    return (config or {}).get(ORIGIN_SURFACE) == WIDGET


@contextmanager
def origin_surface(config: Optional[Mapping[str, Any]]) -> Iterator[None]:
    """Run the block under the turn ``config`` was stamped from: a widget-born
    run's key scopes and team lock. Anything else runs as it is."""
    if widget_born(config):
        with turn_surface(WIDGET, config.get(ORIGIN_SCOPES) or (), config.get(ORIGIN_TEAM),
                          config.get(ORIGIN_AGENT_LOCK)):
            yield
    else:
        yield
