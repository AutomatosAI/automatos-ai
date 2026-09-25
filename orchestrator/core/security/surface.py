"""F155 — the surface a chat turn runs on.

A public widget turn is an anonymous visitor's. The chat service marks the
turn for its duration, and the predicates every gate shares read the mark, so
it holds whatever caller context a tool call built: a widget turn is made for
nobody (core.security.driving_user), never autonomous (the full-autonomy dial,
core.services.auto_autonomy, is the owner's grant to Auto, not to the site's
visitors), never an admin or super admin and never the human whose
instruction approves a card (platform_executor), and is offered no admin tier
(tool_router). The mark is a context variable, not state on the process-wide
tool router, so concurrent turns never see each other's; tasks a turn starts
inherit it.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Iterator, Optional

WIDGET = "widget"

_surface_var: ContextVar[Optional[str]] = ContextVar("turn_surface", default=None)


def widget_turn() -> bool:
    """The current turn is a public widget visitor's."""
    return _surface_var.get() == WIDGET


@contextmanager
def turn_surface(surface: Optional[str]) -> Iterator[None]:
    """Mark every tool call made inside the block with ``surface``."""
    token = _surface_var.set(surface)
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
