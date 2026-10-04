"""F302 (night 9): the owner's own words reach NL2SQL through platform_query_data too.

F077 (A) sends what the person typed this turn to the SQL writer beside Auto's restatement,
because a restatement drops qualifiers ("not counting cancelled ones" became "active", 383 for
400). It rode the chat's ``caller_context["user_query"]`` into smart_query_database. Auto's chat
now asks the database through platform_query_data (modules/tools/data_routes.py), whose handler
takes only its params, and builds its own context with the user id alone: the owner's words
would stop at the platform action's door.

While a platform action runs, ``execute_platform_action`` holds the turn's words here;
``exec_research.owner_words`` reads them when the context it is given carries none. A lane
nobody typed into (a board card, a heartbeat) holds nothing, as before.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator, Optional

_turn_words: ContextVar[Optional[str]] = ContextVar("turn_owner_words", default=None)


@contextmanager
def owner_words_held(caller_context: Any) -> Iterator[None]:
    """Hold what the person typed this turn (``user_query``, set by the chat server-side; never
    a tool argument) for the duration of one platform action. A lane nobody typed into holds
    nothing."""
    typed = caller_context.get("user_query") if isinstance(caller_context, dict) else None
    token = _turn_words.set(str(typed) if typed else None)
    try:
        yield
    finally:
        _turn_words.reset(token)


def held_owner_words() -> Optional[str]:
    """The words the running platform action's turn was typed with, or None outside one."""
    return _turn_words.get()
