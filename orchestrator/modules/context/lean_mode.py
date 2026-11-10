"""A lean agent's chat turn builds its context in the LEAN mode, not CHATBOT.

The chat turn asks ``ContextService.build_context`` for the CHATBOT mode wherever it builds
(``SmartChatOrchestrator.prepare_request``). During a lean agent's turn the turn's grants
are set (``core.security.lean_grants``), and this wrapper builds that request in the
``LEAN`` mode instead (``consumers.chatbot.lean_turn``): the agent's own identity, what it
recalls, the date, the owner's standing rules and the conversation. Every other mode, and
every turn that is not lean, is built as asked.
"""
from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable

from modules.context.modes import ContextMode

Build = Callable[..., Awaitable[Any]]


def builds_a_lean_turn_in_lean_mode(build: Build) -> Build:
    """Wrap ``ContextService.build_context``: CHATBOT becomes LEAN on a lean agent's turn."""
    @functools.wraps(build)
    async def wrapped(service: Any, *args: Any, **kwargs: Any) -> Any:
        from core.security.lean_grants import lean_turn

        if lean_turn():
            if kwargs.get("mode") == ContextMode.CHATBOT:
                kwargs = {**kwargs, "mode": ContextMode.LEAN}
            elif args and args[0] == ContextMode.CHATBOT:
                args = (ContextMode.LEAN, *args[1:])
        return await build(service, *args, **kwargs)
    return wrapped
