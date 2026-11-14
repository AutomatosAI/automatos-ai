"""Request-scoped LLM flags (PRD-201 S5).

The Anthropic **context-editing** and **memory-tool** primitives are adopted on
the *headless* agent loops (TASK_EXECUTION via AgentFactory, HEARTBEAT_AGENT) —
not on chat, which already has ``ContextGuard`` compaction. But chat and the
headless loops share one provider client (``AnthropicProvider.generate_response``)
behind the fixed ``generate_response(messages, tools)`` interface, so the "this
is a headless run" signal can't ride the call signature without refactoring every
provider.

A ``contextvars.ContextVar`` carries it instead: the headless entry
(``AgentFactory.execute_with_prompt``) marks its LLM loop with ``headless_run()``,
and the Anthropic client reads ``is_headless_run()``. ContextVars are async-safe
(each task sees its own value), so a concurrent chat turn in the same process
never sees the flag. Zero signature changes, cleanly scoped, unit-testable
(set the var, assert the request shape).

Voice latency B-1 adds a second flag the same way: which **spoken surface** a turn's LLM
calls answer (``QUICK_ASK`` for a turn sent to ``POST /api/chat`` with ``spokenMode``,
``LIVE`` for an Auto Live call, ``""`` for a turn that is read). The chat turn's wrapper
(``consumers.chatbot.spoken_model.answers_on_the_turns_model``) and the Auto Live stream
(``api.voice_retell``) set it; the Anthropic client reads it to decide whether a call
streams (``ANTHROPIC_STREAMING``). It is set and put back, not reset by token: an async
generator can be closed from another context, where a token reset raises.
"""
from __future__ import annotations

import contextvars
from contextlib import contextmanager
from typing import Iterator

# Default OFF — only the headless entry flips it, and only for the duration of
# its own LLM loop.
_HEADLESS_RUN: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "automatos_headless_run", default=False
)


@contextmanager
def headless_run() -> Iterator[None]:
    """Mark the enclosed LLM work as a headless agent run (PRD-201 S5).

    Reset on exit even if the body raises, so a failed run never leaves the flag
    set for the next task on the same thread/loop.
    """
    token = _HEADLESS_RUN.set(True)
    try:
        yield
    finally:
        _HEADLESS_RUN.reset(token)


def is_headless_run() -> bool:
    """True when running inside a ``headless_run()`` scope (PRD-201 S5)."""
    return bool(_HEADLESS_RUN.get())


QUICK_ASK, LIVE = "quick-ask", "live"
_SPOKEN_TURN: contextvars.ContextVar[str] = contextvars.ContextVar("automatos_spoken_turn", default="")


def set_spoken_turn(surface: str) -> str:
    """Say which spoken surface (QUICK_ASK, LIVE, or "" for none) the LLM calls made from here on
    answer; returns what it was before, for the caller to put back."""
    before = _SPOKEN_TURN.get()
    _SPOKEN_TURN.set(surface or "")
    return before


def spoken_turn() -> str:
    """The spoken surface this call answers: QUICK_ASK, LIVE, or "" on a turn that is read."""
    return _SPOKEN_TURN.get()
