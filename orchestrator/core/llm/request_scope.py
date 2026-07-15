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
