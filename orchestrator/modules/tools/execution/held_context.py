"""P256-FIX-RVW-2: a Composio call checked against the intent keeps the caller's context.

``ToolRouter.execute_and_format`` sends a ``composio_*`` call with an intent through
``execute_tool_with_validation``, which ran it with no caller context. The owner's-click
gate (``owner_only.asks_before_a_send``) reads only that server-built context: who drove
the turn (``driving_user_id``), and the ticket or playbook run the call works for
(``agent_sends.autos_ticket``). Without it a chat's ``composio_execute`` send, and every
Composio action of a playbook step, ran as a call made for nobody, and never asked.

The router's own call holds its context here for the length of the call
(:func:`holds_the_callers_context`, a decorator, so the router's body is untouched), and
the validation path forwards it (:func:`the_callers_context`). A router call made with no
context of its own inside a run's step (:func:`acts_for_its_run`) is made for that run: a
playbook step dispatches the model's own ``composio_execute`` (its hint path, or any call
outside the SDK's matches) with none.
"""
from __future__ import annotations

import functools
import inspect
from contextvars import ContextVar
from typing import Any, Awaitable, Callable, Dict, Optional

Call = Callable[..., Awaitable[Dict[str, Any]]]
CALLER_CONTEXT = "caller_context"

_held: ContextVar[Optional[Dict[str, Any]]] = ContextVar("tool_router_caller_context", default=None)
_run: ContextVar[Optional[Dict[str, Any]]] = ContextVar("tool_router_run_context", default=None)


def holds_the_callers_context(execute_and_format: Call) -> Call:
    """Wrap ``ToolRouter.execute_and_format``: its ``caller_context`` (else its run's) is held
    for the call (nested calls each hold their own, and the outer one is back when they return)."""
    signature = inspect.signature(execute_and_format)

    @functools.wraps(execute_and_format)
    async def wrapped(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        bound = signature.bind(*args, **kwargs)
        context = bound.arguments.get(CALLER_CONTEXT)
        token = _held.set(context if isinstance(context, dict) else _run.get())
        try:
            return await execute_and_format(*args, **kwargs)
        finally:
            _held.reset(token)
    return wrapped


def acts_for_its_run(context_of: Callable[..., Optional[Dict[str, Any]]]) -> Callable[[Call], Call]:
    """Wrap a run's step: a router call it makes with no context of its own is made for the
    run, ``context_of(**the step's arguments)`` (None: for nobody, as before)."""
    def decorate(execute: Call) -> Call:
        signature = inspect.signature(execute)

        @functools.wraps(execute)
        async def wrapped(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            token = _run.set(context_of(**bound.arguments))
            try:
                return await execute(*args, **kwargs)
            finally:
                _run.reset(token)
        return wrapped
    return decorate


def the_callers_context() -> Optional[Dict[str, Any]]:
    """The context of the router call this runs in, or None outside one."""
    return _held.get()


__all__ = ["acts_for_its_run", "holds_the_callers_context", "the_callers_context"]
