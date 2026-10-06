"""The hierarchy gate checks the ticket an agent's call will change (Gerard, 7 Oct).

A ticket reference an agent gives ("#0892", "0892", "892") is the ticket with that board
number (``services.ticket_numbers.read_bare_refs``), and the ticket tools read it so
(``services.ticket_refs.by_ticket_number``). The executor's hierarchy gate (PRD-140) read the
same ``task_id`` as a row id: "892" had #0708's owner checked for a call that changes #0892,
and "#0892" named no row, so every agent's call by number went to Auto.

So, for a call from an agent's platform tool, the gate reads the ticket the way the tools do,
before it looks up the ticket's owner. A ref that names no ticket is checked as given. The
gate's own callers with a row id (and every other target) are unchanged.
"""
from __future__ import annotations

import functools
from typing import Any, Callable

# The executor's source for an agent's platform tool call (platform_executor._run_cleared).
PLATFORM_TOOL = "platform_tool"

Gate = Callable[..., Any]


def reads_the_ticket_named(task_target: str) -> Callable[[Gate], Gate]:
    """Wrap ``can_actor_modify``: a ``task_target`` an agent's tool call names is the ticket the tools act on."""
    def decorate(gate: Gate) -> Gate:
        @functools.wraps(gate)
        def wrapped(db: Any, **kwargs: Any) -> Any:
            if kwargs.get("target_type") == task_target and kwargs.get("source") == PLATFORM_TOOL:
                kwargs = {**kwargs, "target_id": ticket_named(db, kwargs.get("workspace_id"), kwargs.get("target_id"))}
            return gate(db, **kwargs)
        return wrapped
    return decorate


def ticket_named(db: Any, workspace_id: Any, target_id: Any) -> Any:
    """The id of the ticket ``target_id`` names, read as the ticket tools read it; ``target_id``
    as given when it names none. In a savepoint, as the gate's owner lookup is: a lookup that
    fails leaves the call's transaction usable, and the gate denies the call."""
    if target_id in (None, ""):
        return target_id
    from services.ticket_refs import ticket_id_named

    with db.begin_nested():
        found, _ = ticket_id_named(db, workspace_id, target_id)
    return target_id if found is None else found


__all__ = ["PLATFORM_TOOL", "reads_the_ticket_named", "ticket_named"]
