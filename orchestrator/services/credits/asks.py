"""Asks: one user turn, or one run of work, and the credits it may still spend (PRD-265).

Credits are metered (Gerard, 20 Oct 2026): every platform-funded model call costs
its real cost, ``llm_usage.total_cost``, in cents × ``CREDITS_MARKUP``, debited by
the meter (``services/credits/meter.py``). An **ask** — a chat turn with every
agent, tool and sub-call it causes, or one run of a mission task, Task, scheduled
task, heartbeat or playbook — groups those calls, so a reply or a run can say
what it cost, and so a long run stops when the workspace's credits run out
instead of overdrawing them.

:func:`ask_scope` opens one: it mints an ``ask_id`` and puts it on the usage scope,
so every ``llm_usage`` row written inside carries it. A scope opened inside another
ask joins it (a sub-agent is part of the turn that called it) unless ``new=True``.

``AskState`` lives in this process only (a registry keyed by ``ask_id``). The gate
(``gate.before_platform_call``) reads the workspace's balance at the ask's first
platform-funded call and keeps it as the ask's budget; every call then adds its
cost, and the next call is refused once the running cost reaches the budget. A
call that finds the ask in its scope but not in this process (the ask already
ended, or another worker) is checked against the balance again.
"""
from __future__ import annotations

import functools
import inspect
import logging
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Iterator, Optional

logger = logging.getLogger(__name__)

# The ask's lane when the run named none.
LANE_AGENT_RUN = "agent_run"
# Lanes nobody is waiting on: never paid from the trial (PRD-222 FR-17). PRD-268 K14: a
# check-in (services/proactive/runner.py) is one of them.
BACKGROUND_LANES = frozenset({"heartbeat", "scheduled_task", "check_in"})

ASK_ID_MAX = 64


@dataclass
class AskState:
    """One live ask in this process. Mutable on purpose: the gate sets its budget
    and every call adds its cost. Guarded by ``_LOCK``."""

    ask_id: str
    lane: str
    workspace_id: Optional[str]
    background: bool = False
    checked: bool = False
    funded_by: Optional[str] = None
    budget_credits: int = 0
    own_keys_plan: bool = False
    cost_usd: float = 0.0


_ASKS: dict[str, AskState] = {}
_LOCK = threading.Lock()


def is_background_lane(lane: Optional[str]) -> bool:
    return (lane or "") in BACKGROUND_LANES


def current_ask_id() -> Optional[str]:
    from core.llm.usage_context import current_usage_scope

    return current_usage_scope().get("ask_id")


def current_ask() -> Optional[AskState]:
    ask_id = current_ask_id()
    return _ASKS.get(ask_id) if ask_id else None


def _new_ask_id(lane: str) -> str:
    return f"{lane}:{uuid.uuid4().hex}"[:ASK_ID_MAX]


@contextmanager
def ask_scope(
    lane: str, *, workspace_id: Any = None, background: bool = False, new: bool = False,
) -> Iterator[AskState]:
    """Run the block as one ask (or as part of the ask already in scope)."""
    existing = current_ask()
    if existing is not None and not new:
        yield existing
        return
    from core.llm.usage_context import usage_scope

    state = AskState(ask_id=_new_ask_id(lane), lane=lane,
                     workspace_id=str(workspace_id) if workspace_id else None, background=background)
    with _LOCK:
        _ASKS[state.ask_id] = state
    try:
        with usage_scope(ask_id=state.ask_id, workspace_id=workspace_id):
            yield state
    finally:
        with _LOCK:
            _ASKS.pop(state.ask_id, None)


def runs_as_an_ask(
    lane: str, *, background: bool = False, workspace_arg: str = "workspace_id",
) -> Callable[[Callable[..., Awaitable[Any]]], Callable[..., Awaitable[Any]]]:
    """Decorate a coroutine that is one run of work (a heartbeat tick) so it runs
    as one ask, for the workspace named by its ``workspace_arg`` argument."""
    def wrap(run: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        signature = inspect.signature(run)

        @functools.wraps(run)
        async def wrapped(*args: Any, **kwargs: Any) -> Any:
            workspace_id = signature.bind_partial(*args, **kwargs).arguments.get(workspace_arg)
            with ask_scope(lane, workspace_id=workspace_id, background=background):
                return await run(*args, **kwargs)
        return wrapped
    return wrap


def set_budget(state: Optional[AskState], *, funded_by: str, budget_credits: int = 0,
               own_keys_plan: bool = False) -> None:
    """Record on the live ask how it is paid for and what it may spend."""
    if state is None:
        return
    with _LOCK:
        state.checked = True
        state.funded_by = funded_by
        state.budget_credits = int(budget_credits)
        state.own_keys_plan = own_keys_plan


def note_cost(cost_usd: float) -> None:
    """Add one platform-funded call's cost to the ask in scope (the budget reads it)."""
    state = current_ask()
    if state is None or not cost_usd or cost_usd <= 0:
        return
    with _LOCK:
        state.cost_usd += float(cost_usd)
