"""A Playbook run's model calls are booked to the run, whoever started it.

F321 (night 9b, build 15): Auto started "Monday green stock" from chat (run
exec-45d8ac862a79, card #0102, 15:49:29Z). The run's task copied the chat turn's
context, so its ``usage_scope`` was the chat's, and the scope wins over the
agent manager's own ``request_type='recipe'``: all 19 of the step agent's calls
were booked as ``chat:dd0b6649…`` / ``chat`` (llm_usage rows 102294–102320), the
run's report found none of them, and its "LLM calls: 0 … Cost $0.0000" was wrong.
A cron run (cron-61e1db0e63b6, 3 Oct 06:00Z) was booked to a chat the same way.

The run now opens its own scope, inheriting nothing; each step names its agent.
"""
from __future__ import annotations

import functools
import inspect
from typing import Any, Awaitable, Callable

from core.llm.usage_context import LANE_RECIPE, usage_scope


def books_spend_to_the_run(run: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
    """Wrap ``api.recipe_executor.execute_recipe_direct``: every model call made
    while the run executes is booked as ``recipe`` under the run's own id."""
    signature = inspect.signature(run)

    @functools.wraps(run)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        given = signature.bind_partial(*args, **kwargs).arguments
        workspace_id = given.get("workspace_id")
        with usage_scope(request_type=LANE_RECIPE, execution_id=given.get("recipe_execution_id"),
                         workspace_id=str(workspace_id) if workspace_id else None, inherit=False):
            return await run(*args, **kwargs)
    return wrapped


def books_the_step_to_its_agent(execute: Callable[..., Awaitable[dict]]) -> Callable[..., Awaitable[dict]]:
    """Wrap ``api.recipe_executor._execute_step`` (called with keywords): the
    step's helper calls (tool routing, decisions) are booked to the step's agent."""
    @functools.wraps(execute)
    async def wrapped(*args: Any, **kwargs: Any) -> dict:
        with usage_scope(agent_id=getattr(kwargs.get("agent"), "id", None)):
            return await execute(*args, **kwargs)
    return wrapped


__all__ = ["books_spend_to_the_run", "books_the_step_to_its_agent"]
