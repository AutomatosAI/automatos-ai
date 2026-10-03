"""platform_create_task's auto_approve runs only what the tool can run.

With auto_approve, create_board_task executes the ticket's approval_action on the
spot, but it can only publish a blog post. For any other type (create_blog, or one
a model made up) it still marked the ticket done and answered success with
action_executed, though nothing ran. auto_approve now holds for publish_blog
alone: any other approval action waits in Review, where the owner's approval runs
it (api/board_tasks._run_approval_action).
"""
from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable, Dict, Optional

Handler = Callable[[Any, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

RUNS_ON_CREATE = "publish_blog"


def _approval_action(params: Dict[str, Any]) -> Optional[Any]:
    """The approval action as create_board_task reads it: planning_data first."""
    planning = params.get("planning_data")
    if planning:
        return planning.get("approval_action") if isinstance(planning, dict) else None
    return params.get("approval_action")


def auto_approves_only_what_it_runs(handler: Handler) -> Handler:
    """platform_create_task: ``auto_approve`` is kept only for a publish_blog approval."""
    @functools.wraps(handler)
    async def wrapped(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        params = params or {}
        action = _approval_action(params)
        runs = isinstance(action, dict) and action.get("type") == RUNS_ON_CREATE
        if params.get("auto_approve") and not runs:
            params = {**params, "auto_approve": False}
        return await handler(db, workspace_id, params)
    return wrapped
