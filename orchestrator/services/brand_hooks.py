"""Where the platform applies the brand kit at generation (night 9b, prep for night 10).

``services.brand_rules`` holds what is applied; these wrappers are where, one per
place the platform drafts work or ends it, so no long function's body changes:

* the prompt: a board card's run (:func:`card_prompt_with_brand_rules`, on both claim
  paths: the dispatcher's and a Claude Code session's), a mission step's
  (``MissionDispatcher.build_task_prompt``) and a playbook step's
  (``api.recipe_executor._execute_step``) get :func:`brand_rules.brand_rules_block`,
  once, after the agent's lessons (``services.step_lessons``);
* the finished text, before anything else reads it: a card's answer as
  ``finalize_board_task_run`` is handed it (before its notes, so a filled sign-off is
  never also reported as a leftover placeholder), a mission step's as
  ``record_task_completion`` stores it, and a playbook step's as its step returns it;
* a document (the wrappers live beside the code they wrap, which never imports
  ``services`` at import time): its data's placeholder signature is filled before it
  renders (``modules.documents.brand_signing``), and an agent's generate_document call
  is told the banned words its document uses
  (``modules.tools.execution.document_brand_check``).

A Claude Code session agent's mission or playbook step is a ticket: its prompt gets the
rules on the session's claim path, so the step's own prompt does not repeat them.
"""
from __future__ import annotations

import asyncio
import functools
import logging
from typing import Any, Awaitable, Callable, Dict

from services import brand_rules as br

logger = logging.getLogger(__name__)

Async = Callable[..., Awaitable[Any]]


def _session_of(obj: Any) -> Any:
    from sqlalchemy.orm import object_session
    from sqlalchemy.orm.exc import UnmappedInstanceError

    try:
        return object_session(obj)
    except UnmappedInstanceError:      # a plain object standing in for a row
        return None


def _mission_workspace(db: Any, task: Any) -> Any:
    """The workspace of a mission step (its run's)."""
    from core.models.orchestration import OrchestrationRun

    run_id = getattr(task, "run_id", None)
    if not run_id or not callable(getattr(db, "get", None)):
        return None
    with br.without_flushing(db):
        return getattr(db.get(OrchestrationRun, run_id), "workspace_id", None)


def card_prompt_with_brand_rules(prompt: str, task: Any) -> str:
    """A board card's prompt with the brand's rules after it, once. Both claim paths end a
    card's prompt with it: the dispatcher's (``board_dispatcher._claim_and_sweep``) and a
    Claude Code session's (``cli_host_service._ticket_prompt``)."""
    return br.with_brand_rules(prompt, _session_of(task), getattr(task, "workspace_id", None))


def _runs_in_a_session(db: Any, agent_id: Any) -> bool:
    """A Claude Code session agent: its ticket's own prompt carries the rules (the claim path)."""
    from services.cli_ticket_lane import is_cli_agent

    if not agent_id or db is None:
        return False
    with br.without_flushing(db):
        return bool(is_cli_agent(db, agent_id))


def a_steps_prompt_carries_the_brand_rules(build: Callable[..., str]) -> Callable[..., str]:
    """Wrap ``MissionDispatcher.build_task_prompt``: the step's prompt, then the brand's rules."""
    @functools.wraps(build)
    def wrapped(task: Any, *args: Any, **kwargs: Any) -> str:
        prompt = build(task, *args, **kwargs)
        db = _session_of(task)
        if _runs_in_a_session(db, getattr(task, "assigned_agent_id", None)):
            return prompt
        return br.with_brand_rules(prompt, db, _mission_workspace(db, task))
    return wrapped


def a_mission_steps_answer_is_on_brand(record: Callable[..., None]) -> Callable[..., None]:
    """Wrap ``MissionDispatcher.record_task_completion``: the step's answer is stored on brand."""
    @functools.wraps(record)
    def wrapped(db: Any, task: Any, result: Dict[str, Any]) -> None:
        if not isinstance(result, dict) or result.get("status") != "success":
            return record(db, task, result)
        return record(db, task, br.on_brand_result(db, _mission_workspace(db, task), result))
    return wrapped


def a_playbook_step_is_on_brand(execute: Async) -> Async:
    """Wrap ``api.recipe_executor._execute_step`` (called with keywords): the brand's rules
    after the step's prompt, and its answer on brand. Its reads run off the event loop."""
    @functools.wraps(execute)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        db, workspace_id = kwargs.get("db"), kwargs.get("workspace_id")
        agent_id = getattr(kwargs.get("agent"), "id", None)
        if "clean_prompt" in kwargs and not await asyncio.to_thread(_runs_in_a_session, db, agent_id):
            prompt = await br.with_brand_rules_off_loop(kwargs["clean_prompt"], db, workspace_id)
            kwargs = {**kwargs, "clean_prompt": prompt}
        return await br.on_brand_result_off_loop(db, workspace_id, await execute(*args, **kwargs))
    return wrapped


def a_cards_answer_is_on_brand(finalize: Async) -> Async:
    """Wrap ``api.board_tasks.finalize_board_task_run`` (keywords after ``db``): the answer
    it writes is on brand before any of its own checks read it. The kit is read off the loop."""
    @functools.wraps(finalize)
    async def wrapped(db: Any, *args: Any, **kwargs: Any) -> Any:
        if "exec_result" in kwargs:
            result = await br.on_brand_result_off_loop(db, kwargs.get("workspace_id"), kwargs["exec_result"])
            kwargs = {**kwargs, "exec_result": result}
        return await finalize(db, *args, **kwargs)
    return wrapped


__all__ = [
    "a_cards_answer_is_on_brand", "a_mission_steps_answer_is_on_brand", "a_playbook_step_is_on_brand",
    "a_steps_prompt_carries_the_brand_rules", "card_prompt_with_brand_rules",
]
