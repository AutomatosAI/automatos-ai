"""What every run of an agent's work is told: its lessons, and where its answer goes
(F249 and F269, night 7b).

- F249: the owner's lessons reached an agent's next board card (F249, night 7: the
  dispatcher and the CLI host fold them in), never its next mission step or playbook
  step: the coordinator and the playbook runner build those prompts themselves. The
  Content Creator was told on #0176.11 "no line before it" and slipped on #0199.
- F269: answers landed in files, PDFs and reports instead of on the card: #0185 said
  only "saved to email_drafts/…md", #0188.1's working was only in a PDF, and #0194
  read "✅ Task Successfully Completed" with the email only in a delivery report.

Two wrappers, for the two ways the platform runs an agent's work that the board's
claim paths don't (a Claude Code session's ticket has its own prompt, which already
carries the lessons):
- ``a_steps_prompt_carries_its_lessons``: a mission step (``build_task_prompt``);
- ``a_playbook_step_carries_its_lessons``: a playbook step, a timer's included
  (``_execute_step``).
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)

ON_THE_CARD = (
    "## Where your answer goes\n"
    "Your reply is what the owner reads on the card. Put the whole answer in it: the email, the table, "
    "the working, the figures. A file, a PDF or a report may go with it, never instead of it: never end "
    "with only \"saved to …\", \"see the report\" or \"task completed\"."
)


def _with(prompt: str, *blocks: Optional[str]) -> str:
    return "\n\n".join([prompt, *(b for b in blocks if b)])


def _session_of(obj: Any) -> Any:
    from sqlalchemy.orm import object_session
    from sqlalchemy.orm.exc import UnmappedInstanceError

    try:
        return object_session(obj)
    except UnmappedInstanceError:      # a plain object standing in for a row
        return None


def _runs_in_a_session(db: Any, agent_id: Any) -> bool:
    """A Claude Code session agent: its ticket's own prompt carries all this."""
    from services.cli_ticket_lane import is_cli_agent

    return bool(agent_id) and db is not None and is_cli_agent(db, agent_id)


def _mission_step_lessons(task: Any) -> Optional[str]:
    """The lessons for a mission step's agent, from its other cards."""
    from core.models.core import BoardTask
    from core.models.orchestration import OrchestrationRun
    from services.ticket_redo import lessons_block

    db, agent_id = _session_of(task), getattr(task, "assigned_agent_id", None)
    if db is None or not agent_id:
        return None
    run = db.get(OrchestrationRun, task.run_id) if getattr(task, "run_id", None) else None
    own = db.query(BoardTask.id).filter(BoardTask.orchestration_task_id == task.id).first()
    return lessons_block(db, getattr(run, "workspace_id", None), agent_id, but_not=own.id if own else None)


def a_steps_prompt_carries_its_lessons(build: Callable[..., str]) -> Callable[..., str]:
    """Wrap ``MissionDispatcher.build_task_prompt``: the step's agent's lessons, then
    where its answer goes. A session agent's step is its ticket's to tell."""
    @functools.wraps(build)
    def wrapped(task: Any, *args: Any, **kwargs: Any) -> str:
        prompt = build(task, *args, **kwargs)
        if _runs_in_a_session(_session_of(task), getattr(task, "assigned_agent_id", None)):
            return prompt
        return _with(prompt, _mission_step_lessons(task), ON_THE_CARD)
    return wrapped


def a_playbook_step_carries_its_lessons(execute: Callable[..., Awaitable[dict]]) -> Callable[..., Awaitable[dict]]:
    """Wrap ``api.recipe_executor._execute_step`` (called with keywords): the step's
    agent's lessons, then where its answer goes, after the step's own prompt."""
    @functools.wraps(execute)
    async def wrapped(*args: Any, **kwargs: Any) -> dict:
        from services.ticket_redo import lessons_block

        db, agent = kwargs.get("db"), kwargs.get("agent")
        agent_id = getattr(agent, "id", None)
        if "clean_prompt" in kwargs and not _runs_in_a_session(db, agent_id):
            lessons = lessons_block(db, kwargs.get("workspace_id"), agent_id)
            kwargs = {**kwargs, "clean_prompt": _with(kwargs["clean_prompt"], lessons, ON_THE_CARD)}
        return await execute(*args, **kwargs)
    return wrapped


__all__ = ["ON_THE_CARD", "a_playbook_step_carries_its_lessons", "a_steps_prompt_carries_its_lessons"]
