"""What every run of an agent's work is told: its lessons, and where its answer goes
(F249 and F269, night 7b; F249, F288 and F297, night 8).

- F249: the owner's lessons reached an agent's next board card (F249, night 7: the
  dispatcher and the CLI host fold them in), never its next mission step or playbook
  step: the coordinator and the playbook runner build those prompts themselves. The
  Content Creator was told on #0176.11 "no line before it" and slipped on #0199.
- F269: answers landed in files, PDFs and reports instead of on the card: #0185 said
  only "saved to email_drafts/…md", #0188.1's working was only in a PDF, and #0194
  read "✅ Task Successfully Completed" with the email only in a delivery report.

Night 8:
- F249: playbook 102 ignored the owner's standing note on three runs running (#0440
  by Auto, #0441 and #0455 from the board). A playbook step now also carries the
  playbook's own standing notes, read from its last runs' cards, and the agent's
  lessons without them (services/ticket_redo.playbook_lessons).
- F288: the details Auto passed to playbook 102 (cafe_name, owner_email, first_order,
  delivery_day) reached no step, because no step's prompt names them. Every agent
  step of a run now gets them (services/playbook_given.py).
- F297: a plain board card's prompt never said where its answer goes. Cards got a
  description of a saved file (#0256, #0412), a tool's error as the answer's first
  line (#0346, #0360) and a raw Composio error as the whole answer (#0254). The
  card's launch now adds "Where your answer goes", which also says what to do when a
  tool fails.

Three wrappers, for the three ways the platform runs an agent's work through the API
(a Claude Code session's ticket has its own prompt, which already carries the lessons):
- ``a_steps_prompt_carries_its_lessons``: a mission step (``build_task_prompt``);
- ``a_playbook_step_carries_its_lessons``: a playbook step, a timer's included
  (``_execute_step``). A session agent's step gets the run's details only: its ticket
  shows the session nothing else of the run;
- ``a_cards_answer_goes_on_the_card``: a plain board card
  (``api.board_tasks._launch_task_execution``).
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

ON_THE_CARD = (
    "## Where your answer goes\n"
    "Your reply is what the owner reads on the card. Put the whole answer in it: the email, the table, "
    "the working, the figures. A file, a PDF or a report may go with it, never instead of it: never end "
    "with only \"saved to …\", \"see the report\" or \"task completed\".\n"
    "If a tool fails, never put its error, or notes about what you tried, in the answer: do the work another "
    "way, or say plainly what is missing."
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


def _run_of(db: Any, workspace_id: Any, execution_id: Any) -> Any:
    """The playbook run a step belongs to, in its workspace; None without a database
    session to read it from."""
    from sqlalchemy.orm import Session

    from core.models.core import RecipeExecution

    if not isinstance(db, Session) or not execution_id or workspace_id is None:
        return None
    return db.query(RecipeExecution).filter(RecipeExecution.execution_id == str(execution_id),
                                            RecipeExecution.workspace_id == workspace_id).first()


def _card_of(db: Any, run: Any) -> Optional[int]:
    """The id of the run's own card, which follows the run's redos (F243); None without one."""
    if run is None:
        return None
    from core.models.core import BoardTask
    from services.run_cancel import PLAYBOOK_CARD

    card = db.query(BoardTask.id).filter(BoardTask.workspace_id == run.workspace_id,
                                         BoardTask.source_type == PLAYBOOK_CARD,
                                         BoardTask.source_id == run.execution_id).first()
    return card.id if card else None


def _standing_notes(db: Any, workspace_id: Any, agent_id: Any, run: Any) -> List[Optional[str]]:
    """The agent's lessons, then the playbook's own standing notes (F249, night 8), each
    note once. The run's own card is in neither: a redo carries its notes already."""
    from services.ticket_redo import lessons_block, playbook_block, playbook_lessons

    card = _card_of(db, run)
    notes = playbook_lessons(db, workspace_id, getattr(run, "recipe_id", None), but_not=card)
    return [lessons_block(db, workspace_id, agent_id, but_not=card, besides=notes), playbook_block(notes)]


def _playbook_step_prompt(step: Dict[str, Any]) -> str:
    """A playbook step's prompt with what its agent is told besides: what the owner gave
    for the run (F288), the lessons and standing notes (F249) and where its answer goes
    (F269). A session agent's step gets the run's details only."""
    from services.playbook_given import given_for_run

    db, workspace_id = step.get("db"), step.get("workspace_id")
    agent_id = getattr(step.get("agent"), "id", None)
    in_a_session = _runs_in_a_session(db, agent_id)
    given = step.get("input_data")
    run = _run_of(db, workspace_id, step.get("recipe_execution_id")) if given or not in_a_session else None
    prompt = _with(step["clean_prompt"], given_for_run(db, run, given))
    if in_a_session:
        return prompt
    return _with(prompt, *_standing_notes(db, workspace_id, agent_id, run), ON_THE_CARD)


def a_playbook_step_carries_its_lessons(execute: Callable[..., Awaitable[dict]]) -> Callable[..., Awaitable[dict]]:
    """Wrap ``api.recipe_executor._execute_step`` (called with keywords): after the
    step's own prompt, what the owner gave for the run, the agent's lessons, the
    playbook's standing notes, then where its answer goes."""
    @functools.wraps(execute)
    async def wrapped(*args: Any, **kwargs: Any) -> dict:
        if "clean_prompt" in kwargs:
            kwargs = {**kwargs, "clean_prompt": _playbook_step_prompt(kwargs)}
        return await execute(*args, **kwargs)
    return wrapped


def a_cards_answer_goes_on_the_card(launch: Callable[..., None]) -> Callable[..., None]:
    """Wrap ``api.board_tasks._launch_task_execution`` (called with keywords): a plain
    board card's prompt ends with where its answer goes (F297, night 8). A Claude Code
    session's card is parked for its host before the prompt is read; the host builds
    the session's own prompt, which is left as a session's step is."""
    @functools.wraps(launch)
    def wrapped(*args: Any, **kwargs: Any) -> None:
        if isinstance(kwargs.get("prompt"), str):
            kwargs = {**kwargs, "prompt": _with(kwargs["prompt"], ON_THE_CARD)}
        return launch(*args, **kwargs)
    return wrapped


__all__ = ["ON_THE_CARD", "a_cards_answer_goes_on_the_card", "a_playbook_step_carries_its_lessons",
           "a_steps_prompt_carries_its_lessons"]
