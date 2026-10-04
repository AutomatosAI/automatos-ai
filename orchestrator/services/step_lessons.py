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
  description of a saved file (#0256, #0412, #0414, #0415), a tool's output as the
  answer (#0309, #0379, #0391, #0399, #0408.3, #0430), a tool's error as the answer's
  first line (#0346, #0360) and a raw Composio error as the whole answer (#0254). The
  card's launch now adds "Where your answer goes". Its wording asked for "the
  working" with every answer, against the owner's "just the table, no working": it
  now asks for the work itself, in the form the brief and the owner's notes ask for,
  with a saved file beside it, never instead of it, and says what to do when a tool
  fails. (Lifting an answer out of a tool's result is the tool loop's job: FIXER's.)

Night 9 (F300, F304 and F313, services/answer_sources.py): "Where your answer goes" is
followed by "Where your facts come from": a "now" question is answered from the live
system before a dated document, a database's tables and columns are read through its
tool and never asked of the owner, and only what a tool returned in the run is cited.

Three wrappers, for the three ways the platform runs an agent's work through the API
(a Claude Code session's ticket has its own prompt, which already carries the lessons):
- ``a_steps_prompt_carries_its_lessons``: a mission step (``build_task_prompt``);
- ``a_playbook_step_carries_its_lessons``: a playbook step, a timer's included
  (``_execute_step``). A session agent's step gets the run's details and the
  playbook's notes; its ticket brings the agent's lessons;
- ``a_cards_answer_goes_on_the_card``: a plain board card
  (``api.board_tasks._launch_task_execution``), and ``a_cards_run_carries_its_lessons``
  for a card launched without its lessons (Auto's move to In progress).
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional

from services.answer_sources import FACTS_RULES

logger = logging.getLogger(__name__)

ON_THE_CARD = (
    "## Where your answer goes\n"
    "Your reply is the card's answer: the owner reads it on the card as it is. Put the work itself in it (the "
    "email, the table, the figures, the post), in the form the brief and the owner's notes ask for, starting with "
    "the work: no line before it about what you did, a tool or a skill.\n"
    "A file, a PDF or a report you save goes alongside the answer, never instead of it: never end with only "
    "\"saved to …\", a description of what you saved, \"see the report\", \"task completed\" or a tool's output.\n"
    "If a tool fails, leave its error and what you tried out of the answer: do the work another way, or say "
    "plainly what is missing.\n\n"
    f"{FACTS_RULES}"     # F300, F304 and F313 (night 9): the live system first, its schema read, sources read
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
    from services.ticket_redo import MISSION_STEP_ASK, lessons_block

    db, agent_id = _session_of(task), getattr(task, "assigned_agent_id", None)
    if db is None or not agent_id:
        return None
    run = db.get(OrchestrationRun, task.run_id) if getattr(task, "run_id", None) else None
    own = db.query(BoardTask.id).filter(BoardTask.orchestration_task_id == task.id).first()
    words = f"{getattr(task, 'title', '') or ''} {getattr(task, 'description', '') or ''}"
    return lessons_block(db, getattr(run, "workspace_id", None), agent_id, but_not=own.id if own else None,
                         ask=MISSION_STEP_ASK, for_text=words)


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


def _standing_notes(db: Any, workspace_id: Any, agent_id: Any, run: Any, *,
                    in_a_session: bool, for_text: str = "") -> List[Optional[str]]:
    """The agent's lessons, then the playbook's own standing notes (F249, night 8), each
    note once. The run's own card is in neither: a redo carries its notes already. A
    session agent's ticket brings its own lessons (the CLI host's ``_ticket_prompt``),
    so its step gets the playbook's notes only."""
    from services.ticket_redo import lessons_block, playbook_block, playbook_lessons

    card = _card_of(db, run)
    notes = playbook_lessons(db, workspace_id, getattr(run, "recipe_id", None), but_not=card)
    if in_a_session:
        return [playbook_block(notes)]
    return [lessons_block(db, workspace_id, agent_id, but_not=card, besides=notes, for_text=for_text),
            playbook_block(notes)]


def _playbook_step_prompt(step: Dict[str, Any]) -> str:
    """A playbook step's prompt with what its agent is told besides: what the owner gave
    for the run (F288), the lessons and standing notes (F249) and where its answer goes
    (F269). A session agent's step gets the run's details and the playbook's notes."""
    from services.playbook_given import given_for_run

    db, workspace_id = step.get("db"), step.get("workspace_id")
    agent_id = getattr(step.get("agent"), "id", None)
    in_a_session = _runs_in_a_session(db, agent_id)
    run = _run_of(db, workspace_id, step.get("recipe_execution_id"))
    prompt = _with(step["clean_prompt"], given_for_run(db, run, step.get("input_data")),
                   *_standing_notes(db, workspace_id, agent_id, run, in_a_session=in_a_session,
                                    for_text=step["clean_prompt"]))
    return prompt if in_a_session else _with(prompt, ON_THE_CARD)


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


def a_cards_run_carries_its_lessons(read: Callable[..., Awaitable[str]]) -> Callable[..., Awaitable[str]]:
    """Wrap ``services.draft_guides.guides_for_draft``, which every API run of a plain
    card calls with its database session before its first model call
    (``api.board_tasks._launch_task_execution``): a prompt that came without the
    agent's lessons gets them, before where its answer goes.

    F249 (night 8): Auto's move of a card to In progress launches its bare brief
    (modules/tools/discovery/handlers_board_tasks.py), without the lessons the board's
    claim folds in (services/ticket_redo.redo_block): #0346 ran again that way."""
    @functools.wraps(read)
    async def wrapped(db: Any, workspace_id: Any, agent_id: int, brief: str) -> str:
        prompt = await read(db, workspace_id, agent_id, brief)
        return _with_lessons(db, workspace_id, agent_id, prompt, brief)
    return wrapped


def _with_lessons(db: Any, workspace_id: Any, agent_id: Any, prompt: str, brief: str) -> str:
    """``prompt`` with the agent's lessons before "Where your answer goes", unless it
    carries them already or there is no database session to read them from. A card's
    own names are read from its ``brief``, not the prompt: the guide passages a draft's
    prompt gains name the owner's people and places, so any draft would share them (F318)."""
    from uuid import UUID

    from sqlalchemy.orm import Session

    from services.ticket_redo import STANDING_HEADING, lessons_block

    if STANDING_HEADING in prompt or not isinstance(db, Session) or not workspace_id or not agent_id:
        return prompt
    lessons = lessons_block(db, UUID(str(workspace_id)), agent_id, for_text=brief)   # F315: the card's own words
    if not lessons:
        return prompt
    if ON_THE_CARD in prompt:
        return prompt.replace(ON_THE_CARD, f"{lessons}\n\n{ON_THE_CARD}", 1)
    return _with(prompt, lessons)


__all__ = ["ON_THE_CARD", "a_cards_answer_goes_on_the_card", "a_cards_run_carries_its_lessons", "a_playbook_step_carries_its_lessons",
           "a_steps_prompt_carries_its_lessons"]
