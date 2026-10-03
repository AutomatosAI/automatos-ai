"""F270 (night 7b): a playbook with a step that has no agent is refused before it runs.

Auto ran playbook 103, "New Cafe Onboarding", twice (#0183, #0184), and the
board's Run now ran #0183 again. Neither of its two steps has an agent, and each
run failed in under a second with "The run stopped on an internal error, so
nothing after it ran. The details are in the server log." The owner has two
playbooks with that name (102 has the Analyst on both steps), and nothing said
which one was broken, or that a step needed an agent.

``run_refusal`` gives the owner's words for it: the playbook by its number and
name, the steps that have no agent, and what to do. The Run button's route, the
rerun route and the board's redo refuse with those words before anything
changes. Every run, however it started (Auto's tool, the timer, a trigger),
meets them in ``api.recipe_executor.execute_recipe_direct`` through
``refused_before_it_runs``, before its first step. A fixed generate_document
step needs no agent (PRD-251 US-117).
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

from core.models.core import PLAYBOOK_DOCUMENT_STEP

logger = logging.getLogger(__name__)


def steps_without_an_agent(steps: Any) -> List[int]:
    """The step numbers (``order``, as the run names them) of the steps that need
    an agent and have none, in the order they run."""
    rows = [step for step in (steps if isinstance(steps, list) else []) if isinstance(step, dict)]
    ordered = sorted(rows, key=lambda step: _order(step) or 0)
    return [_order(step) or position + 1 for position, step in enumerate(ordered)
            if step.get("type") != PLAYBOOK_DOCUMENT_STEP and not step.get("agent_id")]


def _order(step: dict) -> Optional[int]:
    order = step.get("order")
    return order if isinstance(order, int) and not isinstance(order, bool) else None


def run_refusal(playbook: Any) -> Optional[str]:
    """Why ``playbook`` can't run, in the owner's words; None when it can."""
    missing = steps_without_an_agent(getattr(playbook, "steps", None))
    if not missing:
        return None
    if len(missing) == 1:
        which, what_to_do = f"step {missing[0]} has", f"Give step {missing[0]} an agent"
    else:
        named = ", ".join(str(order) for order in missing[:-1])
        which, what_to_do = f"steps {named} and {missing[-1]} have", "Give each of them an agent"
    return (f'Playbook {playbook.id} "{playbook.name}" can\'t run: {which} no agent. '
            f"{what_to_do}, then run it again.")


async def refused_before_it_runs(execution_id: str, recipe_id: Any, workspace_id: Any,
                                 db_url: Optional[str] = None) -> bool:
    """The net every run passes: a run whose playbook can't run fails before its
    first step, in ``run_refusal``'s words. Its card (made now when the run has
    none yet, as a timer's run hasn't) and the owner's bell say so
    (``api.recipe_executor._fail_execution``). True when the run was refused."""
    from api import recipe_executor as executor
    from core.models.core import WorkflowTemplate

    db, engine = _session(executor, db_url)
    try:
        playbook = db.query(WorkflowTemplate).filter(WorkflowTemplate.id == recipe_id,
                                                     WorkflowTemplate.workspace_id == workspace_id).first()
        words = run_refusal(playbook) if playbook is not None else None
        if words is None:
            return False
        logger.warning("[F270] run %s of playbook %s refused before its first step: %s", execution_id, recipe_id, words)
        _card_for(db, playbook, execution_id)
        await executor._fail_execution(db, execution_id, words)
        return True
    finally:
        db.close()
        if engine is not None:
            engine.dispose()


def _card_for(db: Any, playbook: Any, execution_id: str) -> None:
    """The run's card, as the run makes it when it starts (the bridge finds one
    Auto's tool already made, and leaves it)."""
    from sqlalchemy.exc import IntegrityError

    from core.models.core import RecipeExecution
    from services.board_task_bridge import create_recipe_board_task

    run = db.query(RecipeExecution).filter(RecipeExecution.execution_id == execution_id).first()
    if run is None:
        return
    try:
        create_recipe_board_task(db, playbook, run)
    except IntegrityError:  # made a moment ago by its starter (one card per run): it is there
        db.rollback()


def _session(executor: Any, db_url: Optional[str]):
    """A session the way the run opens its own (``db_url`` for a run bound to another
    database), and the engine to dispose of after, if one was made for it."""
    if not db_url:
        return executor.SessionLocal(), None
    engine = executor.create_engine(db_url)
    return executor.sessionmaker(autocommit=False, autoflush=False, bind=engine)(), engine
