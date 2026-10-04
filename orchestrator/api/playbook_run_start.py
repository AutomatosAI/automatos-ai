"""What the Run button's route (``api.workflow_recipes.execute_recipe``) decides before a run starts.

- Which playbook runs: the one at its address, by template id or by number
  (F277), with steps, and none of them without an agent (F270: refused in
  ``services.playbook_run_refusal``'s words, before anything changes).
- Whether this run waits for the owner (F242, night 7b). The Run button had no
  "wait for me": #0185 went straight to Done with review_mode auto, while the
  timer's card on a playbook set to wait stopped in Review (#0195). The body's
  ``wait_for_me`` is this run's own choice, kept on the run where
  ``services.playbook_wait`` reads it when the run makes its card; left out, the
  playbook's own setting decides, as it does for a timer's run and for Auto's.
- What the run is called in the answer (F294, night 8): "Recipe execution started
  (direct mode)" and an exec code, 10 runs of 10, while the owner works by card
  number. The run's card is made as the run is started, as Auto's run tool makes
  it (F241), and the answer names it.
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, Optional
from uuid import uuid4

from fastapi import HTTPException
from sqlalchemy.orm import Session

from api.playbook_address import playbook_at
from core.models.core import RecipeExecution, WorkflowTemplate
from services.playbook_run_refusal import run_refusal
from services.playbook_wait import WAIT_FOR_ME

NO_STEPS = "This playbook has no steps to run. Add a step, then run it again."
WAIT_FOR_ME_IS_TRUE_OR_FALSE = "wait_for_me must be true or false"
RUN_BUTTON_RUN = "recipe_direct"
RUN_CARD = "recipe"

logger = logging.getLogger(__name__)


def runnable_playbook(db: Session, workspace_id: Any, address: str) -> WorkflowTemplate:
    """The workspace's playbook at ``address``, when it can run: a 404 when there
    is none, a 400 when it has no steps or a step has no agent."""
    playbook = playbook_at(db, workspace_id, address)
    if not playbook.steps:
        raise HTTPException(status_code=400, detail=NO_STEPS)
    refuse_if_it_cannot_run(playbook)
    return playbook


def refuse_if_it_cannot_run(playbook: Any) -> None:
    """A 400 in the owner's words when a step of ``playbook`` has no agent (F270)."""
    words = run_refusal(playbook)
    if words:
        raise HTTPException(status_code=400, detail=words)


def wait_request(body: Optional[Dict[str, Any]]) -> Optional[bool]:
    """This run's own "wait for me": True or False when the body says, None when it
    says nothing (the playbook's own setting decides). A 400 for anything else."""
    wants = (body or {}).get(WAIT_FOR_ME)
    if wants is None or isinstance(wants, bool):
        return wants
    raise HTTPException(status_code=400, detail=WAIT_FOR_ME_IS_TRUE_OR_FALSE)


def start_run_row(db: Session, ctx: Any, playbook: Any, input_data: Dict[str, Any],
                  wants: Optional[bool]) -> str:
    """The run's row, pending, carrying the owner's "wait for me" when they gave
    one; the playbook's use counted. Committed (the run opens a session of its
    own). Returns the run's execution id."""
    execution_id = f"exec-{uuid4().hex[:12]}"
    metadata = {"execution_type": RUN_BUTTON_RUN, "total_steps": len(playbook.steps)}
    execution = RecipeExecution(
        execution_id=execution_id,
        recipe_id=playbook.id,
        workspace_id=ctx.workspace_id,
        status="pending",
        input_data=input_data,
        current_step=0,
        triggered_by=ctx.user.email if ctx.user else "anonymous",
        execution_metadata=metadata if wants is None else {**metadata, WAIT_FOR_ME: wants},
    )
    db.add(execution)
    playbook.use_count = (playbook.use_count or 0) + 1
    playbook.last_used_at = datetime.now()
    db.commit()
    _make_its_card(db, playbook, execution)
    return execution_id


def _make_its_card(db: Session, playbook: Any, execution: Any) -> None:
    """The run's card, made now so the answer can name it (F294). The run makes it a
    moment later otherwise, and finds it made. A card that can't be made now is
    the run's to make, as before: the answer then names none."""
    from services.board_task_bridge import create_recipe_board_task

    try:
        create_recipe_board_task(db, playbook, execution)
    except Exception:  # noqa: BLE001 — the run still starts, and makes its card itself
        db.rollback()
        logger.exception("[playbook-run] the card of run %s was not made at its start", execution.execution_id)


def run_started(db: Session, playbook: Any, execution_id: str) -> Dict[str, Any]:
    """The Run button's answer in words, naming the card the run works on by its
    number (F294), with the card's id and number."""
    from core.models.core import BoardTask
    from services.ticket_numbers import ticket_number

    card = db.query(BoardTask).filter(BoardTask.workspace_id == playbook.workspace_id,
                                      BoardTask.source_type == RUN_CARD,
                                      BoardTask.source_id == execution_id).first()
    number = ticket_number(db, card) if card is not None else None
    where = f" on card {number}" if number else ""
    return {"message": f'Playbook {playbook.id} "{playbook.name}" is running{where}.',
            "task_id": card.id if card is not None else None, "number": number}
