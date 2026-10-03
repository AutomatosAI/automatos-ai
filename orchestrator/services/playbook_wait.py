"""F242 (night 7): "wait for me" holds on a playbook's runs.

Every playbook card closed itself. The card was made with review_mode ``auto``
and the run's end wrote Done whatever the card said: #0112 was set to wait and
went Done in 8 s; a timer's runs never reached the owner (#0057's £3,500-4,000
"order today", #0162, #0170); and #0145's answer ended on a question to the
owner ("Would you like me to help draft the actual purchase orders…?") and went
straight to Done. A playbook had no setting to wait, so Auto wrote "wait for
me" into a step's prompt, which nothing reads.

A run waits for the owner when it was started asking it (its
``execution_metadata``), else when its playbook says so (its
``execution_config``), or when its card is set to wait. A run that waits, or
whose answer ends on a question to the owner, ends in Review, saying why.
"""
from __future__ import annotations

from typing import Any, Optional

WAIT_FOR_ME = "wait_for_me"
OWNER_REVIEWS = "human"
CLOSES_ITSELF = "auto"


def playbook_waits(recipe: Any, execution: Any) -> bool:
    """Whether a run of ``recipe`` waits for the owner: the run's own request
    when it made one, else the playbook's setting."""
    asked = (getattr(execution, "execution_metadata", None) or {}).get(WAIT_FOR_ME)
    if isinstance(asked, bool):
        return asked
    return (getattr(recipe, "execution_config", None) or {}).get(WAIT_FOR_ME) is True


def card_review_mode(recipe: Any, execution: Any) -> str:
    """The review mode a run's card is made with."""
    return OWNER_REVIEWS if playbook_waits(recipe, execution) else CLOSES_ITSELF


def why_it_waits(db: Any, card: Any, execution_id: str, result: Any) -> Optional[str]:
    """Why a run that finished well ends in Review rather than Done: its card is
    set to wait for the owner, or its answer ends on a question to them. None for
    Done. The codes are core.services.ticket_reasons'."""
    from core.services.ticket_reasons import ASKED, ENDS_ON_A_QUESTION
    from services.playbook_owner_ask import ends_with_a_question, writes_to_someone

    if getattr(card, "review_mode", None) == OWNER_REVIEWS:
        return ASKED
    if ends_with_a_question(result) and not writes_to_someone(
            last_step_prompt(db, execution_id, getattr(card, "workspace_id", None))):
        return ENDS_ON_A_QUESTION
    return None


def last_step_prompt(db: Any, execution_id: str, workspace_id: Any) -> str:
    """The prompt of the run's last step (an answer that ends on a question is the
    deliverable when that step writes to someone: F140's rule)."""
    from core.models.core import RecipeExecution, WorkflowTemplate

    run = db.query(RecipeExecution).filter(RecipeExecution.execution_id == execution_id,
                                           RecipeExecution.workspace_id == workspace_id).first()
    recipe_id = getattr(run, "recipe_id", None)
    playbook = db.query(WorkflowTemplate).filter(
        WorkflowTemplate.id == recipe_id, WorkflowTemplate.workspace_id == run.workspace_id).first() if recipe_id else None
    steps = [step for step in (getattr(playbook, "steps", None) or []) if isinstance(step, dict)]
    steps = sorted(steps, key=lambda step: step.get("order") or 0)
    return str(steps[-1].get("prompt_template") or "") if steps else ""
