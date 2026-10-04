"""A playbook's last agent step gives the run's whole answer (F321, night 9b).

The run's card shows its last step's answer. Runs #0085 and #0102 put two of eight
coffees on the card ("saved to the scratchpad"), because the last step summed up its
own step and left what the earlier steps had found behind. The last agent step is
now told its answer is the run's: the whole finished result, carrying forward what
the earlier steps produced.

Which step is last follows the executor (``api.recipe_executor``): the playbook's
steps sorted by ``order``, each step's order its own or its place; the last of them
that is an agent step (not a fixed generate_document step). Without the run's playbook
to read, a step whose order reaches the run's step count is last.
"""
from __future__ import annotations

from typing import Any, List, Optional

LAST_STEP_HEADING = "## Your answer is the run's answer"
LAST_STEP_RULE = (
    f"{LAST_STEP_HEADING}\n"
    "You are this playbook's last step: your answer is what lands on the run's card, and the owner reads it as the "
    "whole result. Give the finished result in full, carrying forward what the earlier steps produced (every item, "
    "figure and draft they found), not a summary of your own step and never a pointer to a saved file."
)


def _agent_step_orders(steps: List[Any]) -> List[Any]:
    """The executor's order for each agent step of ``steps``, in run order."""
    from core.models.core import PLAYBOOK_DOCUMENT_STEP

    ordered = sorted((s for s in steps if isinstance(s, dict)), key=lambda s: s.get("order", 0))
    return [s.get("order", idx + 1) for idx, s in enumerate(ordered) if s.get("type", "agent") != PLAYBOOK_DOCUMENT_STEP]


def is_last_agent_step(db: Any, run: Any, step_order: Any, total_steps: Any) -> bool:
    """Whether the step at ``step_order`` is its run's last agent step."""
    from core.models.core import WorkflowTemplate

    playbook = db.get(WorkflowTemplate, run.recipe_id) if run is not None and db is not None else None
    if playbook is not None and str(getattr(playbook, "workspace_id", None)) == str(run.workspace_id):
        orders = _agent_step_orders(list(playbook.steps or []))
        return bool(orders) and orders[-1] == step_order
    try:
        return int(step_order) >= int(total_steps)
    except (TypeError, ValueError):
        return False


def last_step_rule(db: Any, run: Any, step: dict) -> Optional[str]:
    """The rule for a playbook's last agent step, else None."""
    if is_last_agent_step(db, run, step.get("step_order", 1), step.get("total_steps", 1)):
        return LAST_STEP_RULE
    return None


__all__ = ["LAST_STEP_HEADING", "LAST_STEP_RULE", "is_last_agent_step", "last_step_rule"]
