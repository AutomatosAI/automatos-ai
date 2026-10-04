"""A mission's own card carries its result when it completes (F268, night 7b).

#0188 completed, and its card on the board stayed empty: the summary was only on its
last step's card (#0188.4) and on the mission page. The card follows the mission's
state (``orchestration_board_bridge.sync_mission_board_status``) and never took its
result. Now, when the mission completes, its card takes the mission's final work
product: the output of its last step that produced one, as the mission page shows it
(``coordinator_service.pick_final_output_task``), from the steps of its plan (not
those a re-plan replaced). A card that already says something keeps it.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

MISSION_CARD = "orchestration"


def carries_the_missions_result(sync: Callable[..., None]) -> Callable[..., None]:
    """Wrap ``sync_mission_board_status``: a completed mission's card takes its result."""
    @functools.wraps(sync)
    def wrapped(db: Any, run: Any) -> None:
        from core.models.orchestration_enums import RunState

        sync(db, run)
        if getattr(run, "state", None) == RunState.COMPLETED.value:
            _put_the_result_on_the_card(db, run)
    return wrapped


def _put_the_result_on_the_card(db: Any, run: Any) -> None:
    from core.models.core import BoardTask
    from core.models.orchestration import OrchestrationTask
    from core.models.orchestration_enums import TaskState
    from modules.coordination.mission_ends import superseded
    from services.coordinator_service import pick_final_output_task

    card = db.query(BoardTask).filter(BoardTask.source_type == MISSION_CARD,
                                      BoardTask.orchestration_run_id == run.id).first()
    if card is None or str(card.result or "").strip():
        return
    steps = [t for t in db.query(OrchestrationTask).filter(OrchestrationTask.run_id == run.id).all()
             if not superseded(t)]
    final = pick_final_output_task([t for t in steps if t.state == TaskState.VERIFIED.value] or steps)
    if final is None:
        return
    card.result = str(final.output)
    db.flush()
    logger.info("[F268] mission %s's card %s carries its result (step %s)", run.id, card.id, final.id)


__all__ = ["carries_the_missions_result"]
