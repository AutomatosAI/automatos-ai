"""platform_get_playbook_execution gives a run's results: its final output, each
step's preview, and any step's whole output.

F321 (night 9b, build 15): after run #0102 (exec-45d8ac862a79) put only 2 of 8
coffees on its card, Auto asked for the run as execution_id "0102" (15:52:32Z,
trace 96ce7ba07cef) and got "Execution '0102' not found". With the execution id
it would have got each step's "output" — a key the stored steps do not have, so
every preview was empty — and no final output at all. The 8-coffee report step 2
saved lived only in the expiring scratchpad and in the step's log.

Now the run is found by its card number too, the answer carries the run's final
output and each step's stored preview, and ``step`` reads that step's whole
answer and what it saved from the step's log.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

Handler = Callable[[Session, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]
STEP_PARAM = "step"
NOT_KEPT = "Only a short preview of step {order}'s output was kept: its log was not stored. The preview is below."
UNREADABLE = "Step {order}'s log could not be read just now, so its whole output is not available. Try again later."


def _run_id_for_card(db: Session, workspace_id: Any, ref: Any) -> Optional[str]:
    """The execution id of the run whose card is ``ref`` ("#0102" or "0102"), or None."""
    from core.models.core import BoardTask
    from modules.tools.discovery.card_numbers import RUN_CARD
    from services.ticket_numbers import is_number_ref, resolve_ticket_ref

    if not is_number_ref(ref):
        return None
    card_id = resolve_ticket_ref(db, workspace_id, ref)
    card = db.query(BoardTask).filter(BoardTask.id == card_id, BoardTask.workspace_id == workspace_id).first() \
        if card_id else None
    return card.source_id if card is not None and card.source_type == RUN_CARD else None


def _cap(text: Any) -> str:
    from core.services.playbook_scratchpad import answer_for_next_step

    return answer_for_next_step(str(text or ""))


def _step_summaries(step_results: List[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Each stored step as Auto reads it, numbered as the playbook numbers it."""
    return [{"step": s.get("order"), "agent": s.get("agent_name"), "status": s.get("status", "unknown"),
             "output_preview": s.get("output_preview", ""), "error": s.get("error"),
             "whole_output_kept": bool(s.get("log_url"))}
            for s in step_results if isinstance(s, Mapping)]


def step_output(step_results: List[Mapping[str, Any]], order: Any) -> Dict[str, Any]:
    """Step ``order``'s whole answer and what it saved, from its log; or why not."""
    from services.playbook_run_result import read_step_log, saved_values

    stored = next((s for s in step_results if isinstance(s, Mapping) and str(s.get("order")) == str(order)), None)
    if stored is None:
        return {"step": order, "error": f"This run has no step {order}."}
    if not stored.get("log_url"):
        return {"step": order, "note": NOT_KEPT.format(order=order), "output_preview": stored.get("output_preview")}
    try:
        log = read_step_log(stored["log_url"])
    except Exception:
        logger.exception("[F321] step %s's log %s could not be read", order, stored.get("log_url"))
        return {"step": order, "error": UNREADABLE.format(order=order)}
    saved = {key: _cap(value) for key, value in saved_values(log.get("tool_calls"))}
    return {"step": order, "output": _cap(log.get("agent_output")), "saved": saved}


def _results(db: Session, workspace_id: Any, view: Dict[str, Any], step: Any) -> Dict[str, Any]:
    """``view`` with the run's final output, its steps' previews and, when asked, one step's whole output."""
    from core.models.core import RecipeExecution

    run = db.query(RecipeExecution).filter(RecipeExecution.execution_id == view.get("execution_id"),
                                           RecipeExecution.workspace_id == workspace_id).first()
    if run is None:
        return view
    steps = [s for s in (run.step_results or []) if isinstance(s, Mapping)]
    final = (run.output_data or {}).get("final_output") if isinstance(run.output_data, Mapping) else None
    out = {**view, "final_output": _cap(final) if final else None, "step_results": _step_summaries(steps)}
    return {**out, "step_output": step_output(steps, step)} if step not in (None, "") else out


def gives_the_runs_results(handler: Handler) -> Handler:
    """Wrap ``get_playbook_execution``: a run by its card number, and its results.
    A widget turn's visitor view is left as it is (F155)."""
    @functools.wraps(handler)
    async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        from core.security.surface import widget_turn

        if widget_turn():
            return await handler(db, workspace_id, params)
        run_id = _run_id_for_card(db, workspace_id, params.get("execution_id"))
        asked = {**params, "execution_id": run_id} if run_id else params
        result = await handler(db, workspace_id, asked)
        if not (isinstance(result, dict) and result.get("success") and isinstance(result.get("execution"), dict)):
            return result
        return {**result, "execution": _results(db, workspace_id, result["execution"], params.get(STEP_PARAM))}
    return wrapped


__all__ = ["gives_the_runs_results", "step_output"]
