"""F241 (with F245): Auto's cancel stops what runs the card, as the board's Cancel does.

On night 7, Auto's platform_update_task_status('cancelled') only wrote the status. On
a playbook's card the run went on; on a mission's card the mission did. F245 made the
board's Cancel stop both (services/run_cancel.cancel_ticket). A cancel from this tool
now takes the same path for a card that runs something:
- a playbook's card stops its run, and a mission's card cancels the mission;
- a step of a live mission is refused, naming the mission, because the mission runs
  its steps;
- the person behind the chat must be allowed to stop that run, as its own page asks.
  A call no person drives (an agent's own run) is refused.

Any other status, and a card that runs nothing, keeps the tool's own path. A bulk
cancel splits: the cards that run something go through cancel_ticket, the rest
through the tool, and the answer counts them together.
"""
from __future__ import annotations

import functools
import logging
from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

Handler = Callable[[Session, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

CANCELLED = "cancelled"
BY_AN_AGENT = "platform_tool"
LEFT_AS_IT_IS = "{label} is {status}, so it was left as it is."
STOPPED_ITS_RUN = "Cancelled, and its {what} stopped."
WHAT_RUNS_IT = {"recipe": "playbook run", "orchestration": "mission"}


def stops_what_it_cancels(handler: Handler) -> Handler:
    """Wrap update_board_task_status: a cancel of a card that runs something goes through
    cancel_ticket; any other call is the handler's."""
    @functools.wraps(handler)
    async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        from modules.tools.discovery.handlers_board_tasks import MAX_BULK_TASK_IDS

        listed = params.get("task_ids") if isinstance(params.get("task_ids"), list) and params["task_ids"] else None
        if params.get("status") != CANCELLED or (listed and len(listed) > MAX_BULK_TASK_IDS):
            return await handler(db, workspace_id, params)
        cards = _run_cards(db, workspace_id, [_as_id(r) for r in (listed or [params.get("task_id")])])
        if not cards:
            return await handler(db, workspace_id, params)
        cancelled, refused = _cancel(db, workspace_id, cards, params.get("_user_id"))
        if listed is None:
            return _one_answer(cancelled, refused)
        rest = [r for r in listed if _as_id(r) not in cards]
        tool = await handler(db, workspace_id, {**params, "task_ids": rest}) if rest else {}
        return _bulk_answer(len(listed), cancelled, refused, tool)
    return wrapped


def _as_id(ref: Any) -> Optional[int]:
    if isinstance(ref, bool):
        return None
    if isinstance(ref, int):
        return ref
    return int(ref) if isinstance(ref, str) and ref.strip().isdigit() else None


def _run_cards(db: Session, workspace_id: Any, ids: List[Optional[int]]) -> Dict[int, Any]:
    """The cards among ``ids`` that a playbook run or a mission runs, by id."""
    from core.models.core import BoardTask
    from services.run_cancel import MISSION_CARD, MISSION_STEP, is_playbook_card

    wanted = [i for i in ids if i is not None]
    if not wanted:
        return {}
    rows = db.query(BoardTask).filter(BoardTask.id.in_(wanted), BoardTask.workspace_id == workspace_id).all()
    return {t.id: t for t in rows
            if is_playbook_card(t) or getattr(t, "source_type", None) in (MISSION_CARD, MISSION_STEP)}


def _cancel(db: Session, workspace_id: Any, cards: Dict[int, Any],
            driver: Optional[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Each card through cancel_ticket: what was cancelled (and what of its run stopped),
    and what was refused, with why."""
    from services.run_cancel import cancel_ticket
    from services.ticket_numbers import ticket_label

    by = f"user:{driver}" if driver else BY_AN_AGENT
    may = _may(db, workspace_id, driver)
    cancelled: List[Dict[str, Any]] = []
    refused: List[Dict[str, Any]] = []
    for task_id, card in cards.items():
        live = _runs_live(db, card)
        refusal = cancel_ticket(db, card, by=by, may=may)
        if refusal:
            refused.append({"task_id": task_id, "error": refusal[1]})
            continue
        db.refresh(card)
        if card.status != CANCELLED:
            refused.append({"task_id": task_id, "error": LEFT_AS_IT_IS.format(
                label=ticket_label(card, capital=True), status=card.status)})
            continue
        cancelled.append({"task_id": task_id, "stopped": WHAT_RUNS_IT.get(card.source_type) if live else None})
    return cancelled, refused


def _runs_live(db: Session, card: Any) -> bool:
    """A playbook run or a mission that is still running the card."""
    from services.run_cancel import (FINISHED_RUN_STATUSES, is_playbook_card, mission_is_live, mission_run_of,
                                     playbook_run_of)

    if is_playbook_card(card):
        run = playbook_run_of(db, card)
        return run is not None and run.status not in FINISHED_RUN_STATUSES
    return mission_is_live(mission_run_of(db, card))


def _may(db: Session, workspace_id: Any, driver: Optional[str]) -> Callable[[str], bool]:
    """Whether the person behind the call may stop a run, as its own page asks. The
    local edition's one operator owns the workspace. A call no person drives may not."""
    from config import config
    from core.auth.workspace_permission import workspace_permission_granted

    if not driver:
        return lambda permission: False
    ctx = SimpleNamespace(user=SimpleNamespace(clerk_user_id=driver, system_role=None), workspace_id=workspace_id,
                          auth_type="anonymous" if config.IS_LOCAL_EDITION else "clerk")
    return lambda permission: workspace_permission_granted(db, ctx, permission)


def _one_answer(cancelled: List[Dict[str, Any]], refused: List[Dict[str, Any]]) -> Dict[str, Any]:
    if refused:
        return {"success": False, "task_id": refused[0]["task_id"], "error": refused[0]["error"]}
    done = cancelled[0]
    message = STOPPED_ITS_RUN.format(what=done["stopped"]) if done["stopped"] else "Cancelled."
    return {"success": True, "task_id": done["task_id"], "status": CANCELLED, "triggered_execution": False,
            "message": message}


def _bulk_answer(requested: int, cancelled: List[Dict[str, Any]], refused: List[Dict[str, Any]],
                 tool: Dict[str, Any]) -> Dict[str, Any]:
    """The bulk answer's shape (update_board_task_status), counting both paths."""
    updated = [c["task_id"] for c in cancelled] + list(tool.get("updated") or [])
    failed = refused + list(tool.get("failed") or [])
    if tool and tool.get("success") is False and not isinstance(tool.get("updated"), list):
        failed.append({"task_id": None, "error": tool.get("error")})
    return {"success": not failed, "partial": bool(updated) and bool(failed), "status": CANCELLED,
            "requested": requested, "updated_count": len(updated), "updated": updated, "failed": failed}


__all__ = ["stops_what_it_cancels"]
