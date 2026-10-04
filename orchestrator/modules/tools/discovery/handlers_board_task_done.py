"""``platform_update_task_status``, and F235: a ticket Auto moves to done files its
round's report as a Document.

Only approved work becomes knowledge (``services.report_knowledge``). On the board,
Approve and a drag to done file it through the board's own completion path; Auto
moving a ticket to done in chat is the owner's word too, and its status tool is a
plain write that never passes there. So the tool's result says which tickets ended
done, and their reports are filed after it, single or bulk.

Night 9 (F308): a mission step's card moved to done this way stayed held: its step
waited for the owner's check and its mission stayed paused. The move lets a held step
through, as the board's Approve does (``mission_step_verdicts``).
"""
from __future__ import annotations

from typing import Any, Dict, List
from uuid import UUID

from sqlalchemy.orm import Session

from modules.tools.discovery.handlers_board_tasks import update_board_task_status as _move_status


async def update_board_task_status(db: Session, workspace_id: UUID, params: Dict[str, Any]) -> Dict[str, Any]:
    """The status move (``handlers_board_tasks``), then the report of every ticket it
    moved to done."""
    from modules.tools.discovery.mission_step_verdicts import lets_held_steps_through

    result = await _move_status(db, workspace_id, params)
    if result.get("status") == "done":
        lets_held_steps_through(db, workspace_id, moved_to_done(result), params)  # F308: a held step goes on
        await _file_done(db, workspace_id, moved_to_done(result))
    return result


def moved_to_done(result: Dict[str, Any]) -> List[int]:
    """The tickets a status result moved: the bulk form lists them, a single move
    names one when it succeeded."""
    if result.get("updated") is not None:
        return [int(task_id) for task_id in result["updated"]]
    return [int(result["task_id"])] if result.get("success") and result.get("task_id") is not None else []


async def _file_done(db: Session, workspace_id: UUID, task_ids: List[int]) -> None:
    from core.models.core import BoardTask
    from services.report_knowledge import file_done_ticket

    if not task_ids:
        return
    tasks = db.query(BoardTask).filter(BoardTask.workspace_id == workspace_id, BoardTask.id.in_(task_ids)).all()
    for task in tasks:
        await file_done_ticket(db, workspace_id, task)


__all__ = ["moved_to_done", "update_board_task_status"]
