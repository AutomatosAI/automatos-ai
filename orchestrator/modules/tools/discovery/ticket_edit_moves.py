"""``platform_update_task`` with a status moves the card, its note kept (F309, night 9).

Night 9, iteration 1: "Please approve card 1866 with this note: Good, this is the figure
I'll use. Ignore the June review from now on." Auto's first call was
platform_update_task {task_id: 1866, notes: "…", status: "done"}. The edit tool took no
status: the call was refused before it ran ("this action changes a ticket's details, not
its status … Any other move is platform_update_task_status's"). Auto then split the
approval in two: platform_update_task {note} (a plain note) and
platform_update_task_status {status: "done"} with no note. The move carried no note, so
the approval's own note ("Approved: …", ``services.ticket_verdict.keep_approval_note``, which the owner's lessons
read, F249) was never written, and the owner found their words on none of the card's
fields they checked (#1866). In iteration 2 Auto made the one call,
platform_update_task_status {status: "done", note}, and the note was kept (#1879).

So platform_update_task takes a status (``actions_board_tasks``), and it is that move:
any field it edits is edited first (the board's way, ``ticket_edits.briefs_like_the_board``),
then the card moves through platform_update_task_status with the call's note, which keeps
it as that move does (``notes`` reaches ``note`` through the executor's aliases). A new
brief on a card already worked on is the board's Re-brief, which sends the card back
itself, so it takes no status.
"""
from __future__ import annotations

from typing import Any, Awaitable, Callable, Dict

from sqlalchemy.orm import Session

Handler = Callable[[Session, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

STATUS = "status"
NOTE = "note"
DESCRIPTION = "description"
# The fields platform_update_task edits; a status call with none of them only moves the card.
EDITED = ("title", DESCRIPTION, "priority", "review_mode", "tags")
REBRIEF_MOVES_IT = ("A new brief on a card its agent has already worked on is the board's Re-brief: it sends the "
                    "card back to its agent by itself, so it takes no status. Nothing was done. Send the "
                    "description without a status.")


def moves_the_card(params: Dict[str, Any]) -> bool:
    """Whether an edit call carries a status to move its card to."""
    return isinstance(params.get(STATUS), str) and bool(params[STATUS].strip())


async def edited_then_moved(edit: Handler, db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """The call's edits through ``edit`` (none when it has none), then the card's move
    through platform_update_task_status, with the call's note. The first refusal is
    the answer, and nothing after it is done."""
    from modules.tools.discovery.handlers_board_task_done import update_board_task_status
    from modules.tools.discovery.ticket_edits import _worked_ticket

    if str(params.get(DESCRIPTION) or "").strip() and _worked_ticket(db, workspace_id, params.get("task_id")):
        return {"success": False, "error": REBRIEF_MOVES_IT}
    edits = {k: v for k, v in params.items() if k not in (STATUS, NOTE)}
    edited = await edit(db, workspace_id, edits) if any(edits.get(f) is not None for f in EDITED) else {}
    if edited and edited.get("success") is not True:
        return edited
    moved = await update_board_task_status(db, workspace_id, _the_move(db, workspace_id, params))
    return {**moved, "updated": edited["updated"]} if edited.get("updated") else moved


def _the_move(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """The status call: the card by its number, as the board says it (an id may be
    another card's number), its status, the note, and who drives the call."""
    from modules.tools.discovery.ticket_edits import _ticket
    from services.ticket_numbers import ticket_number

    task = _ticket(db, workspace_id, params.get("task_id"))
    move = {"task_id": (ticket_number(db, task) or task.id) if task is not None else params.get("task_id"),
            STATUS: params[STATUS].strip()}
    note = str(params.get(NOTE) or "").strip()
    extra = {NOTE: note} if note else {}
    driver = {"_user_id": params["_user_id"]} if params.get("_user_id") else {}
    return {**move, **extra, **driver}


__all__ = ["REBRIEF_MOVES_IT", "edited_then_moved", "moves_the_card"]
