"""P256-FIX-N13-7 (F414, night 13): an approval card names a ticket by its board number.

Night 13, Auto and the board named a ticket two ways, its number (#1394) and its row id
(2680), and "did that go through?" got yes, then no. What the model reads names a ticket by
its number alone (``formatting.tickets_by_number``), and its tools take the number. The
approval card's digest of the call (``ToolResultFormatter._tool_params_digest``) showed the
call's ticket parameters as the call gave them, a row id included. The ask now carries how
the card shows each ticket parameter (``CARD_NUMBERS``): the ticket it names, read the way the
ticket tools read it, by its number; the digest shows that (``card_digest.carries_the_question``).
The call itself, and the click's hash of it, are unchanged.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

CARD_NUMBERS = "card_numbers"
# The parameters that name one ticket, and the one that names several.
PARENT_PARAM = "parent_task_id"
TICKET_PARAMS = ("task_id", PARENT_PARAM)
TICKETS_PARAM = "task_ids"
NUMBER_MARK = "#"  # a board number, "#1394": anything else is not shown in its place


def numbers_on_the_card(db: Any, workspace_id: Any, action: str, params: Any) -> Dict[str, Any]:
    """How the card shows each ticket parameter of ``action``'s ``params``: {"task_id": "#1394"};
    {} for none. A parameter that names something else (a scheduled task's id) is left as given."""
    from modules.tools.execution.subject_targets import names_a_ticket

    if not isinstance(params, dict) or db is None:
        return {}
    shown: Dict[str, Any] = {}
    try:
        for param in TICKET_PARAMS:
            if param != PARENT_PARAM and not names_a_ticket(action, param):
                continue
            number = _number_named(db, workspace_id, params.get(param))
            if number is not None:
                shown[param] = number
        listed = params.get(TICKETS_PARAM)
        if isinstance(listed, list) and listed:
            shown[TICKETS_PARAM] = [_number_named(db, workspace_id, ref) or ref for ref in listed]
    except Exception:  # noqa: BLE001 — the card shows the call as given
        logger.exception("[card_ticket_numbers] could not name the call's tickets by number")
        return {}
    return shown


def _number_named(db: Any, workspace_id: Any, ref: Any) -> Optional[str]:
    """The board number of the ticket ``ref`` names in this workspace, or None."""
    from core.models.core import BoardTask
    from services.ticket_numbers import ticket_number
    from services.ticket_refs import ticket_id_named

    if ref in (None, "") or isinstance(ref, bool):
        return None
    task_id = ticket_id_named(db, workspace_id, ref)[0]
    if task_id is None:
        return None
    task = db.query(BoardTask).filter(BoardTask.id == task_id, BoardTask.workspace_id == workspace_id).first()
    number = ticket_number(db, task) if task is not None else None
    return number if isinstance(number, str) and number.startswith(NUMBER_MARK) else None


__all__ = ["CARD_NUMBERS", "numbers_on_the_card"]
