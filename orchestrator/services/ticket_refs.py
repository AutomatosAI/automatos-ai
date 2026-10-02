"""PRD-252 R4 — Auto names tickets by number, and its tools take the number.

Night 6: Auto told the owner about "tasks 2 and 3", their place in a list,
because the tools gave it ids and nothing else. ``by_ticket_number`` wraps a
ticket tool's handler:

- before it runs, a ticket given by number ("#0042", or "#0051.3" for a mission
  step) in ``task_id``, ``task_ids`` or ``parent_task_id`` becomes the ticket's
  id, so the handler is unchanged;
- after it runs, every ticket in the result carries its ``number``, so Auto can
  say it (never on a public widget turn).

An id (an integer, or digits without '#') passes through as before.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

Handler = Callable[[Session, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]

SINGLE_REFS = ("task_id", "parent_task_id")
# Result keys that hold a list of tickets (board_snapshot's scheduled items are not tickets).
TICKET_LISTS = ("tasks", "open_tasks", "recently_finished")
NO_SUCH_NUMBER = "No ticket {ref} in this workspace. platform_list_tasks shows each ticket's number."


def by_ticket_number(handler: Handler) -> Handler:
    """Let ``handler`` take tickets by number and answer with their numbers."""
    @functools.wraps(handler)
    async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        resolved, error = _resolve_refs(db, workspace_id, params or {})
        if error:
            return {"success": False, "error": error}
        result = await handler(db, workspace_id, resolved)
        try:
            return _with_numbers(db, workspace_id, result)
        except Exception:
            # The numbers are a reading aid; the tool's own result still stands.
            logger.exception("[ticket_refs] could not add ticket numbers to %s's result", handler.__name__)
            return result
    return wrapped


def _resolve_refs(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    from services.ticket_numbers import is_number_ref, resolve_ticket_ref

    out = dict(params)
    for key in SINGLE_REFS:
        if is_number_ref(params.get(key)):
            found = resolve_ticket_ref(db, workspace_id, params[key])
            if found is None:
                return params, NO_SUCH_NUMBER.format(ref=params[key].strip())
            out[key] = found
    refs = params.get("task_ids")
    if isinstance(refs, (list, tuple)):
        # A number that matches nothing stays as given; the bulk handler lists it as failed.
        out["task_ids"] = [(resolve_ticket_ref(db, workspace_id, r) or r) if is_number_ref(r) else r
                           for r in refs]
    return out, None


def _with_numbers(db: Session, workspace_id: Any, result: Any) -> Any:
    """``result`` with each ticket's number beside its id (a new dict). Never on a
    public widget turn (F155): a visitor sees a ticket's status, and a number
    would say how many tickets the business has."""
    from core.security.surface import widget_turn

    if not isinstance(result, dict) or widget_turn():
        return result
    ids = _ticket_ids(result)
    if not ids:
        return result
    numbers = _numbers(db, workspace_id, ids)
    out = dict(result)
    if isinstance(out.get("task_id"), int):
        out["number"] = numbers.get(out["task_id"])
    if isinstance(out.get("task"), dict):
        out["task"] = _numbered(out["task"], numbers)
    for key in TICKET_LISTS:
        if isinstance(out.get(key), list):
            out[key] = [_numbered(t, numbers) for t in out[key]]
    card = (out.get("frontend_data") or {}).get("task_card")
    if isinstance(card, dict):
        out["frontend_data"] = {**out["frontend_data"], "task_card": _numbered(card, numbers)}
    return _numbered_bulk(out, numbers)


def _numbered_bulk(out: Dict[str, Any], numbers: Dict[int, Optional[str]]) -> Dict[str, Any]:
    """A bulk status answer names its tickets by number too: ``updated_numbers``
    beside ``updated``, and each failed ticket's ``number``."""
    if not isinstance(out.get("updated"), list):
        return out
    failed = [{**f, "number": numbers.get(f.get("task_id"))} if isinstance(f, dict) else f
              for f in out.get("failed") or []]
    return {**out, "updated_numbers": [numbers.get(i) for i in out["updated"]], "failed": failed}


def _numbered(ticket: Any, numbers: Dict[int, Optional[str]]) -> Any:
    return {**ticket, "number": numbers.get(ticket.get("id"))} if isinstance(ticket, dict) else ticket


def _ticket_ids(result: Dict[str, Any]) -> Set[int]:
    ids = {result.get("task_id"), (result.get("task") or {}).get("id") if isinstance(result.get("task"), dict) else None,
           ((result.get("frontend_data") or {}).get("task_card") or {}).get("id")}
    for key in TICKET_LISTS:
        ids.update(t.get("id") for t in result.get(key) or [] if isinstance(t, dict))
    ids.update(_bulk_ids(result))
    return {i for i in ids if isinstance(i, int) and not isinstance(i, bool)}


def _bulk_ids(result: Dict[str, Any]) -> List[Any]:
    """The tickets a bulk status answer names: those it updated and those it could not."""
    updated = result.get("updated") if isinstance(result.get("updated"), list) else []
    failed = [f.get("task_id") for f in result.get("failed") or [] if isinstance(f, dict)]
    return [*updated, *failed]


def _numbers(db: Session, workspace_id: Any, ids: Set[int]) -> Dict[int, Optional[str]]:
    from core.models.core import BoardTask
    from services.ticket_numbers import ticket_numbers

    tickets = db.query(BoardTask).filter(BoardTask.id.in_(ids), BoardTask.workspace_id == workspace_id).all()
    return ticket_numbers(db, workspace_id, tickets)
