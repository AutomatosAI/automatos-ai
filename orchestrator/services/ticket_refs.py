"""PRD-252 R4 — Auto names tickets by number, and its tools take the number.

Night 6: Auto told the owner about "tasks 2 and 3", their place in a list,
because the tools gave it ids and nothing else. ``by_ticket_number`` wraps a
ticket tool's handler:

- before it runs, a ticket given by number ("#0042", or "#0051.3" for a mission
  step) in ``task_id``, ``task_ids`` or ``parent_task_id`` becomes the ticket's
  id, so the handler is unchanged;
- after it runs, every ticket in the result carries its ``number``, so Auto can
  say it (never on a public widget turn).

F241 (night 7): Auto's calls carried "#0175" as 175 or "0175" as often as with
the '#', and every one was read as an id ("Task 175 not found"; 0 of 11 cards
found by number in a night). A leading zero or a step ("105.4") now reads as a
number. Bare digits are read in the workspace, as ids or as numbers without
their '#' (``read_bare_refs``): the reading that names more of a call's tickets
is the one meant, and when both name as many, different tickets, the call is
refused naming both. Bare digits that name no ticket reach the tool as given,
and its "not found" says the number too.
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
NO_SUCH_TICKET = ("No ticket {number} (and no ticket with id {ref}) in this workspace. "
                  "platform_list_tasks shows each ticket's number.")


def by_ticket_number(handler: Handler) -> Handler:
    """Let ``handler`` take tickets by number and answer with their numbers."""
    @functools.wraps(handler)
    async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        resolved, error = _resolve_refs(db, workspace_id, params or {})
        if error:
            return {"success": False, "error": error}
        result = _not_found_by_number(await handler(db, workspace_id, resolved), resolved)
        try:
            return _with_numbers(db, workspace_id, result)
        except Exception:
            # The numbers are a reading aid; the tool's own result still stands.
            logger.exception("[ticket_refs] could not add ticket numbers to %s's result", handler.__name__)
            return result
    return wrapped


def _resolve_refs(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    out = dict(params)
    for key in SINGLE_REFS:
        if params.get(key) in (None, ""):
            continue
        found, error = _resolve(db, workspace_id, [params[key]], one=True)
        if error:
            return params, error
        out[key] = found[0]
    refs = params.get("task_ids")
    if isinstance(refs, (list, tuple)):
        found, error = _resolve(db, workspace_id, list(refs), one=False)
        if error:
            return params, error
        out["task_ids"] = found
    return out, None


def _resolve(db: Session, workspace_id: Any, refs: List[Any], *, one: bool) -> Tuple[List[Any], Optional[str]]:
    """``refs`` as ticket ids, in order, or why the call is refused. A '#' number
    that names no ticket is refused when it is the one ticket asked for; in a list
    it stays as given, and the bulk handler lists it as failed. So does a ref that
    is neither a number nor digits, and bare digits that name no ticket."""
    from services.ticket_numbers import is_bare_ref, is_number_ref, read_bare_refs, resolve_ticket_ref

    bare, error = read_bare_refs(db, workspace_id, [r for r in refs if is_bare_ref(r)])
    if error:
        return refs, error
    out: List[Any] = []
    for ref in refs:
        if is_number_ref(ref):
            found = resolve_ticket_ref(db, workspace_id, ref)
            if found is None and one:
                return refs, NO_SUCH_NUMBER.format(ref=str(ref).strip())
            out.append(ref if found is None else found)
        else:
            out.append(bare.get(int(str(ref).strip()), ref) if is_bare_ref(ref) else ref)
    return out, None


def _not_found_by_number(result: Any, params: Dict[str, Any]) -> Any:
    """Bare digits that named no ticket reached the tool as given. Its "Task 175 not
    found" then says the number too, so Auto looks the ticket up by its number."""
    from services.ticket_numbers import format_number, is_bare_ref

    ref = params.get("task_id")
    if not (isinstance(result, dict) and result.get("success") is False and is_bare_ref(ref)):
        return result
    n = int(str(ref).strip())
    if not str(result.get("error") or "").startswith(f"Task {n} not found"):
        return result
    return {**result, "error": NO_SUCH_TICKET.format(number=format_number(n), ref=n)}


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
