"""F241 (night 7): a mission's card and a playbook run's card are named by number.

On night 7 Auto never once named a mission or playbook card by its number:
- for a mission it gave the run's id ("The mission ID is c0a86f4e-…"), or made a
  number up ("142");
- for a playbook run it said "I don't have a way to directly retrieve the card
  number of a playbook execution". The card was #0145.
Their tools gave it no number.

- The mission tools' answers now carry each mission's card number (``number``,
  "#0139"), and get_mission carries each step's too ("#0139.1"). This is how the
  ticket tools answer (PRD-252 R4).
- Starting a playbook run makes its card at once, so the answer names it. It is the
  same idempotent bridge the run calls a moment later. Looking a run up names its
  card too.

No number is given on a public widget turn (F155).
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Awaitable, Callable, Dict, Iterable, List
from uuid import UUID

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

Handler = Callable[[Session, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]
Numbering = Callable[[Session, Any, Dict[str, Any]], Dict[str, Any]]

MISSION_CARD, STEP_CARD, RUN_CARD = "orchestration", "orchestration_task", "recipe"


def says_the_mission_cards(handler: Handler) -> Handler:
    """A mission tool's answer names each mission's card, and its steps', by number."""
    return _numbering(handler, _mission_numbers)


def says_the_run_card(handler: Handler) -> Handler:
    """Starting a playbook run makes its card now, and the answer names it by number."""
    return _numbering(handler, functools.partial(_run_numbers, make=True))


def names_the_run_cards(handler: Handler) -> Handler:
    """Looking a playbook run up names its card by number. It never makes one."""
    return _numbering(handler, functools.partial(_run_numbers, make=False))


def _numbering(handler: Handler, add: Numbering) -> Handler:
    @functools.wraps(handler)
    async def wrapped(db: Session, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        from core.security.surface import widget_turn

        result = await handler(db, workspace_id, params)
        if widget_turn() or not (isinstance(result, dict) and result.get("success")):
            return result
        try:
            return add(db, workspace_id, result)
        except Exception:
            # The numbers are a reading aid; the tool's own answer still stands.
            logger.exception("[card_numbers] could not number the cards in %s's answer", handler.__name__)
            return result
    return wrapped


def _mission_numbers(db: Session, workspace_id: Any, result: Dict[str, Any]) -> Dict[str, Any]:
    from core.models.core import BoardTask

    mission = result.get("mission") if isinstance(result.get("mission"), dict) else None
    listed = result.get("missions") if isinstance(result.get("missions"), list) else []
    run_ids = [result.get("mission_id"), (mission or {}).get("id"),
               *(m.get("id") for m in listed if isinstance(m, dict))]
    numbers = _numbers(db, workspace_id, BoardTask.orchestration_run_id, MISSION_CARD, _uuids(run_ids))
    out = dict(result)
    if result.get("mission_id") is not None:
        out["number"] = numbers.get(str(result["mission_id"]))
    if listed:
        out["missions"] = [_numbered(m, numbers, "id") for m in listed]
    if mission is not None:
        out["mission"] = _with_step_numbers(db, workspace_id, _numbered(mission, numbers, "id"))
    return out


def _with_step_numbers(db: Session, workspace_id: Any, mission: Dict[str, Any]) -> Dict[str, Any]:
    from core.models.core import BoardTask

    tasks = mission.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return mission
    steps = _numbers(db, workspace_id, BoardTask.orchestration_task_id, STEP_CARD,
                     _uuids(t.get("id") for t in tasks if isinstance(t, dict)))
    return {**mission, "tasks": [_numbered(t, steps, "id") for t in tasks]}


def _run_numbers(db: Session, workspace_id: Any, result: Dict[str, Any], *, make: bool) -> Dict[str, Any]:
    from core.models.core import BoardTask

    execution = result.get("execution") if isinstance(result.get("execution"), dict) else None
    listed = result.get("executions") if isinstance(result.get("executions"), list) else []
    if make and result.get("execution_id"):
        _make_run_card(db, result["execution_id"])
    ids = [result.get("execution_id"), (execution or {}).get("execution_id"),
           *(e.get("execution_id") for e in listed if isinstance(e, dict))]
    numbers = _numbers(db, workspace_id, BoardTask.source_id, RUN_CARD, [str(i) for i in ids if i])
    out = dict(result)
    if result.get("execution_id") is not None:
        out["number"] = numbers.get(str(result["execution_id"]))
    if execution is not None:
        out["execution"] = _numbered(execution, numbers, "execution_id")
    if listed:
        out["executions"] = [_numbered(e, numbers, "execution_id") for e in listed]
    return out


def _make_run_card(db: Session, execution_id: str) -> None:
    """The run's card, made now. The run makes it a moment later otherwise, and finds it made."""
    from core.models.core import RecipeExecution, WorkflowTemplate
    from services.board_task_bridge import create_recipe_board_task

    execution = db.query(RecipeExecution).filter(RecipeExecution.execution_id == execution_id).first()
    playbook = (db.query(WorkflowTemplate).filter(WorkflowTemplate.id == execution.recipe_id).first()
                if execution is not None else None)
    if playbook is None:
        return
    try:
        create_recipe_board_task(db, playbook, execution)
    except IntegrityError:          # the run made it first: it is there
        db.rollback()


def _numbers(db: Session, workspace_id: Any, column: Any, source_type: str, ids: List[Any]) -> Dict[str, str]:
    """Each card's number, by the id the card is linked with (a run's, a step's or a playbook run's)."""
    from core.models.core import BoardTask
    from services.ticket_numbers import ticket_numbers

    if not ids:
        return {}
    cards = db.query(BoardTask).filter(BoardTask.workspace_id == workspace_id, BoardTask.source_type == source_type,
                                       column.in_(ids)).all()
    numbers = ticket_numbers(db, workspace_id, cards)
    return {str(getattr(card, column.key)): numbers[card.id] for card in cards if numbers.get(card.id)}


def _numbered(item: Any, numbers: Dict[str, str], key: str) -> Any:
    return {**item, "number": numbers.get(str(item.get(key)))} if isinstance(item, dict) else item


def _uuids(values: Iterable[Any]) -> List[UUID]:
    found: List[UUID] = []
    for value in values:
        if value is None:
            continue
        try:
            found.append(value if isinstance(value, UUID) else UUID(str(value)))
        except ValueError:
            continue
    return found


__all__ = ["names_the_run_cards", "says_the_mission_cards", "says_the_run_card"]
