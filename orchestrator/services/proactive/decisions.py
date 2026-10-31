"""PRD-268 §6.2.5, FR-7, FR-14 — a person's Start or Hide of a suggestion.

**Start** (``start``):

1. The suggestion is read in the caller's workspace; another workspace's id is
   ``not_found`` (404).
2. A suggestion that is no longer open is refused by its state (409): ``already_started``,
   ``already_hidden`` or ``expired``. The workspace's open suggestions past their 7 days lapse
   first (``services.proactive.expiry``), so a lapsed one is ``expired`` and files nothing,
   whatever its kind.
3. What Start does is ``proposed_action.kind``: ``board_task`` and ``chat`` in W1. A Mission
   or a playbook run answers ``not_available`` (409, K12) until US-C14, and so does any other
   kind.
4. ``refuse_new_work`` (the day's spend ceiling, as board pickup) comes before anything is
   written: a refusal is ``spend_ceiling`` (409) and leaves the suggestion open.
5. One ``UPDATE … WHERE status = 'open' RETURNING`` moves it to ``started`` with
   ``decided_via`` and ``decided_at``. Of two Starts at once, the second waits on the first's
   row lock, then matches nothing, and is refused by the state it finds: one Start wins.
6. ``board_task``: the ticket (``start_ticket.file_ticket``), its watch
   (``start_ticket.watch_ticket``, FR-8), and ``started_ref`` ``{board_task_id, watch_id}``;
   ``chat``: nothing is made, and the answer's ``opens`` is
   ``{type: 'chat', suggestion_id}`` (the client opens the Auto thread with the suggestion as
   context; its text is never posted as the person's message).
7. One commit; then the suggestion's ``suggestion_changed`` live event
   (``services.proactive.live``), the assigned ticket's wake (``start_ticket.wake``; a failed
   wake is logged and the dispatch loop's poll still finds the ticket) and the audit row.

**Hide** (``hide``) is never gated: no switch, cap or spend check. The person it is
addressed to, or (``any_row``) an owner or admin, moves it from open to ``hidden`` with
``hide_reason``, ``decided_via`` and ``decided_at``, then its ``suggestion_changed`` live
event once that has committed; a hidden suggestion hidden again is answered as it is, and
tells no one. Started or expired (lapsed ones first, as for Start), it is refused by its state.

Audit rows (``proactive:suggestion_started``, ``proactive:suggestion_hidden``; their details
say where, by whom as ``user:<id>``, and what) are written after the decision's commit (K15):
``AuditService.log`` commits its own row and never fails the decision. A database error
before the decision's commit is raised to the route, and nothing is written. The route
shapes the answer: ``Started`` for a Start, the suggestion as it now is for a Hide.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple
from uuid import UUID

from sqlalchemy import update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from core.models.proactive import (
    SUGGESTION_EXPIRED,
    SUGGESTION_HIDDEN,
    SUGGESTION_OPEN,
    SUGGESTION_STARTED,
    ProactiveSuggestion,
)
from modules.tools.discovery.suggestion_fields import BOARD_TASK, CHAT
from services.proactive import start_ticket
from services.proactive.expiry import expire_lapsed
from services.proactive.live import suggestion_changed

logger = logging.getLogger(__name__)

__all__ = ["Decider", "Refused", "START_KINDS", "Started", "hide", "start"]

# K12: what Start makes in W1. Mission and playbook_run are accepted by the schema, refused here.
START_KINDS = (BOARD_TASK, CHAT)
NOT_FOUND, NOT_YOURS, NOT_AVAILABLE, SPEND_CEILING = "not_found", "not_yours", "not_available", "spend_ceiling"
ALREADY_STARTED, ALREADY_HIDDEN, EXPIRED = "already_started", "already_hidden", "expired"
_CLOSED = {SUGGESTION_STARTED: ALREADY_STARTED, SUGGESTION_HIDDEN: ALREADY_HIDDEN, SUGGESTION_EXPIRED: EXPIRED}
HTTP_NOT_FOUND, HTTP_FORBIDDEN, HTTP_CONFLICT = 404, 403, 409

STARTED_ACTION = "proactive:suggestion_started"
HIDDEN_ACTION = "proactive:suggestion_hidden"
AUDIT_RESOURCE = "proactive_suggestion"
OPENS_BOARD_TASK = "board_task"
OPENS_CHAT = "chat"


class Refused(Exception):
    """A Start or Hide that is not made: ``code`` keys the sentence the route answers with."""

    def __init__(self, code: str, http_status: int) -> None:
        super().__init__(code)
        self.code = code
        self.http_status = http_status


@dataclass(frozen=True)
class Decider:
    """The person deciding: their ``users.id`` (the audit row's), the id a ticket records
    as its creator, the ``user:<id>`` an approval records, and where they decided."""

    user_pk: Optional[int]
    created_by: Optional[str]
    actor: str
    via: str


@dataclass(frozen=True)
class Started:
    """A Start made: the suggestion as it now is, and what the client opens."""

    suggestion: Dict[str, Any]
    opens: Dict[str, Any]


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _find(db: Session, workspace_id: UUID, suggestion_id: UUID) -> ProactiveSuggestion:
    row = db.query(ProactiveSuggestion).filter(
        ProactiveSuggestion.id == suggestion_id, ProactiveSuggestion.workspace_id == workspace_id).first()
    if row is None:
        raise Refused(NOT_FOUND, HTTP_NOT_FOUND)
    return row


def _require_open(row: ProactiveSuggestion) -> None:
    if row.status != SUGGESTION_OPEN:
        raise Refused(_CLOSED.get(row.status, ALREADY_STARTED), HTTP_CONFLICT)


def _claim(db: Session, row: ProactiveSuggestion, values: Dict[str, Any]) -> bool:
    """Move ``row`` out of open in one UPDATE; False when another decision moved it first."""
    table = ProactiveSuggestion.__table__
    moved = db.execute(
        update(table)
        .where(table.c.id == row.id, table.c.workspace_id == row.workspace_id, table.c.status == SUGGESTION_OPEN)
        .values(**values)
        .returning(table.c.id)
    ).first()
    if moved is None:
        return False
    db.refresh(row)
    return True


def _lost(db: Session, workspace_id: UUID, suggestion_id: UUID) -> Refused:
    """The refusal for a decision another one beat: the state it left the suggestion in."""
    db.rollback()
    row = _find(db, workspace_id, suggestion_id)
    return Refused(_CLOSED.get(row.status, ALREADY_STARTED), HTTP_CONFLICT)


def _audit(db: Session, row: ProactiveSuggestion, decider: Decider, action: str, details: Dict[str, Any]) -> None:
    """The decision's audit row, after its commit (K15); a failed write is logged by AuditService."""
    from core.workspaces.audit import AuditService

    AuditService(db).log(
        workspace_id=str(row.workspace_id), user_id=decider.user_pk, action=action,
        resource_type=AUDIT_RESOURCE, resource_id=str(row.id), resource_name=row.title,
        details={"via": decider.via, "actor": decider.actor, **details},
    )


def _changed(db: Session, row: ProactiveSuggestion, status: str) -> None:
    """The committed decision's live event: ids and the new status, never the suggestion's words."""
    suggestion_changed(db, workspace_id=row.workspace_id, suggestion_id=row.id, user_id=row.user_id, status=status)


def _wake(db: Session, task: Any, decider: Decider) -> None:
    """The committed ticket's wake; a failure is logged and costs only dispatch latency, since
    the dispatch loop's poll still finds the ticket, and never undoes the Start or its audit."""
    try:
        start_ticket.wake(db, task, decider.actor)
    except SQLAlchemyError:
        db.rollback()
        logger.exception("[proactive] the started ticket %s could not be woken", getattr(task, "id", "?"))


def _refuse_on_spend(db: Session, row: ProactiveSuggestion) -> None:
    from services.daily_spend_guard import refuse_new_work

    if refuse_new_work(db, row.workspace_id, f"suggestion {row.id}"):
        raise Refused(SPEND_CEILING, HTTP_CONFLICT)


def _kind(row: ProactiveSuggestion) -> Any:
    action = row.proposed_action
    return action.get("kind") if isinstance(action, dict) else None


def _begin(db: Session, row: ProactiveSuggestion, kind: str, decider: Decider,
           locale: str) -> Tuple[Dict[str, Any], Optional[Any]]:
    """Make what Start does (uncommitted): what the client opens, and the ticket when it is one,
    with the ticket's watch recorded beside it in ``started_ref``."""
    if kind == CHAT:
        return {"type": OPENS_CHAT, "suggestion_id": str(row.id)}, None
    task = start_ticket.file_ticket(db, row, decider.created_by, locale)
    watch = start_ticket.watch_ticket(db, task, decider.created_by)
    row.started_ref = {"board_task_id": task.id, "watch_id": str(watch.id)}
    db.flush()
    return {"type": OPENS_BOARD_TASK, "board_task_id": task.id}, task


def start(db: Session, workspace_id: UUID, suggestion_id: UUID, decider: Decider, locale: str) -> Started:
    """Start the suggestion, or raise ``Refused``."""
    now = _now()
    expire_lapsed(db, workspace_id, now)
    row = _find(db, workspace_id, suggestion_id)
    _require_open(row)
    kind = _kind(row)
    if kind not in START_KINDS:
        raise Refused(NOT_AVAILABLE, HTTP_CONFLICT)
    _refuse_on_spend(db, row)
    claimed = {"status": SUGGESTION_STARTED, "decided_via": decider.via, "decided_at": now}
    if not _claim(db, row, claimed):
        raise _lost(db, workspace_id, suggestion_id)
    opens, task = _begin(db, row, kind, decider, locale)
    db.commit()
    answer = Started(suggestion=row.to_dict(), opens=opens)
    _changed(db, row, SUGGESTION_STARTED)
    if task is not None:
        _wake(db, task, decider)
    logger.info("[proactive] suggestion %s started (%s) in workspace %s", row.id, kind, workspace_id)
    made = {"board_task_id": opens["board_task_id"]} if task is not None else {}
    _audit(db, row, decider, STARTED_ACTION, {"kind": kind, **made})
    return answer


def hide(db: Session, workspace_id: UUID, suggestion_id: UUID, decider: Decider, reason: Optional[str], *,
         any_row: bool) -> Dict[str, Any]:
    """Hide the suggestion and answer it as it now is, or raise ``Refused``. ``any_row`` is an
    owner's or admin's; anyone else hides only what is addressed to them."""
    now = _now()
    expire_lapsed(db, workspace_id, now)
    row = _find(db, workspace_id, suggestion_id)
    if not any_row and (decider.user_pk is None or row.user_id != decider.user_pk):
        raise Refused(NOT_YOURS, HTTP_FORBIDDEN)
    if row.status == SUGGESTION_HIDDEN:
        return row.to_dict()
    _require_open(row)
    hidden = {"status": SUGGESTION_HIDDEN, "hide_reason": reason, "decided_via": decider.via, "decided_at": now}
    if not _claim(db, row, hidden):
        refusal = _lost(db, workspace_id, suggestion_id)
        if refusal.code == ALREADY_HIDDEN:  # a Hide at the same moment: hidden either way
            return _find(db, workspace_id, suggestion_id).to_dict()
        raise refusal
    db.commit()
    answer = row.to_dict()
    _changed(db, row, SUGGESTION_HIDDEN)
    logger.info("[proactive] suggestion %s hidden (%s) in workspace %s", row.id, reason, workspace_id)
    _audit(db, row, decider, HIDDEN_ACTION, {"reason": reason})
    return answer
