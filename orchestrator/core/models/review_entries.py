"""PRD-266 US-012a (K10) — a ticket that ENTERS Review is announced once, after its commit.

At least ten writers move a ticket into Review by assigning ``BoardTask.status`` (the board's
PATCH and completion, Auto's status tool, the CLI host, session plans, the Mission bridges…)
and no single status-change function exists, so one listener sees them all (the
``core/models/ticket_numbers.py`` precedent):

* ``set`` on ``BoardTask.status`` (``active_history``: the old value is the stored one) flags a
  ticket that moves INTO ``review`` from anything else, a new ticket filed in Review included,
  and unflags one that moves away again before the flush;
* ``after_flush`` checks each flagged ticket's history (Review → Done → Review before a flush
  is no entry), copies its plain values ``(id, workspace_id)`` into ``session.info`` and clears
  the flag;
* ``after_commit`` hands them to ``services.notification_events.review_pending.announce``, which
  re-reads each ticket on a session of its own. It emits no SQL and reads no instance here. A
  savepoint's release waits for the outer commit;
* ``after_rollback`` / ``after_soft_rollback`` forget them when the rollback reaches the
  session's own transaction, and ``after_transaction_end`` when ``close()`` ends it. A
  savepoint's rollback keeps them: ``announce`` re-reads each ticket and skips one the
  savepoint took back out of Review.

Raw SQL never reaches the listener: the dispatch sweeper announces its own move into Review
(``services/board_dispatcher._notify_swept``), and ``api/board_tasks._reopen_review`` (a failed
approval action puts the ticket back as it was) is not an entry.
"""
from __future__ import annotations

import logging
from typing import Any, Dict

from sqlalchemy import event, inspect
from sqlalchemy.orm import Session

from core.models.core import BoardTask

logger = logging.getLogger(__name__)

REVIEW = "review"
# A ticket's flag (on its InstanceState.info) and a session's entries (on Session.info).
_FLAG = "prd266_entered_review"
_ENTRIES = "prd266_review_entries"


@event.listens_for(BoardTask.status, "set", active_history=True, raw=True)
def _flag_entry(state: Any, value: Any, oldvalue: Any, initiator: Any) -> None:
    """Flag a ticket moving into Review from anything else; unflag one that moves away."""
    if value != REVIEW:
        state.info.pop(_FLAG, None)
    elif oldvalue != REVIEW:
        state.info[_FLAG] = True


def _stored_status_entered(task: BoardTask) -> bool:
    """Whether this flush takes the ticket into Review from the value it had before (no SQL)."""
    history = inspect(task).attrs.status.history
    return REVIEW in (history.added or ()) and REVIEW not in (history.deleted or ())


@event.listens_for(Session, "after_flush")
def _record_entries(session: Session, flush_context: Any) -> None:
    """Copy each flagged ticket that entered Review into ``session.info``; clear its flag."""
    entered: Dict[int, str] = {}
    for task in (*session.new, *session.dirty):
        if not isinstance(task, BoardTask) or not inspect(task).info.pop(_FLAG, False):
            continue
        if task.workspace_id is not None and _stored_status_entered(task):
            entered[task.id] = str(task.workspace_id)
    if entered:
        session.info[_ENTRIES] = {**session.info.get(_ENTRIES, {}), **entered}


@event.listens_for(Session, "after_commit")
def _announce_entries(session: Session) -> None:
    """Hand the committed entries to ``review_pending.announce``. Never raises into the commit."""
    if session.in_nested_transaction():
        return  # a savepoint's release: the outer commit announces them
    entries = session.info.pop(_ENTRIES, None)
    if not entries:
        return
    try:
        from services.notification_events.review_pending import announce

        announce(list(entries.items()))
    except Exception:
        logger.exception("[review_pending] %d ticket(s) entered Review and were not announced", len(entries))


@event.listens_for(Session, "after_rollback")
def _forget_on_rollback(session: Session) -> None:
    """A rollback of the session's own transaction forgets the entries; a savepoint's keeps them."""
    if session.get_nested_transaction() is None:
        session.info.pop(_ENTRIES, None)


@event.listens_for(Session, "after_soft_rollback")
def _forget_on_soft_rollback(session: Session, previous_transaction: Any) -> None:
    """As ``_forget_on_rollback``, for a rollback that ended the session's own transaction."""
    if previous_transaction.parent is None:
        session.info.pop(_ENTRIES, None)


@event.listens_for(Session, "after_transaction_end")
def _forget_on_close(session: Session, transaction: Any) -> None:
    """``close()`` ends the session's own transaction with neither rollback event: forget the
    entries, so a reused session never announces them later. After a commit they are gone already."""
    if transaction.parent is None:
        session.info.pop(_ENTRIES, None)
