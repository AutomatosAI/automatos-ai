"""PRD-268 §6.4, §6.11.2 — the brief's "Since your last brief": what started and finished since.

"Since" is the person's last brief: their newest ``mobile_digest_posts`` claim before this
local date (``posted_at``), else ``FALLBACK_WINDOW`` ago. Three reads, each in the caller's
workspace, newest first, each its exact count beside its first ``ROWS_READ`` rows:

* **the person's started suggestions** (decided since): what each became, by the ticket Start
  filed (``started_ref.board_task_id``; Start files a ticket or opens a chat, and a Mission or
  a playbook run answers 409 until US-C14, W1's K12). Finished it is ``ready`` (Ready to look
  at: finished since the last brief, until the seen flag of §12-Q18); otherwise it says what
  the board says (``display_state.ticket_status``). The row opens the ticket, else (a chat)
  the suggestion;
* **tickets finished since** (Done or Closed), ``ready``. A Mission's own card and its steps
  are the Mission's, and a ticket Start filed is its suggestion's row, so neither is listed;
* **Missions completed since**, ``ready``.

A failed ticket or Mission is "Needs you"'s (``needs_you()`` lists it until it is dealt
with), so it is not listed again here, unless it is what a started suggestion became.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func, text
from sqlalchemy.orm import Session

from core.models.core import BoardTask
from core.models.mobile_push import MobileDigestPost
from core.models.orchestration_enums import RunState
from core.models.proactive import SUGGESTION_STARTED, ProactiveSuggestion
from services.brief.model import BriefRow
from services.brief.rows import ROWS_READ, iso, opens, title_of
from services.brief.strings import say
from services.needs_you import MISSION_CARD
from services.proactive.display_state import FAILED, READY, WORKING, ticket_status
from services.proactive.start_ticket import SOURCE_SUGGESTION
from services.ticket_numbers import format_number

__all__ = ["FALLBACK_WINDOW", "FINISHED_STATUSES", "last_brief_at", "since_rows"]

# How far back "since" reaches for a person with no earlier brief.
FALLBACK_WINDOW = timedelta(hours=24)
FINISHED_STATUSES = ["done", "closed"]
MISSION_STEP = "orchestration_task"
# The tickets listed under a row of their own elsewhere: a Mission's card and steps, and Start's.
NOT_LISTED = [MISSION_CARD, MISSION_STEP, SOURCE_SUGGESTION]
_WHY = {READY: "since_ready", FAILED: "since_failed", WORKING: "since_working"}
_STARTED = "since_started"

_FINISHED_TICKETS = text("""
    SELECT bt.id, bt.title, bt.workspace_seq, bt.completed_at AS at, COUNT(*) OVER () AS of_all
      FROM board_tasks bt
     WHERE bt.workspace_id = CAST(:ws AS uuid) AND bt.status = ANY(:finished)
       AND bt.completed_at >= :since AND bt.source_type <> ALL(:not_listed)
  ORDER BY bt.completed_at DESC, bt.id DESC LIMIT :limit
""")
_FINISHED_MISSIONS = text("""
    SELECT r.id, r.goal, r.completed_at AS at, COUNT(*) OVER () AS of_all
      FROM orchestration_runs r
     WHERE r.workspace_id = CAST(:ws AS uuid) AND r.state = :completed AND r.completed_at >= :since
  ORDER BY r.completed_at DESC, r.id LIMIT :limit
""")


def last_brief_at(db: Session, workspace_id: Any, user_id: int, local_date: date) -> Optional[datetime]:
    """When the person's newest brief before ``local_date`` was claimed, or None."""
    return db.query(func.max(MobileDigestPost.posted_at)).filter(
        MobileDigestPost.workspace_id == workspace_id, MobileDigestPost.user_id == user_id,
        MobileDigestPost.local_date < local_date,
    ).scalar()


def since_rows(
    db: Session, workspace_id: Any, user_id: int, local_date: date, language: str, now: datetime,
) -> Tuple[int, List[BriefRow]]:
    """How many things started or finished since the person's last brief, and the newest as rows."""
    start = last_brief_at(db, workspace_id, user_id, local_date) or now - FALLBACK_WINDOW
    read = (_started(db, workspace_id, user_id, start, language), _tickets(db, workspace_id, start, language),
            _missions(db, workspace_id, start, language))
    rows = sorted((row for _, listed in read for row in listed), key=lambda row: row.at or "", reverse=True)
    return sum(count for count, _ in read), rows


def _started(db: Session, workspace_id: Any, user_id: int, start: datetime, language: str) -> Tuple[int, List[BriefRow]]:
    where = (ProactiveSuggestion.workspace_id == workspace_id, ProactiveSuggestion.user_id == user_id,
             ProactiveSuggestion.status == SUGGESTION_STARTED, ProactiveSuggestion.decided_at >= start)
    count = db.query(func.count(ProactiveSuggestion.id)).filter(*where).scalar() or 0
    found = (db.query(ProactiveSuggestion).filter(*where)
             .order_by(ProactiveSuggestion.decided_at.desc(), ProactiveSuggestion.id).limit(ROWS_READ).all())
    tickets = _tickets_by_id(db, workspace_id, [_ticket_id(s) for s in found])
    return int(count), [_started_row(s, tickets.get(_ticket_id(s)), language) for s in found]


def _ticket_id(suggestion: ProactiveSuggestion) -> Optional[int]:
    """The ticket Start filed for the suggestion, None for a chat or an unreadable ref."""
    ref = suggestion.started_ref if isinstance(suggestion.started_ref, dict) else {}
    try:
        return int(ref["board_task_id"])
    except (KeyError, TypeError, ValueError):
        return None


def _tickets_by_id(db: Session, workspace_id: Any, ids: List[Optional[int]]) -> Dict[int, BoardTask]:
    wanted = sorted({i for i in ids if i is not None})
    if not wanted:
        return {}
    found = db.query(BoardTask).filter(BoardTask.workspace_id == workspace_id, BoardTask.id.in_(wanted)).all()
    return {task.id: task for task in found}


def _became(task: Optional[BoardTask], language: str) -> Optional[str]:
    """What the ticket a suggestion started now is: finished is Ready to look at."""
    if task is None:
        return None
    if task.status in FINISHED_STATUSES:
        return READY
    return ticket_status(task, None, language)["display_state"]


def _started_row(suggestion: ProactiveSuggestion, task: Optional[BoardTask], language: str) -> BriefRow:
    state = _became(task, language)
    return BriefRow(
        id=f"suggestion:{suggestion.id}",
        title=title_of(suggestion.title, language),
        why=say(language, _WHY.get(state, _STARTED)),
        at=iso(suggestion.decided_at),
        number=format_number(task.workspace_seq) if task is not None else None,
        display_state=state,
        opens=opens("ticket", task.id) if task is not None else opens("suggestion", suggestion.id),
    )


def _tickets(db: Session, workspace_id: Any, start: datetime, language: str) -> Tuple[int, List[BriefRow]]:
    found = db.execute(_FINISHED_TICKETS, {"ws": str(workspace_id), "finished": FINISHED_STATUSES, "since": start,
                                           "not_listed": NOT_LISTED, "limit": ROWS_READ}).all()
    rows = [BriefRow(id=f"ticket:{r.id}", title=title_of(r.title, language), why=say(language, _WHY[READY]),
                     at=iso(r.at), number=format_number(r.workspace_seq), display_state=READY,
                     opens=opens("ticket", r.id)) for r in found]
    return (int(found[0].of_all) if found else 0), rows


def _missions(db: Session, workspace_id: Any, start: datetime, language: str) -> Tuple[int, List[BriefRow]]:
    found = db.execute(_FINISHED_MISSIONS, {"ws": str(workspace_id), "completed": RunState.COMPLETED.value,
                                            "since": start, "limit": ROWS_READ}).all()
    rows = [BriefRow(id=f"mission:{r.id}", title=title_of(r.goal, language), why=say(language, _WHY[READY]),
                     at=iso(r.at), display_state=READY, opens=opens("mission", r.id)) for r in found]
    return (int(found[0].of_all) if found else 0), rows
