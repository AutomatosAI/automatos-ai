"""PRD-268 §6.2.5, §6.9 — the board ticket a suggestion's Start files.

A ``board_task`` suggestion's Start files one ticket, as a person files one on the board:

* ``source_type = 'suggestion'``, ``source_id`` the suggestion's id, ``created_by_type =
  'user'`` and ``created_by_id`` the person who pressed Start;
* ``review_mode = 'human'`` (§12-Q5): the work waits in Review for a person;
* assigned to the agent the suggestion proposed (``proposed_action.payload.agent_id``) while
  that agent is still in the workspace, so it starts in Assigned; with none, in the Inbox;
* titled with the suggestion's title, its brief the person's Start as the instruction and
  the suggestion and its evidence quoted as data (``decision_text.ticket_brief``).

``file_ticket`` adds and flushes the row (its number is given on insert) and NOTIFYs the
Command Centres, both riding the caller's commit. Once that commit is made, ``wake`` does
what the board's create route does for a ticket filed and assigned at once: on the local
edition the person's click is the approval (``board_consent``), and the dispatch loop is
woken on its channel. Nothing here runs the ticket.
"""
from __future__ import annotations

from typing import Any, Optional

from sqlalchemy.orm import Session

from core.models.core import Agent, BoardTask
from core.models.proactive import ProactiveSuggestion
from modules.tools.discovery.suggestion_fields import PAYLOAD_AGENT
from services.proactive.decision_text import ticket_brief

__all__ = ["SOURCE_SUGGESTION", "file_ticket", "wake"]

SOURCE_SUGGESTION = "suggestion"
CREATED_BY_USER = "user"
REVIEW_BY_A_PERSON = "human"
ASSIGNED = "assigned"
INBOX = "inbox"
TASK_CREATED = "task_created"


def _assignee(db: Session, suggestion: ProactiveSuggestion) -> Optional[int]:
    """The proposed agent while it is still in the suggestion's workspace, else None."""
    payload = (suggestion.proposed_action or {}).get("payload") or {}
    agent_id = payload.get(PAYLOAD_AGENT) if isinstance(payload, dict) else None
    if not isinstance(agent_id, int) or isinstance(agent_id, bool):
        return None
    found = db.query(Agent.id).filter(Agent.id == agent_id, Agent.workspace_id == suggestion.workspace_id).first()
    return agent_id if found else None


def file_ticket(db: Session, suggestion: ProactiveSuggestion, created_by: Optional[str], locale: str) -> BoardTask:
    """The started suggestion's ticket, added and flushed (not committed)."""
    from services.board_events import notify_board_event

    agent_id = _assignee(db, suggestion)
    task = BoardTask(
        workspace_id=suggestion.workspace_id, title=suggestion.title,
        description=ticket_brief(locale, suggestion.title, suggestion.detail, suggestion.evidence),
        status=ASSIGNED if agent_id else INBOX, review_mode=REVIEW_BY_A_PERSON, assigned_agent_id=agent_id,
        created_by_type=CREATED_BY_USER, created_by_id=created_by,
        source_type=SOURCE_SUGGESTION, source_id=str(suggestion.id),
    )
    db.add(task)
    db.flush()
    notify_board_event(db, workspace_id=suggestion.workspace_id, task_id=task.id, status=task.status,
                       event=TASK_CREATED)
    return task


def wake(db: Session, task: Any, actor: str) -> None:
    """After the Start's commit: an assigned ticket's approval (local edition) and the dispatch
    loop's wake, committed. A ticket in the Inbox waits for a person to assign it."""
    from services.board_consent import WHY_CREATED_AND_ASSIGNED, consent_for_created_ticket
    from services.board_dispatcher import notify_task_available

    if task.status != ASSIGNED or not task.assigned_agent_id:
        return
    consent_for_created_ticket(db, workspace_id=task.workspace_id, task=task, actor=actor,
                               why=WHY_CREATED_AND_ASSIGNED)
    notify_task_available(db, workspace_id=task.workspace_id, task_id=task.id)
    db.commit()
