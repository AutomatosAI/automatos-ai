"""POST /api/v1/tasks: what a request may file, and what the create answers (F276).

F276 (night 7b, report friction 3): every card the owner filed on the board
(#0178 to #0181, #0197) came back with ``number: null``, so to learn a new card's
number they had to list the board. The create answered the row's own columns
(``BoardTask.to_dict``), while the list and the ticket view serve every ticket
through ``board_dict``, its number included. The create now answers as they do.
Auto's ``platform_create_task`` already answered with the number (#0182), through
``services.ticket_refs.by_ticket_number``.

What a request may file is checked here, before anything is written. It moved
out of api/board_tasks.py, which is over 800 lines and does not grow.

F293 (night 8): a new card has had no work on it, so it never starts in Review or
Done. Auto made #0251 and #0386 straight into Review, with no run and no answer, and
Needs you counted them as the owner's to judge. The create took no status, so a
request that asked for Review was quietly filed in the Inbox; it is now refused and
told where a new card goes. A card that carries an action for the owner to approve
(publish a post) still starts in Review: that approval is what it asks for.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import HTTPException
from sqlalchemy.orm import Session

from api.board_tasks import (
    USER_CREATABLE_SOURCE_TYPES, VALID_PRIORITIES, VALID_REVIEW_MODES, _agent_id_of, _one_of, _text_of,
)
from core.models import Agent
from core.models.core import BoardTask
from services.board_task_view import board_dict
from services.ticket_numbers import ticket_number

DEFAULT_PRIORITY = "medium"
DEFAULT_REVIEW_MODE = "auto"
DEFAULT_SOURCE_TYPE = "user"
NO_SUCH_AGENT = "Assigned agent not found in workspace"
# F293: the columns that hold finished work, which no new card starts in.
FINISHED_COLUMNS = {"review": "Review", "done": "Done"}
NOTHING_DONE_YET = ("A new card has had no work on it yet, so it can't start in {column}: it goes to the Inbox, "
                    "or to Assigned when it has an agent, and comes to Review once its agent has worked on it. "
                    "Nothing was filed.")


def new_ticket_columns(db: Session, workspace_id: Any, body: Dict[str, Any]) -> Dict[str, Any]:
    """The columns a request's ``body`` gives a new ticket. Every field is checked
    before anything is written: a field of the wrong kind is a 422 (F195), and so
    is a kind of ticket only the platform files (F194); an agent of another
    workspace is a 404."""
    title = _text_of(body.get("title"), "title")
    if not title:
        raise HTTPException(status_code=422, detail="title is required")
    agent_id = _assignee(db, workspace_id, body.get("assigned_agent_id"))
    priority = _named(body, "priority", VALID_PRIORITIES, DEFAULT_PRIORITY)
    review_mode = _named(body, "review_mode", VALID_REVIEW_MODES, DEFAULT_REVIEW_MODE)
    planning_data = _planning_data(body)
    _no_finished_start(body.get("status"), planning_data)
    attachment_ids = _attachments(body)
    return {
        "title": title, "description": body.get("description"), "raw_prompt": body.get("raw_prompt"),
        "status": _first_status(planning_data, agent_id), "priority": priority, "review_mode": review_mode,
        "assigned_agent_id": agent_id, "parent_task_id": body.get("parent_task_id"),
        "source_type": _source_type(body), "source_id": body.get("source_id"), "tags": body.get("tags", []),
        "planning_data": planning_data, "attachment_ids": attachment_ids,
    }


def created_ticket_answer(db: Session, task: BoardTask) -> Dict[str, Any]:
    """What the create answers: the new ticket as the board lists it, with its
    number (#0042), so nobody lists the board to learn it."""
    return board_dict(task, ticket_number(db, task))


def _assignee(db: Session, workspace_id: Any, raw: Any) -> Optional[int]:
    """The agent the ticket is given to, one of this workspace's; None for none."""
    agent_id = _agent_id_of(raw)
    if agent_id is None:
        return None
    agent = db.query(Agent).filter(Agent.id == agent_id, Agent.workspace_id == workspace_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail=NO_SUCH_AGENT)
    return agent_id


def _named(body: Dict[str, Any], field: str, allowed: Any, default: str) -> str:
    """A field that names one of ``allowed``: a list or an object is a 422, not the
    500 its lookup raised (F195)."""
    value = body.get(field, default)
    if not _one_of(value, allowed):
        raise HTTPException(status_code=422, detail=f"Invalid {field}: {value}")
    return value


def _planning_data(body: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The ticket's planning data: an object, or none. Anything else is a 422: the
    board reads it as an object wherever it shows the ticket (services/ticket_redo)."""
    planning_data = body.get("planning_data")
    if planning_data is not None and not isinstance(planning_data, dict):
        raise HTTPException(status_code=422, detail="planning_data must be an object, or null")
    return planning_data


def _no_finished_start(requested: Any, planning_data: Optional[Dict[str, Any]]) -> None:
    """A request that asks for a finished column is a 422 (F293: #0251, #0386), unless
    the card carries an action for the owner to approve, which starts it in Review."""
    column = FINISHED_COLUMNS.get(requested) if isinstance(requested, str) else None
    if column and not (planning_data and planning_data.get("approval_action")):
        raise HTTPException(status_code=422, detail=NOTHING_DONE_YET.format(column=column))


def _first_status(planning_data: Optional[Dict[str, Any]], agent_id: Optional[int]) -> str:
    """In Review when it carries an action for the owner to approve; otherwise
    assigned when it has an agent, and in the Inbox when it has none."""
    if planning_data and planning_data.get("approval_action"):
        return "review"
    return "assigned" if agent_id else "inbox"


def _attachments(body: Dict[str, Any]) -> List[Any]:
    """PRD-127: the ticket's ephemeral attachments, a list."""
    attachment_ids = body.get("attachment_ids", [])
    if attachment_ids and not isinstance(attachment_ids, list):
        raise HTTPException(status_code=422, detail="attachment_ids must be a list")
    return attachment_ids


def _source_type(body: Dict[str, Any]) -> str:
    """What the ticket came from: 'user' when the request does not say. PRD-221 S14:
    a Command Centre activity card says 'activity', so the board card links back.
    F194: only a person's kinds. A mission's or a playbook's step, a session or a
    lane's ticket is filed by the platform: a request claiming one made a ticket
    the dispatcher and the host treat as the platform's own."""
    source_type = body.get("source_type") or DEFAULT_SOURCE_TYPE
    if not isinstance(source_type, str):
        raise HTTPException(status_code=422, detail=(
            f"source_type is one of {sorted(USER_CREATABLE_SOURCE_TYPES)}, or left out."))
    if source_type not in USER_CREATABLE_SOURCE_TYPES:
        raise HTTPException(status_code=422, detail=(
            f"source_type '{source_type}' is filed by the platform, not by a request. "
            f"Use one of {sorted(USER_CREATABLE_SOURCE_TYPES)}, or leave it out."))
    return source_type


__all__ = ["created_ticket_answer", "new_ticket_columns"]
