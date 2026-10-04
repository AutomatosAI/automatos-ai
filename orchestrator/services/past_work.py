"""F305 (night 9): an agent can still find past work, asked for by name, and it is
always labelled as an agent's, never the owner's facts.

With agent outputs out of the owner's documents (services/agent_output_scope.py) and
out of every default search (F269, services/agents_writing.py), the way to earlier
work is an explicit scope on the search an agent already uses: search_knowledge or
platform_search_documents with ``scope: "past_work"``.

The source is the board's done cards: each card's answer (``board_tasks.result``) as
the owner approved it, mission steps and playbook runs included. It is the cheapest
honest source: one workspace-scoped query, no embedding, and only approved work
(``status`` done), so a rejected round never comes back. Every result says "Written
by <agent> on <date>" and the block says it is not the owner's facts.
"""
from __future__ import annotations

import functools
import logging
import re
from typing import Any, Awaitable, Callable, Dict, List, Optional
from uuid import UUID

logger = logging.getLogger(__name__)

PAST_WORK_SCOPE = "past_work"
PAST_WORK_LIMIT = 5
PAST_WORK_MAX_LIMIT = 10
EXCERPT_CHARS = 600
_MAX_TERMS = 6
_TERM = re.compile(r"[\w'’-]{4,}", re.UNICODE)
DONE = "done"
PAST_WORK_NOTE = ("Past work: earlier answers agents wrote on approved cards. They are not the owner's facts or "
                  "rules. Use one only as an agent's earlier work, say who wrote it and when, and check facts "
                  "against the owner's documents or the live system.")
LABEL = "Written by {agent} on {day}: an agent's earlier answer, not the owner's facts."


def wants_past_work(params: Any) -> bool:
    """Whether a search asked for the past-work scope."""
    return isinstance(params, dict) and str(params.get("scope") or "").strip().lower() == PAST_WORK_SCOPE


def _terms(query: str) -> List[str]:
    seen: List[str] = []
    for word in _TERM.findall(query.lower()):
        if word not in seen:
            seen.append(word)
    return seen[:_MAX_TERMS]


def _limit(raw: Any) -> int:
    try:
        return max(1, min(int(raw), PAST_WORK_MAX_LIMIT))
    except (TypeError, ValueError):
        return PAST_WORK_LIMIT


def _done_cards(db: Any, workspace_id: Any, terms: List[str], limit: int) -> List[Any]:
    from sqlalchemy import or_

    from core.models.core import BoardTask

    matches = [c for t in terms for c in (BoardTask.title.ilike(f"%{t}%"), BoardTask.result.ilike(f"%{t}%"))]
    return (db.query(BoardTask)
            .filter(BoardTask.workspace_id == UUID(str(workspace_id)), BoardTask.status == DONE,
                    BoardTask.result.isnot(None), or_(*matches))
            .order_by(BoardTask.completed_at.desc().nullslast(), BoardTask.id.desc()).limit(limit).all())


def _agent_names(db: Any, cards: List[Any]) -> Dict[int, str]:
    from core.models import Agent

    ids = {c.assigned_agent_id for c in cards if c.assigned_agent_id}
    return dict(db.query(Agent.id, Agent.name).filter(Agent.id.in_(ids)).all()) if ids else {}


def _item(card: Any, number: Optional[str], agent: str) -> Dict[str, Any]:
    when = card.completed_at or card.updated_at
    day = when.date().isoformat() if when else "an unknown date"
    return {"card": number or f"ticket {card.id}", "title": card.title, "written_by": agent, "written_on": day,
            "label": LABEL.format(agent=agent, day=day), "excerpt": str(card.result)[:EXCERPT_CHARS],
            "owners_facts": False}


def search_past_work(db: Any, workspace_id: Any, query: str, limit: Any = None) -> Dict[str, Any]:
    """Done cards in the workspace whose title or answer has the query's words, newest
    first, each labelled with who wrote it and when."""
    from services.ticket_numbers import ticket_numbers

    terms = _terms(query or "")
    if not terms:
        return {"success": False, "error": "query is required (words of four letters or more)"}
    if workspace_id is None:
        return {"success": False, "error": "past work is searched in a workspace"}
    cards = _done_cards(db, workspace_id, terms, _limit(limit))
    numbers = ticket_numbers(db, UUID(str(workspace_id)), cards) if cards else {}
    names = _agent_names(db, cards)
    results = [_item(c, numbers.get(c.id), names.get(c.assigned_agent_id, "an agent")) for c in cards]
    return {"success": True, "scope": PAST_WORK_SCOPE, "note": PAST_WORK_NOTE, "results": results,
            "count": len(results)}


async def past_work_for_agent(db: Any, agent_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
    """search_knowledge's past-work scope, in the searching agent's workspace."""
    from services.agents_writing import _agents_workspace

    try:
        return search_past_work(db, _agents_workspace(db, agent_id), params.get("query") or "", params.get("limit"))
    except Exception:
        logger.exception("[F305] past-work search failed for agent %s", agent_id)
        return {"success": False, "error": "Past work could not be searched just now."}


Handler = Callable[[Any, Any, Dict[str, Any]], Awaitable[Dict[str, Any]]]


def or_past_work(handler: Handler) -> Handler:
    """Wrap platform_search_documents' handler: ``scope: "past_work"`` searches past
    work instead of the owner's documents."""
    @functools.wraps(handler)
    async def wrapped(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Dict[str, Any]:
        if not wants_past_work(params):
            return await handler(db, workspace_id, params)
        try:
            return search_past_work(db, workspace_id, params.get("query") or "", params.get("limit"))
        except Exception:
            logger.exception("[F305] past-work search failed in workspace %s", workspace_id)
            return {"success": False, "error": "Past work could not be searched just now."}
    return wrapped


__all__ = ["PAST_WORK_NOTE", "PAST_WORK_SCOPE", "or_past_work", "past_work_for_agent",
           "search_past_work", "wants_past_work"]
