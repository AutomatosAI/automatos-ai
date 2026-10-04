"""F273 (night 7b): a cancelled card says who cancelled it, and when, in its notes.

Night 7b's cancels all worked: two runs of "Weekly Instagram posts" (#0189,
#0190), the mission #0191 with its three step cards, the Inbox card #0193 and the
running task #0197. None of those cards said who cancelled it or when: their notes
were empty, and the mission's own card had no record of who at all. The mission
page and the playbook run said it, and Auto's own changes already leave a note on
the card they change (modules/tools/discovery/ticket_changes.py).

Every card a cancel stops now carries both: the record of who, why and when
(``runtime_ref["cancelled"]``, F015's shape, which the viewer's banner reads) and a
note among its notes (``runtime_ref.session_notes``, the list the ticket view
shows), worded like the other notes there. A person reads "you". When a chat turn
cancelled it, the agent chatting (Auto) is named and the note says "in chat". A
card the platform stopped on its own (a mission that ended, a routine switched
off, a playbook run that failed) names what stopped it, and why.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional, Tuple

CANCELLED_KEY = "cancelled"
# What the note says a person, or an agent, did to the card.
CANCELLED_THIS = "Cancelled this"
CANCELLED_THE_RUN = "Cancelled the playbook run"
CANCELLED_THE_MISSION = "Cancelled the mission"
WITH_ITS_RUN = "Cancelled with its playbook run"
WITH_ITS_MISSION = "Cancelled with its mission"
# A card the platform stopped on its own says why, as its record does (services/board_cancel).
STOPPED_BECAUSE = "Cancelled this because {reason}"
# Who, as the notes say it: the owner's own notes say "you" (services/ticket_verdict),
# Auto's changes "Auto", and "in chat" when a chat made them (ticket_changes).
YOU = "you"
AUTO = "Auto"
AN_AGENT = "An agent"
IN_CHAT = ", in chat"
# A person's record ("user:<id>", or the operator's), and an agent's own tool call with
# no person behind it (modules/tools/discovery/ticket_cancel). Any other ``by`` is the
# platform's own words: "the mission", "the routine", "the playbook run".
PERSON_PREFIX = "user:"
OPERATOR = "operator"
AGENTS_OWN_CALL = "platform_tool"


def with_cancel_recorded(db: Any, task: Any, *, by: str, reason: str, at: datetime,
                         note: str = CANCELLED_THIS) -> Dict[str, Any]:
    """``task``'s runtime_ref with its cancel on record, as a new dict: who, why and
    when (``cancelled``), and the note that says so, after the notes it has. ``note``
    is what a person or an agent did; a card the platform stopped on its own says
    why instead. Nothing is written: the caller stores the dict."""
    from services.cli_host_service import SESSION_NOTES_KEY

    ref = dict(task.runtime_ref or {})
    when = at.isoformat()
    notes = ref.get(SESSION_NOTES_KEY)
    entry = _note(db, task.workspace_id, by=str(by or ""), reason=reason, at=when, note=note)
    return {**ref, CANCELLED_KEY: {"by": by, "reason": reason, "at": when},
            SESSION_NOTES_KEY: [*(notes if isinstance(notes, list) else []), entry]}


def _note(db: Any, workspace_id: Any, *, by: str, reason: str, at: str, note: str) -> Dict[str, str]:
    """One note in the notes' shape (``{note, at, by}``, services/cli_host_service)."""
    if not _someone(by):
        return {"note": f"{STOPPED_BECAUSE.format(reason=reason)}.", "at": at, "by": by}
    who, where = _who(db, workspace_id, by)
    return {"note": f"{note}{where}.", "at": at, "by": who}


def _someone(by: str) -> bool:
    """A person, or an agent's own call: not the platform stopping a run by itself."""
    return _a_person(by) or by == AGENTS_OWN_CALL


def _a_person(by: str) -> bool:
    return by == OPERATOR or by.startswith(PERSON_PREFIX)


def _who(db: Any, workspace_id: Any, by: str) -> Tuple[str, str]:
    """Who cancelled, in the notes' words, and ", in chat" when a chat turn did it. The
    turn's usage scope names the lane and the agent chatting, as it does for Auto's
    other changes (modules/tools/discovery/ticket_changes)."""
    from core.llm.usage_context import LANE_CHAT, current_usage_scope

    scope = current_usage_scope() or {}
    in_chat = scope.get("request_type") == LANE_CHAT
    if _a_person(by) and not in_chat:
        return YOU, ""
    name = _agent_name(db, workspace_id, scope.get("agent_id"))
    return name or (AUTO if in_chat else AN_AGENT), (IN_CHAT if in_chat else "")


def _agent_name(db: Any, workspace_id: Any, agent_id: Any) -> Optional[str]:
    """The name of the workspace's agent ``agent_id``; None for no such agent."""
    if not agent_id:
        return None
    from core.models.core import Agent

    row = db.query(Agent.name).filter(Agent.id == agent_id, Agent.workspace_id == workspace_id).first()
    return row.name if row else None


__all__ = [
    "AN_AGENT", "AUTO", "CANCELLED_KEY", "CANCELLED_THE_MISSION", "CANCELLED_THE_RUN", "CANCELLED_THIS",
    "IN_CHAT", "STOPPED_BECAUSE", "WITH_ITS_MISSION", "WITH_ITS_RUN", "YOU", "with_cancel_recorded",
]
