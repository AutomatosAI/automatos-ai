"""P256-FIX-T3 (Gerard, 11 Oct, benchmark N5-a): a send-back from chat waits for the owner's click.

A card its agent has worked on (Review or Done) moved back to Assigned, In progress or the
Inbox for rework ('turn it down', 'redo it', 'send #1219 back') is owner-only, as approve and
cancel are. Before, ``owner_only.is_owner_only`` read a card move as owner-only only when it
closed the card, so platform_update_task_status {status: 'assigned'} on a Review card ran at
once: 'I've sent #1219 back … It's now in Assigned', and no card.

The rule is ``owner_only.sends_back`` (beside ``closing_status``); this module reads what it
needs from the board, in the caller's workspace: each card's current status, and for the
card the owner reads, the agent it goes back to and the note it carries. Giving a new inbox
card to an agent (inbox → assigned, the hand-off lane) stays free: an inbox card is not worked.

P256-FIX-N17-1 (F438, nights 16–17): a new brief on a worked card is the board's Re-brief, which
sends the card back to its agent whatever status the call names (FX-013 drops that status), so
the Re-brief is the move: platform_update_task {description} on a Review card is a send-back and
waits for the click. Before, a new brief was read as no move at all, and "get it rewritten and
bring it back" re-ran #1713 with no card. A description made only of status orders is no new brief.

F457 (night 18, ask 837): a card its agent is working on (In progress), taken back to Assigned or
the Inbox, is a send-back too. "Turn #1836 down" moved an In progress card to Assigned with no card,
and the board's dispatcher started it over. A card whose run is still going is refused before any
card, in the board's words (``send_back_checks``): the board would refuse the move after the click.
A new brief on an In progress card is no send-back: the board's Re-brief refuses every In progress
card (``api.board_task_rebrief``), so the handler refuses that call at once, and no card is raised.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from modules.tools.discovery.card_question_text import shown

EDIT_ACTION = "platform_update_task"
SEND_BACK_FLAG = "send_back"
SEND_BACK_TO = "assigned"  # where the board's Reject puts the card (ticket_moves.SEND_BACK)
NOTE = "note"
DESCRIPTION = "description"
ACT = "send {what} back to {agent} {words}"
REBRIEF_ACT = "give {what} a new brief and send it back to {agent} {words}"  # N17-1: the Re-brief
WITH_THE_NOTE = "with your note: {note}"
WITH_NO_NOTE = "with no note"
ITS_AGENT = "its agent"
THEIR_AGENTS = "their agents"
# N17-1: what the model is told of the calls that send a worked card back (follows_the_owner,
# actions_board_tasks): they wait for the owner's click in a person's chat.
WAITS_FOR_THE_CLICK = "It waits for the owner's click on the approval card."
REBRIEF_WAITS = ("On a card in Review or Done this is the Re-brief: it goes back to its agent, and it waits for "
                 "the owner's click on the approval card.")


def moved_to(action: str, params: Dict[str, Any]) -> Optional[str]:
    """The status a card move sends its cards to, read the board's way ('send back' is
    Assigned, and so is a Re-brief, whose call's status is dropped, FX-013); None when the
    call names none."""
    from modules.tools.execution.call_effects import STATUS_WORDS

    if params.get(SEND_BACK_FLAG) or rebriefs(action, params):
        return SEND_BACK_TO
    status = str(params.get("status") or "").strip().lower()
    return STATUS_WORDS.get(status, status) or None


def rebriefs(action: str, params: Dict[str, Any]) -> bool:
    """Whether the call gives its card a new brief (platform_update_task's description, less
    the sentences that only move the card): on a worked card, the board's Re-brief."""
    from modules.tools.discovery.ticket_edits import new_brief  # lazy: ticket_edits imports owner_only

    return action == EDIT_ACTION and bool(new_brief(params))


def sends_back_a_card(db: Any, workspace_id: Any, action: str, params: Dict[str, Any]) -> bool:
    """Whether the move sends back a card its agent has worked on or is working on (``owner_only.sends_back``),
    each card's status read from this workspace's board. Without the board (no session), no."""
    return bool(cards_sent_back(db, workspace_id, action, params))


def send_back_act(db: Any, workspace_id: Any, action: str, params: Dict[str, Any], what: str) -> Optional[str]:
    """What the card asks for a send-back: "send 'Wholesale reply' (ticket #1219) back to
    Content Creator with your note: …", or for a Re-brief (N17-1) "give … a new brief and
    send it back to …"; None when the call sends nothing back. ``send_back`` comes first,
    then a new brief, then the status the call names."""
    cards = cards_sent_back(db, workspace_id, action, params)
    if not cards:
        return None
    note = str(params.get(NOTE) or "").strip()
    if not note and params.get(SEND_BACK_FLAG):
        note = str(params.get(DESCRIPTION) or "").strip()  # F279: the owner's words, sent as a description
    words = WITH_THE_NOTE.format(note=shown(note)) if note else WITH_NO_NOTE
    agent = _agent_given(db, workspace_id, params) or _agent_called(db, workspace_id, cards)
    act = REBRIEF_ACT if not params.get(SEND_BACK_FLAG) and rebriefs(action, params) else ACT
    return act.format(what=what, agent=agent, words=words)


def cards_sent_back(db: Any, workspace_id: Any, action: str, params: Dict[str, Any]) -> List[Any]:
    """The worked or at-work cards this move sends back, in this workspace; a mission's own
    card is decided on the mission's page, never here."""
    from api.board_tasks import MISSION_CARD_SOURCE

    new_status = moved_to(action, params)
    if db is None or new_status is None:
        return []
    rebrief = rebriefs(action, params) and not params.get(SEND_BACK_FLAG)
    return [card for card in _cards(db, workspace_id, params)
            if _goes_back(card.status, new_status, rebrief) and card.source_type != MISSION_CARD_SOURCE]


def _goes_back(card_status: Any, new_status: str, rebrief: bool) -> bool:
    """``owner_only.sends_back``, less a Re-brief of an In progress card (F457): the board refuses that."""
    from modules.tools.discovery.owner_only import AT_WORK_STATUSES, sends_back

    return sends_back(card_status, new_status) and not (rebrief and card_status in AT_WORK_STATUSES)


def _cards(db: Any, workspace_id: Any, params: Dict[str, Any]) -> List[Any]:
    """The cards the call names (task_id, and each of task_ids), read by their numbers as
    the ticket tools read them (``ticket_id_named``)."""
    from core.models.core import BoardTask
    from services.ticket_refs import ticket_id_named

    listed = params.get("task_ids") if isinstance(params.get("task_ids"), list) else []
    refs = [ref for ref in [params.get("task_id"), *listed] if ref not in (None, "")]
    ids = [i for i in (ticket_id_named(db, workspace_id, ref)[0] for ref in refs) if i is not None]
    if not ids:
        return []
    return db.query(BoardTask).filter(BoardTask.id.in_(ids), BoardTask.workspace_id == workspace_id).all()


def _agent_given(db: Any, workspace_id: Any, params: Dict[str, Any]) -> Optional[str]:
    """The agent the call also gives the card to (platform_update_task's agent_id, bound
    from a name before the card, FX-010), by name: the click sends it there."""
    from modules.tools.discovery.agent_refs import agent_by_id, agent_id_said

    agent_id = agent_id_said(params.get("agent_id"))
    agent = agent_by_id(db, workspace_id, agent_id)[0] if agent_id is not None else None
    return str(agent.name) if agent is not None else None


def _agent_called(db: Any, workspace_id: Any, cards: List[Any]) -> str:
    """The agent a card goes back to, by name; 'their agents' for several."""
    from core.models import Agent

    agents = {card.assigned_agent_id for card in cards}
    if len(agents) != 1 or None in agents:
        return THEIR_AGENTS if len(cards) > 1 else ITS_AGENT
    row = db.query(Agent.name).filter(Agent.id == agents.pop(), Agent.workspace_id == workspace_id).first()
    return str(row.name) if row is not None else ITS_AGENT


__all__ = ["REBRIEF_WAITS", "WAITS_FOR_THE_CLICK", "cards_sent_back", "moved_to", "rebriefs", "send_back_act",
           "sends_back_a_card"]
