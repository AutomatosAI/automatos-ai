"""PRD-256 P256-FIX-N14-7 (F421): a card Auto files for the owner's approval of an order or a
send is filed where the board's Approve acts on it.

Night 14: Auto filed #1440, an order for the owner to approve, from the owner's chat with no
agent on it. D7 held it for a person (``brief_sends``: its brief orders, so ``review_mode:
human``), but it was filed in the Inbox, where nothing could move it on: no agent works an
unassigned card, the board's Approve refuses an Inbox card ("no one has worked on it yet"),
and Auto's own move to Review was refused for the same reason (``board_drag_rules``: a
column of finished work needs an answer on the card).

Such a card is the owner's decision itself, so it is filed in Review, as a card with an
``approval_action`` is: the server marks its ``planning_data`` (``OWNERS_DECISION``), which
``handlers_board_tasks.initial_board_status`` files in Review past F289's rule (a model's own
``status: review`` is still dropped, ``new_card_checks``), and a mark the call carries is
dropped first, so only this check sets it. The owner's Approve moves it to Done (or Cancel calls
it off), and the review notice goes out as for any card that lands there. The card is the
decision, not the work, so Done asks it for no artifact (``done_needs_an_artifact``).

It is that card only: filed from a person's chat (the server-injected driver), with no agent
named, held for a person, a brief that sends or orders (D7's own reading,
``brief_sends.sends_or_orders``) and a brief that asks for an approval (``APPROVAL_WORDS``, read
from Auto's brief, never the owner's words). A card with an agent keeps its path (the agent
works it, then it waits in Review), and so does any other unassigned card: "Email the supplier"
stays in the Inbox, where an agent can take it on. The answer says where it was filed and that
the approval itself sends or orders nothing.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Tuple

from modules.tools.discovery.brief_sends import BRIEF_FIELDS, DRIVER, HUMAN, REVIEW_MODE, sends_or_orders

REVIEW = "review"
STATUS = "status"
PLANNING = "planning_data"
APPROVAL_ACTION = "approval_action"
# The server's mark on a card filed for the owner's decision (never the model's: dropped from the call first).
OWNERS_DECISION = "owners_decision"
AGENT_KEYS = ("assigned_agent_name", "agent_id")
FILED_FOR_APPROVAL = "filed_for_approval"
# A brief that asks for the owner's approval ("Approve the order", "for sign-off"), as words of the brief.
APPROVAL_WORDS = frozenset({"approve", "approval", "approvals", "authorise", "authorize", "authorisation",
                            "authorization", "signoff"})
SIGN_OFF = ("sign", "off")
_WORD = re.compile(r"[a-z]+")
APPROVAL_NOTE = ("No agent works this card, so it is filed in Review, where the owner approves it: their Approve on "
                 "the board moves it to Done, or Cancel calls it off. Approving records their decision; it sends "
                 "or orders nothing by itself. Tell them so.")


def filed_for_approval(params: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    """``params`` marked for Review (``OWNERS_DECISION`` in its planning_data) when the card
    is the owner's approval of an order or a send; and whether it was. A mark the call
    itself carries is always dropped."""
    params = _without_a_mark(params)
    if not _an_approval_of_a_send(params):
        return params, False
    planning = params.get(PLANNING) if isinstance(params.get(PLANNING), dict) else {}
    return {**params, PLANNING: {**planning, OWNERS_DECISION: True}}, True


def is_an_approval_card(planning_data: Any) -> bool:
    """Whether a card's planning_data carries the server's mark: it was filed for the owner's decision."""
    return isinstance(planning_data, dict) and planning_data.get(OWNERS_DECISION) is True


def _an_approval_of_a_send(params: Dict[str, Any]) -> bool:
    """A person's chat, no agent, held for a person, no approval_action of its own, and a
    brief that sends or orders and asks for an approval."""
    if not params.get(DRIVER) or params.get(REVIEW_MODE) != HUMAN or params.get(APPROVAL_ACTION):
        return False
    if any(params.get(key) not in (None, "") for key in AGENT_KEYS):
        return False
    brief = [params.get(field) for field in BRIEF_FIELDS]
    return sends_or_orders(*brief) and asks_for_approval(*brief)


def _without_a_mark(params: Dict[str, Any]) -> Dict[str, Any]:
    """The call with no ``OWNERS_DECISION`` mark in its planning_data: only the server sets it."""
    planning = params.get(PLANNING)
    if not isinstance(planning, dict) or OWNERS_DECISION not in planning:
        return params
    return {**params, PLANNING: {key: value for key, value in planning.items() if key != OWNERS_DECISION}}


def asks_for_approval(*texts: Any) -> bool:
    """Whether any of ``texts`` asks for an approval: an approval word, or "sign off"."""
    for text in texts:
        words = _WORD.findall(text.lower()) if isinstance(text, str) else []
        if set(words) & APPROVAL_WORDS or any(tuple(words[i:i + 2]) == SIGN_OFF for i in range(len(words))):
            return True
    return False


def says_it_waits_for_approve(result: Any, filed: bool) -> Any:
    """The answer of a card ``filed_for_approval`` put in Review, saying so; any other as it is."""
    if filed and isinstance(result, dict) and result.get("success") and result.get(STATUS) == REVIEW:
        return {**result, FILED_FOR_APPROVAL: True, "approval_note": APPROVAL_NOTE}
    return result


__all__ = ["APPROVAL_NOTE", "APPROVAL_WORDS", "FILED_FOR_APPROVAL", "OWNERS_DECISION", "asks_for_approval",
           "filed_for_approval", "is_an_approval_card", "says_it_waits_for_approve"]
