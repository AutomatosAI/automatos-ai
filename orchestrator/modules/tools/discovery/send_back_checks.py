"""P256-FIX-N17-2 (F444, night 17): a send-back's card is raised only when its click can run.

Night 17: two of the owner's clicks returned 200 and did nothing. #1767 had been filed by
platform_create_task with approval_action create_blog and no agent, so it was born in
Review; "turn #1767 down" became platform_update_task_status {status: 'assigned', note},
whose card ("send … back to its agent") the owner clicked, and the board's Reject refused
it ("Cannot reject a task with no assigned agent"). The grant was given back, the card
never moved, and the owner learned it only by asking. A working send-back would have been
wrong there too: approving #1767 runs a fixed create_blog topic that no redo can change.

So before the card is raised (``owner_only.platform_ask``: for the owner's click and for
the confirmation gate alike), each card the call hands to the board's Reject
(``board_sends_back``, the cards ``ticket_moves`` sends back) is read against the
Reject's own rule (``services.send_back_rules``), as the call would leave it: an agent the
edit tool gives the card in the same call counts.

- A card for the owner's approval (an ``approval_action``, or filed for their decision) is
  refused whatever agent is named: its approval does what it does, and the refusal names
  the cancel that turns it down.
- A card the Reject would refuse is refused in the Reject's words; with no agent, the
  refusal says how to name one.

A bulk move leaves off only the cards that cannot go back, and its ask says which; a call
whose own card cannot go back, or none of whose cards can, raises no card at all.

F457 (night 18): a card its agent is working on, taken back to Assigned or the Inbox
(``board_takes_back``), is not the Reject's: it is read against the board's own rule for
any move (``ticket_moves.board_refusal``), so a card whose run is still going is refused
before any card ("wait for its result, or cancel it"), as the board would refuse the click.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Tuple

TASK_ID, TASK_IDS, AGENT_ID = "task_id", "task_ids", "agent_id"
CREATE_BLOG, PUBLISH_BLOG = "create_blog", "publish_blog"  # the approval actions the board's Approve runs
STARTS_A_BLOG = "starts a blog mission on '{topic}'"
PUBLISHES_A_POST = "publishes its blog post"
RUNS_ITS_ACTION = "runs its '{kind}' action"
RECORDS_THE_DECISION = "records the owner's decision, and sends or orders nothing by itself"
FOR_THE_OWNERS_APPROVAL = ("is a card for the owner's approval{done}: approving it {does}, and an agent's redo "
                           "can't change that")
ALREADY_DONE = ", and it is Done"
TURN_IT_DOWN = (" To turn it down: platform_update_task_status {{task_id: \"{number}\", status: \"cancelled\"}} "
                "(waits for the owner's click).")
A_NEW_CARD = " For a different topic, file a new card with that topic."
IN_REVIEW = "review"
CANNOT_GO_BACK = "can't go back to an agent: {detail}"
NAME_THE_AGENT = (" Name the agent it goes to: platform_update_task {task_id, status: \"assigned\", agent_id, note: "
                  "the owner's words}, or turn it down with status \"cancelled\" (each waits for the owner's click).")
NOTHING_DONE = " Nothing was asked or done."
NONE_CAN_GO_BACK = "Nothing was asked or done: no card this call names can go back to an agent."
OTHERS_NOT_MOVED = (" The call's other cards were not moved either: call again without the cards named here to "
                    "move them.")
LEFT_OFF = "Left off this card: {card}, which {why}."
LEFT_OUT = "left_out"


class _Refused(NamedTuple):
    """One card the Reject would not take: its number, how a sentence names it, why, and what to do instead."""
    number: str
    named: str
    why: str
    advice: str

    def sentence(self) -> str:
        return f"{self.named} {self.why}."


def board_sends_back(db: Any, workspace_id: Any, action: str, params: Any) -> List[Any]:
    """The worked cards this call hands to the board's Reject, in this workspace: those
    ``send_backs`` reads as sent back that ``ticket_moves.sends_it_back`` sends back (to
    Assigned, by the send_back flag or a new brief, or to In progress with a note). A move
    to the Inbox, or to In progress with no note, is not one."""
    from modules.tools.discovery.owner_only import CARD_MOVES
    from modules.tools.discovery.send_backs import cards_sent_back, moved_to
    from modules.tools.discovery.ticket_moves import sends_it_back

    if action not in CARD_MOVES or not isinstance(params, dict):
        return []
    new_status = moved_to(action, params)
    return [card for card in cards_sent_back(db, workspace_id, action, params)
            if sends_it_back(card, new_status, params)]


def board_takes_back(db: Any, workspace_id: Any, action: str, params: Any) -> List[Any]:
    """F457: the cards this call takes off the agent working on them (In progress, sent back
    to Assigned or the Inbox), in this workspace; the Reject takes none of them."""
    from modules.tools.discovery.owner_only import AT_WORK_STATUSES, CARD_MOVES
    from modules.tools.discovery.send_backs import cards_sent_back

    if action not in CARD_MOVES or not isinstance(params, dict):
        return []
    return [card for card in cards_sent_back(db, workspace_id, action, params) if card.status in AT_WORK_STATUSES]


def sendable_back(db: Any, workspace_id: Any, action: str,
                  params: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], List[Dict[str, str]]]:
    """(the call with only the cards that can go back, the refusal when the card it names
    cannot or none it sends back can, the cards a bulk move leaves off as [{card, why}]).
    The card is raised for send-backs only: a bulk move whose send-backs all fail is
    refused whole, saying its other cards (an Inbox card given to its agent) move once
    it is called again without them."""
    sent = [*board_sends_back(db, workspace_id, action, params), *board_takes_back(db, workspace_id, action, params)]
    refusals = {card.id: _refused(db, action, params, card) for card in sent}
    failed = {task_id: why for task_id, why in refusals.items() if why is not None}
    if not failed:
        return params, None, []
    kept = _kept(db, workspace_id, params, failed)
    if kept is None or len(failed) == len(sent):
        return params, {"success": False, "error": _refusal_text(list(failed.values()), others=bool(kept))}, []
    left_out = [{"card": why.number, "why": why.why} for why in failed.values()]
    return {**params, TASK_IDS: kept}, None, left_out


def left_off(left_out: Iterable[Dict[str, str]]) -> str:
    """The ask's words for the cards a bulk send-back leaves off: " Left off this card: #N, which …"."""
    return "".join(f" {LEFT_OFF.format(card=item['card'], why=item['why'])}" for item in left_out)


def _refused(db: Any, action: str, params: Dict[str, Any], card: Any) -> Optional[_Refused]:
    """Why the board's Reject would not take ``card`` from this call, or None."""
    from modules.tools.discovery.owner_only import AT_WORK_STATUSES
    from modules.tools.discovery.send_backs import EDIT_ACTION
    from services.send_back_rules import NO_AGENT, send_back_refusal
    from services.ticket_numbers import ticket_label, ticket_number

    number = ticket_number(db, card) or str(card.id)
    named = f"{ticket_label(card, number, capital=True)} ('{card.title}')"
    if card.status in AT_WORK_STATUSES:
        return _at_work_refused(db, action, params, card, (number, named))
    if _for_the_owners_approval(card):
        return _approval_refused(card, number, named)
    refused = send_back_refusal(db, _as_given(params, card) if action == EDIT_ACTION else card)
    if refused is None:
        return None
    detail = refused[1].rstrip(".")
    return _Refused(number, named, CANNOT_GO_BACK.format(detail=detail), NAME_THE_AGENT if detail == NO_AGENT else "")


def _at_work_refused(db: Any, action: str, params: Dict[str, Any], card: Any, said: Tuple[str, str]
                     ) -> Optional[_Refused]:
    """F457: why the board would not take ``card``, which its agent is working on, off it (a
    run still going waits for its result or is cancelled), or None. ``said``: (number, named)."""
    from modules.tools.discovery.send_backs import moved_to
    from modules.tools.discovery.ticket_moves import board_refusal

    refused = board_refusal(db, card, moved_to(action, params))
    if refused is None:
        return None
    number, named = said
    why = CANNOT_GO_BACK.format(detail=refused.rstrip("."))
    return _Refused(number, named, why, TURN_IT_DOWN.format(number=number))


def _approval_refused(card: Any, number: str, named: str) -> _Refused:
    """An approval card: what its Approve does; in Review, the cancel that turns it down
    (a Done card's approval has already been given: nothing is left to turn down)."""
    in_review = card.status == IN_REVIEW
    why = FOR_THE_OWNERS_APPROVAL.format(done="" if in_review else ALREADY_DONE, does=_what_approving_does(card))
    advice = (TURN_IT_DOWN.format(number=number) if in_review else "") + A_NEW_CARD
    return _Refused(number, named, why, advice)


def _for_the_owners_approval(card: Any) -> bool:
    """A card whose Approve does something of its own (``approval_action``), or one filed
    for the owner's decision (``approval_cards``): no agent's redo changes what it approves."""
    from modules.tools.discovery.approval_cards import APPROVAL_ACTION, is_an_approval_card

    planning = card.planning_data if isinstance(card.planning_data, dict) else {}
    return bool(planning.get(APPROVAL_ACTION)) or is_an_approval_card(planning)


def _what_approving_does(card: Any) -> str:
    """What the board's Approve does with this card (``api.board_tasks._run_approval_action``)."""
    from modules.tools.discovery.approval_cards import APPROVAL_ACTION
    from modules.tools.discovery.card_question_text import shown

    action = card.planning_data.get(APPROVAL_ACTION)
    kind = action.get("type") if isinstance(action, dict) else None
    if kind == CREATE_BLOG:
        return STARTS_A_BLOG.format(topic=shown(action.get("topic")))
    if kind == PUBLISH_BLOG:
        return PUBLISHES_A_POST
    return RUNS_ITS_ACTION.format(kind=kind) if kind else RECORDS_THE_DECISION


def _as_given(params: Dict[str, Any], card: Any) -> Any:
    """``card`` as the edit tool leaves it before its move: with the agent the call gives it
    (bound to an id before the card, FX-010)."""
    from modules.tools.discovery.agent_refs import agent_id_said
    from services.board_drag_rules import as_patched

    agent_id = agent_id_said(params.get(AGENT_ID))
    return as_patched(card, assigned_agent_id=agent_id) if agent_id is not None else card


def _kept(db: Any, workspace_id: Any, params: Dict[str, Any], failed: Dict[int, _Refused]) -> Optional[List[Any]]:
    """The call's task_ids without the cards that cannot go back; None when its own
    task_id cannot, or it lists no cards."""
    listed = params.get(TASK_IDS)
    if _id_of(db, workspace_id, params.get(TASK_ID)) in failed or not isinstance(listed, list):
        return None
    return [ref for ref in listed if _id_of(db, workspace_id, ref) not in failed]


def _id_of(db: Any, workspace_id: Any, ref: Any) -> Optional[int]:
    from services.ticket_refs import ticket_id_named

    return ticket_id_named(db, workspace_id, ref)[0] if ref not in (None, "") else None


def _refusal_text(failed: List[_Refused], *, others: bool = False) -> str:
    """The refusal naming each card that cannot go back and what to do instead, and
    (``others``) that the call's other cards were not moved either."""
    if len(failed) == 1:
        text = f"{failed[0].sentence()}{NOTHING_DONE}{failed[0].advice}"
    else:
        text = " ".join([NONE_CAN_GO_BACK, *(f"{why.sentence()}{why.advice}" for why in failed)])
    return text + (OTHERS_NOT_MOVED if others else "")


__all__ = ["LEFT_OUT", "board_sends_back", "board_takes_back", "left_off", "sendable_back"]
